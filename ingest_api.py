import os
import hmac
import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import requests
import chromadb
from chromadb.config import Settings

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# -----------------------------
# Logging
# -----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest_api")


# -----------------------------
# Environment
# -----------------------------

# Security
# Environment="INGEST_API_KEY=REPLACE_WITH_A_LONG_RANDOM_KEY"
INGEST_API_KEY = os.getenv("INGEST_API_KEY", "").strip()

# Chroma (HTTP mode)
CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "7000"))
CHROMA_SSL = os.getenv("CHROMA_SSL", "false").lower() in ("1", "true", "yes", "on")

# Embeddings (Ollama by default)
# Typical Ollama embeddings endpoint:
#   http://127.0.0.1:11434/api/embeddings
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/embeddings")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# Ingest behavior
DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", "default")
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "1800"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(
    title="RAG Ingest API (Locked)",
    version="2.0.0",
    description="Ingestion-only API for ChromaDB (API-key protected)."
)


# -----------------------------
# Auth dependency
# -----------------------------
def _extract_api_key(request: Request) -> str:
    # 1) X-API-Key
    key = request.headers.get("x-api-key", "").strip()
    if key:
        return key

    # 2) Authorization: Bearer <key>
    auth = request.headers.get("authorization", "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()

    return ""


def require_ingest_key(request: Request) -> None:
    # If you forgot to set INGEST_API_KEY, fail closed (safe default)
    if not INGEST_API_KEY:
        logger.error("INGEST_API_KEY is not set. Refusing requests (fail-closed).")
        raise HTTPException(status_code=500, detail="Server misconfigured: INGEST_API_KEY not set")

    presented = _extract_api_key(request)
    if not presented or not hmac.compare_digest(presented, INGEST_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")


def ingest_auth(request: Request):
    # Dependency wrapper so we can use Depends()
    require_ingest_key(request)


# -----------------------------
# Chroma client
# -----------------------------
def get_chroma_client() -> chromadb.HttpClient:
    # chromadb.HttpClient supports ssl=True for https
    return chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        ssl=CHROMA_SSL,
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(client: chromadb.HttpClient, name: str):
    try:
        return client.get_collection(name=name)
    except Exception:
        return client.get_or_create_collection(name=name)


# -----------------------------
# Embeddings
# -----------------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Calls OLLAMA_URL for embeddings. For Ollama, payload is:
      {"model":"nomic-embed-text","prompt":"..."}
    Returns list of vectors in same order.
    """
    vectors: List[List[float]] = []
    for t in texts:
        payload = {"model": EMBED_MODEL, "prompt": t}
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
            r.raise_for_status()
            data = r.json()
            vec = data.get("embedding")
            if not vec:
                raise RuntimeError(f"Embedding missing in response: {data}")
            vectors.append(vec)
        except Exception as e:
            logger.exception("Embedding request failed")
            raise HTTPException(status_code=502, detail=f"Embedding service error: {str(e)}") from e
    return vectors


# -----------------------------
# Chunking helpers
# -----------------------------
def chunk_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    if max_chars <= 0:
        return [text]

    chunks: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        end = min(i + max_chars, n)
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        i = max(0, end - overlap_chars)

    return chunks


def normalize_metadata(md: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not md:
        return {}
    # Ensure JSON-serializable (Chroma metadata is picky)
    safe: Dict[str, Any] = {}
    for k, v in md.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            safe[k] = v
        else:
            safe[k] = json.dumps(v, default=str)
    return safe


# -----------------------------
# Models
# -----------------------------
class IngestDoc(BaseModel):
    id: Optional[str] = Field(default=None, description="Optional stable id. If omitted, server generates.")
    text: str = Field(..., description="Document text to ingest")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata dict")


class IngestRequest(BaseModel):
    collection: str = Field(default=DEFAULT_COLLECTION, description="Chroma collection name")
    docs: List[IngestDoc] = Field(..., description="Docs to ingest")


class IngestResponse(BaseModel):
    collection: str
    received_docs: int
    inserted_chunks: int
    elapsed_ms: int
    timestamp: str


# -----------------------------
# Error handler
# -----------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


# -----------------------------
# Public (no auth)
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ingest",
        "time": datetime.utcnow().isoformat() + "Z",
    }


# -----------------------------
# Protected endpoints (API key required)
# -----------------------------
@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(ingest_auth)])
def ingest(req: IngestRequest):
    start = time.time()

    collection_name = (req.collection or DEFAULT_COLLECTION).strip()
    if not collection_name:
        raise HTTPException(status_code=400, detail="collection is required")

    if not req.docs:
        raise HTTPException(status_code=400, detail="docs cannot be empty")

    client = get_chroma_client()
    col = get_or_create_collection(client, collection_name)

    # Build chunks + ids + metadata
    all_texts: List[str] = []
    all_ids: List[str] = []
    all_metas: List[Dict[str, Any]] = []

    inserted_chunks = 0
    doc_index = 0

    for d in req.docs:
        doc_index += 1
        base_id = (d.id or f"doc-{int(time.time()*1000)}-{doc_index}")
        md = normalize_metadata(d.metadata)
        md.setdefault("source", "api")
        md.setdefault("ingested_at", datetime.utcnow().isoformat() + "Z")

        chunks = chunk_text(d.text, MAX_CHUNK_CHARS, CHUNK_OVERLAP_CHARS)
        if not chunks:
            continue

        for ci, ctext in enumerate(chunks, start=1):
            chunk_id = f"{base_id}-chunk-{ci}"
            chunk_md = dict(md)
            chunk_md["chunk_index"] = ci
            chunk_md["chunk_total"] = len(chunks)

            all_texts.append(ctext)
            all_ids.append(chunk_id)
            all_metas.append(chunk_md)

        inserted_chunks += len(chunks)

    if not all_texts:
        raise HTTPException(status_code=400, detail="No ingestable text after chunking")

    embeddings = embed_texts(all_texts)

    try:
        col.add(
            ids=all_ids,
            documents=all_texts,
            metadatas=all_metas,
            embeddings=embeddings,
        )
    except Exception as e:
        logger.exception("Chroma add() failed")
        raise HTTPException(status_code=502, detail=f"Chroma error: {str(e)}") from e

    elapsed_ms = int((time.time() - start) * 1000)
    return IngestResponse(
        collection=collection_name,
        received_docs=len(req.docs),
        inserted_chunks=inserted_chunks,
        elapsed_ms=elapsed_ms,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.post("/ingest/file", response_model=IngestResponse, dependencies=[Depends(ingest_auth)])
async def ingest_file(
    request: Request,
    file: UploadFile = File(...),
):
    """
    Upload a .txt/.md/.json file and ingest contents into a collection.
    Collection name is passed via querystring: ?collection=your_collection
    """
    start = time.time()

    collection_name = (request.query_params.get("collection") or DEFAULT_COLLECTION).strip()
    if not collection_name:
        raise HTTPException(status_code=400, detail="collection is required")

    raw = await file.read()
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        text = str(raw)

    doc = IngestDoc(
        id=f"upload-{file.filename}-{int(time.time()*1000)}",
        text=text,
        metadata={"filename": file.filename, "content_type": file.content_type, "source": "upload"},
    )
    req = IngestRequest(collection=collection_name, docs=[doc])
    return ingest(req)


@app.post("/collections/create", dependencies=[Depends(ingest_auth)])
def create_collection(payload: Dict[str, Any]):
    """
    Simple protected helper (not "admin stats" yet).
    Body: {"collection":"name"}
    """
    name = (payload.get("collection") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="collection is required")

    client = get_chroma_client()
    _ = client.get_or_create_collection(name=name)
    return {"status": "ok", "collection": name}
