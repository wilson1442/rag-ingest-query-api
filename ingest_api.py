import os
import hmac
import json
import time
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

import requests
import chromadb
from chromadb.config import Settings

from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File
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
INGEST_API_KEY = os.getenv("INGEST_API_KEY", "").strip()

CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "7000"))
CHROMA_SSL = os.getenv("CHROMA_SSL", "false").lower() in ("1", "true", "yes", "on")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/embeddings")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", "default")
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "1800"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(
    title="RAG Ingest API",
    version="2.2.0",
    description="Ingest-only API with API-key protection and controlled rebuild."
)


# -----------------------------
# Auth
# -----------------------------
def extract_api_key(request: Request) -> str:
    key = request.headers.get("x-api-key", "").strip()
    if key:
        return key
    auth = request.headers.get("authorization", "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def require_ingest_key(request: Request):
    if not INGEST_API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: INGEST_API_KEY not set")

    presented = extract_api_key(request)
    if not presented or not hmac.compare_digest(presented, INGEST_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")


def ingest_auth(request: Request):
    require_ingest_key(request)


# -----------------------------
# Chroma
# -----------------------------
def get_chroma_client() -> chromadb.HttpClient:
    return chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        ssl=CHROMA_SSL,
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(client, name: str):
    try:
        return client.get_collection(name=name)
    except Exception:
        return client.get_or_create_collection(name=name)


# -----------------------------
# Embeddings
# -----------------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    vectors = []
    for t in texts:
        r = requests.post(
            OLLAMA_URL,
            json={"model": EMBED_MODEL, "prompt": t},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        r.raise_for_status()
        vec = r.json().get("embedding")
        if not vec:
            raise HTTPException(status_code=502, detail="Embedding failed")
        vectors.append(vec)
    return vectors


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text: str) -> List[str]:
    chunks = []
    i = 0
    while i < len(text):
        end = min(i + MAX_CHUNK_CHARS, len(text))
        chunks.append(text[i:end])
        i = max(end - CHUNK_OVERLAP_CHARS, end)
    return chunks


# -----------------------------
# Models
# -----------------------------
class IngestDoc(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: Optional[Dict[str, Any]] = None


class IngestRequest(BaseModel):
    collection: str
    docs: List[IngestDoc]


class RebuildRequest(BaseModel):
    collection: str
    confirm: bool = False


# -----------------------------
# Errors
# -----------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ingest",
        "FINGERPRINT": "INGEST_API_V2_REBUILD_JSON_FIX_ACTIVE",
        "time": datetime.utcnow().isoformat() + "Z",
    }


# -----------------------------
# Ingest endpoints
# -----------------------------
@app.post("/ingest", dependencies=[Depends(ingest_auth)])
def ingest(req: IngestRequest):
    client = get_chroma_client()
    col = get_or_create_collection(client, req.collection)

    all_texts = []
    all_ids = []
    all_meta = []

    for idx, d in enumerate(req.docs):
        base_id = d.id or f"doc-{int(time.time()*1000)}-{idx}"
        chunks = chunk_text(d.text)
        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_ids.append(f"{base_id}-{i}")
            meta = d.metadata or {}
            meta["chunk"] = i
            all_meta.append(meta)

    vectors = embed_texts(all_texts)
    col.add(ids=all_ids, documents=all_texts, metadatas=all_meta, embeddings=vectors)

    return {
        "collection": req.collection,
        "inserted_chunks": len(all_texts),
    }


@app.post("/ingest/file", dependencies=[Depends(ingest_auth)])
async def ingest_file(
    request: Request,
    file: UploadFile = File(...),
):
    collection = request.query_params.get("collection") or DEFAULT_COLLECTION
    text = (await file.read()).decode("utf-8", errors="replace")

    req = IngestRequest(
        collection=collection,
        docs=[IngestDoc(text=text, metadata={"filename": file.filename})],
    )
    return ingest(req)


# -----------------------------
# Controlled rebuild
# -----------------------------
@app.post("/admin/rebuild", dependencies=[Depends(ingest_auth)])
def rebuild(req: RebuildRequest):
    if not req.confirm:
        raise HTTPException(
            status_code=400,
            detail="Rebuild not confirmed. Set confirm=true."
        )

    client = get_chroma_client()

    try:
        col = client.get_collection(name=req.collection)
    except Exception:
        # Collection exists logically but Chroma may error if empty
        return JSONResponse(
            status_code=200,
            content={
                "collection": req.collection,
                "deleted_documents": 0,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "note": "Collection empty or not initialized"
            },
        )

    try:
        data = col.get(include=["ids"])
    except Exception:
        # Empty collection edge case
        return JSONResponse(
            status_code=200,
            content={
                "collection": req.collection,
                "deleted_documents": 0,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "note": "No documents to delete"
            },
        )

    raw_ids = data.get("ids") if isinstance(data, dict) else None

    ids: List[str] = []
    if isinstance(raw_ids, list):
        if raw_ids and isinstance(raw_ids[0], list):
            ids = [i for i in raw_ids[0] if i]
        else:
            ids = [i for i in raw_ids if i]

    deleted = len(ids)

    if ids:
        try:
            col.delete(ids=ids)
        except Exception:
            # Deleting an already-empty collection should not fail
            deleted = 0

    return JSONResponse(
        status_code=200,
        content={
            "collection": req.collection,
            "deleted_documents": deleted,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    )
