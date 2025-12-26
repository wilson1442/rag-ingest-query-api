import os
import hmac
import json
import time
import logging
import hashlib
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
    version="2.3.0",
    description="Ingest-only API with API-key protection, controlled rebuild, and idempotent document ingestion."
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
# Deterministic Document IDs
# -----------------------------
def generate_document_id(text: str, metadata: Optional[Dict[str, Any]], provided_id: Optional[str]) -> str:
    """
    Generate a deterministic document ID to ensure idempotent ingestion.

    This function implements the following priority order:

    1. If a doc_id is explicitly provided by the caller, use it directly.
       This allows clients to manage their own ID scheme.

    2. If metadata contains 'internetMessageId' (Microsoft Graph email ID),
       derive the ID from that field: sha256(internetMessageId).
       This ensures emails are uniquely identified and prevents duplicate ingestion.

    3. Otherwise, derive a stable ID by hashing a canonical fingerprint of the content.
       The fingerprint includes:
       - Normalized text (stripped whitespace)
       - Selected stable metadata fields (if present): subject, from, to, receivedDateTime

       This ensures the same content always produces the same ID, making ingestion idempotent.

    Why deterministic IDs are required:
    - Without stable IDs, retries or reprocessing create duplicate vectors
    - Using upsert() with deterministic IDs ensures: same ID → overwrite, different ID → new document
    - This is critical for data integrity and prevents unbounded growth from duplicate ingestion

    Args:
        text: The document text content
        metadata: Optional metadata dictionary
        provided_id: Optional caller-supplied document ID

    Returns:
        A deterministic document ID (string)
    """
    # Priority 1: Use caller-provided ID if present
    if provided_id:
        return provided_id

    # Priority 2: Use internetMessageId for email messages (Microsoft Graph)
    if metadata and "internetMessageId" in metadata:
        msg_id = str(metadata["internetMessageId"])
        # Hash the message ID to produce a valid Chroma ID
        return hashlib.sha256(msg_id.encode("utf-8")).hexdigest()

    # Priority 3: Generate ID from content fingerprint
    # Create a stable fingerprint by combining normalized text and stable metadata
    fingerprint_parts = [text.strip()]

    # Include stable metadata fields that help identify unique documents
    if metadata:
        stable_fields = ["subject", "from", "to", "receivedDateTime", "sender", "filename"]
        for field in stable_fields:
            if field in metadata and metadata[field]:
                fingerprint_parts.append(f"{field}:{str(metadata[field])}")

    # Join all parts and hash to create deterministic ID
    fingerprint = "|".join(fingerprint_parts)
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


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
        "FINGERPRINT": "INGEST_API_V2_3_IDEMPOTENT_UPSERT_ACTIVE",
        "time": datetime.utcnow().isoformat() + "Z",
    }


# -----------------------------
# Ingest endpoints
# -----------------------------
@app.post("/ingest", dependencies=[Depends(ingest_auth)])
def ingest(req: IngestRequest):
    """
    Ingest documents into a Chroma collection with idempotent behavior.

    This endpoint ensures that re-ingesting the same document does not create duplicates.
    It achieves this by:
    1. Generating deterministic document IDs (from caller-provided ID, internetMessageId, or content hash)
    2. Using upsert() instead of add() to overwrite existing documents with the same ID

    Idempotency guarantee:
    - Multiple identical ingest requests result in exactly one stored document
    - Re-ingesting with the same doc_id or content overwrites the previous version
    - No duplicate vectors are created

    Backward compatibility:
    - Existing clients that only send text and metadata will continue to work
    - Document IDs are automatically generated if not provided
    """
    client = get_chroma_client()
    col = get_or_create_collection(client, req.collection)

    all_texts = []
    all_ids = []
    all_meta = []

    for idx, d in enumerate(req.docs):
        # Generate deterministic base document ID
        # This ensures the same document always gets the same ID, preventing duplicates
        base_id = generate_document_id(
            text=d.text,
            metadata=d.metadata,
            provided_id=d.id
        )

        chunks = chunk_text(d.text)
        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            # Create chunk-specific ID by appending chunk index to base document ID
            all_ids.append(f"{base_id}-{i}")
            meta = d.metadata or {}
            meta["chunk"] = i
            # Preserve internetMessageId in metadata for auditing if present
            all_meta.append(meta)

    vectors = embed_texts(all_texts)

    # CRITICAL: Use upsert() instead of add() to ensure idempotent ingestion
    # upsert() will:
    # - Overwrite existing documents if the ID already exists (same content re-ingested)
    # - Insert new documents if the ID is new (different content)
    # This prevents duplicate vectors and ensures data integrity
    col.upsert(ids=all_ids, documents=all_texts, metadatas=all_meta, embeddings=vectors)

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
        data = col.get()
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
