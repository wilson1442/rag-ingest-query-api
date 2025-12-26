import os
import time
import hmac
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings
import requests

from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# -----------------------------
# Logging
# -----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("query_api")


# -----------------------------
# Environment
# -----------------------------
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "").strip()

CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "7000"))
CHROMA_SSL = os.getenv("CHROMA_SSL", "false").lower() in ("1", "true", "yes", "on")
CHROMA_PATH = os.getenv("CHROMA_PATH", "/mnt/ai-data/chroma")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/embeddings")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", "default")
QUERY_TIMEOUT_SECONDS = int(os.getenv("QUERY_TIMEOUT_SECONDS", "60"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "10"))


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(
    title="RAG Query API (Read-Only + Stats)",
    version="2.1.0",
    description="Read-only query API with collection-level stats (safe for OpenWebUI)."
)


# -----------------------------
# Chroma client singleton
# -----------------------------
_chroma_client: Optional[chromadb.HttpClient] = None

def get_chroma_client() -> chromadb.HttpClient:
    """
    Returns a singleton Chroma HttpClient instance.
    Creates the client on first call and reuses it for all subsequent calls.
    """
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            ssl=CHROMA_SSL,
            settings=Settings(anonymized_telemetry=False),
        )
        logger.info(f"Initialized Chroma client: {CHROMA_HOST}:{CHROMA_PORT} (SSL={CHROMA_SSL})")
    return _chroma_client


# -----------------------------
# Auth
# -----------------------------
def extract_api_key(request: Request) -> str:
    """Extract API key from x-api-key header or Authorization Bearer token."""
    key = request.headers.get("x-api-key", "").strip()
    if key:
        return key
    auth = request.headers.get("authorization", "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def require_admin_key(request: Request):
    """Require valid ADMIN_API_KEY for protected endpoints."""
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: ADMIN_API_KEY not set")

    presented = extract_api_key(request)
    if not presented or not hmac.compare_digest(presented, ADMIN_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")


def admin_auth(request: Request):
    """Dependency for admin-protected endpoints."""
    require_admin_key(request)


# -----------------------------
# Helper functions
# -----------------------------
def list_collections() -> List[str]:
    """Returns a list of all collection names in Chroma."""
    client = get_chroma_client()
    collections = client.list_collections()
    return [col.name for col in collections]


# -----------------------------
# Embeddings
# -----------------------------
def embed_query(text: str) -> List[float]:
    payload = {"model": EMBED_MODEL, "prompt": text}
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=QUERY_TIMEOUT_SECONDS)
        r.raise_for_status()
        data = r.json()
        vec = data.get("embedding")
        if not vec:
            raise RuntimeError("Embedding missing in response")
        return vec
    except Exception as e:
        logger.exception("Embedding request failed")
        raise HTTPException(status_code=502, detail=f"Embedding service error: {str(e)}")


# -----------------------------
# Models
# -----------------------------
class QueryRequest(BaseModel):
    query: str
    collections: Optional[List[str]] = None
    top_k: int = 5
    where: Optional[Dict[str, Any]] = None


class QueryResult(BaseModel):
    id: str
    document: str
    metadata: Dict[str, Any]
    distance: float


class QueryResponse(BaseModel):
    collection: str
    results: List[QueryResult]
    elapsed_ms: int
    timestamp: str


class CollectionStats(BaseModel):
    name: str
    document_count: int
    chunk_count: int
    embedding_model: str
    metadata_keys: List[str]


# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "query",
        "mode": "read-only",
        "time": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/admin/health")
def admin_health():
    """
    Confirms Chroma connectivity + collection count.
    """
    try:
        client = get_chroma_client()
        cols = client.list_collections()
        return {
            "status": "ok",
            "chroma": "reachable",
            "collection_count": len(cols),
            "time": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        logger.exception("Chroma health check failed")
        raise HTTPException(status_code=502, detail=str(e))


# -----------------------------
# Collections (read-only)
# -----------------------------
@app.get("/collections", response_model=List[str])
def get_collections():
    """
    List all collection names (read-only, no auth required).
    """
    return list_collections()


# -----------------------------
# Query (read-only)
# -----------------------------
@app.post("/query")
def query(req: QueryRequest):
    # Use explicitly provided collections, otherwise search all collections
    if req.collections and len(req.collections) > 0:
        cols = req.collections
    else:
        cols = list_collections()

    if not cols:
        raise HTTPException(status_code=404, detail="No collections found")

    q_emb = embed_query(req.query)
    hits = []
    client = get_chroma_client()

    for cname in cols:
        col = client.get_collection(name=cname)
        res = col.query(
            query_embeddings=[q_emb],
            n_results=req.top_k,
            where=req.where,
            include=["documents", "metadatas", "distances"],
        )

        for i, cid in enumerate(res["ids"][0]):
            hits.append({
                "collection": cname,
                "id": cid,
                "distance": res["distances"][0][i],
                "document": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
            })

    hits.sort(key=lambda x: x["distance"])
    return {
        "collections": cols,
        "top_k": req.top_k,
        "results": hits[:req.top_k],
    }


# -----------------------------
# Collection stats (read-only)
# -----------------------------
@app.get("/admin/collections", response_model=List[CollectionStats])
def list_collections_with_stats():
    """
    List all collections with detailed statistics including document count,
    chunk count, embedding model, and metadata keys.
    """
    client = get_chroma_client()
    cols = client.list_collections()

    stats: List[CollectionStats] = []

    for c in cols:
        col = client.get_collection(name=c.name)
        data = col.get(include=["metadatas"])
        metadata_keys = set()

        # Track unique base document IDs (before chunk suffix)
        unique_doc_ids = set()

        for md in data.get("metadatas", []):
            if md:
                metadata_keys.update(md.keys())

        # Count chunks (total entries in collection)
        chunk_count = len(data.get("metadatas", []))

        # Estimate document count by counting entries with chunk=0
        # This assumes chunks are numbered starting from 0
        document_count = sum(1 for md in data.get("metadatas", []) if md and md.get("chunk") == 0)

        # If no chunks have chunk=0, fall back to total count
        if document_count == 0 and chunk_count > 0:
            document_count = chunk_count

        stats.append(
            CollectionStats(
                name=c.name,
                document_count=document_count,
                chunk_count=chunk_count,
                embedding_model=EMBED_MODEL,
                metadata_keys=sorted(metadata_keys),
            )
        )

    return stats


@app.get("/admin/collections/{name}", response_model=CollectionStats)
def collection_stats(name: str):
    """
    Get detailed statistics for a specific collection including document count,
    chunk count, embedding model, and metadata keys.
    """
    client = get_chroma_client()
    try:
        col = client.get_collection(name=name)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    data = col.get(include=["metadatas"])
    metadata_keys = set()

    for md in data.get("metadatas", []):
        if md:
            metadata_keys.update(md.keys())

    # Count chunks (total entries in collection)
    chunk_count = len(data.get("metadatas", []))

    # Estimate document count by counting entries with chunk=0
    document_count = sum(1 for md in data.get("metadatas", []) if md and md.get("chunk") == 0)

    # If no chunks have chunk=0, fall back to total count
    if document_count == 0 and chunk_count > 0:
        document_count = chunk_count

    return CollectionStats(
        name=name,
        document_count=document_count,
        chunk_count=chunk_count,
        embedding_model=EMBED_MODEL,
        metadata_keys=sorted(metadata_keys),
    )


@app.delete("/admin/collections/{name}", dependencies=[Depends(admin_auth)])
def clear_collection(name: str):
    """
    Clear all contents of a specific collection (requires API key authentication).
    This endpoint deletes all documents/chunks from the collection but keeps the collection itself.
    """
    client = get_chroma_client()
    try:
        col = client.get_collection(name=name)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")

    try:
        # Get all IDs from the collection (ids are returned by default, no need to specify in include)
        data = col.get()
        raw_ids = data.get("ids") if isinstance(data, dict) else None

        ids = []
        if isinstance(raw_ids, list):
            if raw_ids and isinstance(raw_ids[0], list):
                ids = [i for i in raw_ids[0] if i]
            else:
                ids = [i for i in raw_ids if i]

        deleted = len(ids)

        if ids:
            col.delete(ids=ids)
            logger.info(f"Cleared {deleted} chunks from collection '{name}'")
        else:
            logger.info(f"Collection '{name}' was already empty")

        return {
            "collection": name,
            "deleted_chunks": deleted,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        logger.exception(f"Failed to clear collection '{name}'")
        raise HTTPException(status_code=500, detail=f"Failed to clear collection: {str(e)}")
