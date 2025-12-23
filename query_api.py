import os
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings
import requests

from fastapi import FastAPI, HTTPException
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
    client = get_chroma_client()
    cols = client.list_collections()

    stats: List[CollectionStats] = []

    for c in cols:
        col = client.get_collection(name=c.name)
        data = col.get(include=["metadatas"])
        metadata_keys = set()

        for md in data.get("metadatas", []):
            if md:
                metadata_keys.update(md.keys())

        stats.append(
            CollectionStats(
                name=c.name,
                document_count=len(data.get("metadatas", [])),
                metadata_keys=sorted(metadata_keys),
            )
        )

    return stats


@app.get("/admin/collections/{name}", response_model=CollectionStats)
def collection_stats(name: str):
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

    return CollectionStats(
        name=name,
        document_count=len(data.get("metadatas", [])),
        metadata_keys=sorted(metadata_keys),
    )
