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
# Chroma client
# -----------------------------
def get_chroma_client() -> chromadb.HttpClient:
    return chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        ssl=CHROMA_SSL,
        settings=Settings(anonymized_telemetry=False),
    )


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
    collection: Optional[str] = None
    n_results: Optional[int] = 5
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
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    start = time.time()

    collection_name = (req.collection or DEFAULT_COLLECTION).strip()
    if not collection_name:
        raise HTTPException(status_code=400, detail="collection is required")

    n_results = min(req.n_results or MAX_RESULTS, MAX_RESULTS)

    client = get_chroma_client()
    try:
        col = client.get_collection(name=collection_name)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

    query_vec = embed_query(req.query)

    res = col.query(
        query_embeddings=[query_vec],
        n_results=n_results,
        where=req.where,
        include=["documents", "metadatas", "distances"],
    )

    results: List[QueryResult] = []
    for i in range(len(res["ids"][0])):
        results.append(
            QueryResult(
                id=res["ids"][0][i],
                document=res["documents"][0][i],
                metadata=res["metadatas"][0][i] or {},
                distance=res["distances"][0][i],
            )
        )

    return QueryResponse(
        collection=collection_name,
        results=results,
        elapsed_ms=int((time.time() - start) * 1000),
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


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
