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
    title="RAG Query API (Read-Only)",
    version="2.0.0",
    description="Read-only semantic query API for ChromaDB (safe for OpenWebUI)."
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
    query: str = Field(..., description="User query text")
    collection: Optional[str] = Field(default=None, description="Collection to search")
    n_results: Optional[int] = Field(default=5, description="Number of results")
    where: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter")


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


# -----------------------------
# Public endpoints (safe)
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "query",
        "mode": "read-only",
        "time": datetime.utcnow().isoformat() + "Z",
    }


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

    try:
        res = col.query(
            query_embeddings=[query_vec],
            n_results=n_results,
            where=req.where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.exception("Chroma query failed")
        raise HTTPException(status_code=502, detail=f"Chroma query error: {str(e)}")

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

    elapsed_ms = int((time.time() - start) * 1000)
    return QueryResponse(
        collection=collection_name,
        results=results,
        elapsed_ms=elapsed_ms,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.get("/collections")
def list_collections():
    """
    Safe for OpenWebUI — listing only, no stats yet.
    """
    client = get_chroma_client()
    cols = client.list_collections()
    return {"collections": [c.name for c in cols]}
