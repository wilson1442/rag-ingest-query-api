import os
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from chromadb import PersistentClient


# -----------------------------
# Config
# -----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("rag_query_api")

SERVICE_NAME = "rag-query-api"
SERVICE_VERSION = "1.0.0"
STARTED_AT = datetime.now(timezone.utc).isoformat()

CHROMA_PATH = os.getenv("CHROMA_PATH", "/mnt/ai-data/chroma")

OLLAMA_EMBED_URL = os.getenv(
    "OLLAMA_EMBED_URL",
    "http://127.0.0.1:11434/api/embeddings"
)
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "6"))

chroma = PersistentClient(path=CHROMA_PATH)


# -----------------------------
# Helpers
# -----------------------------
def embed_query(text: str) -> List[float]:
    r = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60
    )
    r.raise_for_status()
    emb = r.json().get("embedding")
    if not emb:
        raise HTTPException(500, "Embedding failed")
    return emb

def list_collections() -> List[str]:
    return sorted(c.name for c in chroma.list_collections())


# -----------------------------
# Models
# -----------------------------
class QueryRequest(BaseModel):
    query: str
    collections: Optional[List[str]] = None
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=50)
    where: Optional[Dict[str, Any]] = None


# -----------------------------
# API
# -----------------------------
app = FastAPI(
    title="RAG Query API",
    version=SERVICE_VERSION,
)

@app.get("/health")
def health():
    chroma.list_collections()
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "started_at": STARTED_AT,
        "chroma_path": CHROMA_PATH,
    }

@app.get("/collections")
def collections():
    return {"collections": list_collections()}

@app.get("/collections/{name}/stats")
def collection_stats(name: str):
    col = chroma.get_collection(name)
    return {
        "collection": name,
        "chunks": col.count(),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }

@app.post("/query")
def query(req: QueryRequest):
    cols = req.collections or list_collections()
    if not cols:
        raise HTTPException(404, "No collections found")

    q_emb = embed_query(req.query)
    hits = []

    for cname in cols:
        try:
            col = chroma.get_collection(cname)
            res = col.query(
                query_embeddings=[q_emb],
                n_results=req.top_k,
                where=req.where,
                include=["documents", "metadatas", "distances", "ids"],
            )
            for i, cid in enumerate(res["ids"][0]):
                hits.append({
                    "collection": cname,
                    "id": cid,
                    "distance": res["distances"][0][i],
                    "document": res["documents"][0][i],
                    "metadata": res["metadatas"][0][i],
                })
        except Exception as e:
            logger.warning(f"Query failed for {cname}: {e}")

    hits.sort(key=lambda x: x["distance"])
    return {
        "collections": cols,
        "top_k": req.top_k,
        "results": hits[:req.top_k],
    }
