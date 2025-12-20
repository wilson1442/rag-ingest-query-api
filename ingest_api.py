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

from fastapi import FastAPI, HTTPException, Request, Depends
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
    title="RAG Ingest API (Controlled Mutation)",
    version="2.1.0",
    description="Ingest-only API with explicit, gated rebuild endpoint."
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


def get_collection(client: chromadb.HttpClient, name: str):
    try:
        return client.get_collection(name=name)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")


# -----------------------------
# Embeddings
# -----------------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    vectors = []
    for t in texts:
        payload = {"model": EMBED_MODEL, "prompt": t}
        r = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        r.raise_for_status()
        vec = r.json().get("embedding")
        if not vec:
            raise HTTPException(status_code=502, detail="Embedding failed")
        vectors.append(vec)
    return vectors


# -----------------------------
# Models
# -----------------------------
class RebuildRequest(BaseModel):
    collection: str = Field(..., description="Collection to rebuild")
    confirm: bool = Field(False, description="Must be true to proceed")


class RebuildResponse(BaseModel):
    collection: str
    deleted_documents: int
    timestamp: str


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
        "time": datetime.utcnow().isoformat() + "Z",
    }


# -----------------------------
# Controlled rebuild endpoint
# -----------------------------
@app.post("/admin/rebuild", response_model=RebuildResponse, dependencies=[Depends(ingest_auth)])
def rebuild_collection(req: RebuildRequest):
    if not req.confirm:
        raise HTTPException(
            status_code=400,
            detail="Rebuild not confirmed. Set confirm=true to proceed."
        )

    client = get_chroma_client()
    col = get_collection(client, req.collection)

    data = col.get(include=["ids"])
    ids = data.get("ids", [])

    deleted = len(ids)

    if ids:
        col.delete(ids=ids)

    return RebuildResponse(
        collection=req.collection,
        deleted_documents=deleted,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )
