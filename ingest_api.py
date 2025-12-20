import os
import io
import re
import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from chromadb import PersistentClient

# Optional parsers
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    import fitz  # pymupdf
except Exception:
    fitz = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import docx
except Exception:
    docx = None


# -----------------------------
# Config
# -----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("rag_ingest_api")

SERVICE_NAME = "rag-ingest-api"
SERVICE_VERSION = "1.0.0"
STARTED_AT = datetime.now(timezone.utc).isoformat()

CHROMA_PATH = os.getenv("CHROMA_PATH", "/mnt/ai-data/chroma")

OLLAMA_EMBED_URL = os.getenv(
    "OLLAMA_EMBED_URL",
    "http://127.0.0.1:11434/api/embeddings"
)
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "80"))

chroma = PersistentClient(path=CHROMA_PATH)


# -----------------------------
# Helpers
# -----------------------------
def now_utc():
    return datetime.now(timezone.utc).isoformat()

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    text = normalize(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if overlap else end
    return chunks

def embed(text: str) -> List[float]:
    r = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60
    )
    r.raise_for_status()
    emb = r.json().get("embedding")
    if not emb:
        raise HTTPException(status_code=500, detail="Embedding failed")
    return emb

def collection(name: str):
    return chroma.get_or_create_collection(name=name)

def upsert(collection_name: str, chunks: List[str], meta: Dict[str, Any], dedupe: str):
    col = collection(collection_name)
    ids, embeds, metas, docs = [], [], [], []

    for i, chunk in enumerate(chunks):
        cid = sha256_text(f"{dedupe}:{i}:{chunk}")
        ids.append(cid)
        embeds.append(embed(chunk))
        m = dict(meta)
        m.update({
            "chunk_index": i,
            "chunk_id": cid,
            "dedupe_key": dedupe,
            "ingested_at": now_utc(),
        })
        metas.append(m)
        docs.append(chunk)

    col.upsert(ids=ids, embeddings=embeds, metadatas=metas, documents=docs)
    return len(ids)


# -----------------------------
# Extraction
# -----------------------------
def extract_pdf(data: bytes) -> str:
    if fitz:
        with fitz.open(stream=data, filetype="pdf") as doc:
            return normalize("\n".join(p.get_text() for p in doc))
    if PdfReader:
        r = PdfReader(io.BytesIO(data))
        return normalize("\n".join(p.extract_text() or "" for p in r.pages))
    raise HTTPException(400, "No PDF parser installed")

def extract_docx(data: bytes) -> str:
    if not docx:
        raise HTTPException(400, "python-docx not installed")
    d = docx.Document(io.BytesIO(data))
    return normalize("\n".join(p.text for p in d.paragraphs if p.text))

def extract_file(name: str, data: bytes) -> str:
    lname = name.lower()
    if lname.endswith(".pdf"):
        return extract_pdf(data)
    if lname.endswith(".docx"):
        return extract_docx(data)
    return normalize(data.decode("utf-8", errors="ignore"))

def scrape(url: str) -> Tuple[str, Dict[str, Any]]:
    if not BeautifulSoup:
        raise HTTPException(400, "beautifulsoup4 not installed")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    text = normalize(soup.get_text("\n", strip=True))
    title = soup.title.get_text(strip=True) if soup.title else ""
    return text, {"url": url, "title": title}


# -----------------------------
# Models
# -----------------------------
class IngestText(BaseModel):
    collection: str
    text: str
    source: str = "text"
    url: Optional[str] = None
    doc_id: Optional[str] = None


class IngestURL(BaseModel):
    collection: str
    url: str
    source: str = "web_scrape"


# -----------------------------
# API
# -----------------------------
app = FastAPI(
    title="RAG Ingest API",
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

@app.post("/ingest-text")
def ingest_text(req: IngestText):
    chunks = chunk_text(req.text, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
    dedupe = req.doc_id or sha256_text(req.text[:1000])
    meta = {
        "source": req.source,
        "url": req.url or "",
        "type": "text",
        "service": SERVICE_NAME,
    }
    count = upsert(req.collection, chunks, meta, dedupe)
    return {"status": "ok", "chunks": count}

@app.post("/ingest-url")
def ingest_url(req: IngestURL):
    text, meta_extra = scrape(req.url)
    chunks = chunk_text(text, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
    dedupe = sha256_text(req.url)
    meta = {
        "source": req.source,
        "type": "web",
        **meta_extra,
    }
    count = upsert(req.collection, chunks, meta, dedupe)
    return {"status": "ok", "chunks": count}

@app.post("/ingest-file")
async def ingest_file(collection: str, file: UploadFile = File(...)):
    data = await file.read()
    if len(data) / (1024 * 1024) > MAX_FILE_MB:
        raise HTTPException(413, "File too large")
    text = extract_file(file.filename, data)
    chunks = chunk_text(text, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
    dedupe = sha256_bytes(data)
    meta = {
        "source": "file",
        "filename": file.filename,
        "type": "file",
    }
    count = upsert(collection, chunks, meta, dedupe)
    return {"status": "ok", "chunks": count}
