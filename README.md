# RAG Split API (Ingest + Query)

This repository contains a **cleanly separated RAG API architecture**:

- **rag-ingest-api** – handles ingestion (text, files, web scraping)
- **rag-query-api** – handles vector search and retrieval
- **ChromaDB** – persistent storage (external, not included)

## Architecture

```
[ Ingest Sources ]
        |
        v
 rag-ingest-api (8011)
        |
        v
   ChromaDB (persistent)
        |
        v
 rag-query-api (8012)
        |
        v
 OpenWebUI / Pipelines / Clients
```

## Requirements

- Python 3.10+
- ChromaDB
- Ollama (for embeddings)
- Persistent Chroma path (e.g. /mnt/ai-data/chroma)

## Environment Variables

Common:
- `CHROMA_PATH=/mnt/ai-data/chroma`
- `OLLAMA_EMBED_URL=http://127.0.0.1:11434/api/embeddings`
- `EMBED_MODEL=nomic-embed-text`

## Running Locally

```bash
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn chromadb requests
uvicorn ingest_api:app --port 8011
uvicorn query_api:app --port 8012
```

## systemd

Example units are provided in the `systemd/` directory.

## Notes

- This repo intentionally excludes:
  - Chroma data
  - Virtual environments
  - Secrets
- Designed for pipeline-safe, read/write separation.
