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
- `CHROMA_HOST=127.0.0.1`
- `CHROMA_PORT=7000`
- `CHROMA_PATH=/mnt/ai-data/chroma`
- `OLLAMA_URL=http://127.0.0.1:11434/api/embeddings`
- `EMBED_MODEL=nomic-embed-text`

Ingest API only:
- `INGEST_API_KEY` – required; ingest endpoints fail closed if unset

Query API only:
- `ADMIN_API_KEY` – required only for the destructive `DELETE /admin/collections/{name}`

## Endpoints

### Ingest API (port 8011, key-protected)

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET  | `/health` | none | Service health |
| POST | `/ingest` | key | Chunk + embed + upsert documents (idempotent) |
| POST | `/ingest/file` | key | Upload and ingest a text file |
| POST | `/admin/rebuild` | key | Delete all docs in a collection (requires `confirm=true`) |

### Query API (port 8012, read-only)

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET  | `/health` | none | Service health |
| POST | `/query` | none | Semantic search across one or more collections |
| GET  | `/collections` | none | List collection names |
| GET  | `/collections/{name}/count` | none | Chunk count for a collection (native `count()`, O(1)) |
| POST | `/collections/{name}/get` | none | Fetch raw documents by `ids` / `where` / `limit` |
| GET  | `/admin/health` | none | Chroma connectivity + collection count |
| GET  | `/admin/collections` | none | Per-collection stats |
| GET  | `/admin/collections/{name}` | none | Stats for one collection |
| DELETE | `/admin/collections/{name}` | key | Clear all chunks in a collection |

> `/collections/{name}/count` and `/collections/{name}/get` are read-only,
> off-box replacements for direct ChromaDB calls. They exist because Chroma is
> bound to **localhost only** (see Deployment Notes) and is therefore not
> reachable from other hosts.

#### Examples

```bash
# Count chunks in a collection
curl http://<query-host>:8012/collections/network_documentation/count
# -> {"collection":"network_documentation","count":484}

# Fetch raw docs (optional ids / where / limit; limit defaults to 50)
curl -X POST http://<query-host>:8012/collections/network_documentation/get \
  -H 'content-type: application/json' \
  -d '{"limit":2}'
# -> {"collection":"...","count":2,"results":[{"id":"...","document":"...","metadata":{...}}]}
```

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

## Deployment Notes

- **Chroma binds to localhost only** (`127.0.0.1:7000`). Nothing off-box talks
  to Chroma directly — all access goes through the Ingest API (8011) and the
  read-only Query API (8012). Raw read access that previously hit Chroma
  directly is now served by `GET /collections/{name}/count` and
  `POST /collections/{name}/get`.
- Embeddings are produced by Ollama (`nomic-embed-text`, 768-dim). Keep the
  embedding model consistent across ingest and query — mixing models corrupts
  similarity results.

## Notes

- This repo intentionally excludes:
  - Chroma data
  - Virtual environments
  - Secrets
- Designed for pipeline-safe, read/write separation.
