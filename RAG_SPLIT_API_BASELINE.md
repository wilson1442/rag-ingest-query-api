# Split RAG API Baseline
## Ingest / Query Separation – Locked Architecture

**Status:** Locked
**Baseline Version:** v1.1
**Last Updated:** 2025-12-26
**Repository:** rag-ingest-query-api  

---

## 1. Purpose

This document defines the **locked baseline architecture** for the split RAG API system.  
It exists to:

- Prevent accidental or malicious data mutation
- Enforce strict separation of responsibilities
- Provide a stable reference for future changes
- Support audits, troubleshooting, and onboarding
- Serve as the authoritative “known good” state

This baseline was created after completing **Steps 1–4** of the hardening plan and should not be modified without a deliberate versioned change.

---

## 2. High-Level Architecture

                ┌────────────────────┐
                │   OpenWebUI / n8n   │
                │  (Read-Only Access)│
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │   Query API         │
                │   (READ-ONLY)       │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │   ChromaDB          │
                │   (Persistent)      │
                └─────────┬──────────┘
                          ▲
                          │
                ┌─────────┴──────────┐
                │   Ingest API        │
                │ (KEY-PROTECTED)    │
                └────────────────────┘

### Key Principles
- **Ingest and Query are physically separate services**
- **Query API cannot mutate data**
- **Ingest API is API-key protected and operator-only**
- **OpenWebUI never touches ingest**
- **Destructive actions require explicit confirmation**

---

## 3. Services Overview

### 3.1 Ingest API
- Path: `ingest_api.py`
- Purpose: Controlled mutation only
- Authentication: **Required**
- Exposure: Operator / automation only

### 3.2 Query API
- Path: `query_api.py`
- Purpose: Read-only semantic search and stats + protected collection management
- Authentication: Optional (required only for destructive operations)
- Exposure: OpenWebUI, tools, dashboards

---

## 4. Ingest API (Mutation Layer)

### 4.1 Security Model

- API Key required for all mutation endpoints
- Fail-closed if `INGEST_API_KEY` is missing
- Supports:
  - `X-API-Key` header
  - `Authorization: Bearer <key>`
- `/health` is the **only unauthenticated endpoint**

### 4.2 Environment Variables

| Variable | Purpose |
|-------|--------|
| `INGEST_API_KEY` | Required mutation key |
| `CHROMA_HOST` | Chroma endpoint |
| `CHROMA_PORT` | Chroma port |
| `OLLAMA_URL` | Embedding endpoint |
| `EMBED_MODEL` | Embedding model |
| `MAX_CHUNK_CHARS` | Chunk size |
| `CHUNK_OVERLAP_CHARS` | Chunk overlap |

---

### 4.3 Ingest Endpoints

#### `GET /health`
Unauthenticated health check.

#### `POST /ingest`
- Adds documents to a collection
- Chunked + embedded
- **Requires API key**

#### `POST /ingest/file`
- Uploads and ingests text-based files
- **Requires API key**

---

### 4.4 Controlled Rebuild Endpoint (Step 4)

#### `POST /admin/rebuild`

**Purpose:** Explicit, operator-only reindex reset.

**Requirements:**
- Valid API key
- `confirm=true`
- Single named collection

**Behavior:**
- Deletes all documents in the specified collection
- Leaves collection intact
- Returns deletion count + timestamp

**Safety Guarantees:**
- No wildcard deletes
- No implicit rebuilds
- No OpenWebUI access
- No accidental execution

---

## 5. Query API (Read-Only Layer)

### 5.1 Design Guarantees

The Query API is **primarily read-only** and cannot:
- Add documents
- Update metadata
- Delete individual documents
- Rebuild collections

**Exception:** The Query API includes one protected destructive endpoint (`DELETE /admin/collections/{name}`) that requires API key authentication. This endpoint can clear entire collections but is separated from read-only operations.

---

### 5.2 Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `ADMIN_API_KEY` | Required for destructive operations | Optional |
| `CHROMA_HOST` | Chroma endpoint | Yes |
| `CHROMA_PORT` | Chroma port | Yes |
| `OLLAMA_URL` | Embedding endpoint | Yes |
| `EMBED_MODEL` | Embedding model name | Yes |

---

### 5.3 Query Endpoints

#### `GET /health`
Read-only service health.

#### `GET /collections`
List all collection names (read-only, no authentication required).

Returns a simple array of collection names.

#### `POST /query`
Semantic search endpoint.

Supports:
- Collection selection
- Result limits
- Metadata filters

---

### 5.4 Admin & Stats Endpoints

#### `GET /admin/health`
- Confirms Chroma connectivity
- Returns collection count

#### `GET /admin/collections`
Returns detailed stats for all collections (read-only, no authentication required):
- Collection name
- Document count (unique documents based on chunk=0)
- Chunk count (total chunks stored)
- Embedding model used
- Metadata keys present

#### `GET /admin/collections/{name}`
Returns detailed stats for a single collection (read-only, no authentication required):
- Collection name
- Document count
- Chunk count
- Embedding model
- Metadata keys

---

### 5.5 Protected Collection Management Endpoints

#### `DELETE /admin/collections/{name}`
**Requires API key authentication** via `ADMIN_API_KEY`.

Clears all contents of a specific collection:
- Deletes all documents/chunks from the collection
- Keeps the collection itself intact
- Returns deletion count and timestamp

**Authentication:**
- Supports `x-api-key` header
- Supports `Authorization: Bearer <token>` header
- Fails with 401 if key is invalid or missing
- Fails with 500 if `ADMIN_API_KEY` not configured

**Safety:**
- Cannot be called without authentication
- Single collection only (no wildcards)
- Returns count of deleted chunks for audit trail

---

## 6. OpenWebUI Integration Rules

OpenWebUI:
- **ONLY points to Query API**
- Never sees ingest endpoints
- Never holds API keys
- Cannot perform destructive operations (no ADMIN_API_KEY)

Recommended usage:
- RAG pipelines → `/query`
- Collection discovery → `/collections`
- Collection stats → `/admin/collections`
- Health checks → `/health`

**Note:** OpenWebUI should NOT have access to `ADMIN_API_KEY`, preventing accidental or malicious collection clearing.

---

## 7. Operational Workflows

### 7.1 Normal Ingest
- Use Ingest API with API key
- Typically via scripts or automation
- Not via UI tools

### 7.2 Rebuild / Reindex
1. Confirm target collection
2. Call `/admin/rebuild` with:
   - API key
   - `confirm=true`
3. Re-ingest data
4. Validate via Query API stats

### 7.3 Clear Collection
1. Confirm target collection via `/admin/collections/{name}`
2. Call `DELETE /admin/collections/{name}` with:
   - API key (`x-api-key` header or `Authorization: Bearer` token)
3. Verify deletion count in response
4. Optional: Re-ingest fresh data

### 7.4 Monitoring
- Query API `/admin/health`
- Collection counts via `/admin/collections`
- Detailed collection stats via `/admin/collections/{name}`

---

## 8. Git & Change Management

This baseline was created through **atomic feature branches**:

1. Lock ingest with API key
2. Make query API read-only
3. Add collection stats & admin endpoints
4. Add controlled rebuild endpoint
5. Add collection management endpoints (v1.1)
   - Simple collection listing
   - Enhanced stats with chunk count and embedding model
   - Protected collection clear endpoint

Any future changes should:
- Use feature branches
- Include PR review
- Update this document if behavior changes
- Increment version number for significant features

---

## 9. Baseline Integrity Statement

This document reflects a **known-good, production-safe state**.

If:
- Mutation occurs without intent
- Query gains write access
- OpenWebUI can alter data

Then **this baseline has been violated** and must be reviewed.

---

## 10. Changelog (v1.1)

**Changes from v1.0 to v1.1:**
- Added `GET /collections` endpoint for simple collection listing
- Enhanced collection stats with `chunk_count` and `embedding_model` fields
- Added `DELETE /admin/collections/{name}` protected endpoint with API key authentication
- Added `ADMIN_API_KEY` environment variable for Query API
- Improved document count accuracy (counting by chunk=0 metadata)

---

## 11. Next Planned Enhancements (Out of Scope)

- Audit logging for collection clear operations
- Snapshot/backup endpoints
- Role-based access layers
- Separate read-only and admin API keys

These are **not part of this baseline**.

---

**End of Baseline**
