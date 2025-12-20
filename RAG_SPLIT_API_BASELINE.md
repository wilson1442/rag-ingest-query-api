# Split RAG API Baseline
## Ingest / Query Separation – Locked Architecture

**Status:** Locked  
**Baseline Version:** v1.0  
**Last Updated:** 2025-12-20  
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
- Purpose: Read-only semantic search and stats
- Authentication: None (safe by design)
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

The Query API **cannot**:
- Add documents
- Update metadata
- Delete documents
- Rebuild collections

Even if misused, there is **no mutation code path**.

---

### 5.2 Query Endpoints

#### `GET /health`
Read-only service health.

#### `POST /query`
Semantic search endpoint.

Supports:
- Collection selection
- Result limits
- Metadata filters

---

### 5.3 Admin & Stats Endpoints (Step 3)

#### `GET /admin/health`
- Confirms Chroma connectivity
- Returns collection count

#### `GET /admin/collections`
Returns per-collection:
- Collection name
- Document count
- Metadata keys present

#### `GET /admin/collections/{name}`
Returns stats for a single collection.

**All admin endpoints are read-only.**

---

## 6. OpenWebUI Integration Rules

OpenWebUI:
- **ONLY points to Query API**
- Never sees ingest endpoints
- Never holds API keys
- Cannot mutate data even if compromised

Recommended usage:
- RAG pipelines → `/query`
- Tools → `/admin/collections`
- Health checks → `/health`

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

### 7.3 Monitoring
- Query API `/admin/health`
- Collection counts via `/admin/collections`

---

## 8. Git & Change Management

This baseline was created through **atomic feature branches**:

1. Lock ingest with API key
2. Make query API read-only
3. Add collection stats & admin endpoints
4. Add controlled rebuild endpoint

Any future changes should:
- Use feature branches
- Include PR review
- Update this document if behavior changes

---

## 9. Baseline Integrity Statement

This document reflects a **known-good, production-safe state**.

If:
- Mutation occurs without intent
- Query gains write access
- OpenWebUI can alter data

Then **this baseline has been violated** and must be reviewed.

---

## 10. Next Planned Enhancements (Out of Scope)

- Optional admin API key for stats
- Audit logging for rebuilds
- Snapshot/backup endpoints
- Role-based access layers

These are **not part of this baseline**.

---

**End of Baseline**
