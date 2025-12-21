# RAG Pipeline

This directory contains the Open WebUI pipeline for RAG integration.

## File: rag_manifold_v3.py

Production-ready RAG (Retrieval-Augmented Generation) pipeline for Open WebUI.

### Features

- ✅ **Fixed Streaming** - Properly parses Ollama's JSON streaming responses
- ✅ **Query API Compatible** - Uses correct payload format (collection/n_results)
- ✅ **Ollama Native API** - Uses `/api/chat` endpoint by default
- ✅ **Manifold Pipeline** - Exposes all Ollama models to Open WebUI
- ✅ **Error Handling** - Gracefully handles RAG failures

### Installation

1. Upload `rag_manifold_v3.py` to Open WebUI → Settings → Pipelines
2. Configure the RAG_API_URL valve to point to your query API (default: http://192.168.4.10:8012/query)
3. Select a model and start chatting

### Configuration

| Valve | Default | Description |
|-------|---------|-------------|
| RAG_API_URL | http://192.168.4.10:8012/query | Query API endpoint |
| DEFAULT_COLLECTION | default | ChromaDB collection name |
| TOP_K | 5 | Number of results to retrieve |
| UPSTREAM_BASE_URL | http://192.168.4.10:11434 | Ollama server URL |
| USE_OLLAMA_NATIVE | True | Use native Ollama API |
| ENABLE_RAG | True | Enable/disable RAG augmentation |

### Version

v3.1.0 - Fixed streaming and query API compatibility

See the main README for query API documentation.
