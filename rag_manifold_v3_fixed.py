"""
title: RAG Manifold v3 (Fixed)
author: OpenWebUI Expert
description: RAG manifold with multi-collection support, Ollama native API, and query API compatibility fixes
required_open_webui_version: 0.4.0+
version: 3.1.0
license: MIT
"""

from typing import List, Dict, Any, Union, Generator
from pydantic import BaseModel, Field
import requests
import logging

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Minimal MANIFOLD pipeline for RAG integration.
    Loader pattern: Classic single-class with instance-level pipelines attribute.

    FIXES in v3.1.0:
    - Changed default RAG API URL to match deployment (192.168.4.10:8012)
    - Fixed payload format to match query API schema (collection/n_results)
    - Enabled Ollama native API by default
    - Better error handling for model loading
    """

    class Valves(BaseModel):
        """Configuration options exposed in OpenWebUI UI"""
        RAG_API_URL: str = Field(
            default="http://192.168.4.10:8012/query",
            description="URL of the external RAG query API"
        )
        DEFAULT_COLLECTION: str = Field(
            default="default",
            description="Default collection name to query (use empty string to query all collections)"
        )
        TOP_K: int = Field(
            default=5,
            description="Number of RAG results to retrieve"
        )
        UPSTREAM_BASE_URL: str = Field(
            default="http://192.168.4.10:11434",
            description="Ollama base URL"
        )
        USE_OLLAMA_NATIVE: bool = Field(
            default=True,
            description="Use Ollama native API instead of OpenAI-compatible endpoint"
        )
        ENABLE_RAG: bool = Field(
            default=True,
            description="Enable RAG augmentation (disable to use as passthrough)"
        )

    def __init__(self):
        """
        Initialize the pipeline.
        CRITICAL: Set self.pipelines here to satisfy loader requirements.
        """
        self.type = "manifold"
        self.name = "RAG Manifold v3 (Fixed)"
        self.valves = self.Valves()

        # CRITICAL: Loader checks for this attribute
        # Must be set BEFORE any method calls during loading
        self.pipelines = []

        # Load available models from upstream
        try:
            self.pipelines = self._get_upstream_models()
            if not self.pipelines:
                logger.warning("No models returned from upstream, using fallback")
                self.pipelines = [{"id": "llama2", "name": "llama2 (fallback)"}]
        except Exception as e:
            logger.error(f"Failed to load upstream models: {e}")
            # Provide a fallback model instead of error model
            self.pipelines = [{"id": "llama2", "name": "llama2 (fallback - check Ollama connection)"}]

    def pipes(self) -> List[Dict[str, str]]:
        """
        Return list of available models (manifold interface).
        Called by OpenWebUI to populate model selector.
        """
        return self.pipelines

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> Union[str, Dict[str, Any]]:
        """
        Main execution pipeline:
        1. Query RAG API (if enabled)
        2. Inject context into user message
        3. Proxy to upstream LLM
        4. Return response
        """
        try:
            logger.info(f"[RAG Pipeline] Starting pipeline for model: {model_id}")
            logger.info(f"[RAG Pipeline] User message: {user_message[:100]}...")

            updated_messages = messages

            # Step 1: Query RAG API (if enabled)
            if self.valves.ENABLE_RAG:
                logger.info("[RAG Pipeline] Step 1: Querying RAG API...")
                rag_results = self._query_rag(user_message)
                logger.info(f"[RAG Pipeline] Got {len(rag_results)} RAG results")

                # Step 2: Build and inject context
                logger.info("[RAG Pipeline] Step 2: Building context...")
                context = self._build_context(rag_results)
                augmented_message = self._inject_context(user_message, context)
                logger.info(f"[RAG Pipeline] Context length: {len(context)} chars")

                # Step 3: Update messages with augmented content
                logger.info("[RAG Pipeline] Step 3: Updating messages...")
                updated_messages = self._update_messages(messages, augmented_message)
            else:
                logger.info("[RAG Pipeline] RAG disabled, using passthrough mode")

            # Step 4: Proxy to upstream
            logger.info(f"[RAG Pipeline] Step 4: Calling upstream {model_id}...")
            body["model"] = model_id
            body["messages"] = updated_messages

            # IMPORTANT: Respect streaming preference from body
            is_streaming = body.get("stream", False)
            logger.info(f"[RAG Pipeline] Streaming mode: {is_streaming}")

            response = self._call_upstream(body)
            logger.info("[RAG Pipeline] ✅ Pipeline completed successfully")

            return response

        except Exception as e:
            logger.exception("Pipeline error")
            # Fail loudly with visible error
            error_response = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": f"❌ PIPELINE ERROR: {type(e).__name__}: {str(e)}"
                    }
                }]
            }
            logger.error(f"[RAG Pipeline] Returning error response: {error_response}")
            return error_response

    def _get_upstream_models(self) -> List[Dict[str, str]]:
        """Fetch available models from Ollama upstream"""
        try:
            response = requests.get(
                f"{self.valves.UPSTREAM_BASE_URL}/api/tags",
                timeout=10
            )
            response.raise_for_status()

            models = response.json().get("models", [])
            if not models:
                logger.warning("Ollama returned empty model list")
                return []

            return [
                {"id": model["model"], "name": model["name"]}
                for model in models
            ]
        except Exception as e:
            logger.error(f"Failed to fetch Ollama models: {e}")
            raise  # Re-raise to be handled by __init__

    def _query_rag(self, query: str) -> List[Dict[str, Any]]:
        """
        Query the external RAG API.
        FIXED: Uses correct API contract matching query_api.py:
        - collection (singular string, not plural list)
        - n_results (not top_k)
        """
        # Build payload matching query_api.py schema
        payload = {
            "query": query,
            "n_results": self.valves.TOP_K  # Changed from top_k to n_results
        }

        # Add collection if specified (singular, not plural)
        if self.valves.DEFAULT_COLLECTION.strip():
            payload["collection"] = self.valves.DEFAULT_COLLECTION  # Changed from collections (list) to collection (string)

        logger.info(f"Querying RAG API with: {payload}")

        try:
            response = requests.post(
                self.valves.RAG_API_URL,
                json=payload,
                timeout=10
            )
            response.raise_for_status()

            # Extract results from response
            data = response.json()
            results = data.get("results", [])

            logger.info(f"RAG API returned {len(results)} results")
            return results
        except requests.exceptions.RequestException as e:
            logger.error(f"RAG API request failed: {e}")
            # Return empty results instead of crashing
            return []

    def _build_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Build context string from RAG results.
        Includes document chunks AND metadata.
        """
        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results, 1):
            doc = result.get("document", "")
            # Handle metadata being None explicitly
            metadata = result.get("metadata") or {}
            distance = result.get("distance", "N/A")

            # Collection might not be in individual results from single-collection query
            collection = result.get("collection", self.valves.DEFAULT_COLLECTION)

            # Format: [index] document + metadata details
            chunk = f"[{i}] {doc}\n"
            chunk += f"    (collection: {collection}, distance: {distance}"

            # Add metadata if available
            if metadata:
                chunk += f", source: {metadata.get('source', 'N/A')}"
                if 'type' in metadata:
                    chunk += f", type: {metadata.get('type')}"
                if 'chunk_index' in metadata:
                    chunk += f", chunk: {metadata.get('chunk_index')}"

            chunk += ")"
            context_parts.append(chunk)

        return "\n\n".join(context_parts)

    def _inject_context(self, user_question: str, context: str) -> str:
        """
        Inject context into user message using specified format.
        """
        if not context:
            return user_question

        return f"""Use the following CONTEXT to answer the user question.
If the answer is not in the context, say you do not know.

--- CONTEXT ---
{context}
--- END CONTEXT ---

User question:
{user_question}"""

    def _update_messages(self, messages: List[dict], new_content: str) -> List[dict]:
        """
        Replace last user message with augmented version.
        """
        updated = [dict(msg) for msg in messages]

        # Find and update last user message
        for i in range(len(updated) - 1, -1, -1):
            if updated[i].get("role") == "user":
                updated[i]["content"] = new_content
                return updated

        # If no user message found, append new one
        updated.append({"role": "user", "content": new_content})
        return updated

    def _call_upstream(self, body: dict) -> Union[Dict[str, Any], Generator]:
        """
        Proxy request to upstream Ollama LLM.
        Supports both native Ollama API and OpenAI-compatible endpoint.
        Handles both streaming and non-streaming responses.
        """
        is_streaming = body.get("stream", False)

        if self.valves.USE_OLLAMA_NATIVE:
            # Use native Ollama /api/chat endpoint
            logger.info("[RAG Pipeline] Using Ollama native API (/api/chat)")

            ollama_payload = {
                "model": body.get("model"),
                "messages": body.get("messages", []),
                "stream": is_streaming,
                "options": {}
            }

            # Map common parameters
            if "temperature" in body:
                ollama_payload["options"]["temperature"] = body["temperature"]
            if "top_p" in body:
                ollama_payload["options"]["top_p"] = body["top_p"]
            if "top_k" in body:
                ollama_payload["options"]["top_k"] = body["top_k"]
            if "max_tokens" in body:
                ollama_payload["options"]["num_predict"] = body["max_tokens"]

            logger.info(f"[RAG Pipeline] Ollama payload: model={ollama_payload['model']}, stream={is_streaming}")

            response = requests.post(
                f"{self.valves.UPSTREAM_BASE_URL}/api/chat",
                json=ollama_payload,
                timeout=120,
                stream=is_streaming
            )
            response.raise_for_status()

            if is_streaming:
                logger.info("[RAG Pipeline] Returning streaming response")
                return response.iter_lines()
            else:
                # Convert Ollama response to OpenAI format
                ollama_response = response.json()
                logger.info("[RAG Pipeline] Got non-streaming response from Ollama")
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": ollama_response.get("message", {}).get("content", "")
                        }
                    }]
                }
        else:
            # Use OpenAI-compatible endpoint
            logger.info("[RAG Pipeline] Using OpenAI-compatible API (/v1/chat/completions)")
            logger.info(f"[RAG Pipeline] Request: model={body.get('model')}, stream={is_streaming}")

            response = requests.post(
                f"{self.valves.UPSTREAM_BASE_URL}/v1/chat/completions",
                json=body,
                timeout=120,
                stream=is_streaming
            )
            response.raise_for_status()

            if is_streaming:
                logger.info("[RAG Pipeline] Returning streaming response")
                return response.iter_lines()
            else:
                logger.info("[RAG Pipeline] Got non-streaming response")
                return response.json()
