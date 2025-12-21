"""
title: RAG Manifold v3 (Fixed)
author: OpenWebUI Expert
description: RAG manifold with query API compatibility and proper streaming support
required_open_webui_version: 0.4.0+
version: 3.1.0
license: MIT
"""

from typing import List, Dict, Any, Union, Generator
from pydantic import BaseModel, Field
import requests
import logging
import json

logger = logging.getLogger(__name__)


class Pipeline:
    """
    MANIFOLD pipeline for RAG integration with Ollama.

    FIXES in v3.1.0:
    - Fixed RAG API payload format (collection/n_results)
    - Fixed streaming response parsing from Ollama
    - Enabled Ollama native API by default
    - Updated default URLs to match deployment
    - Better error handling
    """

    class Valves(BaseModel):
        """Configuration options exposed in OpenWebUI UI"""
        RAG_API_URL: str = Field(
            default="http://192.168.4.10:8012/query",
            description="URL of the external RAG query API"
        )
        DEFAULT_COLLECTION: str = Field(
            default="default",
            description="Collection name to query (leave empty to skip collection parameter)"
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
            description="Use Ollama native API (recommended)"
        )
        ENABLE_RAG: bool = Field(
            default=True,
            description="Enable RAG augmentation (disable for passthrough)"
        )

    def __init__(self):
        """Initialize the pipeline."""
        self.type = "manifold"
        self.name = "RAG Manifold v3"
        self.valves = self.Valves()
        self.pipelines = []

        # Load available models from Ollama
        try:
            self.pipelines = self._get_upstream_models()
            if not self.pipelines:
                logger.warning("No models found, using fallback")
                self.pipelines = [{"id": "llama2", "name": "llama2 (fallback)"}]
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self.pipelines = [{"id": "llama2", "name": "llama2 (check Ollama)"}]

    def pipes(self) -> List[Dict[str, str]]:
        """Return available models for OpenWebUI selector."""
        return self.pipelines

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> Union[str, Dict[str, Any], Generator]:
        """
        Main pipeline execution:
        1. Query RAG API (if enabled)
        2. Inject context into messages
        3. Call Ollama
        4. Return response (streaming or non-streaming)
        """
        try:
            logger.info(f"[RAG Pipeline] Model: {model_id}, Message: {user_message[:100]}...")

            updated_messages = messages

            # RAG augmentation
            if self.valves.ENABLE_RAG:
                logger.info("[RAG Pipeline] Querying RAG API...")
                rag_results = self._query_rag(user_message)
                logger.info(f"[RAG Pipeline] Got {len(rag_results)} results")

                context = self._build_context(rag_results)
                augmented_message = self._inject_context(user_message, context)
                updated_messages = self._update_messages(messages, augmented_message)
                logger.info(f"[RAG Pipeline] Context: {len(context)} chars")
            else:
                logger.info("[RAG Pipeline] RAG disabled")

            # Call upstream
            logger.info(f"[RAG Pipeline] Calling {model_id}...")
            body["model"] = model_id
            body["messages"] = updated_messages

            response = self._call_upstream(body)
            logger.info("[RAG Pipeline] ✅ Success")

            return response

        except Exception as e:
            logger.exception("Pipeline error")
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": f"❌ ERROR: {type(e).__name__}: {str(e)}"
                    }
                }]
            }

    def _get_upstream_models(self) -> List[Dict[str, str]]:
        """Fetch models from Ollama."""
        response = requests.get(
            f"{self.valves.UPSTREAM_BASE_URL}/api/tags",
            timeout=10
        )
        response.raise_for_status()

        models = response.json().get("models", [])
        return [
            {"id": model["model"], "name": model["name"]}
            for model in models
        ]

    def _query_rag(self, query: str) -> List[Dict[str, Any]]:
        """
        Query RAG API.
        FIXED: Uses correct schema (collection/n_results) to match query_api.py
        """
        payload = {
            "query": query,
            "n_results": self.valves.TOP_K  # Changed from top_k
        }

        # Add collection if specified (singular string, not list)
        if self.valves.DEFAULT_COLLECTION.strip():
            payload["collection"] = self.valves.DEFAULT_COLLECTION  # Changed from collections

        logger.info(f"RAG API request: {payload}")

        try:
            response = requests.post(
                self.valves.RAG_API_URL,
                json=payload,
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])
            logger.info(f"RAG API returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"RAG API error: {e}")
            return []  # Don't crash, just return empty results

    def _build_context(self, results: List[Dict[str, Any]]) -> str:
        """Build context from RAG results."""
        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results, 1):
            doc = result.get("document", "")
            metadata = result.get("metadata") or {}
            distance = result.get("distance", "N/A")

            chunk = f"[{i}] {doc}\n"
            chunk += f"    (distance: {distance}"

            if metadata:
                if "source" in metadata:
                    chunk += f", source: {metadata['source']}"
                if "type" in metadata:
                    chunk += f", type: {metadata['type']}"
                if "chunk_index" in metadata:
                    chunk += f", chunk: {metadata['chunk_index']}"

            chunk += ")"
            context_parts.append(chunk)

        return "\n\n".join(context_parts)

    def _inject_context(self, user_question: str, context: str) -> str:
        """Inject context into user message."""
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
        """Replace last user message with augmented version."""
        updated = [dict(msg) for msg in messages]

        # Find and update last user message
        for i in range(len(updated) - 1, -1, -1):
            if updated[i].get("role") == "user":
                updated[i]["content"] = new_content
                return updated

        # No user message found, append new one
        updated.append({"role": "user", "content": new_content})
        return updated

    def _call_upstream(self, body: dict) -> Union[Dict[str, Any], Generator]:
        """
        Call Ollama with proper streaming support.
        FIXED: Properly parse Ollama streaming JSON responses.
        """
        is_streaming = body.get("stream", False)

        if self.valves.USE_OLLAMA_NATIVE:
            # Use Ollama native API
            logger.info("[RAG Pipeline] Using Ollama /api/chat")

            ollama_payload = {
                "model": body.get("model"),
                "messages": body.get("messages", []),
                "stream": is_streaming,
                "options": {}
            }

            # Map parameters
            if "temperature" in body:
                ollama_payload["options"]["temperature"] = body["temperature"]
            if "top_p" in body:
                ollama_payload["options"]["top_p"] = body["top_p"]
            if "top_k" in body:
                ollama_payload["options"]["top_k"] = body["top_k"]
            if "max_tokens" in body:
                ollama_payload["options"]["num_predict"] = body["max_tokens"]

            logger.info(f"[RAG Pipeline] Request: model={ollama_payload['model']}, stream={is_streaming}")

            response = requests.post(
                f"{self.valves.UPSTREAM_BASE_URL}/api/chat",
                json=ollama_payload,
                timeout=120,
                stream=is_streaming
            )
            response.raise_for_status()

            if is_streaming:
                # FIXED: Parse Ollama streaming response properly
                logger.info("[RAG Pipeline] Streaming enabled")
                return self._stream_ollama_response(response)
            else:
                # Non-streaming response
                ollama_response = response.json()
                logger.info("[RAG Pipeline] Non-streaming response")
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
            logger.info("[RAG Pipeline] Using /v1/chat/completions")

            response = requests.post(
                f"{self.valves.UPSTREAM_BASE_URL}/v1/chat/completions",
                json=body,
                timeout=120,
                stream=is_streaming
            )
            response.raise_for_status()

            if is_streaming:
                return response.iter_lines()
            else:
                return response.json()

    def _stream_ollama_response(self, response) -> Generator[str, None, None]:
        """
        Parse Ollama streaming response and yield content.
        FIXED: Properly extract content from Ollama JSON streaming format.

        Ollama streams JSON objects like:
        {"message":{"content":"Hello"},"done":false}
        {"message":{"content":" world"},"done":false}
        {"message":{"content":"!"},"done":true}
        """
        for line in response.iter_lines():
            if line:
                try:
                    # Decode bytes to string
                    line_str = line.decode('utf-8') if isinstance(line, bytes) else line

                    # Parse JSON
                    data = json.loads(line_str)

                    # Extract content from message
                    if "message" in data and "content" in data["message"]:
                        content = data["message"]["content"]
                        if content:  # Only yield if content is not empty
                            yield content

                    # Check if done
                    if data.get("done", False):
                        logger.info("[RAG Pipeline] Streaming complete")
                        break

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse streaming line: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing stream: {e}")
                    continue
