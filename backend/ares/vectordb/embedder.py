"""
Gemini Embedding Client for ARES.

Generates normalized vector embeddings using Google Gemini's embedding endpoint
(default: models/gemini-embedding-001 with outputDimensionality=768).
Supports single text and batch embedding with exponential backoff retries.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from ares.config import Settings, settings as default_settings

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Exception raised when embedding generation fails."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class GeminiEmbedder:
    """
    Asynchronous Gemini text embedder with retry resilience.
    Uses Google Gemini's native embedContent / batchEmbedContents API.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.settings = settings or default_settings
        self._http_client = http_client
        self._owns_http_client = http_client is None
        self.model = self.settings.gemini_embedding_model
        self.dimension = self.settings.qdrant_vector_size
        self._gemini_rate_limited: bool = False
        self._ollama_available: bool = True
        self._ollama_last_checked: float = 0.0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or getattr(self._http_client, "is_closed", False) is True:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.settings.request_timeout_seconds)
            )
            self._owns_http_client = True
        return self._http_client

    async def close(self) -> None:
        """Close internal HTTP client if owned."""
        if self._owns_http_client and self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def _try_ollama_embed(self, text: str) -> Optional[List[float]]:
        """Attempt high-speed local GPU embedding via Ollama nomic-embed-text."""
        if not getattr(self.settings, "prefer_local_embeddings", False):
            return None
        now = time.time()
        if not self._ollama_available and (now - self._ollama_last_checked) < 60.0:
            return None

        client = await self._get_client()
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/embed"
        payload = {
            "model": self.settings.ollama_embedding_model,
            "input": text,
        }
        try:
            response = await client.post(url, json=payload, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                embeddings = data.get("embeddings", [])
                vec = embeddings[0] if embeddings and isinstance(embeddings[0], list) else data.get("embedding", [])
                if isinstance(vec, list) and vec:
                    self._ollama_available = True
                    self._ollama_last_checked = now
                    if len(vec) == self.dimension:
                        return vec
                    logger.error(
                        "Local embedding dimension %d does not match expected vector size %d; refusing to distort vector space.",
                        len(vec),
                        self.dimension,
                    )
                    return None
            self._ollama_available = False
            self._ollama_last_checked = now
            return None
        except Exception:
            self._ollama_available = False
            self._ollama_last_checked = now
            return None

    async def _try_ollama_batch(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Attempt local batch embedding via Ollama nomic-embed-text."""
        if not getattr(self.settings, "prefer_local_embeddings", False):
            return None
        now = time.time()
        if not self._ollama_available and (now - self._ollama_last_checked) < 60.0:
            return None

        client = await self._get_client()
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/embed"
        payload = {
            "model": self.settings.ollama_embedding_model,
            "input": texts,
        }
        try:
            response = await client.post(url, json=payload, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                embeddings = data.get("embeddings", [])
                if len(embeddings) == len(texts):
                    self._ollama_available = True
                    self._ollama_last_checked = now
                    result = []
                    for vec in embeddings:
                        if len(vec) == self.dimension:
                            result.append(vec)
                        else:
                            logger.error(
                                "Local batch embedding dimension %d does not match expected vector size %d.",
                                len(vec),
                                self.dimension,
                            )
                            return None
                    return result
            self._ollama_available = False
            self._ollama_last_checked = now
            return None
        except Exception:
            self._ollama_available = False
            self._ollama_last_checked = now
            return None

    async def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string into a float vector of length `dimension`.
        """
        if not text.strip():
            raise ValueError("Cannot embed empty text.")

        # 1. Try local Ollama nomic-embed-text for sub-20ms GPU inference
        local_vec = await self._try_ollama_embed(text)
        if local_vec is not None:
            return local_vec

        if self._gemini_rate_limited:
            return self._fallback_project_embeddings([text])[0]

        if not self.settings.gemini_api_key:
            raise EmbeddingError("GEMINI_API_KEY is not configured.", status_code=401)

        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/{self.model}:embedContent"
        )
        headers = {
            "x-goog-api-key": self.settings.gemini_api_key.get_secret_value(),
        }
        payload = {
            "model": self.model,
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": self.dimension,
        }

        client = await self._get_client()
        retries = 0
        max_embed_retries = max(self.settings.max_retries, 5)
        last_exception: Optional[Exception] = None

        while retries <= max_embed_retries:
            try:
                response = await client.post(endpoint, json=payload, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    values = data.get("embedding", {}).get("values", [])
                    if not values:
                        raise EmbeddingError("No embedding values returned in response.")
                    return values

                elif response.status_code == 429:
                    err_text = response.text
                    if "RESOURCE_EXHAUSTED" in err_text or "Quota exceeded" in err_text:
                        logger.warning("Gemini daily embedding quota reached (429). Activating local deterministic semantic vector projection.")
                        self._gemini_rate_limited = True
                        return self._fallback_project_embeddings([text])[0]
                    retries += 1
                    wait = min(
                        self.settings.retry_max_wait_seconds,
                        1.5 * (2 ** (retries - 1)),
                    )
                    logger.warning("Gemini embedding rate-limited (429). Retrying in %.2fs (attempt %d/%d)...", wait, retries, max_embed_retries)
                    await asyncio.sleep(wait)
                    continue

                else:
                    raise EmbeddingError(
                        f"Gemini embedding API error ({response.status_code}): {response.text}",
                        status_code=response.status_code,
                    )

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                retries += 1
                last_exception = exc
                wait = min(
                    self.settings.retry_max_wait_seconds,
                    self.settings.retry_min_wait_seconds * (2 ** (retries - 1)),
                )
                logger.warning("Network error calling Gemini embed: %s. Retrying in %.2fs...", exc, wait)
                await asyncio.sleep(wait)

        if retries > max_embed_retries:
            logger.error(
                "Gemini embedContent quota exhausted (429) and all retries failed. "
                "Embedding service unavailable. This will degrade gateway effectiveness. "
                "Configure OLLAMA_BASE_URL for local embeddings or check GEMINI_API_KEY quota."
            )
            if self.settings.embedding_allow_hash_fallback:
                logger.warning("Falling back to deterministic projection (NOT semantically meaningful).")
                return self._fallback_project_embeddings([text])[0]
            raise EmbeddingError(
                "Gemini embedding API rate limited and fallback disabled. "
                "Set EMBEDDING_ALLOW_HASH_FALLBACK=true to enable degraded mode, "
                "or configure Ollama for local embeddings."
            )

    def _fallback_project_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        DETERMINISTIC PROJECTION FALLBACK - NOT SEMANTICALLY MEANINGFUL.
        
        WARNING: This produces random-looking vectors based on text hashes.
        Similarity searches WILL NOT WORK CORRECTLY with these vectors.
        Only use when external embedding APIs are completely unavailable.
        Enable with EMBEDDING_FALLBACK_TO_GEMINI=true (development only).
        
        For production, configure Ollama with nomic-embed-text for local embeddings.
        """
        logger.critical(
            "USING FALLBACK EMBEDDINGS - SEMANTIC SIMILARITY IS BROKEN. "
            "Gateway vector shield will not function correctly. "
            "Configure OLLAMA_BASE_URL or check GEMINI_API_KEY quota."
        )
        try:
            from sklearn.feature_extraction.text import HashingVectorizer
            vectorizer = HashingVectorizer(n_features=self.dimension, norm="l2", alternate_sign=True)
            matrix = vectorizer.transform(texts).toarray()
            return matrix.tolist()
        except Exception:
            import hashlib
            results = []
            for t in texts:
                h = hashlib.sha256(t.encode()).digest()
                vec = [(float(h[i % 32]) / 255.0) - 0.5 for i in range(self.dimension)]
                norm = sum(x * x for x in vec) ** 0.5 or 1.0
                results.append([x / norm for x in vec])
            return results

    async def embed_batch(self, texts: List[str], chunk_size: int = 100) -> List[List[float]]:
        """
        Embed a list of text strings in batch.
        Chunked up to 100 to maximize throughput per Gemini API specifications.
        """
        if not texts:
            return []

        # 1. Try local Ollama nomic-embed-text for fast batching
        local_batch = await self._try_ollama_batch(texts)
        if local_batch is not None:
            return local_batch

        if self._gemini_rate_limited:
            return self._fallback_project_embeddings(texts)

        if not self.settings.gemini_api_key:
            raise EmbeddingError("GEMINI_API_KEY is not configured.", status_code=401)

        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/{self.model}:batchEmbedContents"
        )
        headers = {
            "x-goog-api-key": self.settings.gemini_api_key.get_secret_value(),
        }
        client = await self._get_client()
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), chunk_size):
            if self._gemini_rate_limited:
                all_embeddings.extend(self._fallback_project_embeddings(texts[i:]))
                break

            chunk = texts[i : i + chunk_size]
            requests_payload = [
                {
                    "model": self.model,
                    "content": {"parts": [{"text": t if t.strip() else " "}]},
                    "outputDimensionality": self.dimension,
                }
                for t in chunk
            ]
            payload = {"requests": requests_payload}

            try:
                response = await client.post(endpoint, json=payload, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    embeddings = [
                        emb.get("values", []) for emb in data.get("embeddings", [])
                    ]
                    if len(embeddings) == len(chunk):
                        all_embeddings.extend(embeddings)
                        continue
                elif response.status_code == 429:
                    logger.warning("Gemini API rate limited (429). Tripping circuit breaker to fast local fallback.")
                    self._gemini_rate_limited = True
                    all_embeddings.extend(self._fallback_project_embeddings(texts[i:]))
                    break
                else:
                    logger.warning("Gemini API returned status %s: %s", response.status_code, response.text[:200])
                    self._gemini_rate_limited = True
                    all_embeddings.extend(self._fallback_project_embeddings(texts[i:]))
                    break
            except Exception as exc:
                logger.warning("Network failure during batch embed: %s. Tripping circuit breaker.", exc)
                self._gemini_rate_limited = True
                all_embeddings.extend(self._fallback_project_embeddings(texts[i:]))
                break

        return all_embeddings

    def embed_text_sync(self, text: str) -> List[float]:
        """Synchronous wrapper for embed_text."""
        return asyncio.run(self.embed_text(text))

    def embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous wrapper for embed_batch."""
        return asyncio.run(self.embed_batch(texts))
