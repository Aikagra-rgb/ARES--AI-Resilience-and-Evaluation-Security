"""
Test that the Gemini embedder passes the API key via x-goog-api-key header
and never includes the masked ********** string in the URL.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ares.config import Settings
from ares.vectordb.embedder import GeminiEmbedder, EmbeddingError


class TestEmbedderURL:
    """Test that embedder uses header-based authentication correctly."""

    @pytest.fixture
    def settings(self) -> Settings:
        """Create test settings with a fake API key."""
        return Settings(
            DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/ares_test",
            GEMINI_API_KEY="test-api-key-12345",
            GEMINI_EMBEDDING_MODEL="models/gemini-embedding-001",
            QDRANT_VECTOR_SIZE=768,
            EMBEDDING_ALLOW_HASH_FALLBACK=False,
        )

    @pytest.fixture
    def embedder(self, settings: Settings) -> GeminiEmbedder:
        """Create embedder with mocked HTTP client."""
        embedder = GeminiEmbedder(settings=settings)
        # Mock the HTTP client
        embedder._http_client = AsyncMock()
        embedder._owns_http_client = False
        return embedder

    @pytest.mark.asyncio
    async def test_embed_text_uses_header_not_query_param(self, embedder: GeminiEmbedder):
        """Verify the API key is passed in x-goog-api-key header, not URL query param."""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding": {"values": [0.1] * 768}
        }
        embedder._http_client.post.return_value = mock_response

        # Call embed_text
        result = await embedder.embed_text("test prompt")

        # Verify the call was made
        embedder._http_client.post.assert_called_once()
        call_args = embedder._http_client.post.call_args
        
        # Check URL does NOT contain the API key
        called_url = call_args[0][0] if call_args[0] else call_args[1].get("url", "")
        assert "key=" not in called_url, f"URL should not contain API key, got: {called_url}"
        assert "**********" not in called_url, f"URL should not contain masked key, got: {called_url}"
        
        # Check headers contain the raw API key
        called_headers = call_args[1].get("headers", {})
        assert "x-goog-api-key" in called_headers, "Missing x-goog-api-key header"
        assert called_headers["x-goog-api-key"] == "test-api-key-12345", \
            f"Header should contain raw API key, got: {called_headers.get('x-goog-api-key')}"
        
        # Verify result
        assert len(result) == 768

    @pytest.mark.asyncio
    async def test_embed_batch_uses_header_not_query_param(self, embedder: GeminiEmbedder):
        """Verify batch embed also uses header authentication."""
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [{"values": [0.1] * 768}, {"values": [0.2] * 768}]
        }
        embedder._http_client.post.return_value = mock_response

        result = await embedder.embed_batch(["prompt 1", "prompt 2"])

        embedder._http_client.post.assert_called_once()
        call_args = embedder._http_client.post.call_args
        
        # Check URL does NOT contain the API key
        called_url = call_args[0][0] if call_args[0] else call_args[1].get("url", "")
        assert "key=" not in called_url, f"URL should not contain API key, got: {called_url}"
        assert "**********" not in called_url, f"URL should not contain masked key, got: {called_url}"
        
        # Check headers contain the raw API key
        called_headers = call_args[1].get("headers", {})
        assert "x-goog-api-key" in called_headers, "Missing x-goog-api-key header"
        assert called_headers["x-goog-api-key"] == "test-api-key-12345"

        # Verify result
        assert len(result) == 2
        assert len(result[0]) == 768
        assert len(result[1]) == 768

    @pytest.mark.asyncio
    async def test_no_fallback_when_embedding_fails_and_fallback_disabled(self, embedder: GeminiEmbedder):
        """Verify that when fallback is disabled, embedding failure raises error instead of hash fallback."""
        embedder.settings.embedding_allow_hash_fallback = False
        
        # Mock rate limit response
        mock_response = AsyncMock()
        mock_response.status_code = 429
        mock_response.text = "RESOURCE_EXHAUSTED: Quota exceeded"
        embedder._http_client.post.return_value = mock_response

        # Should raise EmbeddingError, not fall back to hash
        with pytest.raises(EmbeddingError) as exc_info:
            await embedder.embed_text("test prompt")
        
        assert "fallback disabled" in str(exc_info.value).lower() or "disabled" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_fallback_works_when_enabled(self, embedder: GeminiEmbedder):
        """Verify that when fallback is enabled, it returns hash-based vectors on failure."""
        embedder.settings.embedding_allow_hash_fallback = True
        
        # Mock rate limit response
        mock_response = AsyncMock()
        mock_response.status_code = 429
        mock_response.text = "RESOURCE_EXHAUSTED: Quota exceeded"
        embedder._http_client.post.return_value = mock_response

        # Should return hash-based vectors (deterministic fallback)
        result = await embedder.embed_text("test prompt")
        
        assert len(result) == 768
        # Vector should be normalized
        import math
        norm = math.sqrt(sum(x*x for x in result))
        assert abs(norm - 1.0) < 0.01, f"Vector should be normalized, got norm={norm}"

    @pytest.mark.asyncio
    async def test_gemini_api_key_required(self, settings: Settings):
        """Verify that missing GEMINI_API_KEY raises error."""
        settings.gemini_api_key = None
        embedder = GeminiEmbedder(settings=settings)
        
        with pytest.raises(EmbeddingError) as exc_info:
            await embedder.embed_text("test prompt")
        
        assert "not configured" in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])