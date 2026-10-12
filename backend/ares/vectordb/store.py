"""
Qdrant Vector Database Integration for ARES.

Manages collection lifecycle, vector upserts of AttackAttempts, and
semantic similarity retrieval with metadata filtering.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from ares.config import Settings, settings as default_settings
from ares.redteam.attacker import AttackAttempt
from ares.vectordb.embedder import GeminiEmbedder

logger = logging.getLogger(__name__)


class QdrantStoreError(Exception):
    """Exception raised for Qdrant operations errors."""
    pass


_SHARED_LOCAL_CLIENTS: Dict[str, QdrantClient] = {}


def _get_local_client(storage_path: str) -> QdrantClient:
    import os
    norm_path = os.path.abspath(storage_path)
    if norm_path not in _SHARED_LOCAL_CLIENTS:
        try:
            _SHARED_LOCAL_CLIENTS[norm_path] = QdrantClient(path=storage_path)
        except Exception as exc:
            logger.warning("Could not acquire lock for %s (%s); falling back to in-memory Qdrant", storage_path, exc)
            return QdrantClient(":memory:")
    return _SHARED_LOCAL_CLIENTS[norm_path]


class QdrantStore:
    """
    Vector storage and RAG retrieval interface using Qdrant Cloud.
    Handles auto-collection creation, batch embeddings, and similarity search.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedder: Optional[GeminiEmbedder] = None,
        client: Optional[QdrantClient] = None,
    ):
        self.settings = settings or default_settings
        # Prevent embedding-space mismatch:
        # If local embeddings (e.g. nomic-embed-text) are active, isolate vectors
        # in a model-specific collection namespace rather than comparing against Gemini vectors.
        if getattr(self.settings, "prefer_local_embeddings", False) and self.settings.qdrant_collection == "ares_attacks":
            base_col = self.settings.qdrant_collection
            model_tag = getattr(self.settings, "ollama_embedding_model", "local").replace("-", "_").replace(":", "_")
            self.collection_name = f"{base_col}_{model_tag}"
        else:
            self.collection_name = self.settings.qdrant_collection
        self.vector_size = self.settings.qdrant_vector_size
        self.embedder = embedder or GeminiEmbedder(settings=self.settings)

        if client is not None:
            self._client = client
        else:
            url = self.settings.qdrant_url
            if not url or url == ":memory:" or url.startswith("path:"):
                storage_path = url.replace("path:", "") if url and url.startswith("path:") else "./qdrant_data"
                if url == ":memory:":
                    self._client = QdrantClient(":memory:")
                else:
                    self._client = _get_local_client(storage_path)
            else:
                try:
                    self._client = QdrantClient(
                        url=self.settings.qdrant_url,
                        api_key=self.settings.qdrant_api_key.get_secret_value() if self.settings.qdrant_api_key else None,
                        timeout=min(self.settings.request_timeout_seconds, 5.0),
                    )
                    self._client.get_collections()
                except Exception as exc:
                    logger.warning("Remote Qdrant connection failed (%s); switching to local persistent vector store at ./qdrant_data", exc)
                    self._client = _get_local_client("./qdrant_data")

    @property
    def client(self) -> QdrantClient:
        return self._client


    def ensure_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        Create the Qdrant collection if it does not already exist.
        Configures Cosine distance, HNSW graph, and keyword payload indexes.
        """
        col = collection_name or self.collection_name
        try:
            if not self._client.collection_exists(col):
                logger.info("Creating Qdrant collection '%s' (dim=%d, distance=Cosine)...", col, self.vector_size)
                self._client.create_collection(
                    collection_name=col,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE,
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        m=16,
                        ef_construct=128,
                    ),
                )
                # Create payload indexes on frequently filtered fields
                self._client.create_payload_index(
                    collection_name=col,
                    field_name="category",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                self._client.create_payload_index(
                    collection_name=col,
                    field_name="success",
                    field_schema=models.PayloadSchemaType.BOOL,
                )
                self._client.create_payload_index(
                    collection_name=col,
                    field_name="domain",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                logger.info("Collection '%s' successfully created with indexes.", col)
                return True
            return False
        except Exception as exc:
            logger.error("Error ensuring Qdrant collection '%s': %s", col, exc)
            raise QdrantStoreError(f"Failed to ensure collection '{col}': {exc}") from exc

    async def upsert_attempt(
        self,
        attempt: AttackAttempt,
        target_prompt_hash: str,
        collection_name: Optional[str] = None,
    ) -> str:
        """
        Embed and upsert a single AttackAttempt into Qdrant.
        Returns the point UUID.
        """
        col = collection_name or self.collection_name
        self.ensure_collection(col)

        point_id = str(uuid.uuid4())
        vector = await self.embedder.embed_text(attempt.attack_text)

        payload = {
            "attack_text": attempt.attack_text,
            "category": attempt.category,
            "source": getattr(attempt, "source", "redteam_probe"),
            "severity": getattr(attempt, "severity", "high" if attempt.success else "safe"),
            "tags": getattr(attempt, "tags", ["dynamic_attack", attempt.category]),
            "success": attempt.success,
            "rule_verdict": attempt.rule_verdict,
            "llm_verdict": attempt.llm_verdict,
            "reasoning": attempt.reasoning,
            "domain": attempt.domain,
            "mutation_path": attempt.mutation_path,
            "operator_applied": attempt.operator_applied,
            "canary_tokens_found": attempt.canary_tokens_found,
            "attempt_number": attempt.attempt_number,
            "target_prompt_hash": target_prompt_hash,
            "attacker_provider": attempt.attacker_provider,
            "victim_provider": attempt.victim_provider,
            "judge_provider": attempt.judge_provider,
            "latency_ms": attempt.latency_ms,
            "timestamp": attempt.timestamp or datetime.now(timezone.utc).isoformat(),
        }

        try:
            self._client.upsert(
                collection_name=col,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
            logger.debug("Upserted attack point %s to collection '%s'", point_id, col)
            return point_id
        except Exception as exc:
            raise QdrantStoreError(f"Failed to upsert attack attempt: {exc}") from exc

    async def upsert_attempts_batch(
        self,
        attempts: List[AttackAttempt],
        target_prompt_hash: str,
        collection_name: Optional[str] = None,
    ) -> int:
        """
        Embed and upsert multiple AttackAttempts in batch.
        Returns the number of points upserted.
        """
        if not attempts:
            return 0

        col = collection_name or self.collection_name
        self.ensure_collection(col)

        texts = [a.attack_text for a in attempts]
        vectors = await self.embedder.embed_batch(texts)

        points = []
        for attempt, vector in zip(attempts, vectors):
            point_id = str(uuid.uuid4())
            payload = {
                "attack_text": attempt.attack_text,
                "category": attempt.category,
                "success": attempt.success,
                "rule_verdict": attempt.rule_verdict,
                "llm_verdict": attempt.llm_verdict,
                "reasoning": attempt.reasoning,
                "domain": attempt.domain,
                "mutation_path": attempt.mutation_path,
                "operator_applied": attempt.operator_applied,
                "canary_tokens_found": attempt.canary_tokens_found,
                "attempt_number": attempt.attempt_number,
                "target_prompt_hash": target_prompt_hash,
                "attacker_provider": attempt.attacker_provider,
                "victim_provider": attempt.victim_provider,
                "judge_provider": attempt.judge_provider,
                "latency_ms": attempt.latency_ms,
                "timestamp": attempt.timestamp or datetime.now(timezone.utc).isoformat(),
            }
            points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

        try:
            self._client.upsert(collection_name=col, points=points)
            logger.info("Batch upserted %d attack attempts to collection '%s'", len(points), col)
            return len(points)
        except Exception as exc:
            raise QdrantStoreError(f"Failed to batch upsert attack attempts: {exc}") from exc

    async def search_similar_attacks(
        self,
        query_text: str,
        limit: int = 5,
        category: Optional[str] = None,
        success_only: Optional[bool] = None,
        domain: Optional[str] = None,
        score_threshold: Optional[float] = None,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve semantically similar attacks matching the query text.
        Supports filtering by category, success status, and domain.
        """
        col = collection_name or self.collection_name
        self.ensure_collection(col)

        query_vector = await self.embedder.embed_text(query_text)

        # Build filter conditions
        must_conditions: List[models.FieldCondition] = []
        if category is not None:
            must_conditions.append(
                models.FieldCondition(key="category", match=models.MatchValue(value=category))
            )
        if success_only is not None:
            must_conditions.append(
                models.FieldCondition(key="success", match=models.MatchValue(value=success_only))
            )
        if domain is not None:
            must_conditions.append(
                models.FieldCondition(key="domain", match=models.MatchValue(value=domain))
            )

        query_filter = models.Filter(must=must_conditions) if must_conditions else None

        try:
            # Uses Qdrant's universal query_points API
            response = self._client.query_points(
                collection_name=col,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )

            results: List[Dict[str, Any]] = []
            for point in response.points:
                results.append({
                    "id": str(point.id),
                    "score": float(point.score),
                    "payload": point.payload or {},
                })

            return results
        except Exception as exc:
            raise QdrantStoreError(f"Failed to search similar attacks: {exc}") from exc

    def count(self, collection_name: Optional[str] = None) -> int:
        """Return the total number of points in the collection."""
        col = collection_name or self.collection_name
        try:
            if not self._client.collection_exists(col):
                return 0
            res = self._client.count(collection_name=col, exact=True)
            return res.count
        except Exception as exc:
            logger.error("Error counting collection '%s': %s", col, exc)
            return 0

    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """Delete collection if it exists."""
        col = collection_name or self.collection_name
        try:
            if self._client.collection_exists(col):
                self._client.delete_collection(collection_name=col)
                logger.info("Deleted collection '%s'", col)
                return True
            return False
        except Exception as exc:
            raise QdrantStoreError(f"Failed to delete collection '{col}': {exc}") from exc
