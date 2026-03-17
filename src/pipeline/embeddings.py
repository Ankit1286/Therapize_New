"""
Batch embedding generation for therapist profiles.

Embedding strategy:
- Generate from to_embedding_text() — a carefully ordered text representation
- Batch 64 at a time (optimal for CPU inference with sentence-transformers)
- Cache embeddings in DB — only re-embed when profile text changes
- Cost: $0 — all-MiniLM-L6-v2 runs locally via sentence-transformers

Why not generate at query time?
- Query embeddings: generated per search (~5ms, free)
- Profile embeddings: generated once, cached forever in pgvector

Incremental updates:
- Track `last_updated` timestamp
- Only re-embed profiles that changed since last run
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime

from sentence_transformers import SentenceTransformer

from src.config import get_settings
from src.models.therapist import TherapistProfile

logger = logging.getLogger(__name__)
settings = get_settings()

BATCH_SIZE = 64  # Optimal CPU batch size for sentence-transformers


@dataclass
class EmbeddingStats:
    total: int = 0
    embedded: int = 0
    cached: int = 0
    failed: int = 0
    started_at: datetime = field(default_factory=datetime.utcnow)

    def log(self) -> None:
        elapsed = (datetime.utcnow() - self.started_at).total_seconds()
        logger.info(
            "Embedding stats: total=%d, embedded=%d, cached=%d, failed=%d | "
            "elapsed=%.1fs",
            self.total, self.embedded, self.cached, self.failed, elapsed,
        )


class EmbeddingPipeline:
    """
    Generates and updates embeddings for therapist profiles.

    Design: local sentence-transformers model loaded once at init,
    batched encoding to maximize CPU throughput.
    """

    def __init__(self):
        self._model = SentenceTransformer(settings.embedding_model)

    async def embed_profiles(
        self,
        profiles: list[TherapistProfile],
    ) -> list[tuple[TherapistProfile, list[float]]]:
        """
        Generate embeddings for a list of profiles.
        Returns (profile, embedding) pairs.
        """
        stats = EmbeddingStats(total=len(profiles))

        results = []
        for i in range(0, len(profiles), BATCH_SIZE):
            batch = profiles[i:i + BATCH_SIZE]
            texts = [p.to_embedding_text() for p in batch]

            try:
                embeddings = await self._embed_batch(texts)
                for profile, embedding in zip(batch, embeddings):
                    results.append((profile, embedding))
                    stats.embedded += 1
            except Exception as exc:
                logger.error("Embedding batch %d failed: %s", i // BATCH_SIZE, exc)
                stats.failed += len(batch)
                for profile in batch:
                    results.append((profile, []))

            if (i + BATCH_SIZE) % 500 == 0:
                logger.info(
                    "Embedding progress: %d/%d profiles", i + BATCH_SIZE, len(profiles)
                )

        stats.log()
        return results

    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a user query string."""
        if not text.strip():
            text = "therapist search"
        embeddings = await self._embed_batch([text])
        return embeddings[0]

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Encode a batch of texts using sentence-transformers (local, free).

        Runs in a thread to avoid blocking the event loop.
        """
        def _encode():
            return self._model.encode(
                texts, batch_size=BATCH_SIZE, show_progress_bar=False
            ).tolist()

        return await asyncio.to_thread(_encode)
