"""
Redis caching layer.

Caching strategy:
- Cache key: SHA256 hash of (query_text + questionnaire_json)
  Identical queries always hit the same cache entry.
- TTL: 1 hour (therapist data doesn't change minute-to-minute)
- Cache invalidation: TTL-based only (no manual invalidation needed)
- Compression: gzip compress large payloads to reduce memory usage

Cache impact at scale:
- Without cache: every query = 1 LLM call + 1 embedding call + 1 DB query = ~800ms
- With cache: 60% hit rate → 60% of queries served in ~5ms
- Cost: 60% reduction in LLM API spend at scale
"""
import gzip
import hashlib
import json
import logging
from typing import Any

import redis.asyncio as aioredis

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_redis: aioredis.Redis | None = None


async def init_cache() -> None:
    """Initialize Redis connection. Call once at app startup."""
    global _redis
    _redis = aioredis.from_url(
        str(settings.redis_url),
        encoding="utf-8",
        decode_responses=False,  # We handle encoding/decoding manually
        max_connections=20,
        socket_connect_timeout=5,
        socket_timeout=5,
    )
    # Verify connection
    await _redis.ping()
    logger.info("Redis cache initialized")


async def close_cache() -> None:
    """Close the Redis connection. Call once at app shutdown."""
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None


class SearchCache:
    """
    Cache for search results with automatic serialization/deserialization.

    Key design: cache the full SearchResponse (not individual components)
    so cache hits require zero computation.
    """

    CACHE_PREFIX = "therapize:search:"
    RATE_LIMIT_PREFIX = "therapize:ratelimit:"

    def make_cache_key(self, query_text: str, questionnaire_json: str) -> str:
        """
        Deterministic cache key from query content.
        SHA256 so keys are fixed-length regardless of query size.
        """
        content = f"{query_text.strip().lower()}|{questionnaire_json}"
        hash_val = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"{self.CACHE_PREFIX}{hash_val}"

    async def get(self, key: str) -> dict | None:
        """
        Get cached result. Returns None on miss or if Redis is unavailable.
        Never raises — cache failures are non-fatal.
        """
        if _redis is None:
            return None
        try:
            compressed = await _redis.get(key)
            if compressed is None:
                return None
            decompressed = gzip.decompress(compressed)
            return json.loads(decompressed)
        except Exception as exc:
            logger.warning("Cache get failed for key %s: %s", key, exc)
            return None

    async def set(self, key: str, value: dict, ttl: int | None = None) -> None:
        """
        Cache a result with TTL. Compresses payload before storing.
        Never raises — cache write failures are non-fatal.
        """
        if _redis is None:
            return
        try:
            serialized = json.dumps(value, default=str)
            compressed = gzip.compress(serialized.encode(), compresslevel=6)
            ttl = ttl or settings.cache_ttl_seconds
            await _redis.setex(key, ttl, compressed)
            logger.debug("Cached key %s (%d bytes compressed)", key, len(compressed))
        except Exception as exc:
            logger.warning("Cache set failed for key %s: %s", key, exc)

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern. Returns count deleted."""
        if _redis is None:
            return 0
        try:
            keys = [key async for key in _redis.scan_iter(match=pattern)]
            if keys:
                return await _redis.delete(*keys)
            return 0
        except Exception as exc:
            logger.warning("Cache invalidation failed: %s", exc)
            return 0

    async def ping(self) -> bool:
        """Health check."""
        if _redis is None:
            return False
        try:
            return await _redis.ping()
        except Exception:
            return False


class RateLimiter:
    """
    Sliding window rate limiter using Redis sorted sets.

    More accurate than fixed-window (no boundary bursting).
    Each request is stored as a score=timestamp entry.
    Old entries are pruned on each check.
    """

    async def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """
        Check if client is within rate limit.

        Returns: (is_allowed, requests_remaining)
        """
        if _redis is None:
            return True, settings.rate_limit_per_minute  # fail open

        key = f"{SearchCache.RATE_LIMIT_PREFIX}{client_id}"
        limit = settings.rate_limit_per_minute
        window_seconds = 60

        now_ms = int(__import__("time").time() * 1000)
        window_start_ms = now_ms - (window_seconds * 1000)

        try:
            pipe = _redis.pipeline()
            # Remove expired entries
            pipe.zremrangebyscore(key, 0, window_start_ms)
            # Count current window
            pipe.zcard(key)
            # Add current request
            pipe.zadd(key, {str(now_ms): now_ms})
            # Set TTL
            pipe.expire(key, window_seconds + 1)
            results = await pipe.execute()

            current_count = results[1]
            if current_count >= limit:
                return False, 0
            return True, limit - current_count - 1

        except Exception as exc:
            logger.warning("Rate limiter failed: %s — allowing request", exc)
            return True, limit  # fail open


# Module-level singletons
search_cache = SearchCache()
rate_limiter = RateLimiter()
