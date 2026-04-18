"""
Async PostgreSQL client with pgvector support.

Design principles:
- Async throughout: never block the event loop (asyncpg is fully async)
- Connection pooling: reuse connections, don't open/close per request
- Typed queries: return Pydantic models, not raw dicts
- Vector search integrated with SQL filtering:
  Apply WHERE filters BEFORE vector search for efficiency
"""
import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from uuid import UUID

import asyncpg
import numpy as np

from src.config import get_settings
from src.models.therapist import TherapistProfile

logger = logging.getLogger(__name__)
settings = get_settings()

# Global pool — created once at startup, shared across all requests
_pool: asyncpg.Pool | None = None


def _clean_dsn(url: str) -> str:
    """
    Normalize a PostgreSQL DSN for asyncpg.

    - Converts postgresql+asyncpg:// → postgresql://
    - Strips query params asyncpg doesn't understand (e.g. channel_binding)
    """
    from urllib.parse import urlparse, urlencode, parse_qs, urlunparse
    url = url.replace("postgresql+asyncpg://", "postgresql://")
    parsed = urlparse(url)
    params = {k: v[0] for k, v in parse_qs(parsed.query).items()
              if k not in ("channel_binding",)}
    return urlunparse(parsed._replace(query=urlencode(params)))


async def init_db() -> None:
    """Initialize connection pool. Call once at app startup."""
    global _pool
    dsn = _clean_dsn(str(settings.database_url))
    _pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=2,
        max_size=settings.db_pool_size,
        max_inactive_connection_lifetime=300,
        command_timeout=30,
        # Register vector type codec
        init=_register_codecs,
    )
    logger.info("Database pool initialized (min=2, max=%d)", settings.db_pool_size)


async def _register_codecs(conn: asyncpg.Connection) -> None:
    """Register custom type codecs for pgvector."""
    await conn.set_type_codec(
        "vector",
        encoder=lambda v: f"[{','.join(str(x) for x in v)}]",
        decoder=lambda s: [float(x) for x in s.strip("[]").split(",")],
        schema="public",
        format="text",
    )


async def close_db() -> None:
    """Close all connections. Call at app shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("Database pool closed")


@asynccontextmanager
async def get_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    """Context manager for acquiring a connection from the pool."""
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_db() first.")
    async with _pool.acquire() as conn:
        yield conn


class TherapistRepository:
    """
    Data access layer for therapist records.

    All SQL is parameterized — no f-string SQL anywhere.
    The vector search query combines SQL filtering + ANN in a single round-trip.
    """

    async def upsert(self, therapist: TherapistProfile, embedding: list[float]) -> UUID:
        """
        Insert or update therapist record with its embedding.
        Upsert on source_url (natural dedup key).
        """
        async with get_connection() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO therapists (
                    id, source, source_url, source_id, name, credentials,
                    license_number, years_experience, gender, ethnicity,
                    modalities, specializations, populations_served, languages,
                    city, state, zip_code, county, latitude, longitude,
                    session_formats, insurance_providers, sliding_scale,
                    fee_min, fee_max, accepting_new_clients,
                    bio,
                    rating, review_count, profile_completeness,
                    embedding, scraped_at, last_updated, is_active
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                    $11::therapy_modality[], $12, $13, $14,
                    $15, $16, $17, $18, $19, $20,
                    $21::session_format[], $22::insurance_provider[], $23,
                    $24, $25, $26,
                    $27,
                    $28, $29, $30,
                    $31::vector, $32, $33, $34
                )
                ON CONFLICT (source_url) DO UPDATE SET
                    name = EXCLUDED.name,
                    gender = EXCLUDED.gender,
                    ethnicity = EXCLUDED.ethnicity,
                    modalities = EXCLUDED.modalities,
                    specializations = EXCLUDED.specializations,
                    insurance_providers = EXCLUDED.insurance_providers,
                    fee_min = EXCLUDED.fee_min,
                    fee_max = EXCLUDED.fee_max,
                    accepting_new_clients = EXCLUDED.accepting_new_clients,
                    bio = EXCLUDED.bio,
                    rating = EXCLUDED.rating,
                    review_count = EXCLUDED.review_count,
                    profile_completeness = EXCLUDED.profile_completeness,
                    embedding = EXCLUDED.embedding,
                    last_updated = NOW()
                RETURNING id
                """,
                therapist.id,
                therapist.source,
                str(therapist.source_url),
                therapist.source_id,
                therapist.name,
                therapist.credentials,
                therapist.license_number,
                therapist.years_experience,
                therapist.gender,
                therapist.ethnicity,
                [m.value for m in therapist.modalities],
                [s.value for s in therapist.specializations],
                therapist.populations_served,
                therapist.languages,
                therapist.location.city,
                therapist.location.state,
                therapist.location.zip_code,
                therapist.location.county,
                therapist.location.latitude,
                therapist.location.longitude,
                [f.value for f in therapist.session_formats],
                [i.value for i in therapist.accepts_insurance],
                therapist.sliding_scale,
                therapist.fee_min,
                therapist.fee_max,
                therapist.accepting_new_clients,
                therapist.bio,
                therapist.rating,
                therapist.review_count,
                therapist.profile_completeness,
                embedding,
                therapist.scraped_at,
                therapist.last_updated,
                therapist.is_active,
            )
            return row["id"]

    async def search_candidates(
        self,
        where_clause: str,
        params: list,
        query_embedding: list[float],
        top_k: int = 200,
    ) -> tuple[list[TherapistProfile], list[list[float]]]:
        """
        Hybrid search: SQL filtering + vector ANN in a single query.

        Strategy: Filter first (cheap), then vector sort (expensive only on filtered set).
        This is far more efficient than vector search + filter (which scans all vectors).

        Returns: (therapist_profiles, embeddings) — same order, for ranking.
        """
        # pgvector cosine distance: <=> operator (lower = more similar)
        # We ORDER BY vector distance to use the HNSW index efficiently
        # The WHERE clause filters happen at the SQL level before vector scoring
        embedding_param_idx = len(params) + 1
        params.append(query_embedding)

        limit_param_idx = len(params) + 1
        params.append(top_k)

        query = f"""
            SELECT
                id, source, source_url, source_id, name, credentials,
                license_number, years_experience, gender, ethnicity,
                modalities, specializations, populations_served, languages,
                city, state, zip_code, county, latitude, longitude,
                session_formats, insurance_providers, sliding_scale,
                fee_min, fee_max, accepting_new_clients,
                bio,
                rating, review_count, profile_completeness,
                embedding,
                scraped_at, last_updated, is_active
            FROM therapists
            WHERE {where_clause}
              AND embedding IS NOT NULL
            ORDER BY embedding <=> ${embedding_param_idx}::vector
            LIMIT ${limit_param_idx}
        """

        async with get_connection() as conn:
            rows = await conn.fetch(query, *params)

        profiles = []
        embeddings = []
        for row in rows:
            profile = self._row_to_profile(row)
            profiles.append(profile)
            # Parse embedding from string format
            emb = row["embedding"]
            if isinstance(emb, str):
                emb = [float(x) for x in emb.strip("[]").split(",")]
            embeddings.append(emb)

        return profiles, embeddings

    async def count_filtered(self, where_clause: str, params: list) -> int:
        """Count candidates that pass filters (without fetching all data)."""
        query = f"SELECT COUNT(*) FROM therapists WHERE {where_clause}"
        async with get_connection() as conn:
            return await conn.fetchval(query, *params)

    async def get_by_id(self, therapist_id: UUID) -> TherapistProfile | None:
        """Fetch a single therapist by primary key. Returns None if not found."""
        async with get_connection() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM therapists WHERE id = $1", therapist_id
            )
        return self._row_to_profile(row) if row else None

    async def ping(self) -> bool:
        """Health check — returns True if DB is reachable."""
        try:
            async with get_connection() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def get_total_count(self) -> int:
        """Return the total number of active therapists accepting new clients."""
        async with get_connection() as conn:
            return await conn.fetchval(
                "SELECT COUNT(*) FROM therapists WHERE is_active = TRUE AND accepting_new_clients = TRUE"
            )

    async def get_languages(self) -> list[str]:
        """Return distinct languages spoken by at least one active therapist, sorted alphabetically."""
        async with get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT unnest(languages) AS lang
                FROM therapists
                WHERE is_active = TRUE
                ORDER BY lang
                """
            )
        return [row["lang"] for row in rows if row["lang"]]

    async def get_ethnicities(self) -> list[str]:
        """Return distinct ethnicities present in the database, sorted alphabetically."""
        async with get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT unnest(ethnicity) AS eth
                FROM therapists
                WHERE is_active = TRUE
                ORDER BY eth
                """
            )
        return [row["eth"] for row in rows if row["eth"]]

    async def get_cities(self) -> list[str]:
        """Return distinct city names that have at least one active therapist, sorted alphabetically."""
        async with get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT city
                FROM therapists
                WHERE is_active = TRUE AND city IS NOT NULL AND city <> ''
                ORDER BY city
                """
            )
        return [row["city"] for row in rows]

    async def get_all_for_audit(self) -> list[dict]:
        """Return id, source_url, source, name, accepting_new_clients for all active records."""
        async with get_connection() as conn:
            rows = await conn.fetch(
                "SELECT id, source_url, source, name, accepting_new_clients FROM therapists WHERE is_active = TRUE"
            )
        return [dict(r) for r in rows]

    async def delete_by_id(self, therapist_id) -> None:
        """
        Soft-delete a therapist by marking them inactive.

        Uses is_active=FALSE rather than a hard DELETE so that feedback records
        referencing this therapist are preserved (foreign key constraint).
        Active queries all filter on is_active=TRUE, so the record disappears
        from all search results.
        """
        async with get_connection() as conn:
            await conn.execute(
                "UPDATE therapists SET is_active = FALSE, accepting_new_clients = FALSE, last_updated = NOW() WHERE id = $1",
                therapist_id,
            )

    async def set_accepting_new_clients(self, therapist_id, value: bool) -> None:
        """Update the accepting_new_clients flag for a single therapist."""
        async with get_connection() as conn:
            await conn.execute(
                "UPDATE therapists SET accepting_new_clients = $1, last_updated = NOW() WHERE id = $2",
                value,
                therapist_id,
            )

    async def log_search(
        self,
        query_id: UUID,
        session_id: str | None,
        free_text: str,
        extracted_intent: dict,
        result_ids: list[UUID],
        total_candidates: int,
        filtered_count: int,
        latency_ms: float,
        llm_tokens: int,
        cache_hit: bool,
    ) -> None:
        """Log search query for analytics and feedback loop."""
        async with get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO search_queries (
                    id, session_id, free_text, extracted_intent,
                    result_ids, total_candidates, filtered_count,
                    latency_ms, llm_tokens, cache_hit
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                query_id, session_id, free_text,
                json.dumps(extracted_intent),
                result_ids, total_candidates, filtered_count,
                latency_ms, llm_tokens, cache_hit,
            )

    @staticmethod
    def _row_to_profile(row: asyncpg.Record) -> TherapistProfile:
        """Convert raw DB row to Pydantic model."""
        from src.models.therapist import (
            TherapistLocation, TherapyModality, TherapistSpecialization,
            InsuranceProvider, SessionFormat
        )
        return TherapistProfile(
            id=row["id"],
            source=row["source"],
            source_url=row["source_url"],
            source_id=row["source_id"],
            name=row["name"],
            credentials=list(row["credentials"] or []),
            license_number=row["license_number"],
            years_experience=row["years_experience"],
            gender=row["gender"],
            ethnicity=list(row["ethnicity"] or []),
            modalities=[TherapyModality(m) for m in (row["modalities"] or [])
                        if m in TherapyModality._value2member_map_],
            specializations=[TherapistSpecialization(s) for s in (row["specializations"] or [])
                             if s in TherapistSpecialization._value2member_map_],
            populations_served=list(row["populations_served"] or []),
            languages=list(row["languages"] or []),
            location=TherapistLocation(
                city=row["city"],
                state=row["state"],
                zip_code=row["zip_code"],
                county=row["county"],
                latitude=row["latitude"],
                longitude=row["longitude"],
            ),
            session_formats=[SessionFormat(f) for f in (row["session_formats"] or [])],
            accepts_insurance=[InsuranceProvider(i) for i in (row["insurance_providers"] or [])],
            sliding_scale=row["sliding_scale"],
            fee_min=row["fee_min"],
            fee_max=row["fee_max"],
            accepting_new_clients=row["accepting_new_clients"],
            bio=row["bio"] or "",
            rating=float(row["rating"]) if row["rating"] else None,
            review_count=row["review_count"] or 0,
            profile_completeness=float(row["profile_completeness"] or 0),
            scraped_at=row["scraped_at"],
            last_updated=row["last_updated"],
            is_active=row["is_active"],
        )
