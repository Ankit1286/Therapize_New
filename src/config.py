"""
Central configuration — all settings derived from environment variables.
Uses pydantic-settings so misconfigured deployments fail fast at startup.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────────────────────
    app_name: str = "Therapize"
    app_version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # ── LLM ──────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(..., description="Anthropic API key")
    # Model choices: claude-haiku for cost efficiency (~$0.25/1M input tokens)
    # claude-sonnet for higher quality, claude-opus for maximum quality
    llm_model: str = "claude-haiku-4-5-20251001"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    llm_temperature: float = 0.0   # Determinism: always 0 for extraction tasks
    llm_max_tokens: int = 1024

    # ── LangSmith (observability) ─────────────────────────────────────────
    langchain_api_key: str = Field("", description="LangSmith API key (optional)")
    langchain_project: str = "therapize"
    langchain_tracing_v2: bool = False

    # ── Database ─────────────────────────────────────────────────────────
    # PostgreSQL chosen over dedicated vector DB because:
    # 1. Therapist data is structured (insurance, price, location) — needs SQL
    # 2. Filtering BEFORE vector search is far more efficient
    # 3. pgvector supports HNSW for fast ANN with recall > 0.95
    # 4. One system of record — no sync complexity
    database_url: str = Field(
        "postgresql+asyncpg://therapize:therapize@localhost:5432/therapize"
    )

    @field_validator("database_url", mode="before")
    @classmethod
    def fix_postgres_scheme(cls, v: str) -> str:
        # Railway (and most platforms) emit postgresql:// — asyncpg needs +asyncpg
        if isinstance(v, str) and v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30

    # ── Redis (caching + rate limiting) ──────────────────────────────────
    # Cache hit reduces latency from ~800ms → ~5ms and eliminates LLM cost
    redis_url: str = Field("redis://localhost:6379/0")

    @field_validator("redis_url", mode="before")
    @classmethod
    def validate_redis_url(cls, v: str) -> str:
        if isinstance(v, str) and v.startswith(("redis://", "rediss://")):
            return v
        raise ValueError("redis_url must start with redis:// or rediss://")
    cache_ttl_seconds: int = 3600          # 1 hour: therapist data is stable
    rate_limit_per_minute: int = 30        # per IP

    # ── Matching Engine ───────────────────────────────────────────────────
    # Weights must sum to 1.0
    # Modality weight is highest — this is the domain-specific insight
    weight_modality: float = 0.40
    weight_semantic: float = 0.35
    weight_bm25: float = 0.15
    weight_recency: float = 0.05
    weight_rating: float = 0.05
    rerank_top_k: int = 20      # re-rank top 20 with cross-encoder
    final_top_k: int = 10       # return top 10 to user

    # ── Scraping ──────────────────────────────────────────────────────────
    scraper_delay_seconds: float = 2.0     # polite delay between requests
    scraper_max_retries: int = 3
    scraper_timeout_seconds: int = 30
    scraper_user_agent: str = (
        "Mozilla/5.0 (compatible; TherapizeBot/1.0; research purposes)"
    )

    @field_validator("weight_modality", "weight_semantic", "weight_bm25",
                     "weight_recency", "weight_rating", mode="before")
    @classmethod
    def weights_must_be_positive(cls, v: float) -> float:
        assert 0 <= v <= 1, "Weights must be between 0 and 1"
        return v

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    """Cached settings — loaded once per process."""
    return Settings()
