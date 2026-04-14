"""
LangSmith tracing integration.

What gets traced:
- Every LLM call: input, output, latency, token count, model
- Query processor runs: extracted intent quality
- Full search workflow: end-to-end trace with child spans

Why LangSmith over just logging:
- Structured traces (parent/child spans) make latency bottlenecks obvious
- You can replay failed queries for debugging
- Built-in prompt regression testing: compare before/after prompt changes
- Token cost analytics out of the box

Setup:
- Set LANGCHAIN_API_KEY and LANGCHAIN_TRACING_V2=true in .env
- Traces appear at smith.langchain.com under the "therapize" project
"""
import logging
import os
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def configure_tracing() -> None:
    """
    Configure LangSmith tracing at app startup.
    Only activates if LANGCHAIN_API_KEY is set.
    """
    if settings.langchain_api_key and settings.langchain_tracing_v2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        logger.info(
            "LangSmith tracing enabled (project: %s)", settings.langchain_project
        )
    else:
        logger.info(
            "LangSmith tracing disabled. "
            "Set LANGCHAIN_API_KEY and LANGCHAIN_TRACING_V2=true to enable."
        )


def trace_search(func: Callable) -> Callable:
    """
    Decorator to add LangSmith tracing to search functions.
    Falls back to no-op if tracing is disabled.
    """
    if settings.langchain_tracing_v2:
        try:
            from langsmith import traceable
            return traceable(name=func.__name__, run_type="chain")(func)
        except ImportError:
            pass

    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)
    return wrapper


class StructuredLogger:
    """
    JSON-structured logging for machine-parseable logs.

    Why structured logging:
    - Grep/query by any field: grep '"latency_ms":[0-9]*[5-9][0-9][0-9]' = slow queries
    - Sends to log aggregation (Datadog, CloudWatch, etc.) without parsing
    - Consistent schema across all services

    In production: ship these logs to CloudWatch or Datadog for dashboarding.
    """

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def search_completed(
        self,
        query_id: str,
        latency_ms: float,
        num_results: int,
        cache_hit: bool,
        llm_tokens: int,
        top_score: float,
    ) -> None:
        """Emit a structured log event when a search request completes successfully."""
        self._logger.info(
            "search_completed",
            extra={
                "event": "search_completed",
                "query_id": query_id,
                "latency_ms": round(latency_ms, 2),
                "num_results": num_results,
                "cache_hit": cache_hit,
                "llm_tokens": llm_tokens,
                "top_score": round(top_score, 4),
            },
        )

    def search_failed(self, query_id: str, error: str, stage: str) -> None:
        """Emit a structured error log when a search request fails, including the pipeline stage where it failed."""
        self._logger.error(
            "search_failed",
            extra={
                "event": "search_failed",
                "query_id": query_id,
                "error": error,
                "stage": stage,  # query_processor, filter, ranking, etc.
            },
        )

    def slow_query_warning(self, query_id: str, latency_ms: float, stage: str) -> None:
        """Alert on slow queries for SLO tracking."""
        if latency_ms > 2000:
            self._logger.warning(
                "slow_query",
                extra={
                    "event": "slow_query",
                    "query_id": query_id,
                    "latency_ms": round(latency_ms, 2),
                    "stage": stage,
                },
            )

    def feedback_received(
        self,
        query_id: str,
        therapist_id: str,
        rating: int,
        booked: bool | None,
    ) -> None:
        """Emit a structured log event when a user submits feedback on a search result."""
        self._logger.info(
            "feedback_received",
            extra={
                "event": "feedback_received",
                "query_id": query_id,
                "therapist_id": therapist_id,
                "rating": rating,
                "booked": booked,
            },
        )
