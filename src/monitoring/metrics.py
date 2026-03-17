"""
Prometheus metrics registry.

What to track and why:
- search_latency: P50/P99 are the user-facing SLOs
- llm_cost: directly translates to AWS bill
- cache_hit_rate: proxy for cost efficiency
- modality_match_score: quality of the matching engine
- feedback_ratings: ground truth for ranking quality

Tracking philosophy:
- Track what you can act on
- Alert on: search_latency_p99 > 2s, error_rate > 1%, cache_hit_rate < 40%
- Review monthly: ndcg_score, mrr, feedback_ratings

Usage:
    from src.monitoring.metrics import search_latency
    search_latency.observe(latency_ms)
"""
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    CollectorRegistry,
)

# Use default registry (Prometheus scrapes /metrics endpoint)
REGISTRY = CollectorRegistry()

# ── Latency ──────────────────────────────────────────────────────────────────
# Buckets tuned for typical search latency: most requests in 100-500ms range
search_latency = Histogram(
    "therapize_search_latency_ms",
    "End-to-end search latency in milliseconds",
    buckets=[50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000],
    registry=REGISTRY,
)

query_processor_latency = Histogram(
    "therapize_query_processor_latency_ms",
    "LLM query extraction latency in milliseconds",
    buckets=[100, 200, 300, 500, 750, 1000, 2000],
    registry=REGISTRY,
)

db_query_latency = Histogram(
    "therapize_db_query_latency_ms",
    "Database query latency in milliseconds",
    buckets=[5, 10, 20, 50, 100, 200, 500],
    registry=REGISTRY,
)

# ── Throughput ────────────────────────────────────────────────────────────────
search_requests_total = Counter(
    "therapize_search_requests_total",
    "Total number of search requests",
    ["status"],  # labels: success, error, rate_limited
    registry=REGISTRY,
)

# ── Cost tracking ─────────────────────────────────────────────────────────────
# Token costs as of March 2025: gpt-4o-mini = $0.00015/1K input tokens
llm_tokens_used = Counter(
    "therapize_llm_tokens_total",
    "Total LLM tokens consumed",
    registry=REGISTRY,
)

llm_cost_usd = Counter(
    "therapize_llm_cost_usd_total",
    "Estimated LLM cost in USD",
    registry=REGISTRY,
)

# ── Caching ───────────────────────────────────────────────────────────────────
cache_hits_total = Counter(
    "therapize_cache_hits_total",
    "Cache hit count",
    registry=REGISTRY,
)

cache_misses_total = Counter(
    "therapize_cache_misses_total",
    "Cache miss count",
    registry=REGISTRY,
)

# ── Quality ────────────────────────────────────────────────────────────────────
# Average modality match score per request (proxy for matching quality)
modality_match_score = Histogram(
    "therapize_modality_match_score",
    "Distribution of top-1 result modality match scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=REGISTRY,
)

# User feedback ratings
feedback_rating = Histogram(
    "therapize_feedback_rating",
    "User feedback ratings (1-5)",
    buckets=[1, 2, 3, 4, 5],
    registry=REGISTRY,
)

# Number of candidates after filtering (proxy for filter restrictiveness)
candidates_after_filter = Histogram(
    "therapize_candidates_after_filter",
    "Number of therapists remaining after hard filters",
    buckets=[0, 5, 10, 20, 50, 100, 200, 500],
    registry=REGISTRY,
)

# ── System health ─────────────────────────────────────────────────────────────
db_pool_size = Gauge(
    "therapize_db_pool_size",
    "Current database connection pool size",
    registry=REGISTRY,
)

active_searches = Gauge(
    "therapize_active_searches",
    "Currently in-flight search requests",
    registry=REGISTRY,
)

# ── Offline evaluation metrics (updated by evaluation pipeline) ────────────────
ndcg_score = Gauge(
    "therapize_evaluation_ndcg_at_10",
    "Latest NDCG@10 from offline evaluation run",
    registry=REGISTRY,
)

mrr_score = Gauge(
    "therapize_evaluation_mrr",
    "Latest Mean Reciprocal Rank from offline evaluation run",
    registry=REGISTRY,
)


def estimate_llm_cost(tokens: int, model: str = "claude-haiku-4-5-20251001") -> float:
    """
    Estimate LLM cost in USD for given token count.
    Prices as of March 2025 — update when pricing changes.
    """
    # Per-token costs (input tokens; output is ~4x for Haiku)
    COST_PER_TOKEN = {
        "claude-haiku-4-5-20251001": 0.00000025,   # $0.25/1M input
        "claude-sonnet-4-6": 0.000003,              # $3/1M input
        "claude-opus-4-6": 0.000015,                # $15/1M input
        "all-MiniLM-L6-v2": 0.0,                    # local = free
    }
    cost_per_token = COST_PER_TOKEN.get(model, 0.000001)
    return tokens * cost_per_token
