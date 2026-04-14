# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup
```bash
pip install -r requirements.txt
python scripts/setup_dev.py        # Full dev environment setup
```

### Database
```bash
docker-compose up -d postgres redis   # Start infrastructure
python scripts/migrate.py             # Run SQL migrations
python scripts/seed_data.py           # Load synthetic therapist data
```

### Running the App
```bash
uvicorn src.api.main:app --reload --port 8000   # API (hot-reload)
streamlit run frontend/app.py                    # Frontend
docker-compose up                                # Full stack
```

### Testing
```bash
pytest tests/                     # All tests
pytest tests/unit/                # Unit tests only
pytest tests/integration/         # Integration tests
pytest tests/unit/test_hybrid_ranker.py   # Single test file
```

### Demo (no infrastructure required)
```bash
python demo.py                    # Interactive mode
python demo.py "I have anxiety"   # Single query
```

## Architecture

Therapize is an AI-powered therapist search engine for California. The system has three main concerns: data ingestion, search/matching, and serving.

### Request Flow
1. `POST /api/v1/search` → Redis cache check (SHA256 hash key)
2. Cache miss → `src/workflow/search_graph.py` (LangGraph orchestration)
3. `QueryProcessor` calls Claude Haiku via `instructor` to extract emotional concerns → modalities (structured output, temperature=0)
4. `FilterEngine` applies hard SQL constraints (location, insurance, budget)
5. `HybridRanker` scores candidates with 4-signal composite: modality match (0.40) + semantic similarity (0.35) + BM25 (0.15) + quality (0.10)
6. Cross-encoder reranks top-20 → return ranked list with scoring breakdown
7. Result cached in Redis (1hr TTL), query stored for evaluation

### Key Design Decisions

**PostgreSQL + pgvector (not Qdrant/Pinecone):** Therapist data is highly structured. SQL hard filters run *before* vector ANN search, which is far more efficient than post-filtering. HNSW index (`m=16, ef=64`) gives >0.95 recall.

**Local embeddings (`all-MiniLM-L6-v2`, 384 dims):** Zero per-token cost vs OpenAI. Embeddings are cached so the same query is never re-embedded.

**Claude Haiku + instructor:** Query extraction is simple classification. Haiku is cheap (~$0.25/1M tokens); `instructor` enforces structured Pydantic output via tool-use, so parsing never fails. `temperature=0` makes extraction deterministic.

**LangGraph workflow (`src/workflow/search_graph.py`):** Each node is independently testable. Conditional edges handle graceful degradation. LangSmith traces every node for observability.

**Modality-aware ranking:** LLM extracts concerns ("anxiety"), a deterministic JSON map (`src/matching/knowledge/modality_map.json`) converts concerns to evidence-based modalities (CBT, DBT, EMDR, ACT), then the ranker scores therapist profiles against those modalities. Accuracy without per-ranking LLM cost.

### Module Map
| Concern | Location |
|---|---|
| LLM intent extraction | `src/matching/query_processor.py` |
| Concern→modality knowledge | `src/matching/knowledge/modality_map.json` |
| Composite scoring | `src/matching/hybrid_ranker.py` |
| Hard constraint filtering | `src/matching/filter_engine.py` |
| LangGraph orchestration | `src/workflow/search_graph.py` |
| PostgreSQL + pgvector client | `src/storage/database.py` |
| Redis cache | `src/storage/cache.py` |
| DB schema | `src/storage/migrations/001_initial.sql` |
| All config/settings | `src/config.py` |
| Prometheus metrics | `src/monitoring/metrics.py` |
| LangSmith tracing | `src/monitoring/tracing.py` |
| Streamlit frontend | `frontend/app.py` |

### Data Models
- `TherapistProfile` (`src/models/therapist.py`): canonical schema with arrays for modalities, specializations, insurance providers; `embedding` field (384-dim vector); type-safe enums for all categorical fields
- `SearchRequest` / `SearchResponse` (`src/models/query.py`): request includes free-text + structured questionnaire filters; response includes scoring breakdown, `cache_hit`, `llm_cost_usd`

### Configuration
All config lives in `src/config.py` (Pydantic settings), validated at startup. Key env vars (see `.env.example`):
- `ANTHROPIC_API_KEY` — required
- `DATABASE_URL`, `REDIS_URL` — required for non-demo mode
- `LLM_MODEL` — defaults to `claude-haiku-4-5-20251001`
- `EMBEDDING_MODEL` — defaults to `all-MiniLM-L6-v2`
- `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` — optional LangSmith tracing

### Infrastructure
- **PostgreSQL** (pgvector): therapists table with HNSW vector index, GIN indexes on array columns, partial index on `(is_active AND accepting_new_clients)`
- **Redis**: query-level cache with gzip compression, also backs the rate limiter (30 req/min per IP)
- **Prometheus + Grafana**: metrics at `/metrics`, dashboards on port 3000
- **Docker Compose**: full stack with health checks, resource limits, persistent volumes

## Code Style

### Docstrings
Every function and method must have a docstring. Follow these conventions:

- **Single-line docstring** for simple, obvious functions: one sentence describing what it does (not how).
- **Multi-line docstring** for non-trivial functions: include a summary line, then `Args:`, `Returns:`, and any important design notes as needed.
- **LangGraph node methods** (`_*_node`, `_extract_intent`, etc.): prefix the docstring with `LangGraph node:` to make the graph structure scannable.
- **Abstract methods**: describe the contract the subclass must fulfil.
- Docstrings go on the line immediately after the `def` line, before any code.

Example:
```python
def my_func(x: int) -> bool:
    """Return True if x is positive."""
    return x > 0
```
