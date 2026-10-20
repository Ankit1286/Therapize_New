# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup
```bash
pip install -r requirements.txt
python scripts/setup_dev.py        # Migrations + scrape 10 real therapists into Neon
```

### Database
Hosted services, no local Docker: Neon (Postgres + pgvector) via `DATABASE_URL`, Upstash Redis via `REDIS_URL` (must be `rediss://`).
```bash
python scripts/migrate.py             # Run SQL migrations
python scripts/seed_data.py           # Load synthetic therapist data
python scripts/ingest_small.py        # Small real scrape (~5 per source)
python scripts/run_ingestion.py       # Full scrape
```

### Running the App
```bash
uvicorn src.api.main:app --reload --port 8000   # API (hot-reload)
cd web && npm install && npm run dev            # Next.js frontend on :3000
```

### Testing
```bash
pytest tests/                     # All tests (unit only; no integration suite yet)
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
3. `QueryProcessor` calls Claude Haiku via `instructor` to extract emotional concerns + inferred filters (structured output, temperature=0); `ModalityMapper` turns concerns into weighted modalities
4. `FilterEngine` builds hard SQL constraints (location, insurance, budget, format, gender, ethnicity, language, age group); SQL filter + pgvector ANN fetches up to 100 candidates
5. If < 3 candidates, `_progressive_relax` drops filters in order: budget → session format → insurance → city → gender → ethnicity → language → CA-only safety net (age group never dropped)
6. `HybridRanker` scores candidates: modality match (0.40) + semantic similarity (0.35) + BM25 (0.15) + quality (0.10); with a city, a location signal (0.15) is added and the others scale to 0.35/0.30/0.13/0.07. There is no cross-encoder — top 10 returned with scoring breakdown
7. Result cached in Redis (1hr TTL), query stored in `search_queries` for evaluation

### Key Design Decisions

**PostgreSQL + pgvector (not Qdrant/Pinecone):** Therapist data is highly structured. SQL hard filters run *before* vector ANN search, which is far more efficient than post-filtering. HNSW index (`m=16, ef=64`) gives >0.95 recall.

**Local embeddings (`all-MiniLM-L6-v2`, 384 dims):** Zero per-token cost vs OpenAI. Profile embeddings are computed at ingestion and stored in pgvector (only re-embedded when profile text changes); repeat queries are served from the Redis result cache.

**Claude Haiku + instructor:** Query extraction is simple classification. Haiku 4.5 costs $1/1M input and $5/1M output tokens (~$0.002–0.003 per uncached search; `estimate_llm_cost` in `src/monitoring/metrics.py` still uses an outdated $0.25/1M rate and under-reports); `instructor` enforces structured Pydantic output via tool-use, so parsing never fails. `temperature=0` makes extraction deterministic.

**LangGraph workflow (`src/workflow/search_graph.py`):** Each node is independently testable. Conditional edges handle graceful degradation. LangSmith traces every node for observability.

**Modality-aware ranking:** LLM extracts concerns ("anxiety"), a deterministic JSON map (`src/matching/knowledge/modality_map.json`) converts concerns to evidence-based modalities (CBT, DBT, EMDR, ACT), then the ranker scores therapist profiles against those modalities. Accuracy without per-ranking LLM cost.

### Module Map
| Concern | Location |
|---|---|
| LLM intent extraction | `src/matching/query_processor.py` |
| Concern→modality knowledge | `src/matching/knowledge/modality_map.json`, `src/matching/modality_mapper.py` |
| Composite scoring | `src/matching/hybrid_ranker.py` |
| Hard constraint filtering | `src/matching/filter_engine.py` |
| LangGraph orchestration | `src/workflow/search_graph.py` |
| PostgreSQL + pgvector client | `src/storage/database.py` |
| Redis cache | `src/storage/cache.py` |
| DB schema | `src/storage/migrations/` (`001_initial.sql`, `002_add_ethnicity.sql`) |
| Scrapers | `src/scrapers/` (OpenPath via Algolia, GoodTherapy via `__NEXT_DATA__` JSON) |
| Ingestion pipeline | `src/pipeline/` (`ingestion.py`, `cleaner.py`, `embeddings.py`) |
| All config/settings | `src/config.py` |
| Prometheus metrics | `src/monitoring/metrics.py` |
| LangSmith tracing | `src/monitoring/tracing.py` |
| Offline evaluation | `src/monitoring/evaluation.py`, `scripts/evaluate_ranker.py` |
| Next.js frontend | `web/` (`app/page.tsx`, `components/`, `lib/api.ts`) |

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
- **Prometheus**: metrics exposed at `/metrics`; no Prometheus/Grafana server is included or deployed
- **Production**: frontend on Vercel; backend on a GCP e2-micro VM (`scripts/setup_server.sh`: uvicorn under systemd behind nginx, certbot HTTPS, DuckDNS domain); server env in `/etc/therapize.env`; `scripts/weekly_update.sh` must run via a systemd timer with `EnvironmentFile=/etc/therapize.env` (cron won't load the env). `railway.toml` / `Procfile` are legacy and unused

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
