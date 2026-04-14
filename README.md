# Therapize — AI-Powered Therapist Search Engine

> Production-grade AI matching system that connects users with the right therapist in California using emotional-aware query understanding, modality-based ranking, and hybrid semantic search.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE                               │
│                                                                     │
│  OpenPath Collective ──► Algolia API (JSON) ──┐                     │
│  GoodTherapy ──────────► HTML scraping ───────┼──► Cleaner ──► DB   │
│                          (httpx + BS4)        │              │      │
│                               │               │              ▼      │
│                         Rate Limiter          │    PostgreSQL +     │
│                         Robots.txt check      │     pgvector        │
│                                               │    (therapist       │
│                                               │     profiles +      │
│                                               │     embeddings)     │
└──────────────────────────────────────────┬────┴─────────────────────┘
                                           │
┌──────────────────────────────────────────▼──────────────────────────┐
│                       MATCHING ENGINE (Core)                         │
│                                                                     │
│  User Query (free text + optional GUI filters)                      │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────┐   ┌──────────────────────────────────────┐     │
│  │  Query Processor │   │           Scoring Pipeline            │     │
│  │  (Claude Haiku) │   │                                       │     │
│  │                 │   │  1. Hard SQL Filter                   │     │
│  │  Extracts:      │   │     (location/insurance/budget/format)│     │
│  │  - emotional    │   │  2. Vector ANN (pgvector HNSW)        │     │
│  │    concerns     │   │     → top-K candidates                │     │
│  │  - logistical   │   │  3. Modality Match Score   (0.40)     │     │
│  │    filters from │   │     concerns → modality_map.json      │     │
│  │    free text    │   │     → therapist modality overlap      │     │
│  └────────┬────────┘   │  4. Semantic Score         (0.35)     │     │
│           │            │     cosine_sim(query_emb, profile_emb)│     │
│           │            │  5. BM25 Lexical Score      (0.15)    │     │
│           │            │  6. Quality Score            (0.10)   │     │
│           │            │     rating × profile completeness     │     │
│           └───────────►│  → composite score, sorted descending │     │
│                        └──────────────────────────────────────┘     │
└──────────────────────────────────────────┬──────────────────────────┘
                                           │
┌──────────────────────────────────────────▼──────────────────────────┐
│                          API LAYER (FastAPI)                         │
│                                                                     │
│   /search ──► Redis Cache Check ──► Matching Engine ──► Response    │
│   /feedback ──► Feedback Store ──► Metrics Update                   │
│   /health ──► DB ping + cache ping                                  │
│   /metrics ──► Prometheus scrape endpoint                           │
└──────────────────────────────────────────┬──────────────────────────┘
                                           │
┌──────────────────────────────────────────▼──────────────────────────┐
│                       OBSERVABILITY STACK                            │
│                                                                     │
│  LangSmith ──── LLM call traces, extracted intent per query        │
│  Prometheus ─── latency, token cost, cache hit rate, match scores  │
│  Structured Logs ── JSON logs (slow query alerts, search events)   │
│  Feedback Loop ── thumbs up/down → search_queries + feedback tables│
│                   → offline NDCG/MRR evaluation pipeline           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Request Flow (Step by Step)

```
POST /api/v1/search { free_text, questionnaire }
```

1. **Redis cache check** — SHA256 hash of `(free_text + questionnaire)` as key.
   Cache hit → return in ~5ms. Cache miss → continue.

2. **Query Processor (Claude Haiku + instructor)**
   - Extracts `emotional_concerns` from free text (e.g. `["anxiety", "rumination"]`)
   - Extracts logistical filters implied by free text (city, insurance, budget, format)
   - Merges with explicit GUI questionnaire filters — explicit always wins
   - Temperature = 0 for determinism. `instructor` enforces Pydantic output via tool-use.

3. **Embed query** — `all-MiniLM-L6-v2` (local, 384-dim). Zero API cost.

4. **Filter + vector search** (single DB round-trip)
   ```sql
   SELECT ... FROM therapists
   WHERE state = 'CA' AND accepting_new_clients = true
     AND <insurance/city/budget filters>
   ORDER BY embedding <=> $query_vector   -- pgvector HNSW cosine distance
   LIMIT 100
   ```
   SQL hard filters run first (GIN + B-tree indexes), then HNSW ANN on the
   filtered set. Far more efficient than vector search then filter.

5. **Fallback** — if fewer than 3 results, relax all filters to just `state=CA`.

6. **Modality mapping** — `emotional_concerns` → `modality_map.json` → weighted
   modality dict. Deterministic, no LLM cost per ranking.

7. **Hybrid ranking** over the candidate set:
   - Modality score: therapist modalities vs. concern-mapped modalities (weight 0.40)
   - Semantic score: cosine similarity of embeddings (weight 0.35)
   - BM25: keyword overlap on therapist bios (weight 0.15)
   - Quality: log-dampened rating × profile completeness (weight 0.10)
   All signals normalized to [0,1] before combining.

8. **Response** — top 10 results with full score breakdown, matched modalities,
   bio excerpt, and human-readable match explanation.

9. **Cache + log** — result cached in Redis (1hr TTL). Query + result IDs logged
   to `search_queries` table for offline evaluation.

---

## Key Design Decisions

### LLM extracts concerns only — not modalities
The LLM (`query_processor.py`) extracts `emotional_concerns` (plain language:
`["anxiety", "grief"]`). Modality mapping is handled deterministically by
`modality_map.json` (`matching/knowledge/`). This keeps the LLM schema small
(fewer tokens → faster, cheaper) and the ranking logic auditable without LLM cost.

### PostgreSQL + pgvector (not a dedicated vector DB)
Therapist data is highly structured — insurance, location, price need SQL.
Filtering **before** vector search is far more efficient than post-filtering.
Single source of truth — no sync between two databases. HNSW index gives >0.95
recall. Use Qdrant/Pinecone instead only if you need billions of vectors or
multi-tenant isolation.

### Local embeddings (`all-MiniLM-L6-v2`, 384 dims)
Zero per-token cost vs OpenAI. Loaded once at startup, cached in memory.
Same query is never re-embedded (Redis cache on full response).

### Binary feedback (thumbs up/down)
UI collects 👍 (rating=5) / 👎 (rating=1). Stored in `feedback` table linked
to `search_queries`. Feeds offline NDCG/MRR evaluation in `evaluation.py`.
With binary labels, MRR is the most meaningful metric: "how far did the user
scroll before finding a good therapist?"

### Redis cache
Same query → same result in ~5ms instead of ~2-8s. TTL = 1hr. Cache key =
SHA256(free_text + questionnaire JSON). Also backs the rate limiter (30 req/min
per IP, sliding window).

### LangGraph (optional orchestration)
The pipeline runs as a LangGraph graph when installed, or falls back to a plain
sequential function (`_run_pipeline`). The graph adds independent node
testability and LangSmith traces per node. The pipeline is a straight line with
one conditional edge (fallback on empty results) — straightforward enough that
LangGraph is optional overhead.

---

## Matching Algorithm

```
User: "I've been having panic attacks and can't stop ruminating. Feel dysfunctional."

Step 1 — Query Processor (LLM):
  emotional_concerns: ["anxiety", "panic attacks", "rumination", "functional impairment"]
  inferred_filters:   {}  (no logistical info in query)

Step 2 — Modality Mapper (modality_map.json lookup, no LLM):
  anxiety       → CBT (0.95), ACT (0.85), mindfulness (0.75)
  panic attacks → CBT (0.95), exposure_therapy (0.90)
  rumination    → CBT (0.90), mindfulness (0.80), ACT (0.85)
  → merged: { CBT: 0.95, exposure_therapy: 0.90, ACT: 0.85, mindfulness: 0.80 }

Step 3 — Hybrid Scoring (per candidate therapist):
  modality_score  = overlap(therapist.modalities, mapped_modalities)  [0.40]
  semantic_score  = cosine_sim(query_embedding, profile_embedding)     [0.35]
  bm25_score      = bm25(query_tokens, therapist_bio_text)             [0.15]
  quality_score   = log_dampened_rating × profile_completeness         [0.10]

  composite = 0.40×mod + 0.35×sem + 0.15×bm25 + 0.10×quality
  → top 10 sorted descending, with score breakdown returned in response
```

---

## Observability

**LangSmith** — traces every search at `smith.langchain.com` (project: `therapize`).
Each trace shows the `query_processor.extract_intent` span with the full
extracted intent (concerns, inferred filters, confidence, summary).
Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in `.env` to enable.

**Prometheus** — metrics at `/metrics`:
- `therapize_search_latency_ms` — P50/P99 latency histogram
- `therapize_llm_tokens_total` — cumulative token spend
- `therapize_llm_cost_usd_total` — estimated USD cost
- `therapize_cache_hits_total` / `misses` — cache hit rate
- `therapize_modality_match_score` — distribution of top-result modality scores
- `therapize_candidates_after_filter` — filter restrictiveness signal
- `therapize_evaluation_ndcg_at_10` / `mrr` — updated by offline eval pipeline

**Offline evaluation** (`src/monitoring/evaluation.py`) — computes NDCG@10, MRR,
Precision@5 from the `feedback` table. Run after prompt or weight changes to
catch regressions. Results written to `evaluation_runs` table and Prometheus gauges.

---

## Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Required: ANTHROPIC_API_KEY, DATABASE_URL, REDIS_URL
# Optional: LANGCHAIN_API_KEY, LANGCHAIN_TRACING_V2=true

# 3. Run database migrations
python scripts/migrate.py

# 4. Load therapist data
python scripts/ingest_small.py     # ~10 real therapists (dev/testing)
python scripts/seed_data.py        # synthetic data (no scraping needed)

# 5. Start API
python -m uvicorn src.api.main:app --reload --port 8000

# 6. Start frontend
streamlit run frontend/app.py      # http://localhost:8501
```

**Infrastructure** (Postgres + Redis) can be self-hosted via Docker or use
managed services. The codebase is tested against Neon (serverless Postgres) and
Upstash (serverless Redis) for zero-ops dev setup.

---

## Cost Estimation

Per 1000 queries (no cache hits):
- Claude Haiku extraction: ~800 tokens × $0.00000025 = ~$0.0002
- `all-MiniLM-L6-v2` embedding: local, $0
- **Total LLM cost: ~$0.20 per 1,000 queries**

With a 60% cache hit rate (repeat queries):
- Effective cost drops to ~$0.08 per 1,000 queries

Infrastructure (managed Postgres + Redis): ~$20-50/month depending on tier.

---

## Evaluation Metrics

**Online** (automatic, every request):
| Metric | Signal |
|---|---|
| `search_latency_ms` | User-facing SLO |
| `modality_match_score` | Proxy for ranking quality — no feedback needed |
| `candidates_after_filter` | Filter restrictiveness |
| `cache_hit_rate` | Cost efficiency |

**Offline** (requires user feedback, run periodically):
| Metric | What it measures |
|---|---|
| NDCG@10 | Ranking quality — are relevant results ranked first? |
| MRR | How far the user scrolls to find a good match |
| Precision@5 | Fraction of top-5 results that are relevant |
| Booking rate | `booked=true` in feedback — the ground truth signal |

Feedback is binary (👍 = relevant, 👎 = irrelevant). With binary labels, MRR
is the most informative metric.

---

## Project Structure

```
Therapize/
├── src/
│   ├── config.py                    # Pydantic settings (all config in one place)
│   ├── models/
│   │   ├── therapist.py             # TherapistProfile schema + enums
│   │   └── query.py                 # SearchRequest, SearchResponse, ExtractedQueryIntent
│   ├── scrapers/
│   │   ├── base.py                  # Abstract scraper (retry, rate-limit, robots.txt)
│   │   ├── open_path.py             # Algolia API scraper (OpenPath Collective)
│   │   └── good_therapy.py          # HTML scraper (GoodTherapy.org)
│   ├── pipeline/
│   │   ├── ingestion.py             # Orchestrates scraping → embed → DB upsert
│   │   └── embeddings.py            # Local sentence-transformers embedding pipeline
│   ├── matching/
│   │   ├── query_processor.py       # LLM extraction (concerns + inferred filters)
│   │   ├── modality_mapper.py       # concerns → modality weights (JSON map lookup)
│   │   ├── hybrid_ranker.py         # 4-signal composite scorer
│   │   ├── filter_engine.py         # SQL WHERE clause builder
│   │   └── knowledge/
│   │       └── modality_map.json    # Curated concern → evidence-based modality map
│   ├── storage/
│   │   ├── database.py              # asyncpg PostgreSQL + pgvector client
│   │   ├── cache.py                 # Redis async client + rate limiter
│   │   └── migrations/001_initial.sql
│   ├── api/
│   │   ├── main.py                  # FastAPI app + lifespan + middleware
│   │   ├── routes/
│   │   │   ├── search.py            # POST /api/v1/search
│   │   │   ├── feedback.py          # POST /api/v1/feedback
│   │   │   └── health.py            # GET /health, /health/deep
│   │   └── middleware/
│   │       └── rate_limiter.py      # Redis sliding window rate limiter
│   ├── monitoring/
│   │   ├── metrics.py               # Prometheus metrics registry
│   │   ├── tracing.py               # LangSmith config + structured logger
│   │   └── evaluation.py            # Offline NDCG/MRR evaluation pipeline
│   └── workflow/
│       └── search_graph.py          # LangGraph orchestration + sequential fallback
├── frontend/
│   └── app.py                       # Streamlit UI (search + score breakdown + feedback)
├── scripts/
│   ├── ingest_small.py              # Scrape ~10 real therapists (dev)
│   ├── seed_data.py                 # Load synthetic therapist data
│   └── migrate.py                   # Run SQL migrations
├── docker-compose.yml
├── requirements.txt
└── .env.example
```
