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
│                               ▼               │    PostgreSQL +     │
│                         Rate Limiter          │     pgvector        │
│                         Robots.txt check      │    (therapist       │
│                                               │     profiles +      │
│                                               │     embeddings)     │
└──────────────────────────────────────────┬────┴─────────────────────┘
                                           │
┌──────────────────────────────────────────▼──────────────────────────┐
│                       MATCHING ENGINE (Core)                         │
│                                                                     │
│  User Query                                                         │
│  + Questionnaire                                                    │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────────┐    ┌──────────────────────────────────────┐       │
│  │   LangGraph  │    │           Scoring Pipeline            │       │
│  │   Workflow   │    │                                       │       │
│  │              │    │  1. Hard Filter (location/insurance)  │       │
│  │  ┌─────────┐ │    │  2. Modality Match Score (0.40)       │       │
│  │  │  Query  │ │    │     concerns → modalities → therapist │       │
│  │  │Processor│─┼───►│  3. Semantic Vector Score  (0.35)     │       │
│  │  │  (LLM)  │ │    │  4. BM25 Lexical Score     (0.15)     │       │
│  │  └─────────┘ │    │  5. Quality Score           (0.10)    │       │
│  │              │    │  6. Composite Score = weighted sum    │       │
│  │  Extracts:   │    └──────────────────────────────────────┘       │
│  │  - concerns  │                                                   │
│  │  - modalities│                                                   │
│  │  - filters   │                                                   │
│  └──────────────┘                                                   │
└──────────────────────────────────────────┬──────────────────────────┘
                                           │
┌──────────────────────────────────────────▼──────────────────────────┐
│                          API LAYER (FastAPI)                         │
│                                                                     │
│   /search ──► Redis Cache Check ──► Matching Engine ──► Response    │
│   /feedback ──► Feedback Store ──► Metrics Update                   │
│   /health ──► DB ping + cache ping + model ping                     │
│   /metrics ──► Prometheus scrape endpoint                           │
└──────────────────────────────────────────┬──────────────────────────┘
                                           │
┌──────────────────────────────────────────▼──────────────────────────┐
│                       OBSERVABILITY STACK                            │
│                                                                     │
│  LangSmith ──── LLM call traces, latency, token costs              │
│  Prometheus ─── system metrics (latency P50/P99, error rate)       │
│  Structured Logs ── JSON logs → queryable                          │
│  Feedback Loop ── user ratings → offline eval → model improvement  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Recruiter Questions — Answered in Code

| Question | Where in Code | File |
|----------|--------------|------|
| End-to-end system design | Full pipeline: scraper → DB → API | `src/pipeline/`, `src/api/` |
| Cost estimation | Token counter, cost logger | `src/monitoring/metrics.py` |
| Latency reduction | Redis cache, async scraping, ANN search | `src/storage/cache.py`, `src/api/middleware/` |
| Dataset construction | Scraper + cleaner + normalization | `src/scrapers/`, `src/pipeline/cleaner.py` |
| Loss function / ranking | NDCG-optimized composite scorer | `src/matching/hybrid_ranker.py` |
| Database choice | PostgreSQL + pgvector (why: structured + vector in one) | `src/storage/database.py` |
| Metrics tracking | Prometheus: latency, NDCG, user satisfaction | `src/monitoring/metrics.py` |
| System monitoring | LangSmith + structured logging + health checks | `src/monitoring/tracing.py` |
| Feedback loop | Explicit ratings + implicit signals → eval pipeline | `src/monitoring/evaluation.py` |
| Determinism | Temperature=0, structured outputs, rule-based filters | `src/matching/query_processor.py` |

---

## Why These Technology Choices

### PostgreSQL + pgvector (not a dedicated vector DB)
- Therapist profiles have **rich structured data** (insurance, location, price) that needs SQL filters
- Running vector search AFTER SQL filters is more efficient than filtering in a pure vector DB
- Single source of truth — no sync between two databases
- pgvector supports HNSW indexing for fast approximate nearest neighbor
- When to use Qdrant/Pinecone instead: if you need multi-tenant isolation, billions of vectors, or real-time vector updates

### Redis (caching layer)
- Search results cached by query hash — P99 latency drops from ~800ms to ~5ms on cache hit
- TTL = 1 hour (therapist data doesn't change minute to minute)
- Also used for rate limiting (sliding window counter)

### LangGraph (workflow orchestration)
- Stateful graph makes the matching pipeline debuggable and observable
- Each node can be tested independently
- Conditional edges handle edge cases (empty results → fallback strategy)
- Checkpointing allows resuming failed pipelines

### LangSmith (observability)
- Every LLM call is traced with input/output/latency/token count
- Enables cost tracking per query
- Enables prompt regression testing

### Scrapers: Algolia API (OpenPath) + HTML (GoodTherapy)
- **OpenPath**: queries the Algolia search API directly — fast, structured JSON, no HTML parsing
- **GoodTherapy**: BeautifulSoup HTML scraping; selectors centralized for easy maintenance
- Both use `httpx` async HTTP — no Playwright or headless browser required

---

## Matching Algorithm Deep Dive

The core innovation is **modality-aware ranking**:

```
User: "I've been having panic attacks since losing my job, can't sleep"

Step 1 — Query Processor (LLM):
  emotional_concerns: ["anxiety", "panic_disorder", "work_stress", "insomnia"]
  logistical_filters: {insurance: None, location: None, budget: None}

Step 2 — Modality Mapper (knowledge base lookup):
  anxiety       → [CBT, DBT, ACT, EMDR, mindfulness]
  panic_disorder → [CBT, exposure_therapy, EMDR]
  work_stress   → [ACT, solution_focused, coaching]
  insomnia      → [CBT-I, mindfulness, somatic]

  recommended_modalities = {CBT: 0.9, EMDR: 0.7, ACT: 0.8, DBT: 0.6, ...}

Step 3 — Hybrid Scoring (per therapist):
  modality_score  = overlap(therapist.modalities, recommended_modalities)  [weight: 0.40]
  semantic_score  = cosine_sim(query_embedding, therapist_embedding)        [weight: 0.35]
  bm25_score      = bm25(query_tokens, therapist_text)                      [weight: 0.15]
  quality_score   = normalized rating × profile completeness                [weight: 0.10]

  final_score = weighted_sum(all scores)
  → Returns top N results sorted by composite score
```

---

## Running Locally

```bash
# 1. Start infrastructure
docker-compose up -d postgres redis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env
# Edit .env — only ANTHROPIC_API_KEY is required

# 4. Run database migrations
python scripts/migrate.py

# 5. Load therapist data
python scripts/seed_data.py        # fast: synthetic data (no scraping)
python scripts/ingest_small.py     # real data: ~5 from each source (dev)
python scripts/run_ingestion.py    # real data: full scale

# 6. Start API
uvicorn src.api.main:app --reload --port 8000

# 7. Start frontend
streamlit run frontend/app.py
```

**Standalone demo** (no DB or Redis required — just `ANTHROPIC_API_KEY`):
```bash
python demo.py "I have anxiety and trouble sleeping"
```

---

## Metrics & Evaluation

**Online Metrics (Prometheus)**
- `search_latency_p50/p99` — user-facing latency
- `llm_token_cost_per_query` — cost per search
- `cache_hit_rate` — effectiveness of caching
- `result_click_rate` — proxy for relevance

**Offline Metrics (Evaluation Pipeline)**
- `NDCG@10` — ranking quality
- `MRR` — mean reciprocal rank
- `Precision@k` — how many top-k results are relevant

**Feedback Loop**
- User rates results (1-5 stars) or marks "not relevant"
- Feedback stored with query + results for offline eval
- Monthly evaluation run: compare NDCG before/after prompt changes

---

## Cost Estimation

Per 1000 queries:
- Claude Haiku for query extraction: ~500 tokens × $0.00000025 = $0.000125 (effectively free)
- all-MiniLM-L6-v2 for embedding: local, $0 — runs on CPU via sentence-transformers
- Total LLM cost: ~$0.000125 per 1000 queries
- Infrastructure (PostgreSQL + Redis on small EC2): ~$50/month

Cost reduction strategies:
1. Cache embeddings — same query never re-embedded
2. Use Haiku over heavier models — cheapest Claude, good enough for extraction
3. Cache search results in Redis — 60%+ cache hit rate on repeat queries
4. Local embeddings eliminate per-token cost entirely

---

## Project Structure

```
Therapize/
├── src/
│   ├── config.py                    # Pydantic settings (all config in one place)
│   ├── models/                      # Data contracts
│   │   ├── therapist.py             # Therapist profile schema
│   │   └── query.py                 # Search request/response
│   ├── scrapers/                    # Data collection
│   │   ├── base.py                  # Abstract scraper with retry/rate-limit
│   │   ├── open_path.py             # Algolia API scraper (OpenPath Collective)
│   │   └── good_therapy.py          # HTML scraper (GoodTherapy.org)
│   ├── pipeline/                    # Data pipeline
│   │   ├── ingestion.py             # Orchestrates scraping → DB
│   │   ├── cleaner.py               # Normalize + validate scraped data
│   │   └── embeddings.py            # Batch embedding generation (local, free)
│   ├── matching/                    # Core matching engine
│   │   ├── query_processor.py       # LLM query understanding
│   │   ├── modality_mapper.py       # Concerns → modalities
│   │   ├── hybrid_ranker.py         # Multi-signal scoring
│   │   ├── filter_engine.py         # Hard constraint filters
│   │   └── knowledge/
│   │       └── modality_map.json    # Curated concern→modality map
│   ├── storage/
│   │   ├── database.py              # PostgreSQL + pgvector async client
│   │   ├── cache.py                 # Redis async client
│   │   └── migrations/001_initial.sql
│   ├── api/
│   │   ├── main.py                  # FastAPI app
│   │   ├── routes/
│   │   │   ├── search.py
│   │   │   ├── feedback.py
│   │   │   └── health.py
│   │   └── middleware/
│   │       └── rate_limiter.py
│   ├── monitoring/
│   │   ├── metrics.py               # Prometheus metrics registry
│   │   ├── tracing.py               # LangSmith integration
│   │   └── evaluation.py            # Offline NDCG evaluation
│   └── workflow/
│       └── search_graph.py          # LangGraph search workflow
├── frontend/
│   └── app.py                       # Streamlit UI
├── tests/
│   ├── unit/
│   └── integration/
├── scripts/
│   ├── ingest_small.py              # Scrape ~5 from each source (dev)
│   ├── seed_data.py                 # Load synthetic data (no scraping)
│   └── migrate.py                   # Run SQL migrations
├── docker-compose.yml               # Local postgres + redis
├── requirements.txt
└── .env.example
```
