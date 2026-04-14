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
│  User Query (free text + optional filters)                          │
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
│  │    free text    │   │     + bio-corroboration multiplier    │     │
│  └────────┬────────┘   │     + breadth penalty                 │     │
│           │            │  4. Semantic Score         (0.35)     │     │
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
│   POST /api/v1/search   ──► Redis Cache ──► Matching Engine         │
│   POST /api/v1/feedback ──► Feedback Store                          │
│   GET  /api/v1/cities   ──► Distinct cities in DB                   │
│   GET  /api/v1/languages──► Distinct languages in DB                │
│   GET  /api/v1/stats    ──► Total active therapist count            │
│   GET  /health          ──► DB + cache ping                         │
└──────────────────────────────────────────┬──────────────────────────┘
                                           │
┌──────────────────────────────────────────▼──────────────────────────┐
│                       FRONTEND (Next.js 14)                          │
│                                                                     │
│  web/  — React + TypeScript + Tailwind CSS                          │
│  Sticky filter sidebar · Rank-based fit badges · Score breakdown    │
│  Staggered card animation · Coloured modality chips                 │
│  Example prompt chips · Feedback buttons · Mobile drawer            │
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
   - Merges with explicit questionnaire filters — explicit always wins
   - Temperature = 0 for determinism. `instructor` enforces Pydantic output via tool-use.

3. **Embed query** — `all-MiniLM-L6-v2` (local, 384-dim). Zero API cost.

4. **Progressive filter fallback** — tries increasingly relaxed filters until ≥ 3 results:
   - Step 1: all filters applied
   - Step 2: drop budget
   - Step 3: drop insurance
   - Step 4: drop language preference
   - Step 5: drop gender preference
   - Safety net: state=CA only (keeps city+format for in-person; never relaxes city for in-person searches)
   - If in-person + city yields < 3 results, suggests switching to telehealth

5. **Filter + vector search** (single DB round-trip)
   ```sql
   SELECT ... FROM therapists
   WHERE state = 'CA' AND accepting_new_clients = true
     AND <insurance/city/budget filters>
   ORDER BY embedding <=> $query_vector   -- pgvector HNSW cosine distance
   LIMIT 100
   ```

6. **Modality mapping** — `emotional_concerns` → `modality_map.json` → weighted
   modality dict. Deterministic, no LLM cost per ranking.

7. **Hybrid ranking** with three signal refinements:
   - **Modality score** (0.40): therapist modalities vs concern-mapped modalities,
     multiplied by a **bio-corroboration factor** (cosine similarity between modality
     description embeddings and therapist bio — penalises checkbox inflation) and a
     **breadth penalty** (`log(8)/log(n+3)` — discounts therapists listing many modalities)
   - **Semantic score** (0.35): cosine similarity of query + profile embeddings
   - **BM25** (0.15): keyword overlap on therapist bios
   - **Quality** (0.10): log-dampened rating × profile completeness

8. **Response** — top 10 results with score breakdown, fit label (rank-based),
   matched modalities, bio excerpt, and narrative match explanation.

9. **Cache + log** — result cached in Redis (1hr TTL). Query + result IDs logged
   to `search_queries` table for offline evaluation.

---

## Key Design Decisions

### LLM extracts concerns only — not modalities
The LLM (`query_processor.py`) extracts `emotional_concerns` (plain language:
`["anxiety", "grief"]`). Modality mapping is handled deterministically by
`modality_map.json`. This keeps the LLM schema small and the ranking logic
auditable without per-ranking LLM cost.

### Bio-corroboration + breadth penalty on modality score
Therapists self-report modalities via checkbox lists, inflating the signal.
Two corrections applied in `hybrid_ranker.py`:
- **Bio-corroboration**: pre-computed embeddings of 22 modality descriptions
  are compared against the therapist's bio embedding. Low cosine similarity
  → multiplier pulled toward 0.5, capping the modality score contribution.
- **Breadth penalty**: `min(1.0, log(8)/log(n+3))` — no penalty for ≤5 modalities,
  increasing discount above that. Rewards specialists over generalists.

### Progressive filter fallback with in-person city protection
Filter relaxation never drops city for in-person searches — it would be
meaningless to show a therapist 200 miles away. Instead, if too few in-person
results are found, the user is prompted to consider telehealth.

### Rank-based fit labels (not score-based)
Min-max normalisation causes composite scores to cluster in a narrow band,
making score-based thresholds meaningless. Labels are rank-based instead:
`#1 → Strong fit`, `#2–3 → Good fit`, `#4–6 → Decent fit`, `#7+ → Possible match`.

### PostgreSQL + pgvector (not a dedicated vector DB)
Therapist data is highly structured — insurance, location, price need SQL.
Filtering **before** vector search is far more efficient than post-filtering.
HNSW index gives >0.95 recall with zero operational overhead.

### Local embeddings (`all-MiniLM-L6-v2`, 384 dims)
Zero per-token cost vs OpenAI. Loaded once at startup, cached in memory.
Model embeddings are pre-computed at ingestion time and stored in the DB.

### Redis cache
Same query → same result in ~5ms instead of ~2–8s. TTL = 1hr. Also backs
the rate limiter (30 req/min per IP, sliding window).

---

## Matching Algorithm (Example)

```
User: "I've been having panic attacks and can't stop ruminating. Feel dysfunctional."

Step 1 — Query Processor (LLM):
  emotional_concerns: ["anxiety", "panic attacks", "rumination", "functional impairment"]
  inferred_filters:   {}

Step 2 — Modality Mapper (deterministic JSON lookup):
  anxiety       → CBT (0.95), ACT (0.85), mindfulness (0.75)
  panic attacks → CBT (0.95), exposure_therapy (0.90)
  rumination    → CBT (0.90), mindfulness (0.80), ACT (0.85)
  → merged: { CBT: 0.95, exposure_therapy: 0.90, ACT: 0.85, mindfulness: 0.80 }

Step 3 — Hybrid Scoring (per candidate therapist):
  modality_score  = overlap(therapist.modalities, mapped_modalities)
                  × bio_corroboration_factor    ← penalises checkbox inflation
                  × breadth_penalty             ← rewards specialists
  semantic_score  = cosine_sim(query_embedding, profile_embedding)
  bm25_score      = bm25(query_tokens, therapist_bio_text)
  quality_score   = log_dampened_rating × profile_completeness

  composite = 0.40×mod + 0.35×sem + 0.15×bm25 + 0.10×quality
  → top 10, rank-based fit labels: #1 Strong fit, #2–3 Good fit, etc.
```

---

## Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Required: ANTHROPIC_API_KEY, DATABASE_URL (Neon), REDIS_URL (Upstash)
# Optional: LANGCHAIN_API_KEY, LANGCHAIN_TRACING_V2=true

# 3. Run database migrations
python scripts/migrate.py

# 4. Load therapist data
python scripts/seed_data.py        # synthetic data (no scraping needed)

# 5. Start API
uvicorn src.api.main:app --reload --port 8000

# 6. Start frontend (Next.js)
cd web
npm install
npm run dev   # http://localhost:3000
```

**Infrastructure:** uses [Neon](https://neon.tech) (serverless Postgres + pgvector)
and [Upstash](https://upstash.com) (serverless Redis). No local Docker required.

---

## Frontend (Next.js)

The frontend lives in `web/` and is a fully typed React app:

```
web/
  app/
    page.tsx          ← main search page (client component)
    layout.tsx        ← root layout, Inter font, metadata
    globals.css       ← design tokens + animations
  components/
    SearchBar.tsx     ← textarea + gradient Search button
    FilterSidebar.tsx ← sticky sidebar (desktop) / drawer (mobile)
    TherapistCard.tsx ← result card with rank border, chips, feedback
    FitBadge.tsx      ← rank-based fit pill (Strong / Good / Decent / Possible)
    ConcernsBanner.tsx← identified concern pills from extracted intent
    RelaxationNote.tsx← filter relaxation info message
    ScoreBreakdown.tsx← collapsible score bars
    SkeletonCard.tsx  ← loading placeholder
    EmptyState.tsx    ← example prompt chips before first search
  lib/
    api.ts            ← typed fetch functions for all 5 endpoints
    types.ts          ← TypeScript interfaces matching API schema
```

Design tokens: warm off-white canvas (`#FAF7F4`), teal-blue primary (`#4A90A4`),
warm sand sidebar (`#EDE8E3`). Rank accent borders: gold / silver / bronze / teal.

---

## Observability

**LangSmith** — traces every search (project: `therapize`). Each trace shows the
`query_processor.extract_intent` span with extracted intent, concerns, and filters.
Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in `.env` to enable.

**Prometheus** — metrics at `/metrics`:
- `therapize_search_latency_ms` — P50/P99 latency histogram
- `therapize_llm_tokens_total` — cumulative token spend
- `therapize_cache_hits_total` / `misses` — cache hit rate
- `therapize_modality_match_score` — distribution of top-result modality scores
- `therapize_candidates_after_filter` — filter restrictiveness signal

**Offline evaluation** — `scripts/evaluate_ranker.py` computes NDCG@10, MRR,
Precision@5 from the `feedback` table. Run after ranking weight changes to catch regressions.

---

## Cost Estimation

Per 1,000 queries (no cache hits):
- Claude Haiku extraction: ~800 tokens × $0.00000025 ≈ **$0.20**
- `all-MiniLM-L6-v2` embedding: local, **$0**
- **Total: ~$0.20 per 1,000 queries**

With ~60% cache hit rate: effective cost drops to **~$0.08 per 1,000 queries**.

Infrastructure: Neon free tier (500MB) + Upstash free tier (10k commands/day) = **$0/month** for low-traffic deployments.

---

## Evaluation Metrics

**Online** (automatic, every request):
| Metric | Signal |
|---|---|
| `search_latency_ms` | User-facing SLO |
| `modality_match_score` | Proxy for ranking quality |
| `candidates_after_filter` | Filter restrictiveness |
| `cache_hit_rate` | Cost efficiency |

**Offline** (requires user feedback, run periodically via `scripts/evaluate_ranker.py`):
| Metric | What it measures |
|---|---|
| NDCG@10 | Ranking quality — are relevant results ranked first? |
| MRR | How far the user scrolls to find a good match |
| Precision@5 | Fraction of top-5 results that are relevant |

Feedback is binary ("This feels right" = 5, "Not quite" = 1). With binary labels,
MRR is the most informative metric.

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
│   │   ├── hybrid_ranker.py         # 4-signal composite scorer + bio-corroboration
│   │   ├── filter_engine.py         # Progressive SQL WHERE clause builder
│   │   └── knowledge/
│   │       └── modality_map.json    # Curated concern → evidence-based modality map
│   ├── storage/
│   │   ├── database.py              # asyncpg PostgreSQL + pgvector client
│   │   ├── cache.py                 # Redis async client + rate limiter
│   │   └── migrations/001_initial.sql
│   ├── api/
│   │   ├── main.py                  # FastAPI app + lifespan + CORS + middleware
│   │   ├── routes/
│   │   │   ├── search.py            # POST /search, GET /cities, /languages, /stats
│   │   │   ├── feedback.py          # POST /api/v1/feedback
│   │   │   └── health.py            # GET /health, /health/deep
│   │   └── middleware/
│   │       └── rate_limiter.py      # Redis sliding window rate limiter
│   ├── monitoring/
│   │   ├── metrics.py               # Prometheus metrics registry
│   │   └── tracing.py               # LangSmith config + structured logger
│   └── workflow/
│       └── search_graph.py          # LangGraph orchestration + progressive fallback
├── web/                             # Next.js 14 frontend (see Frontend section)
├── scripts/
│   ├── migrate.py                   # Run SQL migrations (required at startup)
│   ├── seed_data.py                 # Load synthetic therapist data
│   ├── run_ingestion.py             # Full-scale scraping pipeline
│   ├── evaluate_ranker.py           # Offline NDCG/MRR evaluation
│   ├── online_metrics.py            # Live query performance tracking
│   ├── clean_cities.py              # Data quality: fix city name typos
│   └── verify_db.py                 # DB health check + stats
├── railway.toml                     # Railway deployment config
├── requirements.txt
└── .env.example
```
