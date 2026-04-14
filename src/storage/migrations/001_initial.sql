-- Initial schema for Therapize
-- PostgreSQL + pgvector
--
-- Design notes:
-- 1. pgvector for embeddings — avoids maintaining two databases (PostgreSQL + Qdrant)
-- 2. HNSW index (not IVFFlat) — better recall at low latency, no training required
-- 3. Enums as PostgreSQL native types — enforces data integrity at DB level
-- 4. GIN index on array columns — fast @> (contains) and && (overlap) operators
-- 5. Partial index on accepting_new_clients — most queries filter this first

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- for fuzzy text search

-- ── Enums ─────────────────────────────────────────────────────────────────
CREATE TYPE session_format AS ENUM (
    'in_person', 'telehealth', 'both'
);

CREATE TYPE insurance_provider AS ENUM (
    'blue_shield', 'blue_cross', 'aetna', 'united_healthcare',
    'cigna', 'kaiser', 'magellan', 'optum', 'self_pay',
    'sliding_scale', 'out_of_network'
);

CREATE TYPE therapy_modality AS ENUM (
    'cognitive_behavioral_therapy', 'dialectical_behavior_therapy', 'emdr',
    'acceptance_commitment_therapy', 'psychodynamic', 'humanistic',
    'mindfulness_based', 'somatic', 'gottman_method', 'emotionally_focused_therapy',
    'exposure_therapy', 'cognitive_processing_therapy', 'narrative_therapy',
    'solution_focused', 'motivational_interviewing', 'play_therapy',
    'art_therapy', 'psychoanalytic', 'integrative', 'cbt_insomnia',
    'trauma_informed', 'family_systems'
);

-- ── Core therapist table ───────────────────────────────────────────────────
CREATE TABLE therapists (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source              VARCHAR(50) NOT NULL,          -- e.g. 'psychology_today'
    source_url          TEXT NOT NULL UNIQUE,          -- dedup by URL
    source_id           VARCHAR(100) NOT NULL,

    -- Identity
    name                VARCHAR(255) NOT NULL,
    credentials         TEXT[] DEFAULT '{}',           -- e.g. {LMFT, PhD}
    license_number      VARCHAR(50),
    years_experience    SMALLINT,
    gender              VARCHAR(50),

    -- Clinical (array types with enum enforcement)
    modalities          therapy_modality[] DEFAULT '{}',
    specializations     TEXT[] DEFAULT '{}',
    populations_served  TEXT[] DEFAULT '{}',
    languages           TEXT[] DEFAULT '{}',

    -- Location
    city                VARCHAR(100),              -- NULL for telehealth-only therapists
    state               CHAR(2) NOT NULL DEFAULT 'CA',
    zip_code            VARCHAR(10),
    county              VARCHAR(100),
    latitude            DOUBLE PRECISION,
    longitude           DOUBLE PRECISION,

    -- Logistics
    session_formats     session_format[] DEFAULT '{}',
    insurance_providers insurance_provider[] DEFAULT '{}',
    sliding_scale       BOOLEAN DEFAULT FALSE,
    fee_min             SMALLINT,
    fee_max             SMALLINT,
    accepting_new_clients BOOLEAN DEFAULT TRUE NOT NULL,

    -- Content (searchable text)
    bio                 TEXT DEFAULT '',

    -- Engagement
    rating              NUMERIC(3,2),
    review_count        INTEGER DEFAULT 0,
    profile_completeness NUMERIC(3,2) DEFAULT 0.0,

    -- Vector embedding (384 dims for all-MiniLM-L6-v2, sentence-transformers local)
    -- Stored here so we can filter first, then do ANN in a single query
    embedding           vector(384),

    -- Metadata
    scraped_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_active           BOOLEAN DEFAULT TRUE NOT NULL,

    CONSTRAINT fee_range_valid CHECK (
        fee_min IS NULL OR fee_max IS NULL OR fee_min <= fee_max
    ),
    CONSTRAINT rating_range CHECK (
        rating IS NULL OR (rating >= 0 AND rating <= 5)
    )
);

-- ── Indexes ───────────────────────────────────────────────────────────────

-- HNSW index for approximate nearest neighbor vector search
-- m=16: connections per layer (higher = better recall, more memory)
-- ef_construction=64: build-time quality (higher = better recall, slower build)
-- Tuning guide: start with m=16, ef_construction=64, increase if recall < 0.95
CREATE INDEX therapists_embedding_hnsw_idx
    ON therapists USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN indexes for array containment/overlap queries
-- These make insurance @> ARRAY[...] and modalities && ARRAY[...] fast
CREATE INDEX therapists_modalities_gin_idx
    ON therapists USING GIN (modalities);

CREATE INDEX therapists_insurance_gin_idx
    ON therapists USING GIN (insurance_providers);

CREATE INDEX therapists_session_formats_gin_idx
    ON therapists USING GIN (session_formats);

-- Partial index: most queries filter active + accepting clients first
CREATE INDEX therapists_active_accepting_idx
    ON therapists (is_active, accepting_new_clients)
    WHERE is_active = TRUE AND accepting_new_clients = TRUE;

-- B-tree for common filter combinations
CREATE INDEX therapists_location_idx ON therapists (state, city);
CREATE INDEX therapists_fee_idx ON therapists (fee_min) WHERE fee_min IS NOT NULL;

-- Full-text search index for bio (BM25 fallback)
CREATE INDEX therapists_bio_trgm_idx
    ON therapists USING GIN (bio gin_trgm_ops);

-- ── Search queries table (for analytics + feedback loop) ──────────────────
CREATE TABLE search_queries (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      VARCHAR(100),
    free_text       TEXT,
    extracted_intent JSONB,         -- full ExtractedQueryIntent as JSON
    filter_criteria JSONB,          -- compiled FilterCriteria
    result_ids      UUID[],         -- ordered list of returned therapist IDs
    total_candidates INTEGER,
    filtered_count  INTEGER,
    latency_ms      NUMERIC(10,2),
    llm_tokens      INTEGER,
    cache_hit       BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX search_queries_created_idx ON search_queries (created_at DESC);
CREATE INDEX search_queries_session_idx ON search_queries (session_id);

-- ── Feedback table ────────────────────────────────────────────────────────
CREATE TABLE feedback (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id        UUID REFERENCES search_queries(id),
    therapist_id    UUID REFERENCES therapists(id),
    rating          SMALLINT NOT NULL CHECK (rating BETWEEN 1 AND 5),
    rank_position   SMALLINT,           -- 1-based position in results when feedback was given
    event_type      VARCHAR(20) NOT NULL DEFAULT 'explicit',  -- 'explicit' | 'profile_view'
    booked          BOOLEAN,
    feedback_text   TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX feedback_query_idx ON feedback (query_id);
CREATE INDEX feedback_therapist_idx ON feedback (therapist_id);

-- ── Evaluation runs table (for tracking NDCG over time) ───────────────────
CREATE TABLE evaluation_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_name        VARCHAR(100),
    ndcg_at_10      NUMERIC(5,4),
    mrr             NUMERIC(5,4),
    precision_at_5  NUMERIC(5,4),
    num_queries     INTEGER,
    config_snapshot JSONB,       -- weights + model used
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── View: evaluation dataset ──────────────────────────────────────────────
-- Queries with enough feedback to evaluate quality
CREATE VIEW evaluation_queries AS
SELECT
    sq.id,
    sq.free_text,
    sq.extracted_intent,
    sq.result_ids,
    COUNT(f.id) AS feedback_count,
    AVG(f.rating) AS avg_rating,
    SUM(CASE WHEN f.booked THEN 1 ELSE 0 END) AS bookings
FROM search_queries sq
JOIN feedback f ON f.query_id = sq.id
GROUP BY sq.id
HAVING COUNT(f.id) >= 1;
