-- Migration 002: add ethnicity column to therapists table
-- gender column already exists from 001_initial.sql but was never populated

ALTER TABLE therapists
    ADD COLUMN IF NOT EXISTS ethnicity TEXT[] DEFAULT '{}';

CREATE INDEX IF NOT EXISTS therapists_ethnicity_gin_idx
    ON therapists USING GIN (ethnicity);
