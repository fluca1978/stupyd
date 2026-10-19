-- Revert stupyd:05_pgvector from pg

BEGIN;

DROP INDEX idx_courses_embedding_hnsw;

ALTER TABLE courses DROP COLUMN IF EXISTS embedding;

COMMIT;
