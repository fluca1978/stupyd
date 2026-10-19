-- Deploy stupyd:05_pgvector to pg

BEGIN;

-- CREATE EXTENSION vector;  -- needs superuser!

ALTER TABLE courses ADD COLUMN embedding vector(384);

CREATE INDEX idx_courses_embedding_hnsw
ON courses
USING hnsw (embedding vector_cosine_ops);

-- The vectors come from StupydEmbedder, in stupyd/embedder.py, which reads
-- the rows, calls the model, and writes them back: PostgreSQL has no model of
-- its own. A course is embedded as `type` plus `description` so that the
-- degree level stays in the vector, and a placeholder name is skipped:
--
--   UPDATE courses SET embedding = %s::vector
--   WHERE pk = %s
--     AND description !~* 'TEST|SOSTEGNO|CORSO SINGOLO';
--
-- and read back with the cosine distance, which pgvector spells `<=>`, so that
-- 1 - distance is a similarity:
--
--   SELECT code, description, type,
--          1 - (embedding <=> %s::vector) AS similarity
--   FROM courses
--   WHERE embedding IS NOT NULL
--   ORDER BY embedding <=> %s::vector
--   LIMIT 10;

COMMIT;
