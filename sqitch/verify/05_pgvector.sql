-- Verify stupyd:05_pgvector on pg

BEGIN;

-- The extension is not created by this migration, because that needs a
-- superuser, so it has to be checked separately. Without it the deploy would
-- have failed on the vector type rather than on anything readable.
DO $$
DECLARE
    installed integer;
BEGIN
    SELECT count(*) INTO installed
    FROM pg_extension
    WHERE extname = 'vector';

    IF installed = 0 THEN
        RAISE EXCEPTION
            'the vector extension is not installed: run CREATE EXTENSION vector as a superuser';
    END IF;
END
$$;

-- The column has to be the width the embedding model produces, or every write
-- from bin/embed_courses.py is refused.
DO $$
DECLARE
    declared text;
BEGIN
    -- format_type renders the stored type, so it gives 'vector(384)' rather
    -- than the raw modifier, whose encoding is a pgvector detail
    SELECT format_type( atttypid, atttypmod ) INTO declared
    FROM pg_attribute
    WHERE attrelid = 'courses'::regclass
      AND attname = 'embedding'
      AND NOT attisdropped;

    IF declared IS NULL THEN
        RAISE EXCEPTION 'courses.embedding is missing';
    END IF;

    IF declared <> 'vector(384)' THEN
        RAISE EXCEPTION
            'courses.embedding is %s, but all-MiniLM-L6-v2 produces 384 dimensions',
            declared;
    END IF;
END
$$;

-- The index the search relies on has to be there and has to be an HNSW one:
-- an index on a different operator class would not answer the cosine order.
DO $$
DECLARE
    indexed integer;
BEGIN
    SELECT count(*) INTO indexed
    FROM pg_indexes
    WHERE schemaname = current_schema()
      AND tablename = 'courses'
      AND indexdef LIKE '%hnsw%'
      AND indexdef LIKE '%vector_cosine_ops%';

    IF indexed = 0 THEN
        RAISE EXCEPTION
            'no cosine HNSW index on courses.embedding';
    END IF;
END
$$;

-- A vector of all zeros has no direction, so the cosine distance to it is
-- undefined and the ordering becomes NaN. The test is written as NOT above
-- zero rather than = 0 so that a NaN is caught too, since a comparison with
-- NaN is false.
--
-- The count itself is deliberately not checked: the column is empty when the
-- migration runs and stays empty until bin/embed_courses.py is called, so
-- requiring a vector here would fail a correct deployment.
DO $$
DECLARE
    degenerate integer;
BEGIN
    SELECT count(*) INTO degenerate
    FROM courses
    WHERE embedding IS NOT NULL
      AND NOT (vector_norm(embedding) > 0);

    IF degenerate > 0 THEN
        RAISE EXCEPTION
            '% vectors have no direction, which makes the cosine distance undefined',
            degenerate;
    END IF;
END
$$;

ROLLBACK;