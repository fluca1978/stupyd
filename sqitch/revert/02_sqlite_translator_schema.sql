-- Revert stupyd:02_sqlite_translator_schema from pg

BEGIN;

DROP SCHEMA src CASCADE;

COMMIT;
