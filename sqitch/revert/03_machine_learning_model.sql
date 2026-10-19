-- Revert stupyd:03_machine_learning_model from pg

BEGIN;

DROP SCHEMA etl CASCADE;
DROP SCHEMA ml_model CASCADE;

COMMIT;
