-- Revert stupyd:04_survival_follow_up from pg

BEGIN;

DROP MATERIALIZED VIEW IF EXISTS ml_model.enrollment_follow_up;

COMMIT;