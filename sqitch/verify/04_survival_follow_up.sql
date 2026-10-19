-- Verify stupyd:04_survival_follow_up on pg

BEGIN;

-- The durations a time axis is built from: every one of them has to be a
-- number of months a human could have waited.
DO $$
DECLARE
    impossible integer;
BEGIN
    SELECT count(*) INTO impossible
    FROM ml_model.enrollment_follow_up
    WHERE duration_months < 0 OR duration_months > 600;

    IF impossible > 0 THEN
        RAISE EXCEPTION
            '% enrollments report a duration outside 0..600 months', impossible;
    END IF;
END
$$;

-- A graduation after the enrollment was closed is impossible, and it is the
-- signature of a closure date borrowed from another career.
DO $$
DECLARE
    incoherent integer;
BEGIN
    SELECT count(*) INTO incoherent
    FROM ml_model.enrollment_follow_up
    WHERE event_observed
      AND graduation_date IS NOT NULL
      AND censor_date IS NOT NULL
      AND graduation_date > censor_date;

    IF incoherent > 0 THEN
        RAISE EXCEPTION
            '% graduations happen after the enrollment was closed', incoherent;
    END IF;
END
$$;

-- The censored enrollments are the ones the regression throws away and the
-- survival model is trained on: there have to be some.
DO $$
DECLARE
    censored integer;
BEGIN
    SELECT count(*) INTO censored
    FROM ml_model.enrollment_follow_up
    WHERE NOT event_observed;

    IF censored = 0 THEN
        RAISE EXCEPTION 'no censored enrollment: nothing to train on';
    END IF;
END
$$;

-- Every enrollment of the feature view has to be reachable, or the survival
-- query silently drops the rows it cannot date.
DO $$
DECLARE
    orphans integer;
BEGIN
    SELECT count(*) INTO orphans
    FROM ml_model.student_features_12m sf
    LEFT JOIN ml_model.enrollment_follow_up fu
      ON fu.enrollment_id = sf.enrollment_id
    WHERE fu.enrollment_id IS NULL;

    IF orphans > 0 THEN
        RAISE EXCEPTION
            '% enrollments have no follow-up', orphans;
    END IF;
END
$$;

ROLLBACK;