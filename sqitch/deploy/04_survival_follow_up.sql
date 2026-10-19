-- Deploy stupyd:04_survival_follow_up to pg

-- A survival model needs to know, for every enrollment, how long the student
-- has been observed and whether the career ended in a graduation or was still
-- in progress. `students.observed_last` cannot answer that: it is a single
-- date per student, so an enrollment is credited with the follow-up of every
-- other career of the same student, and it is capped by a constant that goes
-- stale. The cutoff belongs to the enrollment, and `enrollments.end_date`
-- already carries it.

BEGIN;

CREATE MATERIALIZED VIEW ml_model.enrollment_follow_up AS
WITH bounds AS (
    SELECT
        sf.enrollment_id,
        sf.student,
        sf.observation_date,
        sf.graduation_date,
        sf.event_observed,

        -- The day the enrollment itself stopped: its closure, or the day the
        -- student was last seen, whichever came first. The student date is a
        -- bound rather than the answer, since a student who re-enrolled later
        -- was seen after this career ended.
        LEAST(
            COALESCE(e.end_date, st.observed_last),
            st.observed_last
        ) AS censor_date

    FROM ml_model.student_features_12m sf
    JOIN enrollments e
      ON e.pk = sf.enrollment_id
    JOIN students st
      ON st.pk = sf.student
)
SELECT
    enrollment_id,
    observation_date,
    graduation_date,
    event_observed,
    censor_date,

    -- Months observed since the features were measured. A student that left
    -- before the observation date has no follow-up at all, hence the floor at
    -- zero; the upper bound refuses the dates that no calendar could produce
    -- rather than letting one of them stretch a time axis.
    LEAST(
        GREATEST(
            COALESCE(
                (graduation_date - observation_date) / 30.4375,
                (censor_date - observation_date) / 30.4375,
                0
            ),
            0
        ),
        600
    )::real AS duration_months

FROM bounds;

-- The survival query reads the durations by enrollment, so this is the index
-- it needs: without it the join scans every follow-up for every row.
CREATE UNIQUE INDEX enrollment_follow_up_enrollment_id
    ON ml_model.enrollment_follow_up( enrollment_id );

REFRESH MATERIALIZED VIEW ml_model.enrollment_follow_up WITH DATA;

COMMENT ON MATERIALIZED VIEW ml_model.enrollment_follow_up IS
    'Per-enrollment follow-up for the survival model: duration_months is the'
    ' observed time since the observation date, event_observed tells whether'
    ' the career ended in a graduation.';

COMMIT;