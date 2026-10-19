-- Deploy stupyd:03_machine_learning_model to pg

BEGIN;


CREATE SCHEMA IF NOT EXISTS etl;





CREATE TABLE etl.enrollment_source_map (
    enrollment_id            bigint
    , source_matricola_id bigint NOT NULL

    , PRIMARY KEY( enrollment_id )
    , FOREIGN KEY( enrollment_id ) REFERENCES enrollments( pk )
    , UNIQUE( source_matricola_id )
);




CREATE TABLE etl.new_enrollments (
    new_pk               bigint PRIMARY KEY,
    source_matricola_id  bigint NOT NULL UNIQUE,
    student              bigint NOT NULL,
    begin_date           date NOT NULL,
    end_date             date,
    status               text NOT NULL,
    graduation_date      date,
    graduation_vote      integer,
    graduation_honor     boolean NOT NULL
);



CREATE SCHEMA ml_model;



CREATE MATERIALIZED VIEW ml_model.student_features_12m AS
WITH career_mapping AS (
    SELECT
        e.pk AS enrollment_id,
        e.student,
        e.begin_date,
        e.graduation_date,
        ep.curriculum_version,
        esm.source_matricola_id
    FROM enrollments e
    JOIN enrollment_path ep
      ON ep.enrollment = e.pk
    JOIN etl.enrollment_source_map esm
      ON esm.enrollment_id = e.pk
),
exam_features AS (
    SELECT
        cm.enrollment_id,

        count(*) FILTER (
            WHERE r.voto BETWEEN 18 AND 30
        ) AS graded_exams_completed,

        count(*) FILTER (
            WHERE r.voto IS NULL
              AND nullif(trim(r.tipo_giud_cod), '') IS NOT NULL
        ) AS judgment_activities,

        COALESCE(
            sum(r.peso) FILTER (
                WHERE r.voto BETWEEN 18 AND 30
            ),
            0
        ) AS graded_credits_earned,

        avg(r.voto) FILTER (
            WHERE r.voto BETWEEN 18 AND 30
        ) AS average_grade,

        COALESCE(
            sum(r.voto * r.peso) FILTER (
                WHERE r.voto BETWEEN 18 AND 30
            )::double precision
            /
            NULLIF(
                sum(r.peso) FILTER (
                    WHERE r.voto BETWEEN 18 AND 30
                ),
                0
            ),
            0
        ) AS weighted_average_grade,

        count(*) FILTER (
            WHERE COALESCE(r.lode, 0) <> 0
        ) AS honors_count

    FROM career_mapping cm
    JOIN src.riga_libretto r
      ON r.matricola_id = cm.source_matricola_id

    WHERE nullif(trim(r.data), '') IS NOT NULL
      AND trim(r.data)::timestamp::date >= cm.begin_date
      AND trim(r.data)::timestamp::date
            < cm.begin_date + INTERVAL '12 months'

    GROUP BY cm.enrollment_id
)
SELECT
    cm.enrollment_id,
    cm.student,
    cm.curriculum_version,
    cm.begin_date,
    (cm.begin_date + INTERVAL '12 months')::date
        AS observation_date,

    CASE
        WHEN nullif(trim(rs.data_nascita), '') IS NULL
            THEN NULL
        WHEN trim(rs.data_nascita)::timestamp::date > cm.begin_date
            THEN NULL
        ELSE EXTRACT(
            YEAR FROM age(
                cm.begin_date,
                trim(rs.data_nascita)::timestamp::date
            )
        )::smallint
    END AS age_at_enrollment,

    CASE upper(trim(rs.sesso))
        WHEN 'F' THEN 'F'
        WHEN 'M' THEN 'M'
        WHEN 'X' THEN 'X'
        ELSE 'UNKNOWN'
    END AS audit_sex,

    COALESCE(ef.graded_exams_completed, 0)
        AS graded_exams_completed,

    COALESCE(ef.judgment_activities, 0)
        AS judgment_activities,

    COALESCE(ef.graded_credits_earned, 0)
        AS graded_credits_earned,

    COALESCE(ef.average_grade, 0)
        AS average_grade,

    COALESCE(ef.weighted_average_grade, 0)
        AS weighted_average_grade,

    COALESCE(ef.honors_count, 0)
        AS honors_count,

    cm.graduation_date,

    CASE
        WHEN cm.graduation_date IS NOT NULL
        THEN (
            cm.graduation_date
            - (cm.begin_date + INTERVAL '12 months')::date
        ) / 30.4375
    END AS months_to_graduation,

    cm.graduation_date IS NOT NULL AS event_observed

FROM career_mapping cm
JOIN students s
  ON s.pk = cm.student
JOIN src.studenti rs
  ON rs.id::text = s.src
LEFT JOIN exam_features ef
  ON ef.enrollment_id = cm.enrollment_id

-- La previsione viene effettuata dopo 12 mesi: escludiamo chi aveva
-- già concluso la carriera prima della data di osservazione.
WHERE cm.graduation_date IS NULL
   OR cm.graduation_date
        >= (cm.begin_date + INTERVAL '12 months')::date;



REFRESH MATERIALIZED VIEW ml_model.student_features_12m WITH DATA;

COMMIT;
