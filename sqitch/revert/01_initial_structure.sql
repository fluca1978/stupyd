-- Revert stupyd:01_initial_structure from pg

BEGIN;

DROP TABLE IF EXISTS  students CASCADE;
DROP TABLE IF EXISTS student_demographics CASCADE;
DROP TABLE IF EXISTS  courses CASCADE;
DROP TABLE IF EXISTS  departments CASCADE;

DROP TABLE IF EXISTS  course_versions CASCADE;
DROP TABLE IF EXISTS  curriculum_versions CASCADE;

DROP TABLE IF EXISTS  enrollments CASCADE;
DROP TABLE IF EXISTS  enrollment_path CASCADE;

DROP EXTENSION btree_gist;
COMMIT;
