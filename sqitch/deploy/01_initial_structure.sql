-- Deploy stupyd:01_initial_structure to pg

BEGIN;

CREATE TABLE students (
       pk                bigint GENERATED ALWAYS AS IDENTITY
       , observed_first  date NOT NULL
       , observed_last   date NOT NULL
       , src             text

       , PRIMARY KEY( pk )
);


CREATE TABLE student_demographics (
       pk                bigint GENERATED ALWAYS AS IDENTITY
       , student     bigint NOT NULL
       , gender      text NOT NULL
       , birth_year smallint NOT NULL

       , PRIMARY KEY( pk )
       , CHECK( gender IN ( 'M', 'F' ) )
       , CHECK( birth_year > 1900 )
       , FOREIGN KEY( student ) REFERENCES students( pk )
);


CREATE TABLE courses (
       pk                bigint GENERATED ALWAYS AS IDENTITY
       , code		 text   NOT NULL
       , description	 text
       , type		 text

       , PRIMARY KEY( pk )
       , UNIQUE( code )
);


CREATE TABLE departments (
       pk                bigint GENERATED ALWAYS AS IDENTITY
       , code		 text
       , description	 text

       , PRIMARY KEY( pk )
       , UNIQUE( code )
);


CREATE TABLE course_versions (
       pk                bigint GENERATED ALWAYS AS IDENTITY
       , course          bigint NOT NULL

       , begin_date      date NOT NULL
       , end_date        date   -- if null, still valid!

       , academic_year	 int NOT NULL  -- academic year begin
       , months          int NOT NULL  -- expected duration in months
       , cfu             int NOT NULL  -- expected requested CFU

       , department      bigint NOT NULL

       , PRIMARY KEY( pk )
       , FOREIGN KEY( course ) REFERENCES courses( pk )
       , FOREIGN KEY( department ) REFERENCES departments( pk )
       , CHECK ( end_date IS NULL OR end_date > begin_date )
       , CHECK ( cfu > 0 )
       , CHECK ( academic_year > 1900 )
);


CREATE TABLE curriculum_versions (
       pk                bigint GENERATED ALWAYS AS IDENTITY
       , course_version  bigint NOT NULL

       , code        text NOT NULL
       , description text NOT NULL
       , begin_date  date NOT NULL
       , end_date    date

       , PRIMARY KEY( pk )
       , CHECK( end_date IS NULL or end_date > begin_date )
       , FOREIGN KEY( course_version ) REFERENCES course_versions( pk )

);


CREATE TABLE enrollments (
       pk          bigint GENERATED ALWAYS AS IDENTITY

       , student     bigint NOT NULL
       , begin_date  date   NOT NULL
       , end_date    date
       , status      text   NOT NULL DEFAULT 'active'

       , graduation_date  date
       , graduation_vote  int
       , graduation_honor boolean DEFAULT false

       , CHECK( status IN ( 'active'
       	 	          , 'withdraw'
		          , 'expired'
		          , 'graduated'
		          , 'transferred' ) )

      , PRIMARY KEY( pk )
      , CHECK( end_date IS NULL or end_date > begin_date )
      , CHECK( graduation_vote > 0 AND graduation_vote <= 110 )
      , FOREIGN KEY( student ) REFERENCES students( pk )
);



CREATE TABLE enrollment_path (
       pk          bigint GENERATED ALWAYS AS IDENTITY

       -- these are all 'versions'
       , enrollment          bigint NOT NULL
       , curriculum_version  bigint NOT NULL

       , begin_date  date NOT NULL
       , end_date    date

       , type text NOT NULL default 'initial'

       , PRIMARY KEY( pk )
       , CHECK( end_date IS NULL OR end_date > begin_date )
       , FOREIGN KEY( enrollment ) REFERENCES enrollments( pk )
       , FOREIGN KEY( curriculum_version ) REFERENCES curriculum_versions( pk )
);




CREATE EXTENSION IF NOT EXISTS btree_gist;

ALTER TABLE course_versions
ADD CONSTRAINT course_versions_no_overlap
EXCLUDE USING gist (
    course WITH =,
    daterange(
        begin_date,
        COALESCE(end_date, 'infinity'::date),
        '[)'
    ) WITH &&
);


ALTER TABLE curriculum_versions
ADD CONSTRAINT curriculum_versions_no_overlap
EXCLUDE USING gist (
    course_version WITH =,
    code WITH =,
    daterange(
        begin_date,
        COALESCE(end_date, 'infinity'::date),
        '[)'
    ) WITH &&
);


COMMIT;
