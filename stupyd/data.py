"""PyTorch data access for the 12-month degree-delay training cohort."""

from __future__ import annotations

import os

import psycopg
import torch
from psycopg.rows import dict_row
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


FEATURE_NAMES = (
    "age_at_enrollment",
    "age_at_enrollment_missing",
    "sex_female",
    "sex_male",
    "sex_other_or_unknown",
    "graded_exams_completed",
    "judgment_activities",
    "graded_credits_earned",
    "credits_progress",
    "average_grade",
    "weighted_average_grade",
    "honors_count",
    "expected_months",
    "expected_cfu",
)


COMPLETED_ONLY_FILTER = "WHERE sf.event_observed"


def features_query( completed_only: bool = True,
                    survival: bool = False ) -> str:
    """
    Builds the query that extracts features (and the degree delay) from
    the machine-learning view in the backend database.

    With `completed_only` the query returns only the enrollments with an
    observed event, that is the ones usable to train or evaluate a regressor.

    With `completed_only = False` the censored enrollments are returned too,
    since those are the enrollments still in progress and, therefore, the
    ones to score. Their degree delay is unknown, so it is filled with zeros
    and flagged by the column `degree_delay_observed` (0 for censored rows).

    With `survival` the query adds the two columns a survival analysis needs:

    - `duration_months` the months observed from the observation date, that is
      up to the graduation for a completed enrollment and up to the last date
      the student has been seen for a censored one. That latter is the
      follow-up time, and it is floored at zero, since a student that left
      before the observation date has no follow-up at all.
    - `event_observed` True only when the enrollment graduated

    The time origin is the observation date, 12 months after the enrollment,
    because that is when the features are measured.
    """

    delay = """(
            ( sf.graduation_date - sf.begin_date ) / 30.4375
            - cv.months
        )::real"""

    # censored enrollments have no degree delay, hence the fallback
    target = delay if completed_only else f"COALESCE( {delay}, 0 )"
    filter = COMPLETED_ONLY_FILTER if completed_only else ""

    survival_columns = """,
        fu.duration_months::real AS duration_months,
        sf.event_observed AS event_observed""" if survival else ""

    # the follow-up of an enrollment, that is how long it has been observed
    # and whether it ended in a graduation, lives in a view of its own: the
    # cutoff belongs to the enrollment and not to the student, since one
    # student may have several careers
    follow_up_join = ( "\n    JOIN ml_model.enrollment_follow_up AS fu"
                      "\n      ON fu.enrollment_id = sf.enrollment_id" ) if survival else ""

    return f"""
    SELECT
        sf.enrollment_id,
        sf.student AS student_id,
        EXTRACT(YEAR FROM sf.observation_date)::integer AS observation_year,
        sf.audit_sex AS gender,
        COALESCE(sf.age_at_enrollment, 0)::real AS age_at_enrollment,
        (sf.age_at_enrollment IS NULL)::integer::real AS age_at_enrollment_missing,
        (sf.audit_sex = 'F')::integer::real AS sex_female,
        (sf.audit_sex = 'M')::integer::real AS sex_male,
        (sf.audit_sex NOT IN ('F', 'M'))::integer::real AS sex_other_or_unknown,
        sf.graded_exams_completed::real AS graded_exams_completed,
        sf.judgment_activities::real AS judgment_activities,
        sf.graded_credits_earned::real AS graded_credits_earned,
        (sf.graded_credits_earned / cv.cfu::real) AS credits_progress,
        sf.average_grade::real AS average_grade,
        sf.weighted_average_grade::real AS weighted_average_grade,
        sf.honors_count::real AS honors_count,
        cv.months::real AS expected_months,
        cv.cfu::real AS expected_cfu,
        {target} AS degree_delay_months,
        ( sf.event_observed )::integer::real AS degree_delay_observed{survival_columns}
    FROM ml_model.student_features_12m AS sf
    JOIN public.curriculum_versions AS cur
      ON cur.pk = sf.curriculum_version
    JOIN public.course_versions AS cv
      ON cv.pk = cur.course_version{follow_up_join}
    {filter}
    ORDER BY sf.enrollment_id
"""


# kept for convenience: the query used by the training phase
TRAINING_QUERY = features_query()


class StupydDataset( Dataset[tuple[Tensor, Tensor]] ):
    """
    Dataset for the StuPYD appklication
    that loads all the feature from the backend database.

    Each entry is made by two tensors (float32): feature vector and
    degree delay in months.

    Each item is a pair of float32 tensors: feature vector and degree delay in
    months. The target is total actual time to graduation minus the course's
    expected duration. Censored enrollments are intentionally excluded until a
    survival-model training path is added.
    """

    feature_names = FEATURE_NAMES

    # the columns a survival model needs in addition to the degree delay:
    # the months observed from the observation date and whether the
    # enrollment actually graduated. They are filled only when the dataset
    # is built for a survival analysis.
    durations : Tensor | None = None
    event_observed : Tensor | None = None

    def __init__( self,
                  dsn: str | None = None,
                  completed_only: bool = True,
                  survival: bool = False ) -> None :
        dsn = dsn or os.environ.get( "STUPYD_DATABASE_URL" )
        if not dsn:
            raise ValueError( "Set STUPYD_DATABASE_URL or pass a PostgreSQL DSN." )

        with psycopg.connect( dsn, row_factory = dict_row ) as connection:
            rows = connection.execute(
                features_query( completed_only, survival ) ).fetchall()

        if not rows:
            raise ValueError( "The training query returned no completed enrollments." )


        # collect all the ids out of the database rows
        # NOTE: allow repeated values to keep aligned the tensors
        enrollment_ids    = []
        student_ids       = []
        observation_years = []
        genders           = []
        features          = []
        targets           = []
        targets_observed  = []
        durations         = []
        events            = []

        for row in rows:
            enrollment_ids.append( row[ "enrollment_id" ] )
            student_ids.append( row[ "student_id" ] )
            observation_years.append( row[ "observation_year" ] )
            genders.append( row[ "gender" ] )

            feature_values = []
            for feature_name in FEATURE_NAMES:
                feature_values.append( row[ feature_name ] )
            features.append( feature_values )

            targets.append( row[ "degree_delay_months" ] )
            targets_observed.append( row[ "degree_delay_observed" ] )

            if survival:
                durations.append( row[ "duration_months" ] )
                events.append( row[ "event_observed" ] )


        # now create all the tensors
        self.enrollment_ids    = torch.tensor( enrollment_ids, dtype = torch.int64 )
        self.student_ids       = torch.tensor( student_ids, dtype = torch.int64 )
        self.observation_years = torch.tensor( observation_years, dtype = torch.int64 )
        self.features          = torch.tensor( features, dtype = torch.float32 )
        self.targets           = torch.tensor( targets, dtype = torch.float32 )

        # False for censored enrollments, since their degree delay is unknown
        self.target_observed = torch.tensor( targets_observed, dtype = torch.bool )

        # a survival analysis also needs the observed duration and the event
        if survival:
            self.durations = torch.tensor( durations, dtype = torch.float32 )
            self.event_observed = torch.tensor( events, dtype = torch.bool )

        # genders are used only to filter, so
        # no need for a tensor here
        self.genders = tuple( genders )

    def __len__( self ) :
        return len( self.targets )

    def __getitem__( self, index ) -> tuple[Tensor, Tensor]:
        return self.features[ index ], self.targets[ index ]


def resolve_dsn( configuration : dict ) -> str :
    """
    Extracts the PostgreSQL DSN from the environment variable
    named by the `database` section of the configuration.

    If the environment variable is missing or empty, an exception is thrown.
    """

    database_configuration = configuration.get( "database", {} )
    if not isinstance( database_configuration, dict ):
        raise ValueError( "configuration.database must be a dictionary." )

    dsn_environment = database_configuration.get( "dsn_env", "STUPYD_DATABASE_URL" )

    dsn = os.environ.get( dsn_environment )
    if not dsn:
        raise ValueError( f'Environment variable {dsn_environment} not set!' )

    return dsn


def stupyd_dataset( configuration : dict,
                    completed_only : bool = True,
                    survival : bool = False ) -> StupydDataset :
    """
    Connects to the backend database and returns the dataset described
    by the given configuration.

    With `completed_only = False` the censored enrollments are loaded as well,
    that is the ones the prediction phase has to score.

    With `survival` the dataset carries the observed durations and the events,
    so that the censored enrollments can train a survival model.
    """

    return StupydDataset( resolve_dsn( configuration ),
                          completed_only = completed_only,
                          survival = survival )


class StupydDataLoader( DataLoader[tuple[Tensor, Tensor]] ):
    """
    Dataloader for the StuPYD project.

    Connects to the backend database to extract the tuples used
    for the training phase.

    The connection is performed using the STUPYD_DATABASE_URL
    that is assumed to be a PostgreSQL instance.
    """

    def __init__( self, configuration, dataset = None, completed_only = True, *args ) :

        # check arguments
        if configuration is None :
            raise TypeError( 'Specify a configuration object and a valida dataset!' )

        if not isinstance( configuration, dict ):
            raise TypeError( "configuration must be a dictionary!" )

        loader_configuration = configuration.get( "dataloader", {} )
        if not isinstance( loader_configuration, dict ):
            raise ValueError( "configuration.dataloader must be a dictionary." )

        batch_size = configuration.get( 'batch_size', loader_configuration.get( "batch_size", 256 ) )
        shuffle    = loader_configuration.get( "shuffle", True )
        workers    = loader_configuration.get( "num_workers", 0 )


        # if no dataset has been specified, create one on the fly
        # NOTE: any torch Dataset is accepted (e.g., a Subset), so that
        #       a single cohort can be loaded without querying the database
        if dataset is None:
            dataset = stupyd_dataset( configuration, completed_only )
        elif not isinstance( dataset, Dataset ):
            raise TypeError( "dataset must be a torch Dataset!" )

        # finish the creation of the loader
        super().__init__( dataset,
                          batch_size = batch_size,
                          shuffle = shuffle,
                          num_workers = workers )
