"""
Chronological cohorts and batched inference shared by the StuPYD workflows.

Both the training and the prediction phases work on the same kind of
object: a dataset plus the cohorts of enrollments described by the
`periods` section of the configuration. That common part lives here, in
the base class `StupydCohort`, so that the two workflows only add what
is specific to them.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch import Tensor

from stupyd.data import StupydDataset, stupyd_dataset
from stupyd.model import StupydModel


class StupydCohort :
    """
    Base class for the StuPYD workflows.

    It owns the dataset, the selection of the chronological cohorts,
    and the batched inference used to score a cohort.

    Parameters:
    - configuration the configuration dictionary of the application
    - dataset an already built dataset, if None it is fetched
      from the backend database
    - completed_only keep only the enrollments with an observed event;
      set it to False to include the censored ones, that is the enrollments
      still in progress the prediction phase has to score
    """

    # the ranges used to report the results by age at enrollment
    AGE_RANGES = (
        ( "<20", 0, 19 ),
        ( "20-24", 20, 24 ),
        ( "25-29", 25, 29 ),
        ( "30+", 30, None ),
    )


    def __init__( self,
                  configuration : dict,
                  dataset : StupydDataset | None = None,
                  completed_only : bool = True ) -> None :

        if not isinstance( configuration, dict ):
            raise TypeError( "configuration must be a dictionary." )

        self.configuration  = configuration
        self.device         = configuration.get( "device", "cpu" )
        self.completed_only = completed_only
        self.dataset        = self.create_dataset( dataset )


    survival = False


    def create_dataset( self, dataset : StupydDataset | None = None ) -> StupydDataset :
        """
        Returns the given dataset or, if none has been given,
        connects to the backend database to build a new one.

        Subclasses that need the censored enrollments, or the durations of a
        survival analysis, override the arguments passed to the factory.
        """

        if isinstance( dataset, StupydDataset ):
            return dataset

        return stupyd_dataset( self.configuration,
                              self.completed_only,
                              self.survival )


    def period_indexes( self, period : str ) -> Tensor :
        """
        Returns the indexes of the enrollments whose observation
        year falls into the configured period.
        """

        period_configuration = self.configuration["periods"].get( period )
        if period_configuration is None:
            raise ValueError( f"Unknown configured period: {period}." )

        mask = (
            ( self.dataset.observation_years >= period_configuration["begin"] )
            & ( self.dataset.observation_years <= period_configuration["end"] )
        )
        indexes = torch.where( mask )[0]

        if len( indexes ) == 0:
            raise ValueError( f"No examples for configured period {period}." )

        return indexes


    def age_range( self, age : float, age_is_missing : bool ) -> str :
        """
        Returns the reporting range of the age at enrollment.
        Enrollments without age are reported as UNKNOWN.
        """

        if age_is_missing:
            return "UNKNOWN"

        for label, begin, end in self.AGE_RANGES:
            if age >= begin and ( end is None or age <= end ):
                return label

        return "UNKNOWN"


    def chunk_size( self ) -> int :
        """
        Returns the number of enrollments scored at once.
        It is the batch size used by the data loader.
        """

        loader_configuration = self.configuration.get( "dataloader", {} )
        if not isinstance( loader_configuration, dict ):
            raise ValueError( "configuration.dataloader must be a dictionary." )

        return max( int( self.configuration.get( "batch_size",
                                                 loader_configuration.get( "batch_size", 256 ) ) ), 1 )


    def batch_size( self ) -> int :
        """
        Returns the number of enrollments the survival loss is evaluated on
        at once.

        It is smaller than `chunk_size` on purpose: scoring one enrollment
        costs one row of the time axis, while the loss holds that row for
        every enrollment of the batch, so the survival training needs a
        smaller batch than the point regression to stay within memory.
        """

        return max( 1, int( self.configuration.get( "survival_batch_size",
                                                    max( self.chunk_size() // 8, 1 ) ) ) )


    def prediction_frame( self, model : StupydModel, indexes : Tensor ) -> pd.DataFrame :
        """
        Scores the given enrollments, in chunks of `chunk_size` enrollments,
        and returns one row for each of them.

        The returned frame has a row for every enrollment, with the columns:
        enrollment_id, student_id, observation_year, gender, age_at_enrollment,
        age_range, predicted_delay_months, degree_delay_months and
        degree_delay_observed.
        """

        age_index         = self.dataset.feature_names.index( "age_at_enrollment" )
        age_missing_index = self.dataset.feature_names.index( "age_at_enrollment_missing" )

        rows    = indexes.tolist()
        records = []

        model.eval()

        with torch.inference_mode():
            for start in range( 0, len( rows ), self.chunk_size() ):

                chunk    = rows[ start : start + self.chunk_size() ]
                features = self.dataset.features[ chunk ].to( self.device )

                predictions = model.predict_degree_delay( features ).cpu().tolist()

                for row, prediction in zip( chunk, predictions ):
                    age = self.dataset.features[ row, age_index ].item()
                    age_is_missing = bool(
                        self.dataset.features[ row, age_missing_index ].item()
                    )

                    records.append( {
                        "enrollment_id"          : int( self.dataset.enrollment_ids[ row ] ),
                        "student_id"             : int( self.dataset.student_ids[ row ] ),
                        "observation_year"       : int( self.dataset.observation_years[ row ] ),
                        "gender"                 : self.dataset.genders[ row ],
                        "age_at_enrollment"      : age,
                        "age_range"              : self.age_range( age, age_is_missing ),
                        "predicted_delay_months" : prediction,
                        "degree_delay_months"    : self.dataset.targets[ row ].item(),
                        "degree_delay_observed"  : bool(
                            self.dataset.target_observed[ row ].item()
                        ),
                    } )

        return pd.DataFrame( records )


    def prediction_summary( self,
                            frame : pd.DataFrame,
                            delay_column : str = "predicted_delay_months" ) -> pd.DataFrame :
        """
        Groups a prediction frame by gender and age range,
        reporting how many enrollments there are and
        the mean predicted degree delay in months.

        `delay_column` names the column holding the prediction, since the
        survival workflow names it after the statistic it reports rather than
        after the model that produced it.
        """

        if delay_column not in frame.columns:
            raise ValueError( f"The column {delay_column} is not in the prediction frame." )

        if len( frame ) == 0:
            return pd.DataFrame( columns = [ "gender", "age_range", "count",
                                             "mean_predicted_delay_months" ] )

        return ( frame.groupby( [ "gender", "age_range" ], as_index = False )
                     .agg( count = ( delay_column, "size" ),
                           mean_predicted_delay_months = ( delay_column, "mean" ) ) )