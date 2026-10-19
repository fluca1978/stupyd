"""Prediction workflow for the degree-delay regression model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from stupyd.cohort import StupydCohort
from stupyd.data import StupydDataset
from stupyd.model import StupydModel, StupydRegressorModel


# name of the CSV file where the predictions are saved,
# it is written next to the model that generated them
PREDICTION_FILENAME = "stupyd_prediction.csv"


class StupydRegressorPredictor( StupydCohort ) :
    """
    Load a model saved by the training phase and score a cohort.

    Differently from the training phase, this workflow loads the censored
    enrollments as well, that is the students still in progress: their degree
    delay is unknown, so they cannot train a model but they can be predicted.
    """

    def __init__( self, configuration : dict,
                  model_class : type[StupydModel] = StupydRegressorModel,
                  dataset : StupydDataset | None = None ) -> None :

        # the censored enrollments are exactly the ones to score
        super().__init__( configuration, dataset, completed_only = False )
        self.model_class = model_class


    def load_model( self, model_path : str | Path ) -> StupydModel :
        """
        Loads the model stored by the training phase in `model_path`.

        The features the model has been trained on are compared to the ones
        of the database, so that a mismatching model is refused instead of
        silently producing wrong predictions.
        """

        model_path = Path( model_path )

        if not model_path.is_file():
            raise FileNotFoundError( f'Model file {model_path} not found!' )

        checkpoint = torch.load( model_path, map_location = self.device )

        model_features = checkpoint.get( "feature_names" )
        if model_features is not None and tuple( model_features ) != tuple( self.dataset.feature_names ):
            raise ValueError( "The features of the model do not match the features in the database." )

        return self.model_class.load( model_path, self.device )


    def report_predictions( self,
                            predictions : pd.DataFrame,
                            period : str,
                            output_path : Path ) -> None :
        """
        Prints the summary of the predictions, then saves them all
        in the given CSV file.
        """

        summary = self.prediction_summary( predictions )

        print( f"{period}_predicted_delay_months" )
        print( summary.to_string( index = False,
                                  formatters = { "mean_predicted_delay_months" : "{:.2f}".format } ) )

        # the completed enrollments of the cohort allow to check
        # how accurate the predictions are
        completed = predictions[ predictions[ "degree_delay_observed" ] ]
        if len( completed ) > 0:
            mae = ( completed[ "predicted_delay_months" ]
                    - completed[ "degree_delay_months" ] ).abs().mean()
            print( f"{period}_mae_months {mae:.2f} on {len( completed )} completed enrollments" )

        print( f"\nFirst predictions:\n{predictions.head( 10 ).to_string( index = False )}" )

        predictions.to_csv( output_path, index = False )
        print( f"Predictions saved to [{output_path}]" )


    def run( self ) :
        """
        Loads the model, scores all the enrollments of the given period and
        returns the path of the CSV file with the predictions and the
        predictions themselves.
        """

        model   = self.load_model( self.configuration[ 'model_path' ] )
        indexes = self.period_indexes( self.configuration[ 'period' ] )

        predictions = self.prediction_frame( model, indexes )

        # the CSV is written next to the model that produced it
        output_path = Path( self.configuration[ 'model_path' ] ).with_name( PREDICTION_FILENAME )
        period      = self.configuration[ 'period' ]
        self.report_predictions( predictions, period, output_path )

        return output_path, predictions
