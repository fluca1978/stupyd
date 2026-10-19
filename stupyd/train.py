"""Training workflow for the degree-delay regression model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as functional
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from stupyd.cohort import StupydCohort
from stupyd.data import StupydDataLoader, StupydDataset
from stupyd.model import StupydModel, StupydRegressorModel


class StupydRegressorTrainer( StupydCohort ) :
    """Train, validate, test, and save a degree-delay regression model."""

    def __init__( self,
                  configuration : dict,
                  dataset : StupydDataset | None = None ) -> None :
        # only the enrollments with an observed event can train a regressor
        super().__init__( configuration, dataset, completed_only = True )
        torch.manual_seed( configuration["seed"] )


    def validation_mae( self,
                        model : StupydModel,
                        loader : DataLoader ) -> float :
        """Calculate mean absolute prediction error in months."""
        model.eval()
        absolute_error = 0.0
        row_count      = 0
        with torch.inference_mode():
            for features, targets in loader:
                features    = features.to( self.device, non_blocking = True )
                targets     = targets.to( self.device, non_blocking = True )
                predictions = model.predict_degree_delay( features )
                absolute_error += ( predictions - targets ).abs().sum().item()
                row_count += len( targets )
        return absolute_error / row_count


    def period_metrics( self,
                        model : StupydModel,
                        indexes : Tensor ) -> dict[str, float] :
        """Calculate overall and per-gender MAE for one chronological cohort."""
        metrics = {
            "overall" : self.validation_mae(
                model,
                StupydDataLoader(
                    self.configuration,
                    Subset( self.dataset, indexes.tolist() ),
                ),
            )
        }
        for gender in self.configuration["genders"]:
            gender_indexes = [
                index
                for index in indexes.tolist()
                if self.dataset.genders[index] == gender
            ]
            if gender_indexes:
                metrics[gender] = self.validation_mae(
                    model,
                    StupydDataLoader(
                        self.configuration,
                        Subset( self.dataset, gender_indexes ),
                    ),
                )
        return metrics


    def report_period_predictions( self,
                                  model : StupydModel,
                                  indexes : Tensor,
                                  period : str ) -> pd.DataFrame :
        """
        Prints the table of the mean predicted degree delay by gender and
        age range for the given cohort and returns that table.
        """

        summary = self.prediction_summary( self.prediction_frame( model, indexes ) )

        print( f"{period}_predicted_delay_months" )
        print( summary.to_string( index = False,
                                  formatters = { "mean_predicted_delay_months" : "{:.2f}".format } ) )

        return summary


    def run( self,
            model_class : type[StupydModel] = StupydRegressorModel ) -> Path :
        """Train the configured model and save its checkpoint."""
        indexes = {
            "training"   : self.period_indexes( "training" ),
            "validating" : self.period_indexes( "validating" ),
            "testing"    : self.period_indexes( "testing" ),
        }
        train_features = self.dataset.features[indexes["training"]]
        train_targets  = self.dataset.targets[indexes["training"]]

        model = model_class(
            train_features.mean( dim = 0 ),
            train_features.std( dim = 0, correction = 0 ),
            train_targets.mean(),
            train_targets.std( correction = 0 ),
        ).to( self.device )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr = self.configuration["learning_rate"]
        )
        train_loader = StupydDataLoader(
            self.configuration,
            Subset( self.dataset, indexes["training"].tolist() ),
        )

        epochs = int( self.configuration["epochs"] )
        with tqdm( total = epochs, desc = "Training" ) as progress_bar:
            for _ in range( epochs ):
                model.train()
                for features, targets in train_loader:
                    features = features.to( self.device, non_blocking = True )
                    targets  = targets.to( self.device, non_blocking = True )
                    standardized_targets = (
                        targets - model.target_mean
                    ) / model.target_std

                    optimizer.zero_grad( set_to_none = True )
                    loss = functional.smooth_l1_loss(
                        model.normalized_predictions( features ),
                        standardized_targets,
                    )
                    loss.backward()
                    optimizer.step()
                progress_bar.update( 1 )

        validation_metrics = self.period_metrics( model, indexes["validating"] )
        test_metrics       = self.period_metrics( model, indexes["testing"] )
        for name, metrics in ( ( "validation", validation_metrics ), ( "test", test_metrics ) ):
            print(
                f"{name}_mae_months "
                + " ".join( f"{gender}={mae:.2f}" for gender, mae in metrics.items() )
            )

        self.report_period_predictions( model, indexes["validating"], "validation" )
        self.report_period_predictions( model, indexes["testing"], "test" )

        model_path = model.model_filename( self.configuration.get( "outdir", "./models" ) )
        model_path.parent.mkdir( parents = True, exist_ok = True )
        torch.save(
            {
                "model_state"           : model.state_dict(),
                "feature_names"         : self.dataset.feature_names,
                "periods"               : self.configuration["periods"],
                "validation_mae_months" : validation_metrics,
                "test_mae_months"       : test_metrics,
            },
            model_path,
        )
        return model_path