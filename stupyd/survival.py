"""Discrete-time survival workflow for the degree-delay model.

A regression can only learn from the enrollments that graduated, so every
student still in progress is discarded from the training set. That is not a
minor loss: for the most recent cohorts it is the whole cohort, and the
students who take longer are systematically the ones being dropped.

A survival model does not discard them. Each enrollment contributes its
observed duration, and the censoring simply states that the delay is larger
than the follow-up. The model then estimates the hazard of graduating within
each interval of the time axis, which is a distribution rather than a point.

The time origin is the observation date, that is 12 months after the
enrollment, since the features are measured there. Given the duration `T` in
months from that date, the degree delay is:

    delay = T + 12 - expected_months

The loss is the discrete-time negative log-likelihood. For an enrollment, the
hazard of the intervals it survived is compared to 1 for the interval in which
it graduated, and to 0 for the following intervals, when the enrollment was
censored: a censored enrollment teaches that the student had not graduated yet,
without pretending to know when they will.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm
from torch.nn import functional as functional

from stupyd.cohort import StupydCohort
from stupyd.data import StupydDataset
from stupyd.model import StupydSurvivalModel
from stupyd.predict import PREDICTION_FILENAME


class StupydSurvivalDataset( StupydDataset ) :
    """
    Dataset of the enrollments prepared for a survival analysis.

    It extends the ordinary dataset with the two quantities a survival model
    needs and the regression does not use:

    - `durations` the months observed from the observation date, either up to
      the graduation or up to the censoring
    - `event_observed` True when the enrollment graduated, False when the
      observation was censored, that is the delay is still unknown

    Censored enrollments keep a duration as well: their follow-up time, that
    is the time they have been observed so far.
    """

    def __getitem__( self, index ) -> tuple[Tensor, Tensor, Tensor] :
        return self.features[ index ], self.durations[ index ], self.event_observed[ index ]


class StupydSurvivalCohort( StupydCohort ) :
    """
    Shared behaviour of the two survival workflows, training and scoring.

    It keeps the hazards and the duration metrics, which the point regression
    has no use for, so that `StupydSurvivalTrainer` and
    `StupydSurvivalPredictor` only differ in what they do with the model.
    """

    # every enrollment is loaded with its duration and its event
    survival = True

    # the features are measured at the observation date, which falls this
    # many months after the enrollment, while the degree delay is counted
    # from the enrollment itself
    OBSERVATION_MONTHS = 12

    def __init__( self,
                  configuration : dict,
                  dataset : StupydDataset | None = None ) -> None :

        # a survival model needs the censored enrollments as well,
        # and with them their observed durations
        super().__init__( configuration, dataset, completed_only = False )
        torch.manual_seed( configuration["seed"] )


    def interval_limits( self,
                         months_interval : float = 6.0,
                         months_max : float = 360.0,
                         indexes : Tensor | None = None ) -> Tensor :
        """
        Builds the time axis, that is the limits of the intervals in which
        the hazard is estimated.

        The axis is bounded by `months_max` rather than by the longest
        observed duration, since a single implausible duration would
        otherwise multiply the intervals of every enrollment by the same
        factor, and the loss evaluates the whole dataset at once.

        The durations above the bound are clipped. For a censored enrollment
        that is conservative rather than wrong: the model is only asked to
        state that the student was still in progress at the end of the
        follow-up, which is implied by a longer one too. A graduated
        enrollment above the bound is a data problem instead, so it is
        reported.
        """

        if months_interval <= 0:
            raise ValueError( "The width of the intervals must be positive." )

        if months_max <= 0:
            raise ValueError( "The bound of the follow-up must be positive." )

        durations = self.observed_durations( indexes )

        # the censored durations matter as well: they define the follow-up
        longest = float( durations.clamp( max = months_max ).max() )

        # an enrollment that reports a duration beyond any plausible
        # follow-up points at a bad date rather than at a long delay
        beyond = durations > months_max

        if bool( beyond.any() ):
            graduated = self.dataset.event_observed
            if indexes is not None:
                graduated = graduated[ indexes ]

            impossible = int( ( beyond & graduated ).sum() )

            print( f"durations above {months_max:.0f} months: "
                   f"{int( beyond.sum() )} enrollments "
                   f"({int( ( ~beyond ).sum() )} kept as is), "
                   f"{impossible} of them already graduated, which is "
                   f"most likely a bad date in the database" )

        intervals = max( 1, math.ceil( longest / months_interval ) )

        return torch.arange( intervals + 1, dtype = torch.float32 ) * months_interval


    def observed_durations( self, indexes : Tensor | None = None ) -> Tensor :
        """Returns the observed durations, in months from the observation date."""

        return self.dataset.durations if indexes is None else self.dataset.durations[ indexes ]


    def degree_delay_offset( self, indexes : Tensor | None = None ) -> Tensor :
        """
        Returns the months to add to a duration counted from the observation
        date to obtain the degree delay of each enrollment.

        The observation date falls `OBSERVATION_MONTHS` after the enrollment,
        which have to be added back, and the delay is measured against the
        nominal duration of the curriculum, which is a feature of every
        enrollment:

            delay = duration + OBSERVATION_MONTHS - expected_months
        """

        position = self.dataset.feature_names.index( "expected_months" )

        expected = ( self.dataset.features[ :, position ]
                     if indexes is None
                     else self.dataset.features[ indexes, position ] )

        return self.OBSERVATION_MONTHS - expected


    def hazard_dataset( self,
                        model : StupydSurvivalModel,
                        rows : Tensor | None = None ) -> dict[str, Tensor] :
        """
        Builds, for every enrollment and every interval, the tensors needed
        by the loss, that is the outcome of each interval: 1 when the
        enrollment graduated within it, 0 otherwise, together with the mask
        of the intervals it was at risk in.

        An enrollment only contributes to the intervals it actually survived:
        once it has graduated, the later intervals say nothing about it.
        Censored enrollments contribute 0 outcomes up to their censoring,
        which is precisely the information they carry.

        `rows` restricts the result to a chunk of enrollments, since the
        tensors grow with the product of the enrollments and the intervals.
        """

        if rows is None:
            durations = self.observed_durations()
            graduated = self.dataset.event_observed
        else:
            durations = self.observed_durations( rows )
            graduated = self.dataset.event_observed[ rows ]

        limits = model.interval_limits
        widths = ( limits[ 1 : ] - limits[ : -1 ] )

        # the interval k ends at limits[k + 1]
        entered  = durations.unsqueeze( 1 ) >= limits[ : -1 ].unsqueeze( 0 )
        finished = durations.unsqueeze( 1 ) < limits[ 1 : ].unsqueeze( 0 )

        # An enrollment is at risk in every interval it entered: either it
        # survived the whole interval, or the graduation happened inside it,
        # or it was censored inside it. The last case is a partial exposure,
        # which is counted as a full one on purpose: dropping it would lose
        # the information that the student was still in progress.
        at_risk = entered

        # the outcome of an interval is 1 only for the one
        # in which the enrollment graduated
        event = entered & finished & graduated.unsqueeze( 1 )

        return {
            "durations" : durations,
            "entered"   : entered,
            "at_risk"   : at_risk,
            # the outcomes enter the loss arithmetically, hence as floats
            "event"     : event.to( torch.float32 ),
            "widths"    : widths,
        }


    def training_loss( self,
                       model : StupydSurvivalModel,
                       rows : Tensor ) -> Tensor :
        """
        The loss of one chunk of enrollments, given the tensors
        the chunk is restricted to.
        """

        features = self.dataset.features[ rows ].to( self.device )
        tensors  = self.hazard_dataset( model, rows )

        return self.negative_log_likelihood( model, tensors, features )


    def negative_log_likelihood( self,
                                 model : StupydSurvivalModel,
                                 tensors : dict[str, Tensor],
                                 features : Tensor ) -> Tensor :
        """
        Discrete-time negative log-likelihood, averaged over the enrollments.

        Every interval is a Bernoulli trial: the enrollment either graduated
        within it or it did not, so the loss is the sum, over the intervals
        the enrollment was at risk in, of the negative log of the
        probability of each outcome:

            -log( 1 - h_j )            for every interval j the enrollment survived
            -log( h_k )                for the interval k the enrollment graduated in

        A censored enrollment only contributes the first term, over every
        interval of the follow-up, which equals -log S(k).
        """

        hazard = model( features )

        # The logarithm is computed everywhere and then masked out, hence
        # the hazard has to stay away from 0 and 1 to avoid infinities
        safe_hazard = hazard.clamp( 1e-6, 1.0 - 1e-6 )

        # every interval is a Bernoulli trial: the enrollment either
        # graduated within it or it did not. No survival factor multiplies
        # these terms, since each interval is already a conditional
        # probability and multiplying by the previous survival would count
        # the same history twice.
        outcomes = tensors[ "event" ]
        loss = -( outcomes * torch.log( safe_hazard )
                  + ( 1.0 - outcomes ) * torch.log( 1.0 - safe_hazard ) )

        # only the intervals the enrollment was really at risk in count
        at_risk        = tensors[ "at_risk" ]
        per_enrollment = ( loss * at_risk ).sum( dim = -1 )

        # a row per enrollment, so that every enrollment weighs the same
        return per_enrollment.mean()


    def survival_frame( self,
                        model : StupydSurvivalModel,
                        indexes : Tensor ) -> pd.DataFrame :
        """
        Returns the predicted distribution of the graduation time for every
        enrollment of the cohort: the probability of having graduated by the
        end of each interval of the time axis.

        The columns are named after the index of the interval on the time
        axis: `p_graduated_by_<index>` holds the probability of having
        graduated by the end of that interval, up to the last interval of
        the follow-up, whatever the width of the intervals. `p_graduated` is
        the probability of graduating within the follow-up, which is the
        one that separates the students still in progress from the ones
        already done.

        `median_predicted_delay_months` and `expected_delay_months` are the
        only two columns measured from the enrollment rather than from the
        observation date, since those are the numbers the point regression
        gives and the ones the delay has to be compared against.
        """

        rows    = indexes.tolist()
        records = []

        model.eval()

        age_index         = self.dataset.feature_names.index( "age_at_enrollment" )
        age_missing_index = self.dataset.feature_names.index( "age_at_enrollment_missing" )

        with torch.inference_mode():
            for start in range( 0, len( rows ), self.chunk_size() ):
                chunk    = rows[ start : start + self.chunk_size() ]
                features = self.dataset.features[ chunk ].to( self.device )

                incidence = model.cumulative_incidence( features ).cpu()
                median    = model.median_degree_delay( features ).cpu()
                expected  = model.expected_degree_delay( features ).cpu()

                # the model counts months from the observation date, while
                # the two columns hold the delay the regression predicts
                offset   = self.degree_delay_offset( torch.as_tensor( chunk ) )
                median   = median + offset
                expected = expected + offset

                for row, probabilities, median_delay, expected_delay in zip(
                        chunk, incidence.tolist(), median.tolist(), expected.tolist() ):

                    age = self.dataset.features[ row, age_index ].item()
                    age_is_missing = bool(
                        self.dataset.features[ row, age_missing_index ].item() )

                    record = {
                        "enrollment_id"                 : int( self.dataset.enrollment_ids[ row ] ),
                        "student_id"                    : int( self.dataset.student_ids[ row ] ),
                        "observation_year"              : int( self.dataset.observation_years[ row ] ),
                        "gender"                        : self.dataset.genders[ row ],
                        "age_at_enrollment"             : age,
                        "age_range"                     : self.age_range( age, age_is_missing ),
                        "median_predicted_delay_months" : median_delay,
                        "expected_delay_months"         : expected_delay,
                        # the delay is unknown while the student is in progress
                        "degree_delay_months"           : self.dataset.targets[ row ].item(),
                        "degree_delay_observed"         : bool(
                            self.dataset.target_observed[ row ].item() ),
                        "p_graduated"                   : probabilities[ -1 ],
                    }

                    for index, probability in enumerate( probabilities ):
                        record[ f"p_graduated_by_{index}" ] = probability

                    records.append( record )

        return pd.DataFrame( records )


    def concordance( self,
                     model : StupydSurvivalModel,
                     indexes : Tensor,
                     risk : Tensor | None = None ) -> dict[str, float] :
        """
        Harrell's concordance index, overall and per gender.

        It is the share of comparable pairs ordered correctly: given two
        enrollments, the one that graduates earlier has to receive the higher
        predicted probability of graduating. A value of 0.5 is chance, 1.0 is
        a perfect ordering.

        Tied predictions are credited with half a point.

        The risk score is the negated expected delay, so the enrollment that
        graduates earlier is expected to receive the higher score. It can be
        passed explicitly to measure a given ordering instead of the one the
        model produces.
        """

        durations = self.observed_durations( indexes )
        graduated = self.dataset.event_observed[ indexes ]

        rows = indexes.tolist()

        if risk is None:
            predictions = []

            model.eval()

            with torch.inference_mode():
                for start in range( 0, len( rows ), self.chunk_size() ):
                    chunk    = rows[ start : start + self.chunk_size() ]
                    features = self.dataset.features[ chunk ].to( self.device )

                    # the expected time is used as the risk score: the higher
                    # the chance of graduating early, the lower the delay
                    predictions.append(
                        -model.expected_degree_delay( features ).cpu() )

            risk = torch.cat( predictions )

        risk = risk.cpu()
        risk = risk[ graduated ]

        durations = durations[ graduated ]

        metrics = {}

        for gender in [ "overall" ] + list( self.configuration["genders"] ):
            if gender == "overall":
                selected_risk      = risk
                selected_durations = durations
            else:
                mask = torch.tensor(
                    [ self.dataset.genders[ row ] == gender for row in rows ],
                    dtype              = torch.bool )[ graduated ]
                selected_risk      = risk[ mask ]
                selected_durations = durations[ mask ]

            if len( selected_risk ) < 2:
                continue

            # every pair of comparable enrollments: the first graduates
            # strictly before the second, so its duration is smaller.
            # Both are graduated, since a censored enrollment has no event
            comparable = ( selected_durations.unsqueeze( 1 )
                           < selected_durations.unsqueeze( 0 ) )

            if not bool( comparable.any() ):
                continue

            # the ordering is correct when the earlier graduation has the
            # higher risk, that is a smaller expected delay
            correct = ( selected_risk.unsqueeze( 1 ) > selected_risk.unsqueeze( 0 ) )
            tied    = ( selected_risk.unsqueeze( 1 ) == selected_risk.unsqueeze( 0 ) )

            points = ( correct | tied ).to( torch.float64 )
            points = torch.where( tied, torch.full_like( points, 0.5 ), points )

            metrics[ gender ] = ( points[ comparable ].sum() / comparable.sum() ).item()

        return metrics


    def median_mae( self,
                    model : StupydSurvivalModel,
                    indexes : Tensor ) -> dict[str, float] :
        """
        Mean absolute error, in months, of the predicted median against the
        observed graduation time, overall and per gender.

        It is computed on the graduated enrollments only, since the delay of
        a censored enrollment is unknown. It is the bridge with the point
        regression: the same metric, so the two models can be compared.

        Both sides are counted from the observation date rather than from the
        enrollment, and that is enough: the offset that turns one into the
        other, `OBSERVATION_MONTHS - expected_months`, belongs to the
        enrollment rather than to the prediction, so it is the same on both
        sides of the difference and cancels out. The error is the one of the
        degree delay, without converting anything.
        """

        rows      = indexes.tolist()
        graduated = self.dataset.event_observed[ indexes ]

        genders = [ "overall" ] + list( self.configuration["genders"] )

        absolute_error = { gender : 0.0 for gender in genders }
        count          = { gender : 0     for gender in genders }

        model.eval()

        with torch.inference_mode():
            for start in range( 0, len( rows ), self.chunk_size() ):
                stop = start + self.chunk_size()

                chunk = rows[ start : stop ]

                # the flags are aligned with the chunk by position
                flags = graduated[ start : stop ]

                features = self.dataset.features[ chunk ].to( self.device )
                median   = model.median_degree_delay( features ).cpu()
                observed = self.dataset.durations[ chunk ]

                # the delay of a censored enrollment is unknown,
                # hence it cannot enter the error
                error = ( median - observed ).abs() * flags

                absolute_error[ "overall" ] += float( error.sum() )
                count[ "overall" ]          += int( flags.sum() )

                for gender in self.configuration["genders"]:
                    belongs = torch.tensor(
                        [ self.dataset.genders[ row ] == gender for row in chunk ],
                        dtype = torch.bool )

                    absolute_error[ gender ] += float( ( error * belongs ).sum() )
                    count[ gender ]          += int( ( flags & belongs ).sum() )

        return { gender : absolute_error[ gender ] / count[ gender ]
                 for gender in genders if count[ gender ] }



class StupydSurvivalTrainer( StupydSurvivalCohort ) :
    """
    Train a discrete-time hazard model on the configured cohorts.

    The metrics differ from the ones of the point regression on purpose: a
    model that predicts a distribution has to be judged on how well it orders
    the enrollments and on the calibration of its probabilities, not only on
    the error of a single number.
    """

    def run( self,
             model_class : type = StupydSurvivalModel ) :
        """
        Train the hazard model on the configured cohorts and save its
        checkpoint, whose path is returned.
        """

        indexes = {
            "training"   : self.period_indexes( "training" ),
            "validating" : self.period_indexes( "validating" ),
            "testing"    : self.period_indexes( "testing" ),
        }

        train_features = self.dataset.features[ indexes["training"] ]
        train_targets  = self.observed_durations( indexes["training"] )

        limits = self.interval_limits( months_interval = self.configuration[ 'months_interval' ],
                                       months_max      = self.configuration[ 'months_max' ],
                                       indexes         = indexes["training"] )

        model = model_class(
            train_features.mean( dim = 0 ),
            train_features.std( dim = 0, correction = 0 ),
            train_targets.mean(),
            train_targets.std( correction = 0 ),
            limits,
        ).to( self.device )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr = self.configuration["learning_rate"] )

        epochs        = int( self.configuration["epochs"] )
        training_rows = indexes["training"]

        # The hazard of an enrollment is a whole row of the time axis, so the
        # tensors grow with the product of the enrollments and the intervals:
        # a single batch over the full dataset does not fit in memory. The
        # loss is therefore averaged over the chunks of one epoch, so that a
        # step sees every enrollment while only holding a chunk at a time.
        chunks = [ training_rows[ start : start + self.batch_size() ]
                   for start in range( 0, len( training_rows ), self.batch_size() ) ]

        with tqdm( total = epochs, desc = "Training (survival)" ) as progress_bar:
            for epoch in range( epochs ):
                model.train()

                total = 0.0

                for rows in chunks:
                    optimizer.zero_grad( set_to_none = True )

                    loss = self.training_loss( model, rows )
                    loss.backward()
                    optimizer.step()

                    # detach before converting: the running total is a plain
                    # number, and reading it out of the graph would keep
                    # every chunk of the epoch alive until the end of it
                    total += float( loss.detach() ) * len( rows )

                # print( f"epoch {epoch + 1}/{epochs} "
                #        f"negative_log_likelihood "
                #        f"{total / len( training_rows ):.4f}" )

                progress_bar.update( 1 )

        print( f"time axis: {model.interval_count} intervals "
               f"of {float( limits[1] - limits[0] ):.0f} months, "
               f"up to {float( limits[-1] ):.0f} months" )

        concordance = {}
        mae         = {}

        for name, cohort in ( ( "validation", indexes["validating"] ),
                              ( "test", indexes["testing"] ) ):
            concordance[ name ] = self.concordance( model, cohort )
            mae[ name ] = self.median_mae( model, cohort )

            print( f"{name}_concordance "
                   + " ".join( f"{gender}={value:.3f}"
                               for gender, value in concordance[ name ].items() ) )
            print( f"{name}_median_mae_months "
                   + " ".join( f"{gender}={value:.2f}"
                               for gender, value in mae[ name ].items() ) )

        # the checkpoint goes into its own subdirectory, since it cannot be
        # loaded by the point prediction phase
        model_path = model.model_filename( self.configuration.get( "outdir", "./models" ) )

        model_path.parent.mkdir( parents = True, exist_ok = True )
        torch.save(
            {
                "model_state"                  : model.state_dict(),
                # the time axis travels with the weights: without it the
                # checkpoint could not rebuild the same intervals
                "model_arguments"              : { "interval_limits" : limits.tolist() },
                "feature_names"                : self.dataset.feature_names,
                "periods"                      : self.configuration["periods"],
                "validation_concordance"       : concordance[ "validation" ],
                "test_concordance"             : concordance[ "test" ],
                "validation_median_mae_months" : mae[ "validation" ],
                "test_median_mae_months"       : mae[ "test" ],
            },
            model_path )

        print( f"survival model saved to [{model_path}]" )
        return model_path



class StupydSurvivalPredictor( StupydSurvivalCohort ) :
    """
    Score a cohort with a hazard model saved by `StupydSurvivalTrainer`.

    The output is a distribution rather than a single number: every row holds
    the probability of having graduated by the end of each interval of the time
    axis, plus the two numbers derived from it, the median and the expected
    delay. The probabilities are what distinguishes this file from the one of
    the point regression: they say how certain the model is, and they leave the
    caller free to choose where to cut the distribution.
    """

    def load_model( self, model_path : str | Path ) -> StupydSurvivalModel :
        """
        Loads the hazard model, whose time axis travels with its weights, so
        the columns of the CSV are the intervals the model was trained on.
        """

        model_path = Path( model_path )

        if not model_path.is_file():
            raise FileNotFoundError( f"Model file {model_path} not found!" )

        checkpoint = torch.load( model_path, map_location = self.device )

        model_features = checkpoint.get( "feature_names" )
        if ( model_features is not None
             and tuple( model_features ) != tuple( self.dataset.feature_names ) ):
            raise ValueError( "The features of the model do not match the features in the database." )

        return StupydSurvivalModel.load( model_path, self.device )


    def report_predictions( self,
                            predictions : pd.DataFrame,
                            period : str,
                            output_path : Path ) -> None :
        """
        Prints the summary and the head of the distribution, then saves every
        enrollment in the given CSV file.
        """

        summary = self.prediction_summary( predictions, delay_column = "median_predicted_delay_months" )

        print( f"{period}_median_predicted_delay_months" )
        print( summary.to_string( index = False,
                                  formatters = { "mean_predicted_delay_months" : "{:.2f}".format } ) )

        # the median is the number the point regression produces as well, so
        # the two workflows can be compared on the cohorts they share
        completed = predictions[ predictions[ "degree_delay_observed" ] ]
        if len( completed ) > 0:
            mae = ( completed[ "median_predicted_delay_months" ]
                    - completed[ "degree_delay_months" ] ).abs().mean()
            print( f"{period}_median_mae_months {mae:.2f} "
                   f"on {len( completed )} graduated enrollments" )

        print( f"{period}_predicted_to_have_graduated "
               f"{100 * float( predictions['p_graduated'].mean() ):.1f}% of the cohort" )

        print( f"\nFirst predictions:\n{predictions.head( 10 ).to_string( index = False )}" )

        predictions.to_csv( output_path, index = False )
        print( f"Predictions saved to [{output_path}]" )


    def run( self ) -> tuple[Path, pd.DataFrame] :
        """
        Loads the hazard model, scores the enrollments of the given period and
        returns the path of the CSV file with the predictions and the
        predictions themselves.
        """

        model   = self.load_model( self.configuration[ "model_path" ] )
        indexes = self.period_indexes( self.configuration[ "period" ] )

        predictions = self.survival_frame( model, indexes )

        output_path = Path( self.configuration[ "model_path" ] ).with_name( PREDICTION_FILENAME )
        period      = self.configuration[ "period" ]
        self.report_predictions( predictions, period, output_path )

        return output_path, predictions
