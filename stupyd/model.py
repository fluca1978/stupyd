"""Neural-network models for degree-delay prediction.

The module keeps two concerns apart:

- `StupydNormalizer` owns the feature and target statistics, so that a model
  is trained on standardized inputs and produces predictions on the original
  scale. It also owns the `load` classmethod, so that any model can be
  restored from a checkpoint.
- the models descend from it through `StupydModel`, the common base of both
  families, and each family only implements the prediction head:
  `StupydRegressorModel` squeezes the network down to one degree delay per
  enrollment, `StupydSurvivalModel` broadcasts over a time axis and returns
  the discrete-time hazard that describes the whole degree-delay
  distribution.

Thanks to that split, a workflow such as `StupydRegressorTrainer.run()`
accepts any model class, and switching from a point prediction to a
distribution does not require touching the data access or the reporting
code.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn


class StupydNormalizer( nn.Module ) :
    """
    Base class for every StuPYD model.

    It keeps the statistics of the training set, so that the inputs are
    standardized and the outputs are brought back to the original scale.

    The following tensors are registered as buffers, hence they move with
    `.to(device)` and they are saved and restored with the model.
    The standard deviations are clamped to a minimum value, since a constant
    feature has a null standard deviation and would produce a division by
    zero in the forward pass.
    """

    def __init__( self,
                  feature_mean : Tensor,
                  feature_std : Tensor,
                  target_mean : Tensor,
                  target_std : Tensor ) :

        super().__init__()

        # both models are built on top of a neural network:
        # the base class normalizes, the subclasses decide what it computes
        self.neural_network = None

        self.register_buffer( "feature_mean", feature_mean )
        self.register_buffer( "feature_std",  feature_std.clamp_min( 1e-6 ) )
        self.register_buffer( "target_mean",  target_mean )
        self.register_buffer( "target_std",   target_std.clamp_min( 1e-6 ) )


    def normalize_features( self, features : Tensor ) -> Tensor :
        """
        Normalizes the features.

        Since the features have different scales, for example CFUs range from 0 to 180,
        while age_at_enrollment can span from 18 to 50 or so, genders are only two,
        there is the risk that one feature dominates the other.

        Normalization moves the features so they are 0-centered.
        It exploits the feature means and std provided during the construction of
        this model.

        Parameters:
        - features the tensor with the features

        Returns: the normalized tensor
        """
        return ( features - self.feature_mean ) / self.feature_std


    def do_predictions( self, features ) :
        """
        Performs a prediction on the already normalized feature tensor
        (hence the output of normalize_features).

        Has to be implemented in the base class! In particulare the
        neural network has to be defined.

        Arguments:
        - features a tensor with the *normalized* features

        Returns: the prediction has done by the neural_network
        """

        if self.neural_network is None :
            raise NotImplementedError( 'Need to be implemented in the actual model!' )

        return self.neural_network( features )


    def normalized_predictions( self, features ) :
        """
        Computes the normalization of the features and
        computes the predictions against the normalized features.
        Therefore it returns the predictions normalized.

        Arguments:
        - features the tensor with the (not yet) normalized features

        Returns:
        - the tensor representing the normalized predictions
        """
        # 1) normalize the features so that there is no value dominating the others
        features = self.normalize_features( features )

        # 2) perform a prediction
        predictions = self.do_predictions( features )

        return predictions


    def forward( self, features ) :
        """
        Performs a step in optimization.
        Arguments are:
        - features the tensor with the  (not yet) normalized features

        Returns: the tensor representing the predictions on the original scale
        """

        # 1) get normalized predictions
        predictions = self.normalized_predictions( features )

        # 2) compute the predictions into the output scale
        predictions = predictions * self.target_std + self.target_mean

        return predictions


    @classmethod
    def load( clazz, model_path, device = "cpu", **kwargs ) :
        """
        Loads a model saved by PyTorch on storage.
        Implements it using the specified class, so that the same model
        can be tested against different model implementations.
        Example of invocation:
           model = StupydRegressorModel.load( 'foo/bar.pt' )

        Arguments:
        - clazz the class to load, this is usually the `self` argument
          in the class method call
        - model_path the path to the saved model on disk
        - device the device where to store the model
        - kwargs the extra arguments the model constructor needs,
          typically the arguments saved in the checkpoint

        Returns: the loaded model
        """

        checkpoint = torch.load( model_path, map_location = device )
        state      = checkpoint[ "model_state" ]

        # a model may need more than the normalization statistics,
        # e.g. the survival model needs the intervals of its time axis
        options = dict( checkpoint.get( "model_arguments", {} ) )
        options.update( kwargs )

        model = clazz( state[ "feature_mean" ],
                       state[ "feature_std" ],
                       state[ "target_mean" ],
                       state[ "target_std" ],
                       **options ).to( device )

        model.load_state_dict( state )
        model.eval()
        return model


    @torch.inference_mode()
    def predict_degree_delay( self, features ) :
        """Predict degree delay in months for one feature row or a batch."""
        self.eval()
        return self( features )


class StupydModel( StupydNormalizer ) :
    """
    Common base of the two model families: the point regression and the
    discrete-time hazard.

    The statistics, the scaling in both directions and the checkpoint
    loading all come from `StupydNormalizer`. What this class adds is the
    one thing both families have to agree on: where a checkpoint of theirs
    belongs on disk, which is `model_filename`.

    What they deliberately do not agree on is the shape of the answer, and
    that is the single thing `do_predictions` has to define:

    - `StupydRegressorModel` squeezes the network down to a single degree
      delay per enrollment;
    - `StupydSurvivalModel` broadcasts the batch over the time axis and
      returns one hazard per interval, which is why it overrides `forward`
      as well.

    Naming the common base is what lets a workflow be typed against one
    class and accept either family.
    """


    @classmethod
    def model_filename( cls, basedir = None ) :
        """
        The path this kind of model is saved to, built as
        `<basedir>/<kind>/YYYY_MM_DD/stupyd_model.pt`.

        `kind` is read out of the class name, so `StupydRegressorModel`
        lands in `regressor` and `StupydSurvivalModel` in `survival`: the
        two families share the root but never a folder, since a checkpoint
        of one cannot be loaded by the other workflow. `basedir` defaults
        to `./models`, and the date is the day of the call, so every run of
        a day lands in the same folder.
        """

        if basedir is None :
            basedir = './models'

        # StupydRegressorModel -> regressor, StupydSurvivalModel -> survival
        kind = cls.__name__.removeprefix( 'Stupyd' ).removesuffix( 'Model' )
        if not kind :
            raise ValueError( f'Cannot tell the kind of model out of {cls.__name__}!' )

        today  = datetime.datetime.now()
        folder = '%04d_%02d_%02d' % ( today.year, today.month, today.day )

        return Path( basedir ) / kind.lower() / folder / 'stupyd_model.pt'


class StupydRegressorModel( StupydModel ) :
    """MLP regressor for total degree delay in months."""

    def __init__( self, feature_mean, feature_std,
                  target_mean, target_std ) :

        super().__init__( feature_mean, feature_std, target_mean, target_std )

        # create this neural network
        self.neural_network = nn.Sequential( nn.Linear( feature_mean.numel(), 64 ),
                                             nn.ReLU(),
                                             nn.Dropout( 0.15 ),
                                             nn.Linear( 64, 32 ),
                                             nn.ReLU(),
                                             nn.Linear( 32, 1 ) )


    def do_predictions( self, features ) :
        # the point regression predicts one value per enrollment
        return self.neural_network( features ).squeeze( -1 )


class StupydSurvivalModel( StupydModel ) :
    """
    Discrete-time hazard model for the whole degree-delay distribution.

    The model does not predict a single number: it predicts the hazard of
    graduating within each interval of a fixed time axis, that is the
    intervals measured in months starting from the observation date.

    Given the hazard `h_k` of each interval `k`, the probability of not
    graduating before an interval is the product of the complements:

        S(k) = prod( 1 - h_j, for j <= k )

    The expected and the median degree delay are then derived from `S`,
    so that this model can still be compared with the point regression
    using the very same metrics.

    The interval limits are a buffer, hence they are saved and restored
    together with the network: a checkpoint can always rebuild the same
    time axis, and `median_degree_delay` is consistent with it.

    The target statistics of the base class are not used to standardize the
    hazard, which is a probability, but they store the mean and the standard
    deviation of the durations observed in the training cohort, in months
    from the observation date.
    """

    def __init__( self,
                  feature_mean : Tensor,
                  feature_std : Tensor,
                  target_mean : Tensor,
                  target_std : Tensor,
                  interval_limits : Sequence[float] | Tensor,
                  hidden_sizes : Sequence[int] = ( 64, 32 ) ) :

        super().__init__( feature_mean, feature_std, target_mean, target_std )

        limits = torch.as_tensor( interval_limits, dtype = torch.float32 )

        # a valid axis has at least two limits, that is at least one interval
        if limits.dim() != 1 or limits.numel() < 2:
            raise ValueError( "interval_limits must be at least two limits." )

        if bool( ( limits[1:] <= limits[:-1] ).any() ):
            raise ValueError( "interval_limits must be strictly increasing." )

        self.register_buffer( "interval_limits", limits )

        self.interval_count = limits.numel() - 1

        # one extra input is the index of the interval, since the same
        # network estimates a different hazard for every interval
        layers = []
        inputs = feature_mean.numel() + 1

        for hidden in hidden_sizes:
            layers.append( nn.Linear( inputs, hidden ) )
            layers.append( nn.ReLU() )
            inputs = hidden

        layers.append( nn.Linear( inputs, 1 ) )

        self.neural_network = nn.Sequential( *layers )


    def do_predictions( self, features ) :
        """
        Returns the hazard of graduating within each interval,
        one column per interval of the time axis.
        """

        # the interval index is appended to the features, so that the same
        # network can estimate a different hazard for every interval
        batch_size = features.shape[0]
        intervals = torch.arange( self.interval_count,
                                  device = features.device,
                                  dtype = features.dtype )

        # (batch, intervals, 1) broadcast against the (batch, features)
        interval_features = intervals.view( 1, -1, 1 ).expand( batch_size, -1, 1 )
        repeated          = features.unsqueeze( 1 ).expand( -1, self.interval_count, -1 )

        inputs = torch.cat( [ repeated, interval_features ], dim = -1 )

        # the hazard is a probability: the sigmoid keeps it within (0, 1)
        return torch.sigmoid( self.neural_network( inputs ) ).squeeze( -1 )


    def forward( self, features ) :
        """
        Returns the hazard of graduating within each interval,
        that is a (batch, intervals) tensor.

        The output scale of the base class is not applied here, since the
        hazard is a probability and not a degree delay.
        """

        features = self.normalize_features( features )

        return self.do_predictions( features )


    def cumulative_incidence( self, features ) -> Tensor :
        """
        Returns the cumulative incidence `1 - S`, that is the probability
        of having graduated by the end of each interval.
        """

        hazard = self( features )

        # the complement of the hazard, interval by interval
        survival = torch.cumprod( 1.0 - hazard, dim = -1 )

        return 1.0 - survival


    def survival_function( self, features ) -> Tensor :
        """
        Returns the probability of not having graduated
        at the end of each interval.
        """

        return 1.0 - self.cumulative_incidence( features )


    def expected_degree_delay( self, features ) -> Tensor :
        """
        Returns the expected degree delay in months,
        computed from the area under the survival function.

        This is the expected value of the distribution, so it is
        systematically higher than the median for a right-skewed
        distribution of delays: the long tail pulls the mean up.
        """

        survival = self.survival_function( features )

        # the width of every interval, in months
        widths = ( self.interval_limits[ 1 : ] - self.interval_limits[ : -1 ] )

        # E[T] = sum over the intervals of width * S(begin of the interval)
        return ( survival * widths ).sum( dim = -1 )


    @torch.inference_mode()
    def predict_degree_delay( self, features ) -> Tensor :
        """
        Returns the median degree delay in months.

        The base class would return the output of `forward`, which here is
        the hazard and not a delay, hence the override: a caller that only
        needs a single number per enrollment has to get a delay.
        """

        self.eval()
        return self.median_degree_delay( features )


    def median_degree_delay( self, features ) -> Tensor :
        """
        Returns the median degree delay in months, that is the first
        interval whose cumulative incidence reaches 50%.

        When the follow-up is not long enough to cross the median, the last
        limit of the time axis is returned, so that the result is always
        defined.
        """

        incidence = self.cumulative_incidence( features )

        # True where the median has been crossed: the first such
        # interval is the answer, hence argmax returns its index
        crossed     = incidence >= 0.5
        has_crossed = crossed.any( dim = -1 )

        interval = crossed.to( torch.int32 ).argmax( dim = -1 )

        median = self.interval_limits[ interval ]

        # no median within the time axis: the follow-up was too short
        return torch.where( has_crossed, median, self.interval_limits[ -1 ] )