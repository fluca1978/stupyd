"""
Provide a training/evluating neural network approach for student degree delays.

This has been implemented as a didactic application for the final project
of the 2026 'Python Deep Learning' school.

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in ( None, '' ):
    sys.path.insert( 0, str( Path( __file__ ).resolve().parent.parent ) )

from stupyd import util
from stupyd.embedder import StupydEmbedder
from stupyd.model import StupydRegressorModel
from stupyd.predict import StupydRegressorPredictor
from stupyd.survival import StupydSurvivalPredictor, StupydSurvivalTrainer
from stupyd.train import StupydRegressorTrainer


STUPYD_VERSION = '0.1'

# the two model families take their own training arguments, so the command
# line offers them without a value and `check_training_options` substitutes
# these when the family that reads them leaves them out
DEFAULT_BATCH_SIZE     = 256
DEFAULT_INTERVAL_WIDTH = 6.0
DEFAULT_MAX_MONTHS     = 360.0

# the training options only one family reads: anything set on the other
# family is a mistake rather than something to ignore silently
REGRESSOR_OPTIONS = ( 'batch_size', )
SURVIVAL_OPTIONS  = ( 'interval_width', 'max_months' )


def find_model_file( filename : str, outdir : str ) -> Path :
    """
    Looks for the model file in the current directory first, then
    in the output directory, and returns its absolute path.

    Raises FileNotFoundError if the file cannot be found anywhere.
    """

    given      = Path( filename )
    candidates = [ given, Path( outdir ) / given, Path( outdir ) / given.name ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched = ', '.join( str( candidate ) for candidate in dict.fromkeys( candidates ) )
    raise FileNotFoundError( f"Model file {filename} not found, searched: {searched}" )


def check_training_options( args, parser ) -> None :
    """
    Validates the arguments of the `train` mode for the chosen family, then
    fills in the family-specific options that were left out.

    Those options default to `None` rather than to their value, so that
    "not given" can be told apart from "given the default". Without that
    `train --survival --batch-size 256` would look exactly like a plain
    `train --survival` and the mistake would pass unnoticed.

    Arguments:
    - args the parsed command line
    - parser the `train` subparser, used to report the errors
    """

    if args.epochs <= 0:
        parser.error( "epochs must be positive." )

    if args.learning_rate <= 0:
        parser.error( "the learning rate must be positive." )

    other = SURVIVAL_OPTIONS if args.family == 'regressor' else REGRESSOR_OPTIONS

    for name in other:
        if getattr( args, name ) is not None:
            parser.error( f"--{name.replace( '_', '-' )} is only valid with "
                          f"--{ 'survival' if args.family == 'regressor' else 'regressor' }." )

    if args.family == 'regressor':
        if args.batch_size is None:
            args.batch_size = DEFAULT_BATCH_SIZE
        if args.batch_size <= 0:
            parser.error( "the batch size must be positive." )
    else:
        if args.interval_width is None:
            args.interval_width = DEFAULT_INTERVAL_WIDTH
        if args.max_months is None:
            args.max_months = DEFAULT_MAX_MONTHS
        if args.interval_width <= 0 or args.max_months <= 0:
            parser.error( "the interval width and the bound of the follow-up "
                          "must be positive." )


def add_family_argument( subparser ) -> None :
    """
    Adds the flag that names which of the two model families a mode runs.

    The group is required: neither family is implied by `train` or
    `predict`, so the command line always says which one is meant.

    Arguments:
    - subparser the `train` or `predict` subparser
    """

    family = subparser.add_mutually_exclusive_group( required = True )

    family.add_argument( '--regressor', dest = 'family',
                         action = 'store_const', const = 'regressor',
                         help = 'the point regression: one degree delay per enrollment' )

    family.add_argument( '--survival', dest = 'family',
                         action = 'store_const', const = 'survival',
                         help = 'the discrete-time hazard: a whole distribution' )


def main() -> None:
    """
    Entry point for the StuPYD application.
    Parses the command line with the following arguments:

    mode can be 'train', 'predict', 'embed' or 'query'.

    --outdir is the directory where files can be stored (e.g., model outputs)
    --help to get online help
    --config to specify the path for the configuration file (JSON)

    Both 'train' and 'predict' need one of --regressor or --survival, since
    the two model families neither share a checkpoint nor read the same
    options.

    If the mode is specified as train, these arguments can be specified as
    well:
    --batch-size the size of the batch (regressor only)
    --learning-rate the learning rate for the training phase
    --epochs the number of epochs to train for

    --interval-width the width in months of a follow-up interval (survival only)
    --max-months the bound of the follow-up, so that a single implausible
    duration in the database cannot stretch the time axis (survival only)

    If the mode is specified as predict, a model saved by the training phase
    is loaded and only the prediction phase is performed:
    --filename the file the model has been saved to
    --period the configured period of enrollments to score

    The mode 'embed' fills the courses.embedding column with the vectors of
    the course names, and takes --dry-run to report without writing. The
    mode 'query' searches those vectors, and takes either --query, free text
    that is embedded on the fly, or --like, the code of a course already
    embedded, plus --limit for the number of rows to show.

    """


    # create a parser for the command line options
    parser = argparse.ArgumentParser( prog = f'stupyd v{STUPYD_VERSION}',
                                      description = __doc__ )

    parser.add_argument( '--config', default = 'stupyd.json' )
    parser.add_argument( '--outdir', default = './models' )

    # subparsers
    subparsers   = parser.add_subparsers( dest = 'mode', required = True )

    train_parser = subparsers.add_parser( 'train',
                                          help = 'fit a model on the training cohort' )
    add_family_argument( train_parser )
    train_parser.add_argument( "--learning-rate", type = float, default = 1e-3 )
    train_parser.add_argument( "--epochs", type = int, default = 100 )
    train_parser.add_argument( "--batch-size",
                               type = int,
                               default = None,
                               help = f'regressor only, the size of the batch '
                                      f'(default {DEFAULT_BATCH_SIZE})' )
    train_parser.add_argument( "--interval-width",
                               type = float,
                               default = None,
                               help = f'survival only, the width, in months, of a '
                                      f'follow-up interval (default {DEFAULT_INTERVAL_WIDTH})' )
    train_parser.add_argument( "--max-months",
                               type = float,
                               default = None,
                               help = f'survival only, the bound of the follow-up, since one '
                                      f'implausible duration would stretch the time axis '
                                      f'(default {DEFAULT_MAX_MONTHS})' )

    predict_parser = subparsers.add_parser( 'predict',
                                            help = 'score a cohort with a checkpoint of the training phase' )
    add_family_argument( predict_parser )
    predict_parser.add_argument( '--filename',
                                 required = True,
                                 help = 'The already saved model to load' )
    predict_parser.add_argument( '--period',
                                 default = 'scoring',
                                 help = 'The configured period to score (either "training", "testing", "validating", "scoring")' )


    # the embeddings of the course names are a workflow of their own: they
    # need the sentence-transformers model rather than a checkpoint of ours,
    # and they write into a column rather than scoring a cohort
    embed_parser = subparsers.add_parser(
        'embed',
        help = 'embed the course names into the courses.embedding column' )
    embed_parser.add_argument( "--dry-run",
                               action = "store_true",
                               help = 'report what would be embedded without writing' )

    query_parser = subparsers.add_parser(
        'query',
        help        = 'find the courses whose name is closest to a question' )
    query_group = query_parser.add_mutually_exclusive_group( required = True )
    query_group.add_argument( "--query",
                              help = 'free text, embedded on the fly' )
    query_group.add_argument( "--like",
                              help = 'the code of a course already embedded' )
    query_parser.add_argument( "--limit", type = int, default = 10 )




    # parse arguments
    args = parser.parse_args()


    # load the configuration, this is shared by all the modes
    stupyd_configuration = util.get_configuration( args.config )

    # place the seed and the device into the configuration
    stupyd_configuration[ 'seed' ]   = 1978
    stupyd_configuration[ 'device' ] = 'cpu'
    stupyd_configuration[ 'outdir' ] = args.outdir

    if args.mode in ( 'embed', 'query' ) :
        # the embeddings are read and written by their own class, which owns
        # the model and the two queries against the courses table

        if args.mode == 'query' and args.limit <= 0 :
            parser.error( "--limit must be positive." )

        embedder = StupydEmbedder( stupyd_configuration )

        if args.mode == 'embed' :

            print( f"""
            Starting embedding mode (stupyd v{STUPYD_VERSION})

                dry run       : {args.dry_run}
                model         : { embedder.MODEL_NAME }
                output to     : courses.embedding

            """ )

            embedded = embedder.run( dry_run = args.dry_run )
            print( f"{embedded} courses embedded" )

        else :

            print( f"""
            Starting query mode (stupyd v{STUPYD_VERSION})

                question      : { args.query if args.query else args.like }
                limit         : {args.limit}

            """ )

            rows, embedded = embedder.query( text = args.query,
                                             code = args.like,
                                             limit = args.limit )

            # the question is quoted and the course code is not, since the
            # first is a sentence and the second an identifier
            title = ( f"courses most similar to {args.query!r} " if args.query
                      else f"courses most similar to {args.like} " ) + f"(out of {embedded})"

            embedder.report( rows, title )

        print( 'all done' )
        exit( 0 )


    if args.mode == 'train':
        # training mode: the two families share the shape of the loop but
        # not the options, the trainer, or the folder of the checkpoint

        check_training_options( args, train_parser )

        if args.family == 'regressor':

            # print a summary
            print( f"""
            Starting regressor training mode (stupyd v{STUPYD_VERSION})

                epochs        : {args.epochs}
                batch size    : {args.batch_size}
                learning rate : {args.learning_rate}

                config        : { stupyd_configuration[ 'path' ] }
                output to     : {args.outdir}
                                { StupydRegressorModel.model_filename( args.outdir ) }


            """ )

            # I want to use the configuration as the only shared
            # piece of parameters, so copy all the args into the configuration
            stupyd_configuration[ 'epochs' ]        = int( args.epochs )
            stupyd_configuration[ 'batch_size' ]    = int( args.batch_size )
            stupyd_configuration[ 'learning_rate' ] = float( args.learning_rate )

            trainer_class = StupydRegressorTrainer

        else :

            print( f"""
            Starting survival training mode (stupyd v{STUPYD_VERSION})

                epochs        : {args.epochs}
                learning rate : {args.learning_rate}
                interval width: {args.interval_width} months
                follow-up up  : {args.max_months} months

                config        : { stupyd_configuration[ 'path' ] }
                output to     : {args.outdir}

            """ )

            stupyd_configuration[ 'epochs' ]          = int( args.epochs )
            stupyd_configuration[ 'learning_rate' ]   = float( args.learning_rate )
            stupyd_configuration[ 'months_interval' ] = int( args.interval_width )
            stupyd_configuration[ 'months_max' ]      = int( args.max_months )

            trainer_class = StupydSurvivalTrainer

        model_path = trainer_class( stupyd_configuration ).run()
        print( f"PyTorch {args.family} model saved to [{ model_path }]" )


    else :
        # prediction mode: only a model is loaded and the
        # enrollments of the requested period are scored

        predictor_class = ( StupydSurvivalPredictor if args.family == 'survival'
                            else StupydRegressorPredictor )

        # the model file is looked up in the current directory first and
        # then in the output directory
        stupyd_configuration[ 'period' ]     = args.period
        stupyd_configuration[ 'model_path' ] = find_model_file( args.filename,
                                                                 args.outdir )


        # print a summary
        print( f"""
        Starting {args.family} prediction mode (stupyd v{STUPYD_VERSION})

            period        : {args.period}
            model         : { stupyd_configuration[ 'model_path' ] }
            output to     : {args.outdir}

        """ )


        predictor_class( stupyd_configuration ).run()


    print( 'all done' )
    exit( 0 )

if __name__ == "__main__":
    main()
