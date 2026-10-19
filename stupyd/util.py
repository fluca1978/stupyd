# util.py
#
# Utility functions
#

import os
import json
from pathlib import Path


def get_configuration( filename = 'stupyd.json' ) :
    """
    Loads the configuration file for this application.

    The configuration file is a JSON file, so this methods
    returns the dictionary of the JSON.

    If the file is not found, an exception is thrown.

    The dictionary is augments with a special key 'configuration_path'
    that represents the configuration absolute file path.
    """



    configuration_file = Path( __file__ ).with_name( filename )

    if not configuration_file.exists() or not configuration_file.is_file() :
        raise Exception( f'Configuration file {filename} not found!' )


    with configuration_file.open() as config:
        config = json.load( config )

    config[ 'path' ] = configuration_file.absolute()
    return config



def get_dsn() :
    """
    Extract the database DSN using the STUPYD_DATABASE_URL
    environment variable.

    If not found or empty, an exception is thrown.
    """

    dsn = os.environ.get( "STUPYD_DATABASE_URL" )
    if dsn is None or len( dsn ) <= 0:
        raise TypeError( 'Environment variable STUPYD_DATABASE_URL not set!' )

    return dsn

