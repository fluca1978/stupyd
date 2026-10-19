"""
Embeddings for the course names, and the search that reads them back.

pgvector stores and compares vectors, but it cannot compute them: the model
runs here, in Python, and the database only keeps the result. This module
owns that step as the class `StupydEmbedder`, which does two things:

- `run()` embeds the courses that do not carry a vector yet, which is what
  the `embed` mode of the application calls;
- `query()` searches those vectors, either by embedding a free-text question
  or by reusing the vector of a course already stored, which is what the
  `query` mode calls.

Both need the 05_pgvector migration, and the extension created by hand
first, since `stupyd` is not a superuser:

    CREATE EXTENSION vector;
"""

from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from stupyd.data import resolve_dsn


class StupydEmbedder :
    """
    Embeds the course names with a sentence-transformers model and stores
    the vectors in the `embedding` column of `courses`.

    The model runs here and not in the database: pgvector stores and compares
    vectors, it cannot compute them. So this class reads the rows to embed,
    calls the model, and writes the result back.

    Parameters:
    - configuration the configuration dictionary of the application
    """

    # The dimension has to match the one declared by the migration, and it is
    # a property of the model rather than a free choice: a mismatch is refused
    # by PostgreSQL when the vector is stored.
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    DIMENSIONS = 384

    # A placeholder carries no subject matter, so embedding it produces a
    # vector that mostly reflects the boilerplate of the degree.
    EXCLUDED = "TEST|SOSTEGNO|CORSO SINGOLO"

    # MiniLM similarities sit around 0.2 to 0.6 for related names and near 0
    # for unrelated ones, so 0.5 is where "related" starts rather than the
    # 0.9 that a normalised distance would suggest.
    RELATED = 0.5


    def __init__( self, configuration : dict ) -> None :

        if not isinstance( configuration, dict ) :
            raise TypeError( "configuration must be a dictionary." )

        self.configuration = configuration
        self.model         = None

        try:
            self.dsn = resolve_dsn( configuration )
        except ValueError as error:
            raise SystemExit(
                f"{error}\nFor example:\n"
                "  export STUPYD_DATABASE_URL=postgresql://stupyd@rachel/stupyd"
            )


    def connect( self ) :
        """Returns a connection to the backend database, rows as dictionaries."""

        return psycopg.connect( self.dsn, row_factory = dict_row )


    def load_model( self ) :
        """
        Loads the sentence-transformers model, once.

        The first call downloads about 90 MB of weights, so it needs the
        network once. The import is local to keep this module importable
        without the dependency installed, so that --help of the application
        works on a machine that only holds psycopg.
        """

        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError :
                raise SystemExit(
                    "sentence-transformers is not installed.\n"
                    "Install it with: pip install sentence-transformers\n"
                    "If it wants to replace the existing torch, add --no-deps."
                )

            print( f"loading {self.MODEL_NAME}" )
            self.model = SentenceTransformer( self.MODEL_NAME )

        return self.model


    @staticmethod
    def course_text( row : dict ) -> str :
        """
        Returns the text embedded for a course, that is `type` plus
        `description`.

        The degree level lives in `type` and not in the name, and the same
        subject is offered at several levels, so concatenating the two keeps
        the level in the vector instead of losing it.
        """

        return f"{row['type']}. {row['description']}"


    @staticmethod
    def as_literal( vector : list[float] ) -> str :
        """Renders a vector as the text of a literal, since psycopg has no vector type."""

        return "[" + ",".join( f"{value:.8f}" for value in vector ) + "]"


    def vectorize( self, texts : list[str] ) -> list[list[float]] :
        """
        Embeds the given texts, normalized so that cosine becomes the inner
        product, which is what the vector_cosine_ops index is built for.
        """

        vectors = self.load_model().encode(
            texts,
            batch_size             = 32,
            normalize_embeddings   = True,
            show_progress_bar      = len( texts ) > 1,
        )

        # a wrong dimension would fail on the first write rather than halfway,
        # so it is checked once against every vector
        for vector in vectors :
            if len( vector ) != self.DIMENSIONS :
                raise SystemExit(
                    f"{self.MODEL_NAME} returned {len( vector )} dimensions, "
                    f"but the column is vector({self.DIMENSIONS}). Either use "
                    f"another model or change the migration."
                )

        return [ vector.tolist() for vector in vectors ]


    def embeddable_courses( self, connection ) -> list[dict] :
        """Returns the courses whose description is worth embedding."""

        return connection.execute(
            """
            SELECT pk, code, description, type
            FROM courses
            WHERE description IS NOT NULL
              AND description !~* %s
              AND embedding IS NULL
            ORDER BY pk
            """,
            ( self.EXCLUDED, ),
        ).fetchall()


    def already_embedded( self, connection ) -> int :
        """Counts the courses that already carry a vector."""

        return connection.execute(
            "SELECT count(*) AS total FROM courses WHERE embedding IS NOT NULL"
        ).fetchone()[ "total" ]


    def skipped_courses( self, connection ) -> int :
        """Counts the courses the filter leaves out, for the report."""

        return connection.execute(
            """
            SELECT count(*) AS total
            FROM courses
            WHERE description IS NULL
               OR description ~* %s
            """,
            ( self.EXCLUDED, ),
        ).fetchone()[ "total" ]


    def extension_installed( self, connection ) -> bool :
        """Tells whether pgvector is available in this database."""

        return bool( connection.execute(
            "SELECT count(*) AS total FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()[ "total" ] )


    def store( self, connection, pk, vector : list[float] ) -> None :
        """Writes one vector, as the text of a literal since psycopg has no vector type."""

        connection.execute(
            "UPDATE courses SET embedding = %s::vector WHERE pk = %s",
            ( self.as_literal( vector ), pk ),
        )


    def run( self, dry_run : bool = False ) -> int :
        """
        Embeds the courses that do not carry a vector yet, and returns how
        many of them were written.

        With `dry_run` the report is printed and nothing is written.
        """

        with self.connect() as connection :

            if not self.extension_installed( connection ) :
                raise SystemExit(
                    "The pgvector extension is not installed in this database.\n"
                    "CREATE EXTENSION vector;  needs a superuser, so run it by hand."
                )

            embedded = self.already_embedded( connection )
            skipped  = self.skipped_courses( connection )
            rows     = self.embeddable_courses( connection )

            print( f"courses: {embedded} already embedded, "
                   f"{len( rows )} to embed, {skipped} without a usable name" )

            if not rows :
                print( "nothing to do" )
                return 0

            if dry_run :
                for row in rows[ :5 ] :
                    print( f"  would embed {row['code']}: {row['description']}" )
                print( f"dry run: {len( rows )} courses were not written" )
                return 0

            vectors = self.vectorize( [ self.course_text( row ) for row in rows ] )

            for row, vector in zip( rows, vectors ) :
                self.store( connection, row[ "pk" ], vector )

        print( f"embedded {len( rows )} courses" )
        print( "verify with:" )
        print( "  SELECT count(*) FROM courses WHERE embedding IS NOT NULL;" )

        return len( rows )


    def find_by_text( self, connection, text : str, limit : int ) -> list[dict] :
        """Searches by embedding the question, which is the interesting case."""

        # the model is called once, not once per parameter
        vector = self.as_literal( self.vectorize( [ text ] )[ 0 ] )

        return connection.execute(
            """
            SELECT code, description, type,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM courses
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            ( vector, vector, limit ),
        ).fetchall()


    def find_by_code( self, connection, code : str, limit : int ) -> list[dict] :
        """
        Searches by reusing the vector of a course already stored.

        The reference course is excluded in the WHERE clause rather than by
        ordering it last, and the `<=>` in the ORDER BY is what lets an index
        be used if one ever pays for it.
        """

        return connection.execute(
            """
            WITH reference AS (
                SELECT pk, embedding AS vector
                FROM courses
                WHERE trim(code) = %s AND embedding IS NOT NULL
                LIMIT 1
            )
            SELECT c.code, c.description, c.type,
                   1 - (c.embedding <=> reference.vector) AS similarity
            FROM courses c
            CROSS JOIN reference
            WHERE c.embedding IS NOT NULL
              AND c.pk <> reference.pk
            ORDER BY c.embedding <=> reference.vector
            LIMIT %s
            """,
            ( code, limit ),
        ).fetchall()


    def query( self,
               text  : str | None = None,
               code  : str | None = None,
               limit : int = 10 ) -> tuple[list[dict], int] :
        """
        Returns the courses closest to the given free text or to the given
        course code, the most similar first, together with how many courses
        carry a vector at all, which is what puts the result in context.

        With `text` the question is embedded on the fly, which is the
        interesting case. With `code` the stored vector of an existing course
        is reused, so that "courses like this one" costs no model call and is
        the cheapest way to check that the setup works.
        """

        if ( text is None ) == ( code is None ) :
            raise ValueError( "Give either text or code, exactly one." )

        if limit <= 0 :
            raise ValueError( "The number of courses to return must be positive." )

        with self.connect() as connection :

            embedded = self.already_embedded( connection )

            if not embedded :
                raise SystemExit(
                    "No course carries an embedding yet.\n"
                    "Run the embed mode first: python -m stupyd.main embed"
                )

            if text is not None :
                rows = self.find_by_text( connection, text, limit )
            else :
                rows = self.find_by_code( connection, code, limit )

        return rows, embedded


    def report( self, rows : list[dict], title : str ) -> None :
        """Prints the results, marking the ones that are close enough to matter."""

        if not rows :
            print( f"{title}: nothing to show" )
            return

        print( f"\n{title}" )

        for row in rows :
            mark = "*" if row[ "similarity" ] >= self.RELATED else " "

            print( f" {mark} {row['similarity']:.3f}  {row['code']:<10} "
                   f"{row['type']:<34} {row['description']}" )

        print( f"  (* means a similarity above {self.RELATED})" )
