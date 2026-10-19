# StuPYD - Student Performance Yield Predictor

Welcome to `StuPYD`, my personal playground and final project for the [2026 edition of the school on "Python Deep Learning and Computer Vision"](https://inginf.unimore.it/post/112/) by University of Modena and Reggio Emilia.


## Aim

The project aims to evaluate the *degree delay* of students evaluating a set of features.

## Status

This project is absolutely **production un-ready**!

It is a playground to perform experiments and better understand the usage of PyTorch and Neural Networks concepts.

*It can be used as a starting point, but it is absolutely not a battle-tested application!*


## About the name

**The "StuPYD" name is clearly a joke** around the *intelligence* part of the project
that should predict the sstudents' degree delay.

It is not intended to offend anybody in any case, it's just a name!

## Where is the data?

The project comes with no data inside, the data is obtained on the fly during the project setup and cannot be shared since it is based on effective students' careers.
However, the project contains everything is needed to rebuild the set of features from a given database structure.

## Licence
BSD 3-Clause License, see [LICENSE](./LICENSE).


## Development

This application has been developed using GNU Emacs with [ElPY](https://github.com/jorgenschaefer/elpy).

Parts of the application and of the documentation have been implemented with a local AI agent.


# Glance at architecture

## Technologies involved

The core of the project is developed in Python using PyTorch (and hence, NumPy).

Dependencies are tracked via [requirements.txt](./requirements.txt), so that they can be installed with `pip` easily.

The database layout assumes a PostgreSQL backend, and it is managed via [Sqitch](https://sqtich.org).

There is also a playground part for embeddings, that exploits PostgreSQL extension [PgVector](https://github.com/pgvector/pgvector).


## Data Model

The system imports the data from a student database, and populates the PostgreSQL tables that are organized into the following schemas:

- `src` contains tables that mirrors the original source of data, removing personal information such as name, surname, tax payer number and so on (*quasi-anonymization*);
- `public` contains the tables that aggregate the data imported so that it can be used to build the features;
- `ml` contains the materialized views that are used by the Python part to perform the training and the prediction.

The data tables at glance are:

- `src.studenti`: sex and birth date.
- `src.matricole`: a student's enrollment/career record.
- `src.riga_libretto`: exams, grades, credits, and dates.
- `src.corsi_di_studio`: degree course and department metadata.
- `public.students`: one observed student, with the first and last known activity dates.
- `public.student_demographics`: valid gender and birth year for a student.
- `public.enrollments`: one academic career, including its start, completion, and status.
- `public.course_versions`: course duration (months) and expected credits (cfu) for a regulation year.
- `public.curriculum_versions` and `public.enrollment_path`: the curriculum assigned to an enrollment.
- `etl.enrollment_source_map`: connects a normalized enrollment to its original src.matricole row.
- `ml_model.student_features_12m` is the training-ready view. Each row represents an enrollment observed 12 months after it began.

The features used are:

- `age_at_enrollment`
- `audit_sex`
- `graded_exams_completed`
- `judgment_activities`
- `graded_credits_earned`
- `average_grade`
- `weighted_average_grade`
- `honors_count`
- `curriculum_version`


The main labels are:

- `graduation_date`
- `event_observed` (if the student has graduated)
- `months_to_graduation` the number of months from the observation

The *degree delay in months* can be defined as:
```
degree_delay = (graduation_date - begin_date) / 30.4375 - course_versions.months - 12
```


## Models

Two different models have been implemented:

- a linear regressor one, provides one single output (the number of delay months);
- a survival one, that provides a list of (*when, graduated*) and the probability of graduation.


## Class Hierarchy

```
torch.nn.Module
└── StupydNormalizer                 statistics, scaling, checkpoint loading
    └── StupydModel                  the type both families share
        ├── StupydRegressorModel     one number per enrollment: the MLP
        └── StupydSurvivalModel      hazard per interval: a whole distribution

torch.utils.data.Dataset
└── StupydDataset                    the tensors loaded from PostgreSQL
    └── StupydSurvivalDataset        adds (duration, event) per item

torch.utils.data.DataLoader
└── StupydDataLoader                 batches a dataset for the point regression

StupydCohort                         dataset + periods + batched scoring
├── StupydRegressorTrainer           trains and saves StupydRegressorModel
├── StupydRegressorPredictor         loads a checkpoint and scores a cohort
└── StupydSurvivalCohort             adds the time axis and the hazard metrics
    ├── StupydSurvivalTrainer        trains and saves StupydSurvivalModel
    └── StupydSurvivalPredictor      loads a hazard checkpoint and scores

StupydEmbedder                       standalone: no base class, no cohort
```


# Application Entry Point (how to run it)

The `main.py` file is the application entry point for all  the operations.
Its' `main()` function parses the command line arguments and decides to which part of the application deliver the requests.
The main arguments are:

- `train` or `predict` to start the training or prediction phases. These commands requires the specification of the model to use:
   - `--regressor` or `--survival`
- `embed` or `query` to activate the embeddings part.



## Application Configuration

The application exploits a JSON based configuration file, by default named `stupyd.json`, that contains parameters like the training, testing and prediction periods (expressed as years).

During the application lifecycle, the configuration is shared among classes and objects as a dictionary that is appropriately modified to reflect the runtime status. This means that the number of parameters passed to method and functions is lowered and often becomes only a configuration object.

As an example, a configuration file could be like the following:

```sh
% cat stupyd/stupyd.json
{
  "database": {
    "dsn_env": "STUPYD_DATABASE_URL"
  },
  "dataloader": {
    "batch_size": 256,
    "num_workers": 0
  },
  "periods": {
      "training" : {
          "begin" : 1900,
          "end"   : 2021
      },

      "validating" : {
          "begin" : 2022,
          "end"   : 2022
      },

      "testing" : {
          "begin" : 2023,
          "end"   : 2023
      },

      "scoring" : {
          "begin" : 2024,
          "end"   : 2100
      }
  },

  "genders" : [ "F", "M" ]
}
```


## PyTorch Model Persistence

Models are saved on storage into a directory based on the current day, the model used, so for example into `models/survival/yyyy_mm_dd/stupyd_model.pt`, so that they can be reloaded during the prediction phase.



## Configuring Sqitch

If you need to deploy all the database changes, configure `sqitch` accordinlgy with a `sqitch.conf` file containing the following:

```
[core]
	 engine = pg

[target "test"]
	uri = db:pg://stupydUsername:StupydPassword@your-test-host/stupyd

[target "prod"]
	uri = db:pg://stupydUsername:StupydPassword@your-test-host/stupyd

```

Adjust the URI to `test` and `prod` to your needs (and if required, add more targets).


## Importing the data - ETL

The data can be imported via a script (not shared) `bin/sqlite3_2_postgresql.sh` that copies all the data from the SQLite3 database to the `src` tables into PostgreSQL.

*Details of this script are secret!* - or better, it does not mind to share it here since this is used only for the original database translation.



# Running the application


Before launching the application, it is required to define the `dsn_environment` variable to point to a valid PostgreSQL database (hence, psycopg or libpq URI). This can be done using the `sqitch.conf` file that already owns the connection:

```sh
% export STUPYD_DATABASE_URL=$(sqitch --chdir sqitch config --get target.test.uri | sed 's|^db:pg://|postgresql://|')
```

Then, for every command invoked, the application will show its own configuration and the progress of the operation.
As an example:


```sh
% python stupyd/main.py train --survival

            Starting survival training mode (stupyd v0.1)

                epochs        : 100
                learning rate : 0.001
                interval width: 6.0 months
                follow-up up  : 360.0 months

                config        : stupyd/stupyd.json
                output to     : ./models



Training (survival): 100%|█████████████████████████████████████████████████| 100/100 [14:11<00:00,  8.51s/it]
time axis: 60 intervals of 6 months, up to 360 months
...
PyTorch survival model saved to [models/survival/2026_10_16/stupyd_model.pt]
all done
```

To perform a prediction, use the `predict` command specifying the saved model to use. As an example:

```sh
% python stupyd/main.py predict --survival --file models/survival/2026_10_16/stupyd_model.pt

        Starting survival prediction mode (stupyd v0.1)

            period        : scoring
            model         : models/survival/2026_10_16/stupyd_model.pt
            output to     : ./models


scoring_median_predicted_delay_months
gender age_range  count mean_predicted_delay_months
     F     20-24   7076                       13.30
     F     25-29   1741                        3.54
	 ...
     M       30+   1170                       10.24
     M       <20   7595                       48.97
...

Predictions saved to [models/survival/2026_10_16/stupyd_prediction.csv]
all done
```

As you can see, the output of the `prediction` command is a CSV file that you can analyze later on.


## The Embedding Stuff

The application contains an embedder that exploits PgVector to perform similarity queries for courses.
The embedding is created and driven by `sentence-transformers/all-MiniLM-L6-v2`.

The first step is to embed the course textual data:

```sh
% python stupyd/main.py embed

            Starting embedding mode (stupyd v0.1)

                dry run       : False
                model         : sentence-transformers/all-MiniLM-L6-v2
                output to     : courses.embedding



1252 courses embedded
all done
```

This step has to be done only once (or whenever the course information changes).

Once the embedding is in place, you can query for similarity with the `query` command, for instance:

```sh
% python stupyd/main.py query --query 'excel'

            Starting query mode (stupyd v0.1)

                question      : excel
                limit         : 10


loading sentence-transformers/all-MiniLM-L6-v2
Loading weights: 100%|███████████████████████████████████████████████████| 103/103 [00:00<00:00, 3740.25it/s]

courses most similar to 'excel' (out of 1252)
   0.285  xyz020-6     CP                                 Metodologie, Tecniche e Tool per Analisi di Big Data
   0.265  xyz0-362     LM                                 Informatica
   0.265  xyz0-262     LM                                 Informatica
   0.250  xyz0-209     L2                                 Informatica
   0.250  xyz0-315     L2                                 Informatica
   0.250  xyz0-215     L2                                 Informatica
   0.244  xyz016-FI    FI                                 LAB SCIENZE E TECNOL INFORMATICHE
```

(course codes have been changed!)
