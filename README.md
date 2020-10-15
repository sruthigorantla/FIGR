# Ranking for Individual and Group Fairness Simultaneously (FIGR).
This repository consists of code and datasets for [our paper](https://arxiv.org/abs/2010.06986). We extend the code of [DELTR](https://github.com/MilkaLichtblau/DELTR-Experiments) and [FA\*IR](https://github.com/fair-search) to compare our work with these algorithms as baselines.


## Requirements
``octave-general`` and ``octave-parallel``

## Fair LTR on the ChileSAT dataset

### Preprocessing training data
The bashscript in the root folder pre-processes the ChileSAT data with FIGR and FA\*IR

``preprocess.sh`` pre-processes the training data.

### Model Training

For ChileSAT dataset a bash-script named ``trainEngineering.sh`` is available that trains models for all experimental settings and saves them into ``results/EngineeringStudents/NoSemiPrivate/PROTECTED-ATTRIBUTE/FOLD/EXPERIMENTAL-SETTING/model.m``. 

Training parameters like learning rate or number of iterations can be changed in [``listnet-src/globals.m``](https://github.com/sruthigorantla/FIGR/tree/master/listnet-src/globals.m) and [``deltr-src/globals.m``](https://github.com/sruthigorantla/FIGR/tree/master/deltr-src/globals.m). 

The parameter ``\gamma`` for DELTR is set in a command line argument.
The parameter ``k`` for FIGR is also set in a command line arguments.

### Predictions

For the ChileSAT a bash-script named ``predictEngineering.sh`` is available, that uses the previously trained models and testdata to predict rankings. Predictions are stored in the same folder as model.m.

This script also copies the prediction files from the DELTR experiment with Gamma=0 (color-aware LTR) into folders ``EngineeringStudents/NoSemiPrivate/FA-IR/P-VALUE``, ``EngineeringStudents/NoSemiPrivate/FIGR``, ``EngineeringStudents/NoSemiPrivate/FIGR_PLUS``, ``EngineeringStudents/NoSemiPrivate/FIGR_MINUS`` and , which are then needed by the post-processing script.

### Post-Processing 

The root directory contains bash-script for post-processing LTR predictions on ChileSAT test data with FIGR and also FA\*IR (Baseline)

``postprocess.sh`` 


## Directly applying fair post-processing (re-ranking) methods on the German credit risk and the COMPAS recidivism datasets

### post-processing the true ranking in the dataset.

This bash-script re-ranks the true rankig using FIGR and also FA*IR (Baseline)

``postprocessTrueRanking.sh``

## Result Evaluation

The bash-script evaluates results on all three datasets.

``evaluateResults.sh``
