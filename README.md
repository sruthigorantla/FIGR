# FIGR algorithm and comparision with baselines.
This repository consists of code and datasets used in our paper. We extend the code of DELTR and FA\*IR to compare our work with these algorithms as baselines.

## DELTR-Experiments
Code and Datasets for [DELTR](https://arxiv.org/abs/1805.08716) on disparate exposure in learning to rank is extended in this work. For more details, visit [Fair Search](https://github.com/fair-search) repository.

## Requirements
requires GNU Octave and the packages ``octave-general`` and ``octave-parallel``

This code has been tested with Octave version 5.1.0 and Ubuntu 18.04. 

One known issue is a bug involving osmesa. The scripts already contain a workaround for this bug.

## LTR models on the ChileSAT dataset

### Preprocessing training data
The root directory contains bash-script for pre-processing ChileSAT data with FIGR and FA\*IR

**preprocess.sh:** prepares datasets for pre-processing experiments

All datasets are available in the data folder.

### Model Training

For ChileSAT dataset a bash-script named ``trainEngineering.sh`` is available that trains models for all experimental settings and saves them into ``results/EngineeringStudents/NoSemiPrivate/PROTECTED-ATTRIBUTE/FOLD/EXPERIMENTAL-SETTING/model.m``. 

To check sanity of the model, the code also creates a plot of the cost and gradient development, which should both convert. The figures are stored in the same folder as the model.

Training parameters like learning rate or number of iterations can be changed in [``listnet-src/globals.m``](https://github.com/MilkaLichtblau/DELTR-Experiments/blob/master/listnet-src/globals.m) and [``deltr-src/globals.m``](https://github.com/MilkaLichtblau/DELTR-Experiments/blob/master/deltr-src/globals.m). The only exception is the Gamma parameter for DELTR which is a command line argument. 

### Predictions

For the ChileSAT a bash-script named ``predictEngineering.sh`` is available, that uses the previously trained models and testdata to predict rankings. Predictions are stored in the same folder as model.m.

This script also copies the prediction files from the DELTR experiment with Gamma=0 into folders ``EngineeringStudents/NoSemiPrivate/FA-IR/P-VALUE``, ``EngineeringStudents/NoSemiPrivate/FIGR``, ``EngineeringStudents/NoSemiPrivate/FIGR_PLUS``, ``EngineeringStudents/NoSemiPrivate/FIGR_MINUS`` and , which are then needed by the post-processing script.

### Post-Processing 

The root directory contains bash-script for post-processing LTR predictions on ChileSAT test data with FIGR and also FA\*IR (Baseline)

**postprocess.sh** 


## Directly applying fair post-processing (re-ranking) methods on the German credit risk and the COMPAS recidivism datasets.

### post-processing the true ranking in the dataset.

This bash-script re-ranks the true rankig using FIGR and also FA*IR (Baseline)

**postprocessTrueRanking.sh**

## Result Evaluation

The bash-script evaluates results on all three datasets.

**evaluateResults.sh**
