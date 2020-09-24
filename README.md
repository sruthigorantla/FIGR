# FIGR algorithm and comparision with baselines.
This repository consists of code and datasets used in our paper. We extend the code of DELTR and FA\*IR to compare our work with these algorithms as baselines.


## Requirements
``octave-general`` and ``octave-parallel``

## <font color="blue">LTR models on the ChileSAT dataset</font>

### Preprocessing training data
The bashscript in the root folder pre-processes the ChileSAT data with FIGR and FA\*IR

**preprocess.sh:** pre-processes the training data.

### Model Training

For ChileSAT dataset a bash-script named ``trainEngineering.sh`` is available that trains models for all experimental settings and saves them into ``results/EngineeringStudents/NoSemiPrivate/PROTECTED-ATTRIBUTE/FOLD/EXPERIMENTAL-SETTING/model.m``. 

Training parameters like learning rate or number of iterations can be changed in [``listnet-src/globals.m``](https://github.com/sruthigorantla/FIGR/listnet-src/globals.m) and [``deltr-src/globals.m``](https://github.com/sruthigorantla/FIGR/deltr-src/globals.m). 

The parameter $\gamma$ for DELTR is set in a command line argument.
The parameters $k$ and $\alpha$ for FIGR are also set in command line arguments.

### Predictions

For the ChileSAT a bash-script named ``predictEngineering.sh`` is available, that uses the previously trained models and testdata to predict rankings. Predictions are stored in the same folder as model.m.

This script also copies the prediction files from the DELTR experiment with Gamma=0 (color-aware LTR) into folders ``EngineeringStudents/NoSemiPrivate/FA-IR/P-VALUE``, ``EngineeringStudents/NoSemiPrivate/FIGR``, ``EngineeringStudents/NoSemiPrivate/FIGR_PLUS``, ``EngineeringStudents/NoSemiPrivate/FIGR_MINUS`` and , which are then needed by the post-processing script.

### Post-Processing 

The root directory contains bash-script for post-processing LTR predictions on ChileSAT test data with FIGR and also FA\*IR (Baseline)

**postprocess.sh** 


## <font color="blue">Directly applying fair post-processing (re-ranking) methods on the German credit risk and the COMPAS recidivism datasets.</font>

### post-processing the true ranking in the dataset.

This bash-script re-ranks the true rankig using FIGR and also FA*IR (Baseline)

**postprocessTrueRanking.sh**

## Result Evaluation

The bash-script evaluates results on all three datasets.

**evaluateResults.sh**
