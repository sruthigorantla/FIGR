#!/bin/bash
# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

cd src/

# rerank German Credit dataset folds using FIGR/FA*IR for different values of p
python3 main.py --postprocess_on_groundtruth german figr_auto --k 100 
python3 main.py --postprocess_on_groundtruth german figr_plus --k 100 
python3 main.py --postprocess_on_groundtruth german figr_minus --k 100 
python3 main.py --postprocess_on_groundtruth german p_auto
python3 main.py --postprocess_on_groundtruth german p_plus
python3 main.py --postprocess_on_groundtruth german p_minus

# rerank COMPAS ProPublica dataset folds using FIGR/FA*IR for different values of p
python3 main.py --postprocess_on_groundtruth compas figr_auto --k 100 
python3 main.py --postprocess_on_groundtruth compas figr_plus --k 100 
python3 main.py --postprocess_on_groundtruth compas figr_minus --k 100 
python3 main.py --postprocess_on_groundtruth compas p_auto
python3 main.py --postprocess_on_groundtruth compas p_plus
python3 main.py --postprocess_on_groundtruth compas p_minus

cd ../
