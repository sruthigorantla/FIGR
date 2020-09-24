#!/bin/bash

# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

cd src/

# rerank Chile SAT dataset folds using FIGR/FA*IR for different values of p 
## FIGR
python3 main.py --postprocess engineering-NoSemi figr_auto --k 100 
python3 main.py --postprocess engineering-NoSemi figr_plus --k 100 
python3 main.py --postprocess engineering-NoSemi figr_minus --k 100 

## FA*IR
python3 main.py --postprocess engineering-NoSemi p_auto
python3 main.py --postprocess engineering-NoSemi p_plus
python3 main.py --postprocess engineering-NoSemi p_minus


cd ../
