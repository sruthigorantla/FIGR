#!/bin/bash
# runs all trainings for Engineering students data and saves result models into respective folders

# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

export LD_PRELOAD=libGLX_mesa.so.0 	#very dirty hack to workaround this octave bug: error: __osmesa_print__: 
					#Depth and stencil doesn't match, are you sure you are using OSMesa >= 9.0?

TIMESTAMP=`date +%Y-%m-%d_%H%M%S`
GIT_ROOT="$(git rev-parse --show-toplevel)"
GIT_ROOT=$GIT_ROOT"/FIGR_and_baselines"
PATH_TO_EXECUTABLE_DELTR=$GIT_ROOT/deltr-src
PATH_TO_EXECUTABLE_LISTNET=$GIT_ROOT/listnet-src
PATH_TO_CHILE_NOSEMI_DATASETS=$GIT_ROOT/data/EngineeringStudents/NoSemiPrivate
RESULT_DIR=$GIT_ROOT/results/EngineeringStudents/NoSemiPrivate

GAMMA_SMALL=50000
GAMMA_LARGE=100000

echo $GIT_ROOT

########################################################################################
# all gender experiments, no semi-private highschools
########################################################################################

for EXPERIMENT in gender highschool
do
	echo "###############################  Looping ... experiment $EXPERIMENT ############################### "
	echo ""
	
	for FOLD in fold_1 fold_2 fold_3 fold_4 fold_5
	do
		echo "$FOLD COLORBLIND..."
		cd $PATH_TO_EXECUTABLE_LISTNET
		./train.m $RESULT_DIR/$EXPERIMENT/$FOLD/COLORBLIND/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_${EXPERIMENT}_nosemi_fold${FOLD:5}_train.txt $RESULT_DIR/$EXPERIMENT/$FOLD/COLORBLIND/model.m

		echo "$FOLD PREPROCESSED..."
		cd $PATH_TO_EXECUTABLE_LISTNET
		./train.m $RESULT_DIR/$EXPERIMENT/$FOLD/PREPROCESSED/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_${EXPERIMENT}_nosemi_fold${FOLD:5}_train_RERANKED.txt $RESULT_DIR/$EXPERIMENT/$FOLD/PREPROCESSED/model.m

		./train.m $RESULT_DIR/$EXPERIMENT/$FOLD/PREPROCESSED_FIGR/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_${EXPERIMENT}_nosemi_fold${FOLD:5}_train_RERANKED_FIGR.txt $RESULT_DIR/$EXPERIMENT/$FOLD/PREPROCESSED_FIGR/model.m

		./train.m $RESULT_DIR/$EXPERIMENT/$FOLD/PREPROCESSED_FIGR_PLUS/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_${EXPERIMENT}_nosemi_fold${FOLD:5}_train_RERANKED_FIGR_PLUS.txt $RESULT_DIR/$EXPERIMENT/$FOLD/PREPROCESSED_FIGR_PLUS/model.m

		./train.m $RESULT_DIR/$EXPERIMENT/$FOLD/PREPROCESSED_FIGR_MINUS/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_${EXPERIMENT}_nosemi_fold${FOLD:5}_train_RERANKED_FIGR_MINUS.txt $RESULT_DIR/$EXPERIMENT/$FOLD/PREPROCESSED_FIGR_MINUS/model.m

		./train.m $RESULT_DIR/$EXPERIMENT/$FOLD/PREPROCESSED_PPlus/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_${EXPERIMENT}_nosemi_fold${FOLD:5}_train_RERANKED_PPlus.txt $RESULT_DIR/$EXPERIMENT/$FOLD/PREPROCESSED_PPlus/model.m

		./train.m $RESULT_DIR/$EXPERIMENT/$FOLD/PREPROCESSED_PMinus/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_${EXPERIMENT}_nosemi_fold${FOLD:5}_train_RERANKED_PMinus.txt $RESULT_DIR/$EXPERIMENT/$FOLD/PREPROCESSED_PMinus/model.m

		# echo "$FOLD GAMMA=0..."
		cd $PATH_TO_EXECUTABLE_DELTR
		./train.m $RESULT_DIR/$EXPERIMENT/$FOLD/GAMMA\=0/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_${EXPERIMENT}_nosemi_fold${FOLD:5}_train.txt $RESULT_DIR/$EXPERIMENT/$FOLD/GAMMA\=0/model.m 0

		cp $RESULT_DIR/$EXPERIMENT/$FOLD/GAMMA\=0/model.m $RESULT_DIR/$EXPERIMENT/$FOLD/FA-IR/model.m 

		cp $RESULT_DIR/$EXPERIMENT/$FOLD/GAMMA\=0/model.m $RESULT_DIR/$EXPERIMENT/$FOLD/FIGR/model.m 

		cp $RESULT_DIR/$EXPERIMENT/$FOLD/GAMMA\=0/model.m $RESULT_DIR/$EXPERIMENT/$FOLD/FIGR_PLUS/model.m

		cp $RESULT_DIR/$EXPERIMENT/$FOLD/GAMMA\=0/model.m $RESULT_DIR/$EXPERIMENT/$FOLD/FIGR_MINUS/model.m 

		# echo "$FOLD GAMMA=SMALL..."
		./train.m $RESULT_DIR/$EXPERIMENT/$FOLD/GAMMA\=SMALL/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_${EXPERIMENT}_nosemi_fold${FOLD:5}_train.txt $RESULT_DIR/$EXPERIMENT/$FOLD/GAMMA\=SMALL/model.m $GAMMA_SMALL

		# echo "$FOLD GAMMA=LARGE..."
		./train.m $RESULT_DIR/$EXPERIMENT/$FOLD/GAMMA\=LARGE/ $PATH_TO_CHILE_NOSEMI_DATASETS/$EXPERIMENT/$FOLD/chileDataL2R_${EXPERIMENT}_nosemi_fold${FOLD:5}_train.txt $RESULT_DIR/$EXPERIMENT/$FOLD/GAMMA\=LARGE/model.m $GAMMA_LARGE
	done
done

