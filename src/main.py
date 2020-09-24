'''
Created on May 11, 2018

@author: mzehlike
'''

import argparse, os

from data_preparation import *
from evaluation.evaluate import DELTR_Evaluator
from evaluation.evaluate_on_groundtruth import Postprocessing_Evaluator
from processingWithFair.preprocess import Preprocessing
from processingWithFair.postprocess import Postprocessing
from processingWithFair.postprocess_on_groundtruth import Postprocessing_on_groundtruth


def main():
    # parse command line options
    parser = argparse.ArgumentParser(prog='Disparate Exposure in Learning To Rank',
                                     epilog="=== === === end === === ===")

    parser.add_argument("--create",
                        nargs=1,
                        metavar='DATASET',
                        choices=['engineering-withoutSemiPrivate'],
                        help="creates datasets from raw data and writes them to disk")
    parser.add_argument("--evaluate",
                        nargs=1,
                        metavar='DATASET',
                        choices=['engineering-gender-withoutSemiPrivate',
                                 'engineering-highschool-withoutSemiPrivate'],
                        help="evaluates performance and fairness metrics for DATASET predictions")
    parser.add_argument("--evaluate_on_groundtruth",
                        nargs=1,
                        metavar='DATASET',
                        choices=['compas_sex',
                                 'compas_race',
                                 'german_age25',
                                 'german_age35'],
                        help="evaluates performance and fairness metrics for test DATASET on postprocessing")
    parser.add_argument("--preprocess",
                        nargs=2,
                        metavar=("DATASET", "P"),
                        choices=['engineering-NoSemi',
                                 'figr_auto',
                                 'figr_plus',
                                 'figr_minus',
                                 'p_minus',
                                 'p_auto',
                                 'p_plus'],
                        help="reranks all folds for the specified dataset with FA*IR for pre-processing (alpha = 0.1) or FIGR")
    parser.add_argument("--postprocess_on_groundtruth",
                        nargs=2,
                        metavar=("DATASET", "P"),
                        choices=['compas',
                                 'german',
                                 'figr_minus',
                                 'figr_auto',
                                 'figr_plus',
                                 'p_minus',
                                 'p_auto',
                                 'p_plus'],
                        help="reranks all folds for the specified dataset's test fold with FA*IR for pre-processing (alpha = 0.1) or FIGR")
    parser.add_argument("--postprocess",
                        nargs=2,
                        metavar=("DATASET", "P"),
                        choices=['engineering-NoSemi',
                                 'figr_minus',
                                 'figr_auto',
                                 'figr_plus',
                                 'p_minus',
                                 'p_auto',
                                 'p_plus'],
                        help="reranks all folds for the specified dataset with FA*IR for post-processing (alpha = 0.1) or FIGR")
    parser.add_argument('--k', type=int,  default=10,
                        help='interval length to audit fairness')
    args = parser.parse_args()

    ################### argparse create #########################
    if args.create == ['engineering-withoutSemiPrivate']:
        EngineeringStudPrep.prepareNoSemi()

    #################### argparse evaluate ################################
    elif args.evaluate == ['engineering-gender-withoutSemiPrivate']:
        resultDir = '../results/EngineeringStudents/NoSemiPrivate/gender/results/'
        binSize = 100
        protAttr = 3
        evaluator = DELTR_Evaluator('engineering-gender-withoutSemiPrivate',
                                    resultDir,
                                    binSize,
                                    protAttr)
        evaluator.evaluate()

    elif args.evaluate == ['engineering-highschool-withoutSemiPrivate']:
        resultDir = '../results/EngineeringStudents/NoSemiPrivate/highschool/results/'
        binSize = 100
        protAttr = 3
        evaluator = DELTR_Evaluator('engineering-highschool-withoutSemiPrivate',
                                    resultDir,
                                    binSize,
                                    protAttr)
        evaluator.evaluate()

    #################### argparse evaluate on groundtruth ################################

    elif args.evaluate_on_groundtruth == ['german_age25']:
        resultDir = '../results/GermanCredit/age25/'
        binSize = 100
        protAttr = 3
        evaluator = Postprocessing_Evaluator('german_age25',
                                    resultDir,
                                    binSize,
                                    protAttr)
        evaluator.evaluate_on_groundtruth()

    elif args.evaluate_on_groundtruth == ['german_age35']:
        resultDir = '../results/GermanCredit/age35/'
        binSize = 100
        protAttr = 3
        evaluator = Postprocessing_Evaluator('german_age35',
                                    resultDir,
                                    binSize,
                                    protAttr)
        evaluator.evaluate_on_groundtruth()
    
    elif args.evaluate_on_groundtruth == ['compas_race']:
        resultDir = '../results/COMPAS/race/'
        binSize = 100
        protAttr = 3
        evaluator = Postprocessing_Evaluator('compas_race',
                                    resultDir,
                                    binSize,
                                    protAttr)
        evaluator.evaluate_on_groundtruth()
    
    elif args.evaluate_on_groundtruth == ['compas_sex']:
        resultDir = '../results/COMPAS/gender/'
        binSize = 100
        protAttr = 3
        evaluator = Postprocessing_Evaluator('compas_sex',
                                    resultDir,
                                    binSize,
                                    protAttr)
        evaluator.evaluate_on_groundtruth()

    #################### argparse pre-process ################################
    elif (args.preprocess != None):
        if ( 'engineering-NoSemi' in args.preprocess) and \
           ('figr_auto' in args.preprocess or 'figr_plus' in args.preprocess or 'figr_minus' in args.preprocess or 'p_minus' in args.preprocess or 'p_auto' in args.preprocess or 'p_plus' in args.preprocess) and \
           len(args.preprocess) == 2:
            preprocessor = Preprocessing(args)
            preprocessor.preprocess_dataset()

    #################### argparse post-process-on-groundtruth ################################
    elif (args.postprocess_on_groundtruth != None):
        if ('compas' in args.postprocess_on_groundtruth  or 'german' in args.postprocess_on_groundtruth) and \
           ('figr_minus' in args.postprocess_on_groundtruth or 'figr_auto' in args.postprocess_on_groundtruth or 'figr_plus' in args.postprocess_on_groundtruth or 'p_minus' in args.postprocess_on_groundtruth or 'p_auto' in args.postprocess_on_groundtruth or 'p_plus' in args.postprocess_on_groundtruth) and \
           len(args.postprocess_on_groundtruth) == 2:
            postprocessor_on_groundtruth = Postprocessing_on_groundtruth(args)
            postprocessor_on_groundtruth.postprocess_on_groundtruth_result()


    #################### argparse post-process ################################
    elif (args.postprocess != None):
        if ('engineering-NoSemi' in args.postprocess) and \
           ('figr_auto' in args.postprocess or 'figr_plus' in args.postprocess or 'figr_minus' in args.postprocess or 'p_minus' in args.postprocess or 'p_auto' in args.postprocess or 'p_plus' in args.postprocess) and \
           len(args.postprocess) == 2:
            postprocessor = Postprocessing(args)
            postprocessor.postprocess_result()
    else:
        parser.error("choose one command line option")


if __name__ == '__main__':
    main()
