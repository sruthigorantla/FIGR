import processingWithFair.rerank_with_fair as rerank
import pandas as pd
from processingWithFair.DatasetDescription import DatasetDescription


class Postprocessing():

    def __init__(self, args):
        self.dataset = args.postprocess[0]
        p = args.postprocess[1]

        if p == "figr_auto":
            self.p = 0.0
            self.p_classifier = "FIGR"
            self.k = args.k
        elif p == 'figr_minus':
            self.p = -0.1
            self.p_classifier = "FIGR_MINUS"
            self.k = args.k
        elif p == 'figr_plus':
            self.p = 0.1
            self.p_classifier = "FIGR_PLUS"
            self.k = args.k
        elif p == "p_minus":
            self.p = -0.1
            self.p_classifier = "P-Minus"
        elif p == "p_plus":
            self.p = 0.1
            self.p_classifier = "P-Plus"
        else:
            self.p = 0.0
            self.p_classifier = "P-Star"

    def postprocess_result (self):

        if self.dataset == "engineering-NoSemi":

            """
            Engineering Students Data - NoSemi - gender

            """
            print("Start reranking of Engineering Students Data - No Semi Private - gender")
            header = ["query_id", "doc_id", "score", "prot_attr"]
            protected_attribute = 3
            protected_group = "prot_attr"
            score_attribute = 2
            judgment = "score"
                
            fold_count = 1
            for fold in ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]:
                print("post-processing for " + fold + " with " + self.p_classifier)

                if "figr" in self.p_classifier.lower():
                    origPredictions = "../results/EngineeringStudents/NoSemiPrivate/gender/" + fold + "/"+self.p_classifier+"/predictions_ORIG.pred"
                    rerankedPredictions = "../results/EngineeringStudents/NoSemiPrivate/gender/" + fold + "/"+self.p_classifier+"/predictions.pred"
                else:
                    origPredictions = "../results/EngineeringStudents/NoSemiPrivate/gender/" + fold + "/FA-IR/" + self.p_classifier + "/predictions_ORIG.pred"
                    rerankedPredictions = "../results/EngineeringStudents/NoSemiPrivate/gender/" + fold + "/FA-IR/" + self.p_classifier + "/predictions.pred"
                EngineeringData = DatasetDescription(rerankedPredictions,
                                                     origPredictions,
                                                     protected_attribute,
                                                     score_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)

                if "figr" in self.p_classifier.lower():
                    rerank.rerank_featurevectors_figr(EngineeringData, self.dataset, self.p, self.k, post_process=True)
                else:
                    rerank.rerank_featurevectors(EngineeringData, self.dataset, self.p, post_process=True)

                fold_count += 1

            # """
            # Engineering Students Data - NoSemi - highschool

            # """
            print("Start reranking of Engineering Students Data - No Semi Private - highschool")
            header = ["query_id", "doc_id", "score", "prot_attr"]
            protected_attribute = 3
            protected_group = "prot_attr"
            score_attribute = 2
            judgment = "score"
            fold_count = 1
            for fold in ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]:
                print("post-processing for " + fold + " with " + self.p_classifier)
                if "figr" in self.p_classifier.lower():
                    origPredictions = "../results/EngineeringStudents/NoSemiPrivate/highschool/" + fold + "/"+self.p_classifier+"/predictions_ORIG.pred"
                    rerankedPredictions = "../results/EngineeringStudents/NoSemiPrivate/highschool/" + fold + "/"+self.p_classifier+"/predictions.pred"
                else:
                    origPredictions = "../results/EngineeringStudents/NoSemiPrivate/highschool/" + fold + "/FA-IR/" + self.p_classifier + "/predictions_ORIG.pred"
                    rerankedPredictions = "../results/EngineeringStudents/NoSemiPrivate/highschool/" + fold + "/FA-IR/" + self.p_classifier + "/predictions.pred"
                EngineeringData = DatasetDescription(rerankedPredictions,
                                                     origPredictions,
                                                     protected_attribute,
                                                     score_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)

                if "figr" in self.p_classifier.lower():
                    rerank.rerank_featurevectors_figr(EngineeringData, self.dataset, self.p, self.k, post_process=True)
                else:
                    rerank.rerank_featurevectors(EngineeringData, self.dataset, self.p, post_process=True)


                fold_count += 1

