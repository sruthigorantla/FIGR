import processingWithFair.rerank_with_fair as rerank
from processingWithFair.DatasetDescription import DatasetDescription


class Preprocessing():

    def __init__(self, args):
        self.dataset = args.preprocess[0]
        p = args.preprocess[1]

        if p == "figr_auto":
            self.p = 0.0
            self.p_classifier = "FIGR"
            self.k = args.k
            self.description_classifier = "RERANKED_FIGR"
        elif p == 'figr_minus':
            self.p = -0.1
            self.p_classifier = "FIGR_MINUS"
            self.k = args.k
            self.description_classifier = "RERANKED_FIGR_MINUS"
        elif p == 'figr_plus':
            self.p = 0.1
            self.p_classifier = "FIGR_PLUS"
            self.k = args.k
            self.description_classifier = "RERANKED_FIGR_PLUS"
        elif p == "p_minus":
            self.p_classifier = "PMINUS"
            self.p = -0.1
            self.description_classifier = "RERANKED_PMinus"
        elif p == "p_plus":
            self.p_classifier = "PPLUS"
            self.p = 0.1
            self.description_classifier = "RERANKED_PPlus"
        else:
            self.p = 0.0
            self.p_classifier = "PAUTO"
            self.description_classifier = "RERANKED"

    def preprocess_dataset(self):

        if self.dataset == "engineering-NoSemi":

            """
            Engineering Students Data - NoSemi - gender

            """
            print("Start reranking of Engineering Students Data - No Semi Private - gender")
            protected_attribute = 1
            score_attribute = 6
            protected_group = "hombre"
            header = ['query_id', 'hombre', 'psu_mat', 'psu_len', 'psu_cie', 'nem', 'score']
            judgment = "score"

            fold_count = 1
            for fold in ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]:

                print("Reranking for " + fold)
                origFile = "../data/EngineeringStudents/NoSemiPrivate/gender/" + fold + "/chileDataL2R_gender_nosemi_fold" + str(fold_count) + "_train.txt"
                resultFile = "../data/EngineeringStudents/NoSemiPrivate/gender/" + fold + "/chileDataL2R_gender_nosemi_fold" + str(fold_count) + "_train_" + self.description_classifier + ".txt"
                EngineeringData = DatasetDescription(resultFile,
                                                     origFile,
                                                     protected_attribute,
                                                     score_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)
            
                if "figr" in self.p_classifier.lower():
                    rerank.rerank_featurevectors_figr(EngineeringData, self.dataset, self.p, self.k, pre_process=True)
                else:
                    rerank.rerank_featurevectors(EngineeringData, self.dataset, self.p, pre_process=True)

                fold_count += 1

            """
            Engineering Students Data - NoSemi - highschool

            """
            print("Start reranking of Engineering Students Data - No Semi Private - highschool")
            protected_attribute = 1
            score_attribute = 6
            protected_group = "highschool_type"
            header = ['query_id', 'highschool_type', 'psu_mat', 'psu_len', 'psu_cie', 'nem', 'score']
            judgment = "score"

            fold_count = 1
            for fold in ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]:

                print("Reranking for " + fold)
                origFile = "../data/EngineeringStudents/NoSemiPrivate/highschool/" + fold + "/chileDataL2R_highschool_nosemi_fold" + str(fold_count) + "_train.txt"
                resultFile = "../data/EngineeringStudents/NoSemiPrivate/highschool/" + fold + "/chileDataL2R_highschool_nosemi_fold" + str(fold_count) + "_train_" + self.description_classifier + ".txt"
                EngineeringData = DatasetDescription(resultFile,
                                                     origFile,
                                                     protected_attribute,
                                                     score_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)

                if "figr" in self.p_classifier.lower():
                    rerank.rerank_featurevectors_figr(EngineeringData, self.dataset, self.p, self.k, pre_process=True)
                else:
                    rerank.rerank_featurevectors(EngineeringData, self.dataset, self.p, pre_process=True)

                fold_count += 1

