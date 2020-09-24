import processingWithFair.rerank_with_fair as rerank
from processingWithFair.DatasetDescription import DatasetDescription


class Postprocessing_on_groundtruth():

    def __init__(self, args):
        self.dataset = args.postprocess_on_groundtruth[0]
        p = args.postprocess_on_groundtruth[1]
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

    def postprocess_on_groundtruth_result(self):

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
                origFile = "../data/EngineeringStudents/NoSemiPrivate/gender/" + fold + "/chileDataL2R_gender_nosemi_fold" + str(fold_count) + "_test.txt"
                resultFile = "../data/EngineeringStudents/NoSemiPrivate/gender/" + fold + "/chileDataL2R_gender_nosemi_fold" + str(fold_count) + "_test_" + self.description_classifier + ".txt"
                EngineeringData = DatasetDescription(resultFile,
                                                     origFile,
                                                     protected_attribute,
                                                     score_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)
                if "figr" in self.p_classifier.lower():
                    rerank.rerank_featurevectors_figr(EngineeringData, self.dataset, self.alpha, self.k)
                else:
                    rerank.rerank_featurevectors(EngineeringData, self.dataset, self.p)


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
                origFile = "../data/EngineeringStudents/NoSemiPrivate/highschool/" + fold + "/chileDataL2R_highschool_nosemi_fold" + str(fold_count) + "_test.txt"
                resultFile = "../data/EngineeringStudents/NoSemiPrivate/highschool/" + fold + "/chileDataL2R_highschool_nosemi_fold" + str(fold_count) + "_test_" + self.description_classifier + ".txt"
                EngineeringData = DatasetDescription(resultFile,
                                                     origFile,
                                                     protected_attribute,
                                                     score_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)

                if "figr" in self.p_classifier.lower():
                    rerank.rerank_featurevectors_figr(EngineeringData, self.dataset, self.alpha, self.k)
                else:
                    rerank.rerank_featurevectors(EngineeringData, self.dataset, self.p)


                fold_count += 1

        elif self.dataset == "german":

            """
            German Credit dataset - age 25

            """
            print("Start reranking of German Credit - Age 25")
            protected_attribute = 3
            score_attribute = 2
            protected_group = "age25"
            header = ['DurationMonth', 'CreditAmount', 'score', 'age25']
            judgment = "score"

            origFile = "../data/GermanCredit/GermanCredit_age25.csv"
            resultFile = "../data/GermanCredit/GermanCredit_age25_" + self.description_classifier + ".txt"
            GermanCreditData = DatasetDescription(resultFile,
                                                     origFile,
                                                     protected_attribute,
                                                     score_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)
            if "figr" in self.p_classifier.lower():
                rerank.rerank_featurevectors_figr(GermanCreditData, self.dataset, self.p, self.k)
            else:
                rerank.rerank_featurevectors(GermanCreditData, self.dataset, self.p)


            """
            German Credit dataset - age 35

            """
            print("Start reranking of German Credit - Age 35")
            protected_attribute = 3
            score_attribute = 2
            protected_group = "age35"
            header = ['DurationMonth', 'CreditAmount', 'score', 'age35']
            judgment = "score"

            origFile = "../data/GermanCredit/GermanCredit_age35.csv"
            resultFile = "../data/GermanCredit/GermanCredit_age35_" + self.description_classifier + ".txt"
            GermanCreditData = DatasetDescription(resultFile,
                                                     origFile,
                                                     protected_attribute,
                                                     score_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)
            if "figr" in self.p_classifier.lower():
                rerank.rerank_featurevectors_figr(GermanCreditData, self.dataset, self.p, self.k)
            else:
                rerank.rerank_featurevectors(GermanCreditData, self.dataset, self.p)


            """
            German Credit dataset - gender

            """
            print("Start reranking of German Credit - gender")
            protected_attribute = 3
            score_attribute = 2
            protected_group = "sex"
            header = ['DurationMonth', 'CreditAmount', 'score', 'sex']
            judgment = "score"

            origFile = "../data/GermanCredit/GermanCredit_sex.csv"
            resultFile = "../data/GermanCredit/GermanCredit_sex_" + self.description_classifier + ".txt"
            GermanCreditData = DatasetDescription(resultFile,
                                                     origFile,
                                                     protected_attribute,
                                                     score_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)
            if "figr" in self.p_classifier.lower():
                rerank.rerank_featurevectors_figr(GermanCreditData, self.dataset, self.p, self.k)
            else:
                rerank.rerank_featurevectors(GermanCreditData, self.dataset, self.p)

        elif self.dataset == 'compas':

            """
            COMPAS propublica dataset - race

            """
            print("Start reranking of COMPAS propublica - Race")
            protected_attribute = 3
            score_attribute = 2
            protected_group = "race"
            header = ['priors_count','Violence_rawscore','Recidivism_rawscore','race']
            judgment = "Recidivism_rawscore"

            origFile = "../data/COMPAS/ProPublica_race.csv"
            resultFile = "../data/COMPAS/ProPublica_race_" + self.description_classifier + ".txt"
            CompasData = DatasetDescription(resultFile,
                                                     origFile,
                                                     protected_attribute,
                                                     score_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)
            if "figr" in self.p_classifier.lower():
                rerank.rerank_featurevectors_figr(CompasData, self.dataset, self.p, self.k)
            else:
                rerank.rerank_featurevectors(CompasData, self.dataset, self.p)


            """
            COMPAS propublica dataset - gender

            """
            print("Start reranking of COMPAS propublica - gender")
            protected_attribute = 3
            score_attribute = 2
            protected_group = "sex"
            header = ['priors_count','Violence_rawscore','Recidivism_rawscore','sex']
            judgment = "Recidivism_rawscore"

            origFile = "../data/COMPAS/ProPublica_sex.csv"
            resultFile = "../data/COMPAS/ProPublica_sex_" + self.description_classifier + ".txt"
            CompasData = DatasetDescription(resultFile,
                                                     origFile,
                                                     protected_attribute,
                                                     score_attribute,
                                                     protected_group,
                                                     header,
                                                     judgment)
            if "figr" in self.p_classifier.lower():
                rerank.rerank_featurevectors_figr(CompasData, self.dataset, self.p, self.k)
            else:
                rerank.rerank_featurevectors(CompasData, self.dataset, self.p)

