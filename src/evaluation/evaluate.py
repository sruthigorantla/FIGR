'''
Created on May 11, 2018

@author: mzehlike
'''



from adjustText import adjust_text
import pandas as pd
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats as stats
import math
import os
from fileinput import filename


class DELTR_Evaluator():
    '''
    
    :field __trainingDir:             directory in which training scores and predictions are stored
    :field __resultDir:               path where to store the result files
    :field __protectedAttribute:      int, defines what the protected attribute in the dataset was
    :field __dataset:                 string, specifies which dataset to evaluate
    :field __chunkSize:               int, defines how many items belong to one chunk in the bar plots
    :field __columnNames:             predictions are read into dataframe with defined column names
    :field __predictions:             np-array with predicted scores
    :field __original:                np-array with training scores
    :field __experimentNamesAndFiles: collects experiment names and result filenames to use for scatter plot
    '''

    def __init__(self, dataset, resultDir, binSize, protAttr):
        self.__trainingDir = '../results/'
        self.__resultDir = resultDir
        if not os.path.exists(resultDir):
            os.makedirs(resultDir)
        self.__protectedAttribute = protAttr
        self.__dataset = dataset
        self.__prot_attr_name = self.__dataset.split('-')[1]
        self.__chunkSize = binSize
        self.__columnNames = ["query_id", "doc_id", "prediction", "prot_attr"]
        
    def evaluate(self):
        if self.__dataset == 'engineering-gender-withoutSemiPrivate' or self.__dataset == 'engineering-highschool-withoutSemiPrivate':
            mpl.rcParams.update({'font.size': 10, 'lines.linewidth': 5, 'lines.markersize': 40, 'font.family':'Times New Roman'})
            # avoid type 3 (i.e. bitmap) fonts in figures
            mpl.rcParams['ps.useafm'] = True
            mpl.rcParams['pdf.use14corefonts'] = True
            mpl.rcParams['text.usetex'] = True

            fig_group = plt.figure(figsize=(15, 10))
            group_fair_ax = plt.axes()
            
            fig_ind = plt.figure(figsize=(15, 10))
            ind_fair_ax = plt.axes()
            
            self.__trainingDir = self.__trainingDir + 'EngineeringStudents/NoSemiPrivate/'+self.__prot_attr_name+'/'
            NUM_FOLDS = 5
            N_BINS_TO_SHOW = 1000
            EXPERIMENT_NAMES = ['true','Gamma=colorblind', 'Gamma=0', 'Gamma=small', 'Gamma=large', 'FAIR_PStar', 'FAIR_PMinus', 'FAIR_PPlus', 'FIGR', 'FIGR_PLUS', 'FIGR_MINUS',
                               'Gamma=PREPROCESSED', 'Gamma=PREPROCESSED_PMinus', 'Gamma=PREPROCESSED_PPlus', 'Gamma=PREPROCESSED_FIGR', 'Gamma=PREPROCESSED_FIGR_PLUS', 'Gamma=PREPROCESSED_FIGR_MINUS']
            PATH_TO_SCORES_map = {'true': 'GAMMA=0', 'Gamma=colorblind': 'COLORBLIND', 'Gamma=0': 'GAMMA=0', 'Gamma=small': 'GAMMA=SMALL',
                                   'Gamma=large': 'GAMMA=LARGE', 'FAIR_PStar': 'FA-IR/P-Star', 'FAIR_PMinus': 'FA-IR/P-Minus',
                                   'FAIR_PPlus': 'FA-IR/P-Plus', 'FIGR': 'FIGR', 'FIGR_PLUS': 'FIGR_PLUS', 'FIGR_MINUS': 'FIGR_MINUS', 'Gamma=PREPROCESSED': 'PREPROCESSED',
                                   'Gamma=PREPROCESSED_PMinus': 'PREPROCESSED_PMinus', 'Gamma=PREPROCESSED_PPlus': 'PREPROCESSED_PPlus', 
                                   'Gamma=PREPROCESSED_FIGR': 'PREPROCESSED_FIGR', 'Gamma=PREPROCESSED_FIGR_PLUS': 'PREPROCESSED_FIGR_PLUS', 'Gamma=PREPROCESSED_FIGR_MINUS': 'PREPROCESSED_FIGR_MINUS'}
            PLOT_LABEL_map = {'true': 'True', 'Gamma=colorblind': 'color-blind LTR', 'Gamma=0': 'color-aware LTR', 'Gamma=small': 'DELTR small',
                                   'Gamma=large': 'DELTR', 'FAIR_PStar': 'FA*IR post ${p^*}$', 'FAIR_PMinus': 'FA*IR post  ${p^-}$',
                                   'FAIR_PPlus': 'FA*IR post ${p^+}$', 'FIGR': 'FIGR post ${p^*}$','FIGR_PLUS': 'FIGR post ${p^+}$','FIGR_MINUS': 'FIGR post ${p^-}$', 'Gamma=PREPROCESSED': 'FA*IR pre ${p^*}$',
                                   'Gamma=PREPROCESSED_PMinus': 'FA*IR pre ${p^-}$', 'Gamma=PREPROCESSED_PPlus': 'FA*IR pre ${p^+}$', 
                                   'Gamma=PREPROCESSED_FIGR': 'FIGR pre ${p^*}$', 'Gamma=PREPROCESSED_FIGR_PLUS': 'FIGR pre ${p^+}$', 'Gamma=PREPROCESSED_FIGR_MINUS': 'FIGR pre ${p^-}$'}
            
            PLOT_BOOL_map = {'true': True, 'Gamma=colorblind': True, 'Gamma=0': True, 'Gamma=small': False,'Gamma=large': True, 
                             'FAIR_PStar': True, 'FAIR_PMinus': True, 'FAIR_PPlus': True, 
                             'FIGR': True, 'FIGR_PLUS': True, 'FIGR_MINUS': True,  
                             'Gamma=PREPROCESSED': True, 'Gamma=PREPROCESSED_PMinus': True, 'Gamma=PREPROCESSED_PPlus': True, 
                             'Gamma=PREPROCESSED_FIGR': True, 'Gamma=PREPROCESSED_FIGR_PLUS': True, 'Gamma=PREPROCESSED_FIGR_MINUS': True}
            
            colormap = {'true': 'dimgray', 'Gamma=colorblind': 'black', 'Gamma=0': 'limegreen', 'Gamma=small': 'white', 'Gamma=large': 'cornflowerblue', 
                        'FAIR_PStar': 'mediumvioletred', 'FAIR_PMinus': 'goldenrod', 'FAIR_PPlus': 'brown', 
                        'FIGR': 'forestgreen', 'FIGR_PLUS': 'tomato', 'FIGR_MINUS':'mediumblue',
                        'Gamma=PREPROCESSED': 'mediumvioletred', 'Gamma=PREPROCESSED_PMinus': 'goldenrod', 'Gamma=PREPROCESSED_PPlus': 'brown',
                        'Gamma=PREPROCESSED_FIGR': 'forestgreen', 'Gamma=PREPROCESSED_FIGR_PLUS': 'tomato', 'Gamma=PREPROCESSED_FIGR_MINUS': 'mediumblue'}

            linemap = {'true': (0, (1, 1)),'Gamma=colorblind': 'solid' , 'Gamma=0': 'dotted', 'Gamma=small': 'dotted', 'Gamma=large': 'dotted', 
                       'FAIR_PStar': 'dashdot', 'FAIR_PMinus': 'dashed','FAIR_PPlus': (0, (3, 1, 1, 1, 1, 1)), 
                       'FIGR': 'solid', 'FIGR_PLUS': (0, (3, 5, 1, 5)), 'FIGR_MINUS': (0, (5, 5)),
                       'Gamma=PREPROCESSED': 'dashdot', 'Gamma=PREPROCESSED_PMinus': 'dashed', 'Gamma=PREPROCESSED_PPlus': (0, (3, 1, 1, 1, 1, 1)),
                       'Gamma=PREPROCESSED_FIGR': 'solid', 'Gamma=PREPROCESSED_FIGR_PLUS': (0, (3, 5, 1, 5)), 'Gamma=PREPROCESSED_FIGR_MINUS': (0, (5, 5))}

            markermap = {'true': 's', 'Gamma=colorblind': '.' , 'Gamma=0': '>', 'Gamma=small': ',', 'Gamma=large': 'o', 
                       'FAIR_PStar': '*', 'FAIR_PMinus': '.','FAIR_PPlus': 'p', 
                       'FIGR': 'X', 'FIGR_PLUS': 'P', 'FIGR_MINUS': '^',
                       'Gamma=PREPROCESSED': '*', 'Gamma=PREPROCESSED_PMinus': '.', 'Gamma=PREPROCESSED_PPlus': 'p',
                       'Gamma=PREPROCESSED_FIGR': 'X', 'Gamma=PREPROCESSED_FIGR_PLUS': 'P', 'Gamma=PREPROCESSED_FIGR_MINUS': '^'}
            
            for experiment in EXPERIMENT_NAMES:
                if len(experiment.split('=')) > 1 and experiment.split('=')[1] == 'colorblind':   #### do this only for colorblind
                    pathsForColorblind = [self.__trainingDir + 'fold_'+str(i+1)+'/GAMMA=0/' for i in range(NUM_FOLDS)]

                
                pathsToScores = [self.__trainingDir + 'fold_'+str(i+1)+'/'+PATH_TO_SCORES_map[experiment]+'/' for i in range(NUM_FOLDS)]

                if len(experiment.split('=')) > 1 and experiment.split('=')[1] == 'colorblind':   #### do this only for colorblind
                    self.__original, self.__predictions = self.__prepareData(pathsToScores, pathsForColorblind)
                else:
                    self.__original, self.__predictions = self.__prepareData(pathsToScores)

                if experiment == 'true':
                    self.__predictions = self.__original
                if PLOT_BOOL_map[experiment]:
                    result_protected, result_nonprotected, prot_group_percentage = self.__group_fairness_average_all_queries(NUM_FOLDS=NUM_FOLDS)
                    group_x_ticks_pos = np.arange(0, (len(EXPERIMENT_NAMES)+2)*len(result_protected), len(EXPERIMENT_NAMES)+2)[:N_BINS_TO_SHOW]*100
                    group_x_ticks = np.arange(1, len(group_x_ticks_pos)+1, 1)[:N_BINS_TO_SHOW]
                    group_fair_ax.plot(group_x_ticks_pos, result_protected[:N_BINS_TO_SHOW], label=PLOT_LABEL_map[experiment], color=colormap[experiment], linestyle=linemap[experiment], marker=markermap[experiment])

                    ind_fairness, length  = self.__individual_fairness_average_all_queries()
                    ind_x_ticks_pos = np.arange(0, (len(EXPERIMENT_NAMES)+2)*(length/self.__chunkSize), len(EXPERIMENT_NAMES)+2)[:N_BINS_TO_SHOW]
                    ind_x_ticks = np.arange(1, len(ind_x_ticks_pos)+1, 1)[:N_BINS_TO_SHOW]
                    ind_fair_ax.plot(ind_x_ticks_pos, ind_fairness[:N_BINS_TO_SHOW], label=PLOT_LABEL_map[experiment], color=colormap[experiment], linestyle=linemap[experiment], marker=markermap[experiment])
                
                    
            # plot line y = p^*
            group_fair_ax.plot(group_x_ticks_pos, [prot_group_percentage]*len(group_x_ticks_pos),  label='y = ${p^*}$', color='deepskyblue', linestyle='dashed')
                
            
            #######################################################################################
            #######################################################################################
            if self.__prot_attr_name == 'sex':
                group_fair_ax.set_title('Protected attrbute = gender', fontsize=16)
            else:
                group_fair_ax.set_title('Protected attribute = '+self.__prot_attr_name, fontsize=16)
            group_fair_ax.set_xlabel ("${k}$", fontsize=16)
            group_fair_ax.set_ylabel("proportion of the protected group", fontsize=16)
            group_fair_ax.set_xticks(group_x_ticks_pos)
            group_fair_ax.set_xticklabels(group_x_ticks)
            group_fair_ax.tick_params(axis='both', which='major', labelsize=40)
            group_fair_ax.tick_params(axis='both', which='minor', labelsize=40)
            group_fair_ax.set_facecolor("white")
            group_fair_ax.spines['bottom'].set_color('black')
            group_fair_ax.spines['left'].set_color('black')
            group_fair_ax.spines['right'].set_color('black')
            group_fair_ax.spines['top'].set_color('black')
            group_fair_ax.grid(color='silver', linestyle='dotted', linewidth=0.7)
            group_fair_ax.legend( prop={'size': 6}, facecolor='white', loc='upper right',ncol=3, framealpha=1)
            fig_group.savefig(self.__resultDir + 'group_fairness_measure_' + self.__dataset + '.png', dpi=300)
            fig_group.savefig(self.__resultDir + 'group_fairness_measure_' + self.__dataset + '.pdf')

            if self.__prot_attr_name == 'sex':
                ind_fair_ax.set_title('Protected attribute = gender', fontsize=16)
            else:
                ind_fair_ax.set_title('Protected attribute = '+self.__prot_attr_name, fontsize=16)
            ind_fair_ax.set_xlabel ("Blocks of size "+str(self.__chunkSize), fontsize=50)
            ind_fair_ax.set_ylabel(r"$\min \alpha$", fontsize=16)
            ind_fair_ax.set_xticks(ind_x_ticks_pos)
            ind_fair_ax.set_xticklabels(ind_x_ticks)
            ind_fair_ax.tick_params(axis='both', which='major', labelsize=40)
            ind_fair_ax.tick_params(axis='both', which='minor', labelsize=40)
            ind_fair_ax.set_facecolor("white")
            ind_fair_ax.spines['bottom'].set_color('black')
            ind_fair_ax.spines['left'].set_color('black')
            ind_fair_ax.spines['right'].set_color('black')
            ind_fair_ax.spines['top'].set_color('black')
            ind_fair_ax.grid(color='silver', linestyle='dotted', linewidth=0.7)
            ind_fair_ax.legend(loc="upper left", prop={'size': 6}, facecolor='white', framealpha=1, ncol=3)
            fig_ind.savefig(self.__resultDir + 'individual_fairness_measure_' + self.__dataset + '.png', dpi=300)
            fig_ind.savefig(self.__resultDir + 'individual_fairness_measure_' + self.__dataset + '.pdf')

            
    ######################################################################################################
    ############################## TRUE AND PRED DATA PROCESSING HERE ####################################
    ###################################################################################################### 
    def __prepareData(self, pathsToScores, pathsForColorblind=None):
        '''
        reads training scores and predictions from disc and arranges them NICELY into a dataframe
        '''
        trainingfiles = list()
        predictionfiles = list()
        for dirName in pathsToScores:
            for _, _, filenames in os.walk(dirName):
                for fileName in filenames:
                    if 'trainingScores_ORIG.pred' in fileName:
                        trainingfiles.append(str(dirName + fileName))
                    if 'predictions.pred' in fileName:
                        predictionfiles.append(str(dirName + fileName))
        trainingScores = pd.concat((pd.read_csv(file,
                                                sep=",",
                                                names=self.__columnNames) \
                                    for file in trainingfiles))
        predictedScores = pd.concat((pd.read_csv(file,
                                                sep=",",
                                                names=self.__columnNames) \
                                    for file in predictionfiles))
        if pathsForColorblind is not None:
            # if we want to evaluate a colorblind training, we have to put the protectedAttribute
            colorblindTrainingFiles = (dirname + 'trainingScores_ORIG.pred' for dirname in pathsForColorblind)
            trainingScoresWithProtected = pd.concat((pd.read_csv(file, sep=",", names=self.__columnNames) \
                                         for file in colorblindTrainingFiles))

            trainingScores, \
            predictedScores = self.__add_prot_to_colorblind(trainingScoresWithProtected,
                                                            trainingScores,
                                                            predictedScores)
        return trainingScores, predictedScores

    ######################################################################################################
    ############################## METRICS CALCULATION HERE ##############################################
    ###################################################################################################### 
    
    
    def __group_fairness_average_all_queries(self, NUM_FOLDS = 1, plotGroundTruth=False):
        '''
        calculates percentage of protected (non-protected resp.) for each chunk of the ranking
        plots them into a figure

        averages results over all queries
        '''
        if plotGroundTruth:
            rankingsPerQuery = self.__original.groupby(self.__original['query_id'], as_index=False, sort=False)
        else:
            rankingsPerQuery = self.__predictions.groupby(self.__predictions['query_id'], as_index=False, sort=False)
        p = float(len(self.__predictions.query("prot_attr" + "==1")) / len(self.__predictions))
        
        shortest_query = math.inf

        data_matriks = pd.DataFrame()

        for name, rank in rankingsPerQuery:
            # find length shortest query to make plotting easy
            if (len(rank) < shortest_query):
                shortest_query = len(rank)
        if 'compas' in self.__dataset:
            shortest_query = 1000
        # shortest_query = 100 # change this later
        for name, rank in rankingsPerQuery:
            if 'engineering' in self.__dataset:
                temp = rank['prot_attr'].head(shortest_query)
            else:
                temp = rank[self.__prot_attr_name].head(shortest_query)
            data_matriks[name] = temp.reset_index(drop=True)


        # chunkStartPositions = np.arange(0, shortest_query, self.__chunkSize)
        chunkStartPositions = np.arange(0, shortest_query, self.__chunkSize)

        result_protected = np.empty(len(chunkStartPositions))
        result_nonprotected = np.empty(len(chunkStartPositions))

        cumulative_result_protected = np.empty(len(chunkStartPositions))
        cumulative_result_nonprotected = np.empty(len(chunkStartPositions))

        for idx, start in enumerate(chunkStartPositions):
            if idx == (len(chunkStartPositions) - 1):
                # last Chunk
                end = shortest_query
            else:
                end = chunkStartPositions[idx] + self.__chunkSize
            chunk = data_matriks.iloc[start:end]
            chunk_protected = 0
            for col in chunk:
                try:
                    chunk_protected += chunk[col].value_counts()[1]
                except KeyError:
                    # no protected elements in this chunk
                    chunk_protected += 0

            chunk_nonprotected = 0
            for col in chunk:
                try:
                    chunk_nonprotected += chunk[col].value_counts()[0]
                except KeyError:
                    # no nonprotected elements in this chunk
                    chunk_nonprotected = 0
            
            result_protected[idx] = chunk_protected 
            result_nonprotected[idx] = chunk_nonprotected 
        
           
        
        # cumulative results
        divisor = np.arange(len(chunkStartPositions))+1
        cumulative_result_protected = np.cumsum(result_protected) / ((NUM_FOLDS*self.__chunkSize) * divisor )
        cumulative_result_nonprotected = np.cumsum(result_nonprotected) / ((NUM_FOLDS*self.__chunkSize) * divisor )
        
        
        # proportions in bins
        result_protected /= (NUM_FOLDS*self.__chunkSize)
        result_nonprotected /= (NUM_FOLDS*self.__chunkSize)

        return cumulative_result_protected, cumulative_result_nonprotected, p
        
    def __individual_fairness_average_all_queries(self, plotGroundTruth=False):
        '''
        calculates alpha for individual fairness for each chunk of the ranking
        plots them into a figure

        averages results over all queries
        '''
        if plotGroundTruth:
            rankingsPerQuery = self.__original.groupby(self.__original['query_id'], as_index=False, sort=False)
        else:
            rankingsPerQuery = self.__predictions.groupby(self.__predictions['query_id'], as_index=False, sort=False)
        shortest_query = math.inf

        data_matriks = []
        

        for name, rank in rankingsPerQuery:
            # find length shortest query to make plotting easy
            if (len(rank) < shortest_query):
                shortest_query = len(rank)

        disp = []

        for name, rank in rankingsPerQuery:
            temp = rank['doc_id'].head(shortest_query)
            data_matriks.append(temp.reset_index(drop=True).to_numpy())
        data_matriks = np.asarray(data_matriks)
        
        for query in data_matriks:
            sorted_indices = np.argsort(query)+1
            disp.append([float(i+1)/sorted_indices[i] for i in range(len(query))])
        disp = np.mean(np.asarray(disp), axis=0)

        bin_size = int(self.__chunkSize)
        
        disp = np.concatenate((disp, np.ones((self.__chunkSize - len(disp)%bin_size)) ))
        length = len(disp)
        
        reshaped_disp = disp[:length].reshape(-1, bin_size)
        binned_disp = []
        for row in reshaped_disp:
            try:
                binned_disp.append(np.amin(row))
            except ZeroDivisionError:
                binned_disp.append(1.0)
        
        return binned_disp, length

     
    def __add_prot_to_colorblind(self, trainingScoresWithProtected, colorblind_orig, colorblind_pred):
        orig_prot_attr = trainingScoresWithProtected['prot_attr']
        colorblind_orig["prot_attr"] = orig_prot_attr
        new_prot = []
        for i in range(5):
            new_prot.append(len((colorblind_orig.loc[colorblind_orig['query_id']==int(i+1)]))*[0])
        lengths = [len(row) for row in new_prot]
        cumulative_lengths = [np.sum(lengths[:i]) for i in range(len(lengths))]
        for doc_id in colorblind_orig['doc_id']:
            prot_status_for_pred = colorblind_orig.loc[colorblind_orig['doc_id'] == doc_id]['prot_attr'].values
            query_ids = colorblind_orig.loc[colorblind_orig['doc_id'] == doc_id]['query_id'].values
            positions = colorblind_pred[colorblind_pred['doc_id']==doc_id].index.to_numpy()
            i = 0
            for query_id in query_ids:
                new_prot[query_id-1][positions[i]] = prot_status_for_pred[i]
                i += 1
            
        prot = [item for sublist in new_prot for item in sublist]
        prot = np.asarray(prot)
        colorblind_pred['prot_attr'] = prot.astype(int)
        return colorblind_orig, colorblind_pred
        

    