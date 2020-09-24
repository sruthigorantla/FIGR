'''
Created on May 11, 2018

@author: mzehlike
'''

from adjustText import adjust_text
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats as stats
import math
import os
from fileinput import filename



class Postprocessing_Evaluator():
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
        self.__trainingDir = '../data/'
        self.__resultDir = resultDir
        if not os.path.exists(resultDir):
            os.makedirs(resultDir)
        self.__protectedAttribute = protAttr
        self.__dataset = dataset
        self.__chunkSize = binSize
        if 'german' in dataset:
            self.__prot_attr_name = self.__dataset.split('_')[1]
            self.__columnNames = ['DurationMonth','CreditAmount','score',self.__prot_attr_name,'query_id','doc_id']
            
        elif 'compas' in dataset:
            self.__prot_attr_name = self.__dataset.split('_')[1]
            self.__columnNames = ['priors_count','Violence_rawscore','Recidivism_rawscore',self.__prot_attr_name,'query_id','doc_id']


    def evaluate_on_groundtruth(self):
        
        if 'german' in self.__dataset:
            self.__trainingDir = self.__trainingDir + 'GermanCredit/'
        elif 'compas' in self.__dataset:
            self.__trainingDir = self.__trainingDir + 'COMPAS/'
        else:
            raise ValueError("Choose dataset from (enginering/compas/german)")
        mpl.rcParams.update({'font.size': 10, 'lines.linewidth': 5, 'lines.markersize': 25, 'font.family':'Times New Roman'})
        # avoid type 3 (i.e. bitmap) fonts in figures
        mpl.rcParams['ps.useafm'] = True
        mpl.rcParams['pdf.use14corefonts'] = True
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams["legend.handlelength"] = 6.0
        

        fig_group = plt.figure(figsize=(15, 10))
        group_fair_ax = plt.axes()
        
        fig_ind = plt.figure(figsize=(15, 10))
        ind_fair_ax = plt.axes()
        

        
        NUM_FOLDS = 1
        N_BINS_TO_SHOW = 1000
        EXPERIMENT_NAMES = ['true','FAIR_PStar', 'FAIR_PPlus', 'FIGR', 'FIGR_PLUS', 'FIGR_MINUS', 'FAIR_PMinus']
        PATH_TO_SCORES_map = {'true': '_true', 'FAIR_PStar': '', 'FAIR_PMinus': '_PMinus',
                                'FAIR_PPlus': '_PPlus', 'FIGR': '_FIGR', 'FIGR_PLUS': '_FIGR_PLUS', 'FIGR_MINUS': '_FIGR_MINUS'}
        
        PLOT_LABEL_map = {'true': 'True', 'FAIR_PStar': 'FA*IR ${p^*}$', 'FAIR_PMinus': 'FA*IR ${p^-}$',
                                'FAIR_PPlus': 'FA*IR ${p^+}$', 'FIGR': 'FIGR ${p^*}$', 'FIGR_PLUS': 'FIGR ${p^+}$', 'FIGR_MINUS': 'FIGR ${p^-}$'}
        PLOT_BOOL_map = {'true': True, 'FAIR_PStar': True, 'FAIR_PMinus': True,
                                'FAIR_PPlus': True, 'FIGR': True, 'FIGR_PLUS': True, 'FIGR_MINUS': True}
        colormap = {'true': 'dimgray', 'FAIR_PStar': 'mediumvioletred', 'FAIR_PMinus': 'goldenrod',
                                'FAIR_PPlus': 'brown', 'FIGR': 'forestgreen', 'FIGR_PLUS': 'tomato', 'FIGR_MINUS': 'mediumblue'}
        
        linemap = {'true': (0, (1, 1)), 'FAIR_PStar': 'dashdot', 'FAIR_PMinus': 'dashed',
                                        'FAIR_PPlus': (0, (3, 1, 1, 1, 1, 1)), 'FIGR': 'solid',
                                        'FIGR_PLUS': (0, (3, 5, 1, 5)), 'FIGR_MINUS': (0, (5, 5))}
        markermap = {'true': 's', 'FAIR_PStar': '*', 'FAIR_PMinus': '.',
                                'FAIR_PPlus': 'p', 'FIGR': 'X', 'FIGR_PLUS': 'P', 'FIGR_MINUS': '^'}
        for experiment in EXPERIMENT_NAMES:
            
            pathsToFold = [self.__trainingDir + '' for i in range(NUM_FOLDS)]

            self.__predictions = self.__prepareData(pathsToFold, experiment, self.__prot_attr_name, PATH_TO_SCORES_map)

            if PLOT_BOOL_map[experiment]:
                result_protected, result_nonprotected, prot_group_percentage = self.__group_fairness_average_all_queries(NUM_FOLDS=NUM_FOLDS)
                group_x_ticks_pos = np.arange(0, (len(EXPERIMENT_NAMES)+2)*len(result_protected), len(EXPERIMENT_NAMES)+2)[:N_BINS_TO_SHOW]*100
                group_x_ticks = np.arange(1, len(group_x_ticks_pos)+1, 1)[:N_BINS_TO_SHOW]
                group_fair_ax.plot(group_x_ticks_pos, result_protected[:N_BINS_TO_SHOW], label=PLOT_LABEL_map[experiment], color=colormap[experiment], linestyle=linemap[experiment], marker=markermap[experiment])

                ind_fairness, length  = self.__individual_fairness_average_all_queries()
                ind_x_ticks_pos = np.arange(0, (len(EXPERIMENT_NAMES)+2)*(length/self.__chunkSize), len(EXPERIMENT_NAMES)+2)[:N_BINS_TO_SHOW]
                ind_x_ticks = np.arange(1, len(ind_x_ticks_pos)+1, 1)[:N_BINS_TO_SHOW]
                ind_fair_ax.plot(ind_x_ticks_pos, ind_fairness[:N_BINS_TO_SHOW], label=PLOT_LABEL_map[experiment], color=colormap[experiment], linestyle=linemap[experiment], marker=markermap[experiment])
            
                
        group_fair_ax.plot(group_x_ticks_pos, [prot_group_percentage]*len(group_x_ticks_pos),  label='y = ${p^*}$', color='deepskyblue', linestyle='dashed')
            
            
        #######################################################################################
        #######################################################################################
        if self.__prot_attr_name == 'sex':
            group_fair_ax.set_title('Protected group = female', fontsize=40)
        elif self.__prot_attr_name == 'age35':
            group_fair_ax.set_title('Protected group = ${age < 35}$', fontsize=40)
        elif self.__prot_attr_name == 'age25':
            group_fair_ax.set_title('Protected group = ${age < 25}$', fontsize=40)
        elif self.__prot_attr_name == 'race':
            group_fair_ax.set_title('Protected group = African American', fontsize=40)
        else:
            group_fair_ax.set_title('Protected group = '+self.__prot_attr_name, fontsize=40)
        group_fair_ax.set_xlabel ("${k}$", fontsize=50)
        group_fair_ax.set_ylabel("Proportion of the protected group", fontsize=40)
        group_fair_ax.set_xticks(group_x_ticks_pos)
        group_fair_ax.set_xticklabels(group_x_ticks*self.__chunkSize)
        group_fair_ax.tick_params(axis='both', which='major', labelsize=40)
        group_fair_ax.tick_params(axis='both', which='minor', labelsize=40)
        group_fair_ax.set_facecolor("white")
        group_fair_ax.spines['bottom'].set_color('black')
        group_fair_ax.spines['left'].set_color('black')
        group_fair_ax.spines['right'].set_color('black')
        group_fair_ax.spines['top'].set_color('black')
        group_fair_ax.grid(color='silver', linestyle='dotted', linewidth=0.7)
        group_fair_ax.legend( prop={'size': 8}, facecolor='white', loc='lower center',ncol=4, framealpha=1)
        fig_group.savefig(self.__resultDir + 'group_fairness_measure_' + self.__dataset + '.png', dpi=300)
        fig_group.savefig(self.__resultDir + 'group_fairness_measure_' + self.__dataset + '.pdf')

        if self.__prot_attr_name == 'sex':
            ind_fair_ax.set_title('Protected group = female', fontsize=40)
        elif self.__prot_attr_name == 'age35':
            ind_fair_ax.set_title('Protected group = ${age < 35}$', fontsize=40)
        elif self.__prot_attr_name == 'age25':
            ind_fair_ax.set_title('Protected group = ${age < 25}$', fontsize=40)
        elif self.__prot_attr_name == 'race':
            ind_fair_ax.set_title('Protected group = African American', fontsize=40)
        else:
            ind_fair_ax.set_title('Protected group = '+self.__prot_attr_name, fontsize=40)
        ind_fair_ax.set_xlabel ("Blocks of size "+str(self.__chunkSize), fontsize=40)
        ind_fair_ax.set_ylabel(r"$\min \alpha$", fontsize=40)
        ind_fair_ax.set_xticks(ind_x_ticks_pos)
        ind_fair_ax.set_xticklabels(ind_x_ticks)
        ind_fair_ax.tick_params(axis='both', which='major', labelsize=40)
        ind_fair_ax.tick_params(axis='both', which='minor', labelsize=40)
        ind_fair_ax.set_facecolor("white")
        ind_fair_ax.spines['bottom'].set_color('black')
        ind_fair_ax.spines['left'].set_color('black')
        ind_fair_ax.spines['right'].set_color('black')
        ind_fair_ax.spines['top'].set_color('black')
        ind_fair_ax.set_ylim(0.45,1.05)
        ind_fair_ax.grid(color='silver', linestyle='dotted', linewidth=0.7)
        ind_fair_ax.legend(loc="lower right", prop={'size': 6}, facecolor='white', framealpha=1)
        fig_ind.savefig(self.__resultDir + 'individual_fairness_measure_' + self.__dataset + '.png', dpi=300)
        fig_ind.savefig(self.__resultDir + 'individual_fairness_measure_' + self.__dataset + '.pdf')

            
        
        
    ######################################################################################################
    ############################## TRUE AND PRED DATA PROCESSING HERE ####################################
    ###################################################################################################### 
    def __prepareData(self, pathsToFold, experiment, prot_attr_name, exp_map):
        '''
        reads training scores and predictions from disc and arranges them NICELY into a dataframe
        '''
        test_ground_truth_files = list()
        test_pred_files = list()

        for pathToFold in pathsToFold:
            if 'german' in self.__dataset:
                test_pred_files.append(pathToFold+'GermanCredit_'+prot_attr_name+'_RERANKED'+exp_map[experiment]+'.txt')
            elif 'compas' in self.__dataset:
                test_pred_files.append(pathToFold+'ProPublica_'+prot_attr_name+'_RERANKED'+exp_map[experiment]+'.txt')

        predictedScores = pd.concat((pd.read_csv(file,
                                                sep=",",
                                                header=0) \
                                    for file in test_pred_files))
        return predictedScores

    ######################################################################################################
    ############################## METRICS CALCULATION HERE ##############################################
    ###################################################################################################### 
    
  
    def __group_fairness_average_all_queries(self, NUM_FOLDS = 1, plotGroundTruth=False):
        '''
        calculates percentage of protected (non-protected resp.) for each chunk of the ranking
        plots them into a figure

        averages results over all queries
        '''
        rankingsPerQuery = self.__predictions.groupby(self.__predictions['query_id'], as_index=False, sort=False)
        p = float(len(self.__predictions.query(self.__prot_attr_name + "==1")) / len(self.__predictions))
        
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
            temp = rank[self.__prot_attr_name].head(shortest_query)
            data_matriks[name] = temp.reset_index(drop=True)


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
                # end = chunkStartPositions[idx + 1]
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
        rankingsPerQuery = self.__predictions.groupby(self.__predictions['query_id'], as_index=False, sort=False)
        shortest_query = math.inf

        data_matriks = []
        

        for name, rank in rankingsPerQuery:
            # find length shortest query to make plotting easy
            if (len(rank) < shortest_query):
                shortest_query = len(rank)
        # shortest_query = 100 # change this later
        
        
        
        if 'compas' in self.__dataset:
            shortest_query = 1000
        
        disp = []

        for name, rank in rankingsPerQuery:
            if 'doc_id' in rank.columns.values:
                temp = rank['doc_id'].head(shortest_query)
            else:
                temp = rank['rank'].head(shortest_query)
            data_matriks.append(temp.reset_index(drop=True).to_numpy())
        data_matriks = np.asarray(data_matriks)
        
        for query in data_matriks:
            sorted_indices = np.argsort(query)+1
            disp.append([float(i+1)/sorted_indices[i] for i in range(len(query))])
        disp = np.mean(np.asarray(disp), axis=0)

        bin_size = int(self.__chunkSize)
        length = len(disp) - len(disp)%bin_size
        
        reshaped_disp = disp[:length].reshape(-1, bin_size)
        binned_disp = []
        for row in reshaped_disp:
            try:
                binned_disp.append(np.amin(row))
            except ZeroDivisionError:
                binned_disp.append(1.0)

        return binned_disp, length
