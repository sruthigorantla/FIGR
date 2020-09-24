import processingWithFair.fair.post_processing_methods.fair_ranker.create as fair
from processingWithFair.fair.dataset_creator.candidate import Candidate


from processingWithFair.metrics import precision_at
from figr.figr import figr

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def rerank_featurevectors_figr(dataDescription, dataset, p_deviation=0.0, k=100, post_process=False, pre_process=False):

    if 'engineering' in dataset:
        if post_process or pre_process:
            data = pd.read_csv(dataDescription.orig_data_path, names=dataDescription.header)
        else:
            data = pd.read_csv(dataDescription.orig_data_path, names=dataDescription.header, header=0)
        print(data.columns.values)
        NUM_GROUPS = len(data.groupby(dataDescription.protected_group).count())
            
        data.rename(columns = {dataDescription.protected_group:'prot_attr'}, inplace = True)

        if not post_process and not pre_process:
            data['doc_id'] = data.index+1
            dataDescription.header = np.append(dataDescription.header, 'doc_id')

        train_final_ranking = None
        for query in data['query_id'].unique():
            data_query = data.query("query_id==" + str(query))
            p = float(len(data_query.query("prot_attr" + "==1")) / len(data_query) )
            print(dataDescription.protected_group, query)
            print("proportion of protected group elements: ", p)
            print(len(data_query))
            np_data = np.array(data_query)
            try:
                LTR_ranking = [f"id-{int(x)}" for x in data["doc_id"]] # 2nd column is ranking
            except KeyError:
                LTR_ranking = [f"id-{int(x+1)}" for x in range(len(data_query))]
            
            id_2_protected = {}
            id_2_row = {}
            for idx, (id, protected) in enumerate(zip(LTR_ranking, data_query["prot_attr"])):
                id_2_protected[id] = int(protected)
                id_2_row[id] = np_data[idx]
            
            # Now run the algorithm figr

            final_ranking = figr(LTR_ranking, id_2_protected, NUM_GROUPS, 1-p, p_deviation, k)
        
            # Contruct the dataframe back for dumping purposes
            length = len(final_ranking)
            final_data = []
            counter = 0
            for id in final_ranking:
                counter += 1
                final_data.append(id_2_row[id])
                final_data[-1][dataDescription.score_attribute] = 1.0 - float(counter-1)/length
                
            if train_final_ranking is None:
                train_final_ranking = final_data
            else:
                train_final_ranking = np.concatenate((train_final_ranking, final_data), axis=0)

            
        final_data_to_write = pd.DataFrame(data=train_final_ranking, columns=dataDescription.header)
        # bring file into expected format for evaluation, if used for post-processing
        if post_process:
            final_data_to_write = final_data_to_write.astype({'prot_attr' : 'int64', 'doc_id' : 'int64', 'query_id' : 'int64'})
            final_data_to_write = final_data_to_write[dataDescription.header]
        else:
            final_data_to_write.rename(columns = {'prot_attr':dataDescription.protected_group}, inplace = True)

        # write the data back    
        if pre_process:
            final_data_to_write.to_csv(dataDescription.result_path, sep=',', index=False, header=False)
        elif post_process:
            final_data_to_write.to_csv(dataDescription.result_path, sep=',', index=False, header=False)
        else:
            final_data.to_csv(dataDescription.result_path, sep=',', index=False, header=True)

                
        return  
        
    elif 'german' in dataset:
        data = pd.read_csv(dataDescription.orig_data_path, names=dataDescription.header, header=0)
        p = float(len(data.query(dataDescription.protected_group + "==1")) / len(data)) 
        
        NUM_GROUPS = len(data.groupby(dataDescription.protected_group).count())
        data.rename(columns = {dataDescription.protected_group:'prot_attr'}, inplace = True)
    
        # for German credit, COMPAS and other such datasests, add query_id as all 1s.
        data['query_id'] = np.ones(len(data))
        dataDescription.header = np.append(dataDescription.header, 'query_id')
        
        # sort the training data based on true scores
        data = (data.sort_values(by=[dataDescription.judgment], ascending=False)).reset_index(drop=True)
        
        # add 'doc_id' here itself
        data['doc_id'] = data.index+1
        dataDescription.header = np.append(dataDescription.header, 'doc_id')
        
    
    elif 'compas' in dataset:
        data = pd.read_csv(dataDescription.orig_data_path, names=dataDescription.header, header=0)
        p = float(len(data.query(dataDescription.protected_group + "==1")) / len(data))
        
        NUM_GROUPS = len(data.groupby(dataDescription.protected_group).count())
        data.rename(columns = {dataDescription.protected_group:'prot_attr'}, inplace = True)
    
        data['query_id'] = np.ones(len(data))
        dataDescription.header = np.append(dataDescription.header, 'query_id')
        data[dataDescription.judgment] = data[dataDescription.judgment].apply(lambda val: 1-val)
        data = (data.sort_values(by=[dataDescription.judgment], ascending=False)).reset_index(drop=True)
        data['doc_id'] = data.index+1
        dataDescription.header = np.append(dataDescription.header, 'doc_id')

       
    np_data = np.array(data)
    try:
        LTR_ranking = [f"id-{int(x)}" for x in data["doc_id"]] # 2nd column is ranking
    except KeyError:
        LTR_ranking = [f"id-{int(x+1)}" for x in range(len(data))]
    
    id_2_protected = {}
    id_2_row = {}
    for idx, (id, protected) in enumerate(zip(LTR_ranking, data["prot_attr"])):
        id_2_protected[id] = int(protected)
        id_2_row[id] = np_data[idx]
    
    # Now run the algorithm figr

    final_ranking = figr(LTR_ranking, id_2_protected, NUM_GROUPS, 1-p, p_deviation, k)
    
    # Contruct the dataframe back for dumping purposes
    length = len(final_ranking)
    final_data = []
    counter = 0
    for id in final_ranking:
        counter += 1
        final_data.append(id_2_row[id])
        final_data[-1][dataDescription.score_attribute] = 1.0 - float(counter-1)/length
        
    final_data = np.asarray(final_data)
    final_data = pd.DataFrame(data=final_data, columns=dataDescription.header)
    
    
    # bring file into expected format for evaluation, if used for post-processing
    if post_process:
        final_data = final_data.astype({'prot_attr' : 'int64', 'doc_id' : 'int64', 'query_id' : 'int64'})
        final_data = final_data[dataDescription.header]
    else:
        if pre_process:
            final_data = final_data.drop(columns=['doc_id'])
            new_header = dataDescription.header
            del new_header['doc_id']
            dataDescription.header = new_header
        final_data.rename(columns = {'prot_attr':dataDescription.protected_group}, inplace = True)

    # write the data back    
    if pre_process or post_process:
        final_data.to_csv(dataDescription.result_path, sep=',', index=False, header=False)
    else:
        final_data.to_csv(dataDescription.result_path, sep=',', index=False, header=True)


def rerank_featurevectors(dataDescription, dataset, p_deviation=0.0, post_process=False, pre_process=False):
    if post_process or pre_process:
        data = pd.read_csv(dataDescription.orig_data_path, names=dataDescription.header)
    else:
        data = pd.read_csv(dataDescription.orig_data_path, names=dataDescription.header, header=0)
    data['uuid'] = 'empty'
    
    reranked_features = pd.DataFrame()
    
    
    # for German credit, COMPAS and other such datasests, add query_id as all 1s.
    if 'query_id' not in data.columns.values:
        data['query_id'] = np.ones(len(data))
        dataDescription.header = np.append(dataDescription.header, 'query_id')
    

    if dataset == 'german':
        data = (data.sort_values(by=[dataDescription.judgment], ascending=False)).reset_index(drop=True)
    if dataset == 'compas':
        data[dataDescription.judgment] = data[dataDescription.judgment].apply(lambda val: 1-val)
        data = (data.sort_values(by=[dataDescription.judgment], ascending=False)).reset_index(drop=True)
    
    # only engineering with after LTR already has doc_id
    if not pre_process and not post_process:
        data['doc_id'] = data.index+1
        dataDescription.header = np.append(dataDescription.header, 'doc_id')

    np_data = np.array(data)
    # re-rank with fair for every query
    for query in data['query_id'].unique():

        print("Rerank for query " + str(query))
        data_query = data.query("query_id==" + str(query))
        data_query, protected, nonProtected = create(data_query, dataDescription)
        # protected attribute value is always 1
        # p = (len(data_query.query(dataDescription.protected_attribute + "==1")) / len(data_query) + p_deviation)
        
        p = float(len(data_query.query(dataDescription.protected_group + "==1")) / len(data_query) + p_deviation)
        
        print("proportion of protected group elements: ", p)
        fairRanking, _ = fair.fairRanking(data_query.shape[0], protected, nonProtected, p, dataDescription.alpha)
        fairRanking = setNewQualifications(fairRanking)

        # swap original qualification with fair qualification
        for candidate in fairRanking:
            candidate_row = data_query[data_query.uuid == candidate.uuid]
            candidate_row.iloc[0, data_query.columns.get_loc(dataDescription.judgment)] = (candidate.qualification / len(fairRanking))

            reranked_features = reranked_features.append(candidate_row.iloc[0], sort=False)
            reranked_features = reranked_features[candidate_row.columns]
    
    # sort by judgment to ease evaluation of output
    reranked_features_sorted = pd.DataFrame()
    reranked_features_orig_order = reranked_features
    for query in data['query_id'].unique():
        sortet = reranked_features.query("query_id==" + str(query)).sort_values(by=dataDescription.judgment, ascending=False)
        reranked_features_sorted = reranked_features_sorted.append(sortet)

    reranked_features_sorted.update(reranked_features_sorted['query_id'].astype(int).astype(str))
    
    # bring file into expected format for evaluation, if used for post-processing
    if post_process:
        reranked_features_sorted = reranked_features_sorted.drop(columns=['uuid'])
        reranked_features_sorted = reranked_features_sorted.astype({'prot_attr' : 'int64', 'doc_id' : 'int64'})
        reranked_features_sorted = reranked_features_sorted[dataDescription.header]
    else:
        reranked_features_sorted = reranked_features_sorted.drop(columns=['uuid'])
    print(reranked_features_sorted[:15])
    # write the data back    
    if pre_process or post_process:
        reranked_features_sorted.to_csv(dataDescription.result_path, sep=',', index=False, header=False)
    else:
        reranked_features_sorted.to_csv(dataDescription.result_path, sep=',', index=False, header=True)



def create(data, dataDescription):
    protected = []
    nonProtected = []

    for row in data.itertuples():
        # change to different index in row[.] to access other columns from csv file
        if row[data.columns.get_loc(dataDescription.protected_group) + 1] == 0.:
            candidate = Candidate(row[data.columns.get_loc(dataDescription.judgment) + 1], [])
            nonProtected.append(candidate)
            data.loc[row.Index, "uuid"] = candidate.uuid
            
        else:
            candidate = Candidate(row[data.columns.get_loc(dataDescription.judgment) + 1], dataDescription.protected_group)
            protected.append(candidate)
            data.loc[row.Index, "uuid"] = candidate.uuid
            
    # sort candidates by judgment
    protected.sort(key=lambda candidate: candidate.qualification, reverse=True)
    nonProtected.sort(key=lambda candidate: candidate.qualification, reverse=True)

    return data, protected, nonProtected


def setNewQualifications(fairRanking):
    qualification = len(fairRanking)
    for candidate in fairRanking:
        candidate.qualification = qualification
        qualification -= 1
    return fairRanking