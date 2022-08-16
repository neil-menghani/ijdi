import pandas as pd, numpy as np
from q import *

def get_aggregates(coordinates,probs,outcomes,values_to_choose,column_name,penalty,direction='positive'):
    #print("Calling get_aggregates with column_name=",column_name)
    if values_to_choose:
        to_choose = coordinates[list(values_to_choose.keys())].isin(values_to_choose).all(axis=1)
        temp_df=pd.concat([coordinates.loc[to_choose], outcomes[to_choose], pd.Series(data=probs[to_choose],index=outcomes[to_choose].index,name='prob')],axis=1)
    else:
        temp_df= pd.concat([coordinates, outcomes, pd.Series(data=probs,index=outcomes.index,name='prob')],axis=1)
    aggregates = {}
    thresholds = set()
    for name, group in temp_df.groupby(column_name):
        thesum = group.iloc[:,-2].sum() 
        theprobs = group.iloc[:,-1]
        positive, q_mle, q_min, q_max = compute_q(thesum,theprobs,penalty)
        #print("name=",name,"q_mle=",q_mle,"q_min=",q_min,"q_max=",q_max)
        if positive:
            if direction == 'positive':
                if q_max < 1:
                    positive = 0
                elif q_min < 1:
                    q_min = 1
            else: 
                if q_min > 1:
                    positive = 0
                elif q_max > 1:
                    q_max = 1
            if positive:
                aggregates[name]={'positive':positive, 'q_mle':q_mle,'q_min':q_min,'q_max':q_max,'thesum':thesum,'theprobs':theprobs}
                thresholds.update([q_min,q_max])
                
    allsum = temp_df.iloc[:,-2].sum()
    allprobs = temp_df.iloc[:,-1]
    return [aggregates,sorted(thresholds),allsum,allprobs]

def choose_aggregates(aggregates,thresholds,penalty,allsum,allprobs,direction='positive'):
    #print('thresholds=',thresholds)
    best_score = 0
    best_q = 0
    best_names = []
    for i in range(len(thresholds)-1):
        thethreshold = (thresholds[i]+thresholds[i+1])/2
        names = []
        thesum = 0.0
        theprobs = []
        for key, value in aggregates.items():
            if (value['positive']) & (value['q_min'] < thethreshold) & (value['q_max'] > thethreshold):
                names.append(key)
                thesum += value['thesum']
                theprobs = theprobs + value['theprobs'].tolist()
        theprobs_series = pd.Series(theprobs)
        current_q_mle = binary_search_on_slopeterm(thesum,theprobs_series)
        if ((direction == 'positive') & (current_q_mle < 1)) | ((direction != 'positive') & (current_q_mle > 1)):
            current_q_mle = 1
        current_score = compute_score_given_q(thesum,theprobs_series,penalty*len(names),current_q_mle)
        #print("In choose_aggregates, current_score = ",current_score+penalty*len(names),"-",penalty*len(names),"=",current_score)
        if current_score > best_score:
            best_score = current_score
            best_q = current_q_mle
            best_names = names
        #print('current',names,current_score,current_q_mle,'with penalty of',penalty*len(names))
    # also have to consider case of including all attributes values including those that never make positive contributions to the score
    allprobs_series = pd.Series(allprobs)
    current_q_mle = binary_search_on_slopeterm(allsum,allprobs_series)
    if ((direction == 'positive') & (current_q_mle < 1)) | ((direction != 'positive') & (current_q_mle > 1)):
        current_q_mle = 1
    current_score = compute_score_given_q(allsum,allprobs_series,0,current_q_mle)
    #print("In choose_aggregates, current_score = ",current_score,"-[no penalty]=",current_score)
    if current_score > best_score:
        best_score = current_score
        best_q = current_q_mle
        best_names = []
    return [best_names,best_score]