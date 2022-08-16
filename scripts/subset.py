import pandas as pd, numpy as np
from q import *
from prep import *

def mk_subset_all_values(coordinates):
    subset_all_values = {}
#    for theatt in coordinates:
#        subset_all_values[theatt]=coordinates[theatt].unique().tolist()
    return subset_all_values

def mk_subset_random_values(coordinates,prob,minelements=0):
    subset_random_values = {}
    shuffled_column_names = np.random.permutation(coordinates.columns.values)
    for theatt in shuffled_column_names:
        temp = coordinates[theatt].unique()
        mask = np.random.rand(len(temp)) < prob
        if mask.sum() < len(temp):
            subset_random_values[theatt] = temp[mask].tolist()
            remaining_records = len(coordinates.loc[coordinates[list(subset_random_values.keys())].isin(subset_random_values).all(axis=1)])
            if remaining_records < minelements:
                del subset_random_values[theatt]
    return subset_random_values
      
def score_current_subset(coordinates,probs,outcomes,penalty,current_subset,direction='positive'):
    if current_subset:
        to_choose = coordinates[list(current_subset.keys())].isin(current_subset).all(axis=1)
        temp_df=pd.concat([coordinates.loc[to_choose], outcomes[to_choose], pd.Series(data=probs[to_choose],index=outcomes[to_choose].index,name='prob')],axis=1)
    else:
        temp_df= pd.concat([coordinates, outcomes, pd.Series(data=probs,index=outcomes.index,name='prob')],axis=1)
    thesum = temp_df.iloc[:,-2].sum()
    theprobs = temp_df.iloc[:,-1]
    current_q_mle = binary_search_on_slopeterm(thesum,theprobs)
    if ((direction == 'positive') & (current_q_mle < 1)) | ((direction != 'positive') & (current_q_mle > 1)):
        current_q_mle = 1
    # totalpenalty = penalty * sum of list lengths in current_subset
    totalpenalty = 0
    for i in current_subset.values():
        totalpenalty += len(i)
    totalpenalty *= penalty  
    penalized_score = compute_score_given_q(thesum,theprobs,totalpenalty,current_q_mle)
    #print("In score_current_subset, current_score = ",penalized_score+totalpenalty,"-",totalpenalty,"=",penalized_score)
    
    #wrong_q = thesum/theprobs.sum()
    #print("Score =",penalized_score,"at q =",current_q_mle,"vs.",compute_score_given_q(thesum,theprobs,totalpenalty,wrong_q),"at q = ",wrong_q)
    return penalized_score

def score_current_subset_ijdi(df, features, proba_confusion_col, proba_ijdi_col, outcomes_col, threshold, lambda_param, current_subset, 
                              constant_threshold=True, confusion_metrics=True, ijdi_metrics=True, verbose=False, 
                              p_bar='p_bar'):
    
    # generate metrics (if not already done, either in previous recursive call or before running scan)
    if confusion_metrics:
        df = generate_metrics(df, proba_confusion_col, outcomes_col, threshold, constant_threshold=constant_threshold)
       
    if ijdi_metrics:
        df = generate_ijdi_metrics(df, proba_ijdi_col, lambda_param, p_bar=p_bar)

    current_score = score_current_subset(df[features], df['p_hat'], df['positives'], penalty=0.1, current_subset=current_subset, direction='positive')
    
    print("Current Score:", current_score)
    
    if current_score > 0:
        
        if current_subset:
            in_current_subset = df[list(current_subset.keys())].isin(current_subset).all(axis=1)
        else:
            in_current_subset = pd.Series(True, index=df.index)
            
        if 'p_old' not in df.columns:
            print("First Iteration")
            df['p_old'] = df[proba_ijdi_col]
            df['p_bar_old'] = df['p_bar']

        df['in_current_subset'] = in_current_subset
        in_current_subset_df = df[in_current_subset]
        p_delta_avg = in_current_subset_df['p_delta'].mean()
        p_censor_avg = in_current_subset_df['p_censor'].mean()
        print("Average p_delta:", p_delta_avg)
        print("Average p_censor:", p_censor_avg)
        
        if verbose:
            print(in_current_subset_df[[proba_ijdi_col, 'p_old', 'p_bar', 'p_bar_old', 'p_delta', 
                                        'positives_constprobs', 'p_hat_raw', 'p_hat']])
        
        if p_delta_avg < -1E-6: # check p_delta condition
            print("Subset violates p_delta condition. Adjusting and Re-scoring...")

            not_in_current_subset_df = df[~in_current_subset]
            p_not_s_avg = not_in_current_subset_df['p_old'].mean()
            in_current_subset_violates_df = in_current_subset_df.loc[in_current_subset_df[proba_ijdi_col] < p_not_s_avg]
            alpha_ratio = (p_not_s_avg - in_current_subset_df[proba_ijdi_col]).sum() / \
                          (p_not_s_avg - in_current_subset_violates_df[proba_ijdi_col]).sum()
            print("Alpha Ratio:", alpha_ratio)
            df[proba_ijdi_col] = df.apply(lambda x : (x[proba_ijdi_col] + alpha_ratio * (p_not_s_avg-x[proba_ijdi_col])) 
                                                     if (x['in_current_subset'] and (x[proba_ijdi_col] < p_not_s_avg)) 
                                                     else x[proba_ijdi_col], axis=1)
            in_current_subset_df = df[in_current_subset]
            print("p(S):", in_current_subset_df[proba_ijdi_col].mean(), "p(~S):", p_not_s_avg)
            
            return score_current_subset_ijdi(df, features, proba_confusion_col, proba_ijdi_col, outcomes_col, threshold, lambda_param, 
                                             current_subset=current_subset, constant_threshold=constant_threshold, 
                                             confusion_metrics=True, ijdi_metrics=True, verbose=verbose, 
                                             p_bar='p_bar_old')

        if p_censor_avg < -1E-6: # check p_censor condition
            print("Subset violates p_censor condition. Adjusting and Re-scoring...")
            
            if in_current_subset_df['p_hat_raw'].mean() >= 1: # check if E[uncensored] >= 1
                print("E[uncensored] >= 1. Pushing all p_hat values below 1 to 1.")
                df['p_hat'] = df.apply(lambda x : 1
                                                  if (x['in_current_subset'])
                                                  else x['p_hat'], axis=1)
                df['p_censor'] = df.apply(lambda x : 0
                                 if (x['in_current_subset'])
                                 else x['p_censor'], axis=1)
            else: # E[uncensored] < 1
                print("E[uncensored] < 1. Pushing all p_hat values below 1 toward 1 so that E[censored] = E[uncensored].")
                numerator = in_current_subset_df['p_hat_raw'].sum() - in_current_subset_df['p_hat'].sum()
                denominator = (1 - in_current_subset_df['p_hat']).sum()
                beta = numerator / denominator
                print("Beta Ratio:", beta)
                df['p_hat'] = df.apply(lambda x : x['p_hat'] + beta * (1 - x['p_hat']) 
                                                  if (x['in_current_subset'] and (x['p_hat'] < 1)) 
                                                  else x['p_hat'], axis=1)
                df['p_censor'] = df['p_hat'] - df['p_hat_raw'] # should be 0 for subset
            
            return score_current_subset_ijdi(df, features, proba_confusion_col, proba_ijdi_col, outcomes_col, threshold, lambda_param, 
                                             current_subset=current_subset, constant_threshold=constant_threshold, 
                                             confusion_metrics=False, ijdi_metrics=False, verbose=verbose, 
                                             p_bar='p_bar_old')
        return current_score
    else:
        return 0
    
def correct_ijdi_subset(df, features, proba_confusion_col, proba_ijdi_col, outcomes_col, 
                        threshold, lambda_param, current_subset):

    df = generate_metrics(df, proba_confusion_col, outcomes_col, threshold, constant_threshold=False)
    df = generate_ijdi_metrics(df, proba_ijdi_col, lambda_param)

    if current_subset:
        in_current_subset = df[list(current_subset.keys())].isin(current_subset).all(axis=1)
    else:
        in_current_subset = pd.Series(True, index=df.index)

    df['in_current_subset'] = in_current_subset
    in_current_subset_df = df[in_current_subset]
    not_in_current_subset_df = df[~in_current_subset]

    pr_not_s = not_in_current_subset_df['positives'].mean()
    p_s_avg = in_current_subset_df[proba_ijdi_col].mean()
    p_not_s_avg = not_in_current_subset_df[proba_ijdi_col].mean()
    target_pr_s = set_edge(pr_not_s + lambda_param * (p_s_avg - p_not_s_avg))
    print("Target positive rate for subset:", target_pr_s)

    new_threshold_s = np.percentile(in_current_subset_df[proba_confusion_col], 100*(1-target_pr_s))

    print("New threshold for subset:", new_threshold_s)

    new_threshold = (new_threshold_s*in_current_subset) + (df[threshold]*~in_current_subset)

    return new_threshold