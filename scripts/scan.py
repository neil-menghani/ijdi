import pandas as pd, numpy as np
from subset import *
from aggregate import *
from prep import *
import time

def md_scan(coordinates, probs, outcomes, penalty, num_iters, direction='positive'):
    best_subset = {}
    best_score = -1E10
    for i in range(num_iters):
        flags = np.empty(len(coordinates.columns))
        flags.fill(0)
        # starting subset
        current_subset = mk_subset_all_values(coordinates) if i == 0 else mk_subset_random_values(coordinates,np.random.rand(),10)
        current_score = score_current_subset(coordinates,probs,outcomes,penalty,current_subset,direction)
        #print("Starting subset with score of",current_score,":")
        #print(current_subset)
        while flags.sum() < len(coordinates.columns):
        
            # choose random
            attribute_number_to_scan = np.random.choice(len(coordinates.columns))
            while flags[attribute_number_to_scan]:
                attribute_number_to_scan = np.random.choice(len(coordinates.columns))
            attribute_to_scan = coordinates.columns.values[attribute_number_to_scan]
            
            #print('SCANNING:',attribute_to_scan)
            if attribute_to_scan in current_subset:
                del current_subset[attribute_to_scan]  
            aggregates,thresholds,allsum,allprobs = get_aggregates(coordinates,probs,outcomes,current_subset,attribute_to_scan,penalty,direction)
            temp_names,temp_score=choose_aggregates(aggregates,thresholds,penalty,allsum,allprobs,direction)
            temp_subset = current_subset.copy()
            if temp_names: # if temp_names is not empty (or null)
                temp_subset[attribute_to_scan]=temp_names
            temp_score =  score_current_subset(coordinates,probs,outcomes,penalty,temp_subset,direction)
            #print("Temp subset with score of",temp_score,":")
            #print(temp_subset)
            if temp_score > current_score+1E-6:
                flags.fill(0)
            elif temp_score < current_score-1E-6:
                print("WARNING SCORE HAS DECREASED from",current_score,"to",temp_score)
                    
            flags[attribute_number_to_scan] = 1
            current_subset = temp_subset
            current_score = temp_score
        
        # print("Subset found on iteration",i+1,"of",num_iters,"with score",current_score,":")
        # print(current_subset)
        if (current_score > best_score):
            best_subset = current_subset.copy()
            best_score = current_score
            # print("Best score is now",best_score)
        else:
            pass
            # print("Current score of",current_score,"does not beat best score of",best_score)
            
    return [best_subset,best_score]

def run_scan(df, features, proba_col, outcomes_col, scan_type="bias", penalty=0.1, num_iters=10, direction='positive'):
    
    treatments = df[features]
    probs = df[proba_col]
    outcomes = df[outcomes_col]
    
    start_time = time.time()
    current_subset, current_score = md_scan(treatments, probs, outcomes, penalty, num_iters, direction)
    print("Required time = ", time.time() - start_time, "seconds")
    print("Found", direction, "subset for", scan_type, "scan:")
    print(current_subset)
    print("Score:",current_score)
    return current_subset, current_score

def summarize_scan(df, features, proba_ijdi_col, outcomes_col, current_subset, include=None):
    # If subset is non-empty, set conditions on it. Otherwise set condition to all rows.
    if current_subset:
        to_choose = df[list(current_subset.keys())].isin(current_subset).all(axis=1)
    else:
        to_choose = pd.Series(True, index=df.index)
        
    temp_treatments = df[features][to_choose]
    temp_probs = df[proba_ijdi_col][to_choose]
    temp_outcomes = df[outcomes_col][to_choose]
    
    print("Number of people in subset:", len(temp_treatments), "out of", len(df), "in population")
    print("Proportion of", outcomes_col, ":", temp_outcomes.mean())
    print("Expected proportion of", outcomes_col, ":", temp_probs.mean())
    print("Summary statistics:")
    print(temp_treatments.describe(include=include))

# IJDI-Scan (Algorithm 1; outlined in Section 4.2 of the paper)
def run_ijdi_scan(df, features, proba_confusion_col, proba_ijdi_col, outcomes_col, threshold, lambda_param, 
                  constant_threshold=True, confusion_metrics=True, ijdi_metrics=True, verbose=False, 
                  p_bar='p_bar', recalc_subset=[]):
    
    # generate metrics (if not already done, either in previous recursive call or before running scan)
    if confusion_metrics:
        df = generate_metrics(df, proba_confusion_col, outcomes_col, threshold, constant_threshold=constant_threshold, recalc_subset=recalc_subset)

    if ijdi_metrics:
        df = generate_ijdi_metrics(df, proba_ijdi_col, lambda_param, p_bar=p_bar, recalc_subset=recalc_subset)
        
    current_subset, current_score = run_scan(df, features, 'p_hat', 'positives', scan_type='ijdi', penalty=0.1, num_iters=10, direction='positive')
    
    if current_score == 0:
        print("No subset was detected.")
        return current_subset, current_score
    
    in_current_subset = get_subset_series(df, current_subset)
        
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
    
    not_in_current_subset_df = df[~in_current_subset]
    p_not_s_avg = not_in_current_subset_df['p_old'].mean()
    print("p(S):", in_current_subset_df[proba_ijdi_col].mean(), "p(~S):", p_not_s_avg, "E[censored]", in_current_subset_df['p_hat'].mean(), "E[uncensored]", in_current_subset_df['p_hat_raw'].mean())

    if verbose:
        print(in_current_subset_df[[proba_ijdi_col, 'p_old', 'p_bar', 'p_bar_old', 'p_delta', 'positives_constprobs', 'p_hat_raw', 'p_hat']])

    if p_delta_avg < -1E-6: # check p_delta condition
        print("Subset violates p_delta condition. Adjusting and Re-running...")
        
        in_current_subset_violates_df = in_current_subset_df.loc[in_current_subset_df[proba_ijdi_col] < p_not_s_avg]
        alpha_ratio = (p_not_s_avg - in_current_subset_df[proba_ijdi_col]).sum() / \
                      (p_not_s_avg - in_current_subset_violates_df[proba_ijdi_col]).sum()
        print("Alpha Ratio:", alpha_ratio)
        df[proba_ijdi_col] = df.apply(lambda x : (x[proba_ijdi_col] + alpha_ratio * (p_not_s_avg-x[proba_ijdi_col])) 
                                                 if (x['in_current_subset'] and (x[proba_ijdi_col] < p_not_s_avg)) 
                                                 else x[proba_ijdi_col], axis=1)
        in_current_subset_df = df[in_current_subset]
        print("p(S):", in_current_subset_df[proba_ijdi_col].mean(), "p(~S):", p_not_s_avg)
        
        return run_ijdi_scan(df, features, proba_confusion_col, proba_ijdi_col, outcomes_col, threshold, lambda_param, 
                             constant_threshold=constant_threshold, confusion_metrics=True, ijdi_metrics=True, verbose=verbose, p_bar='p_bar_old', recalc_subset=current_subset)

    if p_censor_avg < -1E-6: # check p_censor condition
        print("Subset violates p_censor condition. Adjusting and Re-running...")
        
        if in_current_subset_df['p_hat_raw'].mean() >= 1: # check if E[uncensored] >= 1
            print("E[uncensored] >= 1. Pushing all p_hat values below 1 to 1.")
            df['p_hat'] = df.apply(lambda x : 1
                                              if (x['in_current_subset'])
                                              else x['p_hat'], axis=1)
            df['p_censor'] = df.apply(lambda x : 0
                                                 if (x['in_current_subset'])
                                                 else x['p_censor'], axis=1)
            in_current_subset_df = df[in_current_subset]
            print("E[censored]:", in_current_subset_df['p_hat'].mean())
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
            in_current_subset_df = df[in_current_subset]
            print("E[censored]:", in_current_subset_df['p_hat'].mean(), "E[uncensored]:", in_current_subset_df['p_hat_raw'].mean())
        
        return run_ijdi_scan(df, features, proba_confusion_col, proba_ijdi_col, outcomes_col, threshold, lambda_param, 
                             constant_threshold=constant_threshold, confusion_metrics=False, ijdi_metrics=False, verbose=verbose, p_bar='p_bar_old')
    
    print("Subset does not violate p_delta or p_censor conditions!")
    return current_subset, current_score
