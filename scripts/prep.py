import pandas as pd, numpy as np
import random

def set_edge(x):
    if x < 0:
        return 0
    elif x > 1:
        return 1
    else:
        return x

# function to generate metrics from probabilities, outcomes, and threshold
# if constant_threshold=True pass in a constant otherwise pass in a column name
def generate_metrics(df, proba_col, outcomes_col, threshold, constant_threshold=True, recalc_subset=[]):
    
    if constant_threshold:
        df['threshold'] = threshold
        threshold_col = 'threshold'
    else:
        threshold_col = threshold
        
    df['errorprobs'] = calculate_metric(df, 'errorprobs', np.minimum(df[proba_col],1.0-df[proba_col]), recalc_subset)
    df['positives'] = calculate_metric(df, 'positives', 1*(df[proba_col] > df[threshold_col]), recalc_subset)
    df['positives_constprobs'] = calculate_metric(df, 'positives_constprobs', pd.Series(df['positives'].mean(),index=df.index), recalc_subset)
    df['falsepositives'] = calculate_metric(df, 'falsepositives', (df['positives'] * (1-df[outcomes_col])), recalc_subset)
    df['falsepositives_constprobs'] = calculate_metric(df, 'falsepositives_constprobs', pd.Series(df['falsepositives'].mean(),index=df.index), recalc_subset)
    df['falsenegatives'] = calculate_metric(df, 'falsenegatives', ((1-df['positives']) * df[outcomes_col]), recalc_subset)
    df['falsenegatives_constprobs'] = calculate_metric(df, 'falsenegatives_constprobs', pd.Series(df['falsenegatives'].mean(),index=df.index), recalc_subset)
    df['errors'] = calculate_metric(df, 'errors', df['falsepositives'] + df['falsenegatives'], recalc_subset)
    df['errors_constprobs'] = calculate_metric(df, 'errors_constprobs', pd.Series(df['errors'].mean(),index=df.index), recalc_subset)
    
    if constant_threshold:
        df = df.drop(columns=['threshold'])
        
    return df

# function to generate metrics required for ijdi scan
def generate_ijdi_metrics(df, proba_col, lambda_param, p_bar='p_bar', recalc_subset=[]):
    df['p_bar'] = calculate_metric(df, 'p_bar', pd.Series(df[proba_col].mean(), index=df.index), recalc_subset)
    # either use p_bar column calculated here or original p_bar column specified in function
    df['p_delta'] = calculate_metric(df, 'p_delta', df[proba_col] - df[p_bar], recalc_subset)
    df['p_hat_raw'] = calculate_metric(df, 'p_hat_raw', df['positives_constprobs'] + lambda_param*(df['p_delta']), recalc_subset)
    df['p_hat'] = calculate_metric(df, 'p_hat', df['p_hat_raw'].apply(lambda x : set_edge(x)), recalc_subset)
    df['p_censor'] = calculate_metric(df, 'p_censor', df['p_hat'] - df['p_hat_raw'], recalc_subset)
    return df

# function to generate probabilities from a uniform distribution
def generate_proba(x, subgroup_mu, non_subgroup_mu, k):
    if x:
        return np.random.uniform(subgroup_mu-0.01*k, subgroup_mu+0.01*k)
    else:
        return np.random.uniform(non_subgroup_mu-0.01*k, non_subgroup_mu+0.01*k)

# function to generate outcomes from a bernoulli distribution
def generate_outcomes(x):
    return int(np.random.uniform() < x)

# function to calculate the required shift in probability for a specified shift in log-odds
def calculate_proba_shift(proba, epsilon):
    return 1 / (1 + ((1 - proba) / (proba * (np.e ** epsilon))))

# remove dataframe from memory
def release_df(x_df):
    x_list = [x_df]
    del x_list
    return

# get truth series for subset
def get_subset_series(df, subset):
    if subset:
        return df[list(subset.keys())].isin(subset).all(axis=1)
    else:
        return pd.Series(True, index=df.index)

def calculate_metric(df, col_name, formula, recalc_subset=[]):
    if recalc_subset:
        df['in_recalc_subset'] = get_subset_series(df, recalc_subset)
        df['formula'] = formula
        return df.apply(lambda x : x['formula'] if x['in_recalc_subset'] else x[col_name], axis=1)
    else:
        return formula

def pick_random_protected_group(df, cols):
    df_s = df[cols]
    unique_values = []
    for i in list(df_s):
        for k in df_s[i].unique():
            unique_values.append((i,k))
    key, key_value = random.choice(unique_values)
    return key, key_value

def pick_random_subgroup(df, cols):
    df_s = df[cols]
    subgroup = {}
    while subgroup == {}:
        for i in cols:
            subset_values = []
            while len(subset_values) == 0:
                unique_values = df_s[i].unique()
                subset_values = [x for x in unique_values if random.choice((True, False))]
            if len(unique_values) != len(subset_values):
                subgroup[i] = list(subset_values)
    return subgroup