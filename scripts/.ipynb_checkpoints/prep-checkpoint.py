import pandas as pd, numpy as np

def set_edge(x):
    if x < 0:
        return 0
    elif x > 1:
        return 1
    else:
        return x

# function to generate metrics from probabilities, outcomes, and threshold
# if constant_threshold=True pass in a constant otherwise pass in a column name
def generate_metrics(df, proba_col, outcomes_col, threshold, constant_threshold=True):
    
    if constant_threshold:
        df['threshold'] = threshold
        threshold_col = 'threshold'
    else:
        threshold_col = threshold
        
    df['errorprobs'] = np.minimum(df[proba_col],1.0-df[proba_col])
    df['positives'] = 1*(df[proba_col] > df[threshold_col])
    df['positives_constprobs'] = pd.Series(df['positives'].mean(),index=df.index)
    df['falsepositives'] = (df['positives'] * (1-df[outcomes_col]))
    df['falsepositives_constprobs'] = pd.Series(df['falsepositives'].mean(),index=df.index)
    df['falsenegatives'] = ((1-df['positives']) * df[outcomes_col])
    df['falsenegatives_constprobs'] = pd.Series(df['falsenegatives'].mean(),index=df.index)
    df['errors'] = df['falsepositives'] + df['falsenegatives']
    df['errors_constprobs'] = pd.Series(df['errors'].mean(),index=df.index)
    
    if constant_threshold:
        df = df.drop(columns=['threshold'])
        
    return df

# function to generate metrics required for ijdi scan
def generate_ijdi_metrics(df, proba_col, lambda_param, p_bar='p_bar'):
    df['p_bar'] = pd.Series(df[proba_col].mean(), index=df.index)
    # either use p_bar column calculated here or original p_bar column specified in function
    df['p_delta'] = df[proba_col] - df[p_bar]
    df['p_hat_raw'] = df['positives_constprobs'] + lambda_param*(df['p_delta'])
    df['p_hat'] = df['p_hat_raw'].apply(lambda x : set_edge(x))
    df['p_censor'] = df['p_hat'] - df['p_hat_raw']
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