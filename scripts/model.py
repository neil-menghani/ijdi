import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression

def lr_build_and_predict(df, exclude):
    
    compas_df = df.drop(columns=exclude)

    for cat_col in compas_df.select_dtypes(include=['object', 'bool']).columns:
        compas_df[cat_col] = compas_df[cat_col].astype('category')

    X_num = compas_df.select_dtypes(exclude=['category'])
    X_cat = compas_df.select_dtypes(include=['category'])

    # one-hot encoding of categorical features
    X_encoded = pd.get_dummies(X_cat)
    frames = [X_encoded, X_num]
    compas_df = pd.concat(frames, axis=1)

    extra_cols = { 'sex':'sex_Female'
                 , 'under_25': 'under_25_False'
                 , 'race': 'race_Other'
                 , 'prior_offenses':'prior_offenses_None'
                 , 'charge_degree':'charge_degree_M'
                 }
    for extra_col in extra_cols.keys():
        if extra_col not in exclude:
            compas_df = compas_df.drop(columns=extra_cols[extra_col])
    
    X = compas_df.drop(columns=['outcomes'])
    y = compas_df['outcomes']
    
    lr = LogisticRegression(random_state=0, solver='lbfgs')
    lr.fit(X, y)
    proba = lr.predict_proba(X)[:,1]
    
    return proba

def check_accuracy(df):
    # Note: model results can vary environment to environment, which can impact IJDI scores.
    df['test_outcomes'] = df['proba'].apply(lambda x : 1 if x > 0.5 else 0)
    df['check'] = df.apply(lambda x : 1 if x['outcomes'] == x['test_outcomes'] else 0, axis=1)
    accuracy = np.sum(df['check']) / len(df)
    df = df.drop(columns=['test_outcomes', 'check'])
    return accuracy