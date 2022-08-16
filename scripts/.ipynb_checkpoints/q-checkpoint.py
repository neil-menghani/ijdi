import pandas as pd, numpy as np

# this actually computes q times the slope, which has the same sign as the slope
def compute_slopeterm_given_q(thesum,theprobs,q):
    return thesum-theprobs.apply(lambda x: q*x/(1-x+q*x)).sum()

def binary_search_on_slopeterm(thesum,theprobs):
    q_temp_min = 0.000001
    q_temp_max = 1000000.0
    while np.abs(q_temp_max-q_temp_min) > 0.000001:
        q_temp_mid = (q_temp_min+q_temp_max)/2
        if np.sign(compute_slopeterm_given_q(thesum,theprobs,q_temp_mid)) > 0:
            q_temp_min = q_temp_min+(q_temp_max-q_temp_min)/2
        else:
            q_temp_max = q_temp_max-(q_temp_max-q_temp_min)/2
    return (q_temp_min+q_temp_max)/2
    
# penalty should be >0 and is subtracted from the score
def compute_score_given_q(thesum,theprobs,penalty,q):
    if (q <= 0):
        print("Warning: calling compute_score_given_q with thesum=",thesum,"theprobs of length",len(theprobs),"penalty=",penalty,"q=",q)
    return thesum*np.log(q)-np.log(1-theprobs+q*theprobs).sum() - penalty

def binary_search_on_score_for_q_min(thesum,theprobs,penalty,q_mle):
    q_temp_min = 0.000001
    q_temp_max = q_mle
    while np.abs(q_temp_max-q_temp_min) > 0.000001:
        q_temp_mid = (q_temp_min+q_temp_max)/2
        if np.sign(compute_score_given_q(thesum,theprobs,penalty,q_temp_mid)) > 0:
            q_temp_max = q_temp_max-(q_temp_max-q_temp_min)/2
        else:
            q_temp_min = q_temp_min+(q_temp_max-q_temp_min)/2
    return (q_temp_min+q_temp_max)/2

def binary_search_on_score_for_q_max(thesum,theprobs,penalty,q_mle):
    q_temp_min = q_mle
    q_temp_max = 1000000.0
    while np.abs(q_temp_max-q_temp_min) > 0.000001:
        q_temp_mid = (q_temp_min+q_temp_max)/2
        if np.sign(compute_score_given_q(thesum,theprobs,penalty,q_temp_mid)) > 0:
            q_temp_min = q_temp_min+(q_temp_max-q_temp_min)/2
        else:
             q_temp_max = q_temp_max-(q_temp_max-q_temp_min)/2
    return (q_temp_min+q_temp_max)/2

# q_mle = argmax_q thesum*\ln(q) - \sum_{p_i \in theprobs} \ln(1-p_i+qp_i) + penalty 
# [q_min, q_max] = q: thesum*\ln(q) - \sum_{p_i \in theprobs} \ln(1-p_i+qp_i) + penalty = 0, if these exist
# positive = 1 if q_min, q_max exist, 0 otherwise
def compute_q(thesum,theprobs,penalty):
    q_mle = binary_search_on_slopeterm(thesum,theprobs)
    if compute_score_given_q(thesum,theprobs,penalty,q_mle) > 0:
        positive = 1
        q_min = binary_search_on_score_for_q_min(thesum,theprobs,penalty,q_mle)
        q_max = binary_search_on_score_for_q_max(thesum,theprobs,penalty,q_mle)
    else:
        positive = 0
        q_min = 0
        q_max = 0
    return positive, q_mle, q_min, q_max