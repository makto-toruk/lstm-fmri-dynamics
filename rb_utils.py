'''
utils for bhv regression (rb)
'''
import numpy as np
import scipy as sp
import pandas as pd
import pickle

from sklearn.metrics import mean_squared_error
import torch
from torch import backends

from utils import _info, _to_cpu
from cc_utils import _get_mask

'''
SCORES:
'mse': mean squared error
'p': pearson correlation
's': spearman correlation
'''
SCORES = ['mse', 'p', 's']

def _get_t_score(y_hat, y, k_time):
    '''
    score as f(time)
    '''
    s = {}
    for score in SCORES:
        s[score] = np.zeros(k_time)
    
    for ii in range(k_time):
        y_i = y[ii::k_time]
        y_hat_i = y_hat[ii::k_time]
        s['mse'][ii] = mean_squared_error(y_hat_i, y_i)
        s['p'][ii] = np.corrcoef(y_hat_i, y_i)[0, 1]
        s['s'][ii] = sp.stats.spearmanr(y_hat_i, y_i)[0]
        
    return s

def _lstm_score(model, X, Y, C, X_len, max_length,
    clip_time, return_states=False):
    '''
    masked accuracy for lstm
    '''
    # mask to ignore padding
    mask = _get_mask(X_len, max_length)
    
    # forward pass
    if return_states:
        _, Y_HAT = model(X, X_len, max_length).squeeze(2)
    else:
        Y_HAT = model(X, X_len, max_length).squeeze(2)

    '''
    cuda variables to numpy for saving
    '''
    Y = _to_cpu(Y).numpy()
    Y_HAT = _to_cpu(Y_HAT).numpy()
    C = _to_cpu(C).numpy()
    
    # remove padded values
    # converts matrix to vec
    y_hat = Y_HAT[mask==True]
    y = Y[mask==True]
    c = C[mask==True]

    # mean scores (across all t)
    s = {}
    s['mse'] = mean_squared_error(y_hat, y)
    s['p'] = np.corrcoef(y_hat, y)[0, 1]
    s['s'] = sp.stats.spearmanr(y_hat, y)[0]

    # score as a function of t
    k_clip = len(clip_time)
    s_t = {}
    for ii in range(k_clip):
        y_i = y[c==ii]
        y_hat_i = y_hat[c==ii]
        k_time = clip_time[ii]
        s_t[ii] = _get_t_score(y_hat_i, y_i, k_time)

    return s, s_t, Y, Y_HAT, C

def _cpm_score(model, X, y, c):
    '''
    scores for cpm
    '''
    y_hat = model.predict(X)['glm']
    
    # mean scores (across all t)
    s = {}
    s['mse'] = mean_squared_error(y_hat, y)
    s['p'] = np.corrcoef(y_hat, y)[0, 1]
    s['s'] = sp.stats.spearmanr(y_hat, y)[0]

    # score as a function of t
    s_c = {}
    k_clip = len(np.unique(c))
    for ii in range(k_clip):
        y_i = y[c==ii]
        y_hat_i = y_hat[c==ii]
        s_c[ii] = {}
        s_c[ii]['mse'] = mean_squared_error(y_hat_i, y_i)
        s_c[ii]['p'] = np.corrcoef(y_hat_i, y_i)[0, 1]
        s_c[ii]['s'] = sp.stats.spearmanr(y_hat_i, y_i)[0]

    return s, s_c, y_hat