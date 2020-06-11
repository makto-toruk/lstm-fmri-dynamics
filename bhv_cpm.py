'''
model: rnn (lstm)
task: predict behavioral score (at every t***: predict for each hidden state)
data: all clips used together
behavioral measures: see notebook
'''
import numpy as np
import pandas as pd
import pickle
import os
import argparse
import time
'''
ml
'''
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from cpm import cpm
'''
Helpers
'''
from utils import _info
from rb_utils import _cpm_score
from dataloader import _bhv_class_df as _bhv_reg_df
from dataloader import _get_bhv_cpm_seq as _get_seq

# results directory
RES_DIR = 'results/bhv_cpm'
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)

K_SEED = 330
'''
SCORES:
'mse': mean squared error
'p': pearson correlation
's': spearman correlation
'''
SCORES = ['mse', 'p', 's']

def _train(df, bhv_df, args):
    
    # get X-y from df   
    features = [ii for ii in df.columns if 'feat' in ii]
    k_feat = len(features)
    print('number of features = %d' %(k_feat))
    k_clip = len(np.unique(df['c']))
    print('number of clips = %d' %(k_clip))

    # init dict for all results
    results = {}
    
    # true and predicted scores and clip label
    results['y'] = {}
    results['y_hat'] = {}
    results['c'] = {}

    for score in SCORES:

        # mean scores across time
        results['train_%s'%score] = np.zeros(args.k_fold)
        results['val_%s'%score] = np.zeros(args.k_fold)
        
        # per clip temporal score
        results['c_train_%s'%score] = {}
        results['c_val_%s'%score] = {}

        for ii in range(k_clip):
            results['c_train_%s'%score][ii] = np.zeros(args.k_fold)
            results['c_val_%s'%score][ii] = np.zeros(args.k_fold)

    kf = KFold(n_splits=args.k_fold, random_state=K_SEED)
    
    # get participant lists for each assigned class
    class_list = {}
    for ii in range(args.k_class):
        class_list[ii] = bhv_df[bhv_df['y']==ii]['Subject'].values
    '''    
    split participants in each class with kf
    nearly identical ratio of train and val,
    in all classes
    '''
    split = {}
    for ii in range(args.k_class):
        split[ii] = kf.split(class_list[ii])
        
    for i_fold in range(args.k_fold):
        
        _info('fold: %d/%d' %(i_fold+1, args.k_fold))

        # ***between-subject train-val split
        train_subs, val_subs = [], []
        for ii in range(args.k_class):
            train, val = next(split[ii])
            for jj in train:
                train_subs.append(class_list[ii][jj])
            for jj in val:
                val_subs.append(class_list[ii][jj])     
        '''
        model main
        '''
        model = cpm(corr_thresh=args.corr_thresh)

        X_train, y_train, c_train = _get_seq(df, 
            train_subs, args)
        X_val, y_val, c_val = _get_seq(df, 
            val_subs, args)
        '''
        train regression model
        '''
        _, _ = model.fit(X_train, y_train)
        '''
        results on train data
        '''
        s, s_c, _ = _cpm_score(model, X_train, y_train, c_train)
        for score in SCORES:
            results['train_%s'%score][i_fold] = s[score]
            for ii in range(k_clip):
                results['c_train_%s'%score][ii][i_fold] = s_c[ii][score]
        print('train p = %0.3f' %s['p'])
        '''
        results on val data
        '''
        s, s_c, y_hat = _cpm_score(model, X_val, y_val, c_val)
        for score in SCORES:
            results['val_%s'%score][i_fold] = s[score]
            for ii in range(k_clip):
                results['c_val_%s'%score][ii][i_fold] = s_c[ii][score]
        print('val p = %0.3f' %s['p'])
        
        results['y'][i_fold] = y_val
        results['y_hat'][i_fold] = y_hat
        results['c'][i_fold] = c_val
        
    return results

def run(args):   
    
    _info(args.bhv)
    _info(args.subnet)
    # set to regression mode
    args.mode = 'reg'
    '''
    get dataframe
    '''
    # all-but-subnetwork (invert_flag)
    if 'minus' in args.subnet:
        args.invert_flag = True
    
    res_path = (RES_DIR + 
        '/roi_%d_net_%d' %(args.roi, args.net) + 
        '_nw_%s' %(args.subnet) +
        '_bhv_%s_cutoff_%0.1f' %(args.bhv, args.cutoff) +
        '_corrthresh_%0.1f' %(args.corr_thresh) +
        '_kfold_%d_z_%d.pkl' %(args.k_fold, args.zscore))
    if not os.path.isfile(res_path):
        df, bhv_df = _bhv_reg_df(args)
        results = _train(df, bhv_df, args)
        with open(res_path, 'wb') as f:
            pickle.dump(results, f)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')

    # data parameters
    parser.add_argument('-d', '--input-data', type=str,
        default='data/roi_ts', help='path/to/roi/data')
    parser.add_argument('-r', '--roi', type=int,
        default=300, help='number of ROI')
    parser.add_argument('-n', '--net', type=int,
        default=7, help='number of networks (7 or 17)')
    parser.add_argument('--subnet', type=str,
        default='wb', help='name of subnetwork')
    
    # behavioral parameters
    parser.add_argument('-b', '--bhv', type=str,
        default='ListSort_Unadj',
        help='behavioral measure: PMAT24_A_CR, ListSort_Unadj, ... ')
    parser.add_argument('--cutoff', type=float,
        default=0.1, help='percentile of subjects')
    
    # preprocessing
    parser.add_argument('--zscore', type=int,
        default=1, help='zscore = 1 or 0')

    # training parameters
    parser.add_argument('-k', '--k_fold', type=int,
        default=10, help='number of folds for cross validation')
    parser.add_argument('--corr_thresh', type=float,
        default=0.2, help='fc edge p thresh')
    
    args = parser.parse_args()

    run(args)

    print('finished!')