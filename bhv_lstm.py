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
from models import LSTMRegression
'''
Helpers
'''
from utils import _info
from rb_utils import _lstm_score, _get_mask
from dataloader import _bhv_class_df as _bhv_reg_df
from dataloader import _get_bhv_seq as _get_seq

# results directory
RES_DIR = 'results/bhv_lstm'
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
    
    # set pytorch device
    torch.manual_seed(K_SEED)
    use_cuda = torch.cuda.is_available()
    args.device = torch.device('cuda:0' if use_cuda else 'cpu')
    if use_cuda:
        _info('cuda')
    else:
        _info('cpu')

    # get X-y from df   
    features = [ii for ii in df.columns if 'feat' in ii]
    k_feat = len(features)
    print('number of features = %d' %(k_feat))
    k_clip = len(np.unique(df['c']))
    print('number of clips = %d' %(k_clip))

    # length of each clip
    clip_time = np.zeros(k_clip)
    for ii in range(k_clip):
        class_df = df[df['c']==ii]
        clip_time[ii] = np.max(np.unique(class_df['timepoint'])) + 1
    clip_time = clip_time.astype(int) # df saves float
    _info('seq lengths = %s' %clip_time)

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
        results['t_train_%s'%score] = {}
        results['t_val_%s'%score] = {}
    
        for ii in range(k_clip):
            results['t_train_%s'%score][ii] = np.zeros(
                (args.k_fold, clip_time[ii]))
            results['t_val_%s'%score][ii] = np.zeros(
                (args.k_fold, clip_time[ii]))
        
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
        model = LSTMRegression(k_feat, args.k_hidden, 
            args.k_layers)
        model.to(args.device)
        print(model)

        lossfn = nn.MSELoss()
        # if input is cuda, loss function is auto cuda
        opt = torch.optim.Adam(model.parameters())

        # get train, val sequences
        X_train, train_len, y_train, c_train = _get_seq(df, 
            train_subs, args)
        X_val, val_len, y_val, c_val = _get_seq(df, 
            val_subs, args)

        max_length = torch.max(train_len)

        '''
        train regression model
        '''
        permutation = torch.randperm(X_train.size()[0])
        losses = np.zeros(args.num_epochs)
        #
        then = time.time()
            
        for epoch in range(args.num_epochs):
            for i in range(0, X_train.size()[0], args.batch_size):

                indices = permutation[i:i + args.batch_size]
                batch_x, batch_y = X_train[indices], y_train[indices]
                batch_x_len = train_len[indices]
                batch_mask = torch.BoolTensor(
                    _get_mask(batch_x_len, max_length)).to(args.device)
                
                y_pred = model(batch_x, batch_x_len, max_length).squeeze(2)
                loss = lossfn(y_pred[batch_mask==True], 
                    batch_y[batch_mask==True])
                    
                opt.zero_grad()
                loss.backward()
                opt.step()

            losses[epoch] = loss

        _info(losses)
        #
        print('--- train time =  %0.4f seconds ---' %(time.time() - then))
        model.eval()
        '''
        results on train data
        '''
        s, s_t, _, _, _ = _lstm_score(model, X_train, y_train,
            c_train, train_len, max_length, clip_time)
        for score in SCORES:
            results['train_%s'%score][i_fold] = s[score]
            for ii in range(k_clip):
                results['t_train_%s'%score][ii][i_fold] = s_t[ii][score]
        print('train p = %0.3f' %s['p'])
        '''
        results on val data
        '''
        s, s_t, y, y_hat, c = _lstm_score(model, X_val, y_val,
            c_val, val_len, max_length, clip_time)
        for score in SCORES:
            results['val_%s'%score][i_fold] = s[score]
            for ii in range(k_clip):
                results['t_val_%s'%score][ii][i_fold] = s_t[ii][score]
        print('val p = %0.3f' %s['p'])
        
        results['y'][i_fold] = y
        results['y_hat'][i_fold] = y_hat
        results['c'][i_fold] = c
        
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
        '_kfold_%d_k_hidden_%d' %(args.k_fold, args.k_hidden) +
        '_k_layers_%d_batch_size_%d' %(args.k_layers, args.batch_size) +
        '_num_epochs_%d_z_%d.pkl' %(args.num_epochs, args.zscore))
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
    parser.add_argument('-f', '--cutoff', type=float,
        default=0.1, help='percent of total participants')
    
    # preprocessing
    parser.add_argument('--zscore', type=int,
        default=1, help='zscore = 1 or 0')

    # training parameters
    parser.add_argument('-k', '--k_fold', type=int,
        default=10, help='number of folds for cross validation')
    parser.add_argument('--k_hidden', type=int,
        default=150, help='size of hidden state')
    parser.add_argument('--k_layers', type=int,
        default=1, help='no. of lstm layers')
    parser.add_argument('--batch_size', type=int,
        default=16, help='batch size for training')
    parser.add_argument('--num_epochs', type=int,
        default=20, help='no. of epochs for training')
    
    args = parser.parse_args()

    run(args)

    print('finished!')