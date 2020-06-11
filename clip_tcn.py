'''
model: temporal CNN
task: predict clip (15 way classifier)
data: all runs used together
input to model: clip time series/seq
output: label time series
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
from models import TCNClassifier
'''
Helpers
'''
from utils import _info
from cc_utils import _ff_acc, _ff_test_acc
from dataloader import _get_clip_seq as _get_seq
from dataloader import _clip_class_df

# results directory
RES_DIR = 'results/clip_tcn'
if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)

K_SEED = 330
    
def _train(df, args):
    '''
    cross-validation results
    '''
    # set pytorch device
    torch.manual_seed(K_SEED)
    use_cuda = torch.cuda.is_available()
    args.device = torch.device('cuda:0' if use_cuda else 'cpu')
    if use_cuda:
        _info('cuda')
    else:
        _info('cpu')

    # get X-y from df
    subject_list = df['Subject'].unique()
    train_list = subject_list[:args.train_size]
    test_list = subject_list[args.train_size:]

    print('number of subjects = %d' %(len(subject_list)))
    features = [ii for ii in df.columns if 'feat' in ii]
    k_feat = len(features)
    print('number of features = %d' %(k_feat))
    args.k_class = len(np.unique(df['y']))
    print('number of classes = %d' %(args.k_class))
    
    # length of each clip
    clip_time = np.zeros(args.k_class)
    for ii in range(args.k_class):
        class_df = df[df['y']==ii]
        clip_time[ii] = np.max(np.unique(class_df['timepoint'])) + 1
    clip_time = clip_time.astype(int) # df saves float
    _info('seq lengths = %s' %clip_time)

    # results dict init
    results = {}
    
    # mean accuracy across time
    results['train'] = np.zeros(args.k_fold)
    results['val'] = np.zeros(args.k_fold)
    
    # confusion matrices
    results['train_conf_mtx'] = np.zeros((args.k_class, args.k_class))
    results['val_conf_mtx'] = np.zeros((args.k_class, args.k_class))
    
    # per class temporal accuracy
    results['t_train'] = {}
    results['t_val'] = {}
    for ii in range(args.k_class):
        results['t_train'][ii] = np.zeros((args.k_fold, clip_time[ii]))
        results['t_val'][ii] = np.zeros((args.k_fold, clip_time[ii]))

    i_fold = 0
    kf = KFold(n_splits=args.k_fold, random_state=K_SEED)
    
    for train, val in kf.split(train_list):

        _info('fold: %d/%d' %(i_fold+1, args.k_fold))
        
        # ***between-subject train-val split
        train_subs = [train_list[ii] for ii in train]
        val_subs = [train_list[ii] for ii in val]
        '''
        init model
        '''
        model = TCNClassifier(k_feat, args.k_hidden, 
            args.k_wind, args.k_class)
        model.to(args.device)
        print(model)
        
        lossfn = nn.CrossEntropyLoss(ignore_index=-100) 
        # if input is cuda, loss function is auto cuda
        opt = torch.optim.Adam(model.parameters())

        # get train, val sequences
        X_train, train_len, y_train = _get_seq(df, 
            train_subs, args)
        X_val, val_len, y_val = _get_seq(df, 
            val_subs, args)
        
        max_length = torch.max(train_len)
        '''
        train classifier
        '''
        permutation = torch.randperm(X_train.size()[0])
        losses = np.zeros(args.num_epochs)
        #
        then = time.time()

        for epoch in range(args.num_epochs):
            for i in range(0, X_train.size()[0], args.batch_size):
                
                indices = permutation[i:i + args.batch_size]
                batch_x, batch_y = X_train[indices], y_train[indices]
                
                y_pred = model(batch_x)
                loss = lossfn(y_pred.view(-1,args.k_class), 
                    batch_y.view(-1))
                
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            losses[epoch] = loss
        
        _info(losses)
        #
        print('--- train time =  %0.4f seconds ---' %(time.time() - then))

        '''
        results on train data
        '''
        a, a_t, c_mtx = _ff_acc(model, X_train, y_train,
            train_len, max_length, clip_time)
        results['train'][i_fold] = a
        print('tacc = %0.3f' %a)
        for ii in range(args.k_class):
            results['t_train'][ii][i_fold] = a_t[ii]
        results['train_conf_mtx'] += c_mtx
        '''
        results on val data
        '''
        a, a_t, c_mtx = _ff_acc(model, X_val, y_val,
            val_len, max_length, clip_time)
        results['val'][i_fold] = a
        print('vacc = %0.3f' %a)
        for ii in range(args.k_class):
            results['t_val'][ii][i_fold] = a_t[ii]
        results['val_conf_mtx'] += c_mtx
        
        i_fold += 1
        
    return results

def _test(df, args):
    '''
    test subject results
    view only for best cross-val parameters
    '''
    _info('test mode')
        
    # get X-y from df
    subject_list = df['Subject'].unique()
    train_list = subject_list[:args.train_size]
    test_list = subject_list[args.train_size:]

    features = [ii for ii in df.columns if 'feat' in ii]
    k_feat = len(features)
    print('number of classes = %d' %(args.k_class))
    
    # length of each clip
    clip_time = np.zeros(args.k_class)
    for ii in range(args.k_class):
        class_df = df[df['y']==ii]
        clip_time[ii] = np.max(np.unique(class_df['timepoint'])) + 1
    clip_time = clip_time.astype(int) # df saves float
    _info('seq lengths = %s' %clip_time)

    # results dict init
    results = {}
    
    # mean accuracy across time
    results['train'] = np.zeros(len(test_list))
    results['val'] = np.zeros(len(test_list))

    # per class temporal accuracy
    results['t_train'] = {}
    results['t_test'] = {}
    for ii in range(args.k_class):
        results['t_train'][ii] = np.zeros(
            (len(test_list), clip_time[ii]))
        results['t_test'][ii] = np.zeros(
            (len(test_list), clip_time[ii]))
    '''
    init model
    '''
    model = TCNClassifier(k_feat, args.k_hidden, 
        args.k_wind, args.k_class)
    model.to(args.device)
    print(model)
    
    lossfn = nn.CrossEntropyLoss(ignore_index=-100) 
    # if input is cuda, loss function is auto cuda
    opt = torch.optim.Adam(model.parameters())

    # get train, val sequences
    X_train, train_len, y_train = _get_seq(df, 
        train_list, args)
    X_test, test_len, y_test = _get_seq(df, 
        test_list, args)
    
    max_length = torch.max(train_len)
    '''
    train classifier
    '''
    permutation = torch.randperm(X_train.size()[0])
    losses = np.zeros(args.num_epochs)
    #
    then = time.time()

    for epoch in range(args.num_epochs):
        for i in range(0, X_train.size()[0], args.batch_size):
            
            indices = permutation[i:i + args.batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            
            y_pred = model(batch_x)
            loss = lossfn(y_pred.view(-1,args.k_class), 
                batch_y.view(-1))
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        losses[epoch] = loss
    
    _info(losses)
    #
    print('--- train time =  %0.4f seconds ---' %(time.time() - then))

    '''
    results on train data
    '''
    a, a_t, c_mtx = _ff_test_acc(model, X_train, y_train,
        train_len, max_length, clip_time, len(train_list))
    results['train'] = a
    print('tacc = %0.3f' %np.mean(a))
    for ii in range(args.k_class):
        results['t_train'][ii] = a_t[ii]
    results['train_conf_mtx'] = c_mtx
    '''
    results on test data
    '''
    a, a_t, c_mtx = _ff_test_acc(model, X_test, y_test,
        test_len, max_length, clip_time, len(test_list))
    results['test'] = a
    print('sacc = %0.3f' %np.mean(a))
    for ii in range(args.k_class):
        results['t_test'][ii] = a_t[ii]
    results['test_conf_mtx'] = c_mtx
            
    return results

def run(args):   
    
    _info(args.subnet)
    '''
    get dataframe
    '''
    # all-but-subnetwork (invert_flag)
    if 'minus' in args.subnet:
        args.invert_flag = True

    res_path = (RES_DIR + 
        '/roi_%d_net_%d' %(args.roi, args.net) + 
        '_nw_%s' %(args.subnet) +
        '_trainsize_%d' %(args.train_size) +
        '_kfold_%d_k_hidden_%d' %(args.k_fold, args.k_hidden) +
        '_k_wind_%d_batch_size_%d' %(args.k_wind, args.batch_size) +
        '_num_epochs_%d_z_%d.pkl' %(args.num_epochs, args.zscore))
    if not os.path.isfile(res_path):
        df = _clip_class_df(args)
        results = {}
        results['train_mode'] = _train(df, args)
        results['test_mode'] = _test(df, args)
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
    
    # preprocessing
    parser.add_argument('--zscore', type=int,
        default=1, help='zscore = 1 or 0')

    # training parameters
    parser.add_argument('-k', '--k_fold', type=int,
        default=10, help='number of folds for cross validation')
    parser.add_argument('--k_hidden', type=int,
        default=150, help='size of hidden state')
    parser.add_argument('--k_wind', type=int,
        default=30, help='length of temporal window')
    parser.add_argument('--batch_size', type=int,
        default=16, help='batch size for training')
    parser.add_argument('--num_epochs', type=int,
        default=50, help='no. of epochs for training')
    parser.add_argument('--train_size', type=int,
        default=100, help='number of participants in training data')
    
    args = parser.parse_args()
    
    run(args)

    print('finished!')