import numpy as np
import pandas as pd
import random
import pickle

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
from torch.nn.utils.rnn import pad_sequence
from utils import _get_parcel, _get_behavioral
from cc_utils import _get_clip_labels

K_RUNS = 4
K_SEED = 330

def _get_clip_seq(df, subject_list, args):
    '''
    return:
    X: input seq (batch_size x time x feat_size)
    y: label seq (batch_size x time)
    X_len: len of each seq (batch_size x 1)
    batch_size <-> number of sequences
    time <-> max length after padding
    '''
    features = [ii for ii in df.columns if 'feat' in ii]
    
    X = []
    y = []
    for subject in subject_list:
        for i_class in range(args.k_class):
            
            if i_class==0: # split test-retest into 4
                seqs = df[(df['Subject']==subject) & 
                    (df['y'] == 0)][features].values
                label_seqs = df[(df['Subject']==subject) & 
                    (df['y'] == 0)]['y'].values

                k_time = int(seqs.shape[0]/K_RUNS)
                for i_run in range(K_RUNS):
                    seq = seqs[i_run*k_time:(i_run+1)*k_time, :]
                    label_seq = label_seqs[i_run*k_time:(i_run+1)*k_time]
                    if args.zscore:
                        # zscore each seq that goes into model
                        seq = (1/np.std(seq))*(seq - np.mean(seq))

                    X.append(torch.FloatTensor(seq))
                    y.append(torch.LongTensor(label_seq))
            else:
                seq = df[(df['Subject']==subject) & 
                    (df['y'] == i_class)][features].values
                label_seq = df[(df['Subject']==subject) & 
                    (df['y'] == i_class)]['y'].values
                if args.zscore:
                    # zscore each seq that goes into model
                    seq = (1/np.std(seq))*(seq - np.mean(seq))
                
                X.append(torch.FloatTensor(seq))
                y.append(torch.LongTensor(label_seq))
            
    X_len = torch.LongTensor([len(seq) for seq in X])

    # pad sequences
    X = pad_sequence(X, batch_first=True, padding_value=0)
    y = pad_sequence(y, batch_first=True, padding_value=-100)
            
    return X.to(args.device), X_len.to(args.device), y.to(args.device)

def _clip_class_df(args):
    '''
    data for 15-way clip classification

    args.roi: number of ROIs
    args.net: number of subnetworks (7 or 17)
    args.subnet: subnetwork; 'wb' if all subnetworks
    args.invert_flag: all-but-one subnetwork
    args.r_roi: number of random ROIs to pick
    args.r_seed: random seed for picking ROIs

    save each timepoint as feature vector
    append class label based on clip

    return:
    pandas df
    '''
    # optional arguments
    d = vars(args)
    if 'invert_flag' not in d:
        args.invert_flag = False
    if 'r_roi' not in d:
        args.r_roi = 0
        args.r_seed = 0

    load_path = (args.input_data + '/data_MOVIE_runs_' +
        'roi_%d_net_%d_ts.pkl' %(args.roi, args.net))

    with open(load_path, 'rb') as f:
        data = pickle.load(f)
        
    # where are the clips within the run?
    timing_file = pd.read_csv('data/videoclip_tr_lookup.csv')
    
    # pick either all ROIs or subnetworks
    if args.subnet!='wb':      
        if 'minus' in args.subnet:
            # remove 'minus_' prefix
            args.subnet = args.subnet.split('minus_')[1]

        _, nw_info = _get_parcel(args.roi, args.net)
        # ***roi ts sorted in preprocessing
        nw_info = np.sort(nw_info)
        idx = (nw_info == args.subnet)
    else:
        idx = np.ones(args.roi).astype(bool)

    # all-but-one subnetwork
    if args.subnet and args.invert_flag:
        idx = ~idx

    # if random selection,
    # overwrite everything above
    if args.r_roi > 0:
        random.seed(args.r_seed)
        idx = np.zeros(args.roi).astype(bool)
        # random sample without replacement
        samp = random.sample(range(args.roi), k=args.r_roi)
        idx[samp] = True
    '''
    main
    '''
    clip_y = _get_clip_labels()
    
    table = []
    for run in range(K_RUNS):
        
        print('loading run %d/%d' %(run+1, K_RUNS))
        run_name = 'MOVIE%d' %(run+1) #MOVIEx_7T_yz

        # timing file for run
        timing_df = timing_file[
            timing_file['run'].str.contains(run_name)]  
        timing_df = timing_df.reset_index(drop=True)

        for subject in data:

            # get subject data (time x roi x run)
            roi_ts = data[subject][:, idx, run]

            for jj, clip in timing_df.iterrows():

                start = int(np.floor(clip['start_tr']))
                stop = int(np.ceil(clip['stop_tr']))
                clip_length = stop - start
                
                # assign label to clip
                y = clip_y[clip['clip_name']]

                for t in range(clip_length):
                    act = roi_ts[t + start, :]
                    t_data = {}
                    t_data['Subject'] = subject
                    t_data['timepoint'] = t
                    for feat in range(roi_ts.shape[1]):
                        t_data['feat_%d' %(feat)] = act[feat]
                    t_data['y'] = y
                    table.append(t_data)

    df = pd.DataFrame(table)
    df['Subject'] = df['Subject'].astype(int)
        
    return df

def _get_bhv_seq(df, subject_list, args):
    '''
    return:
    X: input seq (batch_size x time x feat_size)
    y: label seq (batch_size x time)
        in {0, 1, ..} if args.mode=='class'
        in R if args.mode=='reg'
    c: clip seq (batch_size x time)
    X_len: len of each seq (batch_size x 1)
    batch_size <-> number of sequences
    time <-> max length after padding
    '''
    # optional arguments
    d = vars(args)

    # regression or classification
    if 'mode' not in d:
        args.mode = 'class'
    if args.mode=='class':
        label = 'y'
    elif args.mode=='reg':
        label = args.bhv

    # permutation test
    if 'shuffle' not in d:
        args.shuffle = False
    if args.shuffle:
        # different shuffle for each iteration
        np.random.seed(args.i_seed)
        # get scores for all participants without bhv_df
        train_label = df[(df['Subject'].isin(subject_list)) &
            (df['c']==1) & (df['timepoint']==0)][label].values
        np.random.shuffle(train_label) # inplace

    k_clip = len(np.unique(df['c']))
    features = [ii for ii in df.columns if 'feat' in ii]
    
    X = []
    y = []
    c = []
    for ii, subject in enumerate(subject_list):
        for i_clip in range(k_clip):
            
            if i_clip==0: #handle test retest differently
                seqs = df[(df['Subject']==subject) & 
                    (df['c'] == 0)][features].values
                if args.shuffle:
                    label_seqs = np.ones(seqs.shape[0])*train_label[ii]
                else:
                    label_seqs = df[(df['Subject']==subject) & 
                        (df['c'] == 0)][label].values
                clip_seqs = df[(df['Subject']==subject) & 
                    (df['c'] == 0)]['c'].values

                k_time = int(seqs.shape[0]/K_RUNS)
                for i_run in range(K_RUNS):
                    seq = seqs[i_run*k_time:(i_run+1)*k_time, :]
                    label_seq = label_seqs[i_run*k_time:(i_run+1)*k_time]
                    clip_seq = clip_seqs[i_run*k_time:(i_run+1)*k_time]
                    if args.zscore:
                        # zscore each seq that goes into model
                        seq = (1/np.std(seq))*(seq - np.mean(seq))

                    X.append(torch.FloatTensor(seq))
                    if args.mode=='class':
                        y.append(torch.LongTensor(label_seq))
                    elif args.mode=='reg':
                        y.append(torch.FloatTensor(label_seq))
                    c.append(torch.LongTensor(clip_seq))
            else:
                seq = df[(df['Subject']==subject) & 
                    (df['c'] == i_clip)][features].values
                if args.shuffle:
                    label_seq = np.ones(seq.shape[0])*train_label[ii]
                else:
                    label_seq = df[(df['Subject']==subject) & 
                        (df['c'] == i_clip)][label].values
                clip_seq = df[(df['Subject']==subject) & 
                    (df['c'] == i_clip)]['c'].values
                if args.zscore:
                    # zscore each seq that goes into model
                    seq = (1/np.std(seq))*(seq - np.mean(seq))
                
                X.append(torch.FloatTensor(seq))
                if args.mode=='class':
                    y.append(torch.LongTensor(label_seq))
                elif args.mode=='reg':
                    y.append(torch.FloatTensor(label_seq))
                c.append(torch.LongTensor(clip_seq))

    X_len = torch.LongTensor([len(seq) for seq in X])

    # pad sequences
    X = pad_sequence(X, batch_first=True, padding_value=0)
    y = pad_sequence(y, batch_first=True, padding_value=-100)
    c = pad_sequence(c, batch_first=True, padding_value=-100)
            
    return (X.to(args.device), X_len.to(args.device), 
        y.to(args.device), c.to(args.device))

def _group_bhv_df(args, subject_list):
    '''
    based on behavioral score,
    group participants into clusters
    if k_class==2:
        group top cutoff and bot cutoff
    if k_class > 2:
        use k_means for grouping
    return:
    if args.mode=='class'
        bhv_df: ['Subject', bhv, 'y']
    if args.mode=='reg'
        bhv_df: ['Subject', bhv, 'y']

    *** return 'y' in reg mode
    for kfold balancing
    '''
    # for kfold balancing
    if args.mode=='reg':
        args.k_class = 2

    # get behavioral data for subject_list
    bhv_df = _get_behavioral(subject_list)
    bhv_df = bhv_df[['Subject', args.bhv]]

    '''
    ***normalize bhv scores
    must be explicitly done for pytorch
    '''
    b = bhv_df[args.bhv].values
    bhv_df[args.bhv] = (b - np.min(b))/(np.max(b) - np.min(b))

    # reduce subjects by picking top and bottom 'cutoff' percent
    _x = np.sort(bhv_df[args.bhv].values)
    percentile = int(np.floor(args.cutoff*len(subject_list)))

    bot_cut = _x[percentile]
    top_cut = _x[-percentile]

    bhv_df = bhv_df[(bhv_df[args.bhv] >= top_cut) |
        (bhv_df[args.bhv] <= bot_cut)]
    
    '''
    behavioral groups: into 'k_class'
    '''
    if args.k_class > 2:
        _x = bhv_df[[args.bhv]].values
        model = KMeans(n_clusters=args.k_class, 
            random_state=K_SEED)
        y = model.fit_predict(_x)
        # each participant assigned a label
        bhv_df['y'] = y
    else:
        b = bhv_df[args.bhv].values
        y = [1 if ii>=top_cut else 0 for ii in b]
        bhv_df['y'] = np.array(y)

    return bhv_df

def _bhv_class_df(args):
    '''
    data for k_class bhv classification
    *** used for both classification and regression
    args.mode: 'class' or bhv'

    args.roi: number of ROIs
    args.net: number of subnetworks (7 or 17)
    args.subnet: subnetwork; 'wb' if all subnetworks
    args.bhv: behavioral measure
    args.k_class: number of behavioral groups
    args.cutoff: percentile for participant cutoff
    args.invert_flag: all-but-one subnetwork

    save each timepoint as feature vector
    append 'c' based on clip
    append 'y' based on behavioral group
    '''
    # optional arguments
    d = vars(args)
    if 'invert_flag' not in d:
        args.invert_flag = False
    if 'mode' not in d:
        args.mode = 'class'

    load_path = (args.input_data + '/data_MOVIE_runs_' +
        'roi_%d_net_%d_ts.pkl' %(args.roi, args.net))

    with open(load_path, 'rb') as f:
        data = pickle.load(f)

    subject_list = np.sort(list(data.keys()))
    bhv_df = _group_bhv_df(args, subject_list)

    cutoff_list = bhv_df['Subject'].values.astype(str)

    # where are the clips within the run?
    timing_file = pd.read_csv('data/videoclip_tr_lookup.csv')
    
    # pick either all ROIs or subnetworks
    if args.subnet!='wb':      
        if 'minus' in args.subnet:
            # remove 'minus_' prefix
            args.subnet = args.subnet.split('minus_')[1]

        _, nw_info = _get_parcel(args.roi, args.net)
        # ***roi ts sorted in preprocessing
        nw_info = np.sort(nw_info)
        idx = (nw_info == args.subnet)
    else:
        idx = np.ones(args.roi).astype(bool)

    # all-but-one subnetwork
    if args.subnet and args.invert_flag:
        idx = ~idx

    '''
    main
    '''
    clip_y = _get_clip_labels()
    
    table = []
    for run in range(K_RUNS):
        
        print('loading run %d/%d' %(run+1, K_RUNS))
        run_name = 'MOVIE%d' %(run+1) #MOVIEx_7T_yz

        # timing file for run
        timing_df = timing_file[
            timing_file['run'].str.contains(run_name)]  
        timing_df = timing_df.reset_index(drop=True)

        for subject in data:
            if subject in cutoff_list:
                # get subject data (time x roi x run)
                roi_ts = data[subject][:, idx, run]

                for jj, clip in timing_df.iterrows():

                    start = int(np.floor(clip['start_tr']))
                    stop = int(np.ceil(clip['stop_tr']))
                    clip_length = stop - start
                    
                    # assign label to clip
                    c = clip_y[clip['clip_name']]

                    for t in range(clip_length):
                        act = roi_ts[t + start, :]
                        t_data = {}
                        t_data['Subject'] = subject
                        t_data['timepoint'] = t
                        for feat in range(roi_ts.shape[1]):
                            t_data['feat_%d' %(feat)] = act[feat]
                        t_data['c'] = c
                        table.append(t_data)

    df = pd.DataFrame(table)
    df['Subject'] = df['Subject'].astype(int)
    # merges on all subject rows!
    df = df.merge(bhv_df, on='Subject', how='inner')
    
    return df, bhv_df

def _get_bhv_cpm_seq(data_df, subject_list, args):
    '''
    return:
    X: input seq (batch_size x (FC_size))
    y: label seq (batch_size)
        in {0, 1, ..} if args.mode=='class'
        in R if args.mode=='reg'
    c: clip seq (batch_size)
    X_len: len of each seq (batch_size x 1)
    batch_size <-> number of sequences
    time <-> max length after padding
    '''
    # optional arguments
    d = vars(args)
    if 'mode' not in d:
        args.mode = 'class'
        
    k_clip = len(np.unique(data_df['c']))
    features = [ii for ii in data_df.columns if 'feat' in ii]
    
    X, y, b, c = [], [], [], []
    for subject in subject_list:
        for i_clip in range(k_clip):
            
            if i_clip==0: #split test retest into 4
                seqs = data_df[(data_df['Subject']==subject) & 
                               (data_df['c'] == 0)][features].values
                label_seqs = data_df[(data_df['Subject']==subject) & 
                                     (data_df['c'] == 0)]['y'].values
                bhv_seqs = data_df[(data_df['Subject']==subject) & 
                                    (data_df['c'] == 0)][args.bhv].values
                clip_seqs = data_df[(data_df['Subject']==subject) & 
                                    (data_df['c'] == 0)]['c'].values

                k_time = int(seqs.shape[0]/K_RUNS)
                for i_run in range(K_RUNS):
                    seq = seqs[i_run*k_time:(i_run+1)*k_time, :]
                    label_seq = label_seqs[i_run*k_time:(i_run+1)*k_time]
                    bhv_seq = bhv_seqs[i_run*k_time:(i_run+1)*k_time]
                    clip_seq = clip_seqs[i_run*k_time:(i_run+1)*k_time]
                    if args.zscore:
                        # zscore each seq that goes into model
                        seq = (1/np.std(seq))*(seq - np.mean(seq))

                    FC = np.corrcoef(seq.T)
                    vecFC = FC[np.triu_indices_from(FC, k=1)]
                    X.append(vecFC)
                    # sanity check
                    if (np.all(label_seq==label_seq[0]) and
                        np.all(bhv_seq==bhv_seq[0]) and
                        np.all(clip_seq==clip_seq[0])):
                        
                        y.append(label_seq[0])
                        b.append(bhv_seq[0])
                        c.append(clip_seq[0])
                    else:
                        print('FATAL ERROR')

            else:
                seq = data_df[(data_df['Subject']==subject) & 
                              (data_df['c'] == i_clip)][features].values
                label_seq = data_df[(data_df['Subject']==subject) & 
                                    (data_df['c'] == i_clip)]['y'].values
                bhv_seq = data_df[(data_df['Subject']==subject) & 
                                   (data_df['c'] == i_clip)][args.bhv].values
                clip_seq = data_df[(data_df['Subject']==subject) & 
                                   (data_df['c'] == i_clip)]['c'].values
                if args.zscore:
                    # zscore each seq that goes into model
                    seq = (1/np.std(seq))*(seq - np.mean(seq))
                
                FC = np.corrcoef(seq.T)
                vecFC = FC[np.triu_indices_from(FC, k=1)]
                X.append(vecFC)
                # sanity check
                if (np.all(label_seq==label_seq[0]) and
                    np.all(bhv_seq==bhv_seq[0]) and
                    np.all(clip_seq==clip_seq[0])):

                    y.append(label_seq[0])
                    b.append(bhv_seq[0])
                    c.append(clip_seq[0])
                else:
                    print('FATAL ERROR')

    if args.mode=='class':
        return np.array(X), np.array(y), np.array(b), np.array(c)
    elif args.mode=='reg':
        return np.array(X), np.array(b), np.array(c)
