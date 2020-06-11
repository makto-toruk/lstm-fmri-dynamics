import numpy as np
import pandas as pd
import torch
import pickle

PARCEL_DIR = 'data/parcellation/cifti'

def _info(s):
    print('---')
    print(s)
    print('---')

def _get_parcel(roi, net=7):
    '''
    return:
    parcel: grayordinate -> ROI map
    nw_info: subnetwork tags for each ROI
    '''
    parcel_path = (PARCEL_DIR + 
        '/Schaefer2018_%dParcels_%dNetworks_order.csv' %(roi, net))
    
    df = pd.read_csv(parcel_path)
    parcel = np.array(df['ROI'])

    info_path = parcel_path.replace('.csv', '_info_condensed.csv')
    df = pd.read_csv(info_path)
    nw_info = np.array(df['network'])

    return parcel, nw_info

def _to_cpu(a, clone=True):
    '''
    cuda to cpu for numpy operations
    '''
    if clone:
        a = a.clone()
    a = a.detach().cpu()

    return a

def _get_behavioral(subject_list):
    '''
    load behavioral measures for all HCP subjects
    '''
    bhv_path = 'data/unrestricted_behavioral.csv'
    bhv_df = pd.read_csv(bhv_path)

    bhv_df = bhv_df.loc[bhv_df['Subject'].isin(subject_list)]
    bhv_df = bhv_df.reset_index(drop=True)

    return bhv_df

def _vectorize(Q):
    '''
    Q: symmetric matrix (FC)
    return: unique elements as an array
    ignore diagonal elements
    '''
    # extract lower triangular matrix
    tri = np.tril(Q, -1)

    vec = []
    for ii in range(1, tri.shape[0]):
        for jj in range(ii):
            vec.append(tri[ii, jj])
    
    return np.asarray(vec)

def _getfc(scan):
    '''
    Functional Connectivity matrix
    using Pearson correlation
    
    scan: timeseries of ROI x t
    output: FC of ROI x ROI
    '''
    return np.corrcoef(scan)

def _get_clip_lengths():
    '''
    return:
    clip_length: dict of lengths of each clip
    '''
    K_RUNS = 4
    # where are the clips within the run?
    timing_file = pd.read_csv('data/videoclip_tr_lookup.csv')

    clip_length = {}
    for run in range(K_RUNS):

        run_name = 'MOVIE%d' %(run+1) #MOVIEx_7T_yz
        timing_df = timing_file[timing_file['run'].str.contains(run_name)]  
        timing_df = timing_df.reset_index(drop=True)

        for jj, clip in timing_df.iterrows():
            start = int(np.floor(clip['start_tr']))
            stop = int(np.ceil(clip['stop_tr']))
            t_length = stop - start
            clip_length[clip['clip_name']] = t_length
            
    return clip_length