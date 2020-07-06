'''
collect participants' runs into data cell
1. extract roi timeseries (ts) from grayordinate ts
2. zscore (normalize) each time series
3. save subject tag
'''
import numpy as np
import pandas as pd
import nibabel as nib
import argparse
import os
from glob import glob
import pickle

from utils import _info, _get_parcel

def _zscore_ts(ts):
    '''
    zscore each ROIs time series
    ts: (time x roi)
    '''
    ts = ts.T
    z_ts = []
    for ii in range(ts.shape[0]):
        t = ts[ii, :]
        z_ts.append((t - np.mean(t))/np.std(t))
        
    z_ts = np.array(z_ts)
    z_ts = z_ts.T

    return z_ts

def _get_roi_ts(path, parcel, nw_info, args):
    '''
    path: path to cifti time series
    parcel: grayordinate -> roi map
        labels range {1, 2, ... , roi}
    nw_info: roi -> nw map
    args: {roi, net}

    return
    roi_ts: time x roi
    '''
    # load cifti, (time x grayordinate)
    ts = nib.load(path).get_fdata()
    # truncate ts to only cortex (rest are subcortical)
    ts = ts[:, :parcel.shape[0]]

    # zscore ts
    z_ts = _zscore_ts(ts) # time x grayordinate
    t = z_ts.shape[0]

    roi_ts = np.zeros((t, args.roi))
    for ii in range(args.roi):
        roi_ts[:, ii] = np.mean(
            z_ts[:, parcel==(ii+1)], axis=1)

    # ***reorder based on nw info
    roi_ts = roi_ts[:, np.argsort(nw_info)]

    return roi_ts

def run(args):

    # load parcellation file
    parcel, nw_info = _get_parcel(args.roi, args.net)

    # use glob to get all files with `ext`
    ext = '*MSMAll_hp2000_clean.dtseries.nii'
    files = [y for x in os.walk(args.input_data) 
        for y in glob(os.path.join(x[0], ext))]

    # get list of participants
    # ID <=> individual
    participants = set()
    for file in files:
        ID = file.split('/MNINonLinear')[0][-6:]
        participants.add(ID)
    participants = np.sort(list(participants))
    _info('Number of participants = %d' %len(participants))

    data = {}
    for ii, ID in enumerate(participants):
        ID_files = [file for file in files if ID in file]
        ID_files = np.sort(ID_files)

        # if individual has all 4 runs
        if len(ID_files)==4:
            _info('%s: %d/%d' %(ID, (ii+1), len(participants)))
            ID_ts, t = [], []
            for path in ID_files:
                roi_ts = _get_roi_ts(
                    path, parcel, nw_info, args)
                ID_ts.append(roi_ts)
                t.append(roi_ts.shape[0])
            
            k_time = np.max(t)
            '''
            ID_ts have different temporal length
            pad zeros
            (time x roi x number of runs)
            '''
            save_ts = np.zeros((k_time, args.roi, 4))
            for run in range(4):
                run_ts = ID_ts[run]
                t = run_ts.shape[0]
                save_ts[:t, :, run] = run_ts

            data[ID] = save_ts

        else:
            _info('%s not processed'%ID)

    SAVE_DIR = args.output_data
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    save_path = (SAVE_DIR + 
        '/data_MOVIE_runs_roi_%d_net_%d_ts.pkl' %(
            args.roi, args.net))
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='preprocess movie data')
    
    # preprocessed hcp data
    parser.add_argument('-i', '--input-data', type=str,
        default='data/hcp_movies', help='path/to/movie/data')
    parser.add_argument('-o', '--output-data', type=str,
        default='data/roi_ts', help='path/to/roi/data')

    # parcellation
    parser.add_argument('-r', '--roi', type=int,
        default=300, help='number of ROI')
    parser.add_argument('-n', '--net', type=int,
        default=7, help='number of networks (7 or 17)')
    args = parser.parse_args()

    run(args)