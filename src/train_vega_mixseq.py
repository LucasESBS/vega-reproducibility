#!/usr/bin/env python3

import torch
import os
from vega_model import VEGA
from utils import *
from learning_utils import *
import scanpy as sc
from scipy import sparse
from sklearn import preprocessing
import argparse
import itertools



def train_vega():
    """ Main """
    indir_name = './data/mixseq/'
    # Train path
    pathway_file = "./data/hallmarks.gmt"
    LR = 1e-4
    N_EPOCHS=500
    p_drop = 0.5

    # Set model
    random_seed = 1
    # Out dir
    local_out = '/trained_models/mixseq/'
    torch.backends.cudnn.enabled = True
    torch.manual_seed(random_seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', dev, flush=True)
    dir_name = '.'+local_out

    # Get all datasets
    list_f = os.listdir(indir_name)
    list_f = [f for f in list_f if 'h5ad' in f]
    list_f.sort()
    for fname in list_f:
        print('Training models for '+fname, flush=True)
        # Read pathway file and data
        train_data = sc.read(indir_name+fname)
        labels = train_data.obs['singlet_ID']
        le = preprocessing.LabelEncoder()
        le.fit(labels)
        y = torch.Tensor(le.transform(labels))
        # Load pathway
        pathway_dict = read_gmt(pathway_file, min_g=0, max_g=1000)
        print(train_data.shape, flush=True)
        pathway_mask = create_pathway_mask(train_data.var.index.tolist(), pathway_dict, add_missing=5, fully_connected=True)
        if sparse.issparse(train_data.X):
            train_ds = train_data.X.A
        else:
            train_ds = train_data.X
        train_ds = torch.Tensor(train_ds)
        train_ds = UnsupervisedDataset(train_ds, targets=y)
        # Initialize CV
        model_pref = 'vega_'+fname.split('.')[0]+'_'
        kfold = KFoldTorch(cv=10, n_epochs=N_EPOCHS, lr=LR, train_p=10, test_p=10, num_workers=0, save_all=True, save_best=False, path_dir=dir_name, model_prefix=model_pref)
        dict_params = {'pathway_mask':pathway_mask, 'beta':0.00005, 'dropout':p_drop, 'path_model':None, 'device':dev, 'positive_decoder':True}
        kfold.train_kfold(VEGA, dict_params, train_ds, batch_size=64)
        np.save(dir_name+'10CV_'+model_pref+'res.npy', kfold.cv_res_dict)
    return

if __name__=="__main__":
    train_vega()
