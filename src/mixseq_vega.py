#!/usr/bin/env python3

import torch
import sys
import os
sys.path.append('./vega/')
sys.path.append('./')
from vega import VEGA
from utils import *
#from learning_utils import *
import scanpy as sc
import scvi
from scipy import sparse
from sklearn import preprocessing
import numpy as np
import itertools
import argparse

def train_vega():
    """ Main """
    indir_name = "./data/mixseq/"
    pathway_file = "./data/hallmarks.gmt"
    LR = 1e-4
    N_EPOCHS=500
    p_drop = 0.5
    z_drop = 0.5

    # Set model
    random_seed = 1
    # Out dir
    local_out = '/trained_models/mixseq/'
    torch.backends.cudnn.enabled = True
    torch.manual_seed(random_seed)

    # ------ VEGA ---------
    # Init and train VEGA
    # Get all datasets
    list_f = os.listdir(indir_name)
    list_f = [f for f in list_f if 'h5ad' in f]
    list_f.sort()
    for fname in list_f:
        print('Training models for '+fname, flush=True)
        # Read pathway file and data
        adata = sc.read(indir_name+fname)
        vega.utils.setup_anndata(adata)
        print('Initializing VEGA and training...', flush=True)
        model = VEGA(adata, 
                        add_nodes=5,
                        dropout=p_drop,
                        z_dropout=z_drop,
                        positive_decoder=True,
                        gmt_paths=pathway_file,
                        use_cuda=True)

        model.train_vega(n_epochs=N_EPOCHS, lr=LR, train_size=0.8, use_gpu=True)
        # Save
        model_pref = 'vega_'+fname.split('.')[0]+'/'
        model.save('.'+local_out+model_pref, save_adata=True, save_history=True)
    print('DONE', flush=True)
    return

if __name__ == "__main__":
    train_vega()

