#!/usr/bin/env python3

import torch
import sys
sys.path.append('/root/vega/')
sys.path.append('/root/')
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
    train_path = "./data/kang_pbmc.h5ad"
    pathway_file = "./data/reactomes.gmt"
    LR = 1e-4
    N_EPOCHS=500
    p_drop = 0.5

    # Set model
    random_seed = 1
    # Out dir
    local_out = '/trained_models/vega_ct_out/'
    torch.backends.cudnn.enabled = True
    torch.manual_seed(random_seed)

    print('Loading and preprocessing...', flush=True)
    adata = sc.read(train_path)
    for cell_type in adata.obs['cell_type'].unique():
        print('Training model leaving out '+cell_type, flush=True)
        train_data = adata[~((adata.obs['cell_type'] == cell_type) & (adata.obs['condition'] == 'stimulated'))].copy()
        path_model = 'vega_kang_pbmc_%s_out/'%(cell_type)
        # Setup Anndata for VEGA
        vega.utils.setup_anndata(train_data)
        # ------ VEGA ---------
        # Init and train VEGA
        print('Initializing VEGA and training...', flush=True)
        model = VEGA(train_data,
                        add_nodes=1,
                        dropout=p_drop,
                        z_dropout=0.5,
                        positive_decoder=True,
                        gmt_paths=pathway_file,
                        use_cuda=True)
    
        model.train_vega(n_epochs=N_EPOCHS, lr=LR, train_size=0.8, use_gpu=True)
        model.save('.'+local_out+path_model, save_adata=True, save_history=True)
    # Done
    print('DONE', flush=True)
    return

if __name__ == "__main__":
    train_vega()

