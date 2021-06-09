#!/usr/bin/env python3

import torch
import sys
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
    train_path = "./data/kang_pbmc.h5ad"
    pathway_file = "./data/reactomes.gmt"
    LR = 1e-4
    N_EPOCHS=500
    p_drop = 0.2

    # Set model
    random_seed = 1
    # Out dir
    local_out = '/trained_models/corr_z/'
    torch.backends.cudnn.enabled = True
    torch.manual_seed(random_seed)

    print('Loading and preprocessing...', flush=True)
    adata = sc.read(train_path)
    # Setup Anndata for VEGA
    vega.utils.setup_anndata(adata)
    # ------ VEGA ---------
    # Init and train VEGA
    print('Initializing VEGA and training...', flush=True)
    model_0= VEGA(adata, 
                    add_nodes=1,
                    dropout=p_drop,
                    z_dropout=0.,
                    positive_decoder=True,
                    gmt_paths=pathway_file,
                    use_cuda=True)

    model_0.train_vega(n_epochs=N_EPOCHS, lr=LR, train_size=0.8, use_gpu=True)
    # Save
    model_0.save('.'+local_out+'vega_zdrop0/', save_adata=True, save_history=True)
    # Z dropout 0.5 - reset seed to get same split
    torch.manual_seed(random_seed)
    adata = adata.copy()
    vega.utils.setup_anndata(adata)
    model_05 = VEGA(adata,
                    add_nodes=1,
                    dropout=p_drop,
                    z_dropout=0.5,
                    positive_decoder=True,
                    gmt_paths=pathway_file,
                    use_cuda=True)
    
    model_05.train_vega(n_epochs=N_EPOCHS, lr=LR, train_size=0.8, use_gpu=True)
    model_05.save('.'+local_out+'vega_zdrop05/', save_adata=True, save_history=True)
    # Done
    print('DONE', flush=True)
    return

if __name__ == "__main__":
    train_vega()

