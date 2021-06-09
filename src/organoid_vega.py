#!/usr/bin/env python3

import torch
import sys
sys.path.append('./vega/')
sys.path.append('./')
from vega import VEGA
from utils import *
import scanpy as sc
import scvi
from scipy import sparse
from sklearn import preprocessing
import numpy as np
import itertools
import argparse

def train_vega():
    """ Main """
    train_path = "./data/wk02_af_hvg.h5ad"
    pathway_file = "./data/reactomes_organoid_ct.gmt"
    LR = 1e-4
    N_EPOCHS=500
    p_drop = 0.5

    # Set model
    random_seed = 1
    # Out dir
    local_out = '/trained_models/'
    torch.backends.cudnn.enabled = True
    torch.manual_seed(random_seed)

    print('Loading and preprocessing...', flush=True)
    adata = sc.read(train_path)
    # Setup Anndata for VEGA
    vega.utils.setup_anndata(adata)
    # ------ VEGA ---------
    # Init and train VEGA
    print('Initializing VEGA and training...', flush=True)
    model = VEGA(adata,
                    add_nodes=1,
                    dropout=p_drop,
                    z_dropout=0.5,
                    positive_decoder=True,
                    gmt_paths=pathway_file,
                    use_cuda=True)
    
    model.train_vega(n_epochs=N_EPOCHS, lr=LR, train_size=0.8, use_gpu=True)
    model.save('.'+local_out+'organoid_vega/', save_adata=True, save_history=True)
    # Done
    print('DONE', flush=True)
    return

if __name__ == "__main__":
    train_vega()

