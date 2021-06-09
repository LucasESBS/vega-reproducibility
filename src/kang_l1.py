#!/usr/bin/env python3

# Training VEGA model with different L1 to discover the 3 missing IFN-alpha pathway genes.
# Used as a decent L1 parameter.



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
    train_path = "./data/kang_pbmc.h5ad"
    pathway_file = "./data/reactomes_ifna_3_rm.gmt"
    LR = 1e-4
    N_EPOCHS=500
    p_drop = 0.5
    z_drop = 0.5

    # Set model
    random_seed = 1
    # Out dir
    local_out = '/trained_models/kang_l1/'
    torch.backends.cudnn.enabled = True
    torch.manual_seed(random_seed)

    print('Loading and preprocessing...', flush=True)
    adata = sc.read(train_path)
    # ------ VEGA ---------
    # Init and train VEGA
    l1_list = [0.1, 1., 10., 100., 1000.]
    for l in l1_list:
        print('Initializing VEGA and training...', flush=True)
        print('L1: %s'%str(l), flush=True)
        vega.utils.setup_anndata(adata)
        model = VEGA(adata, 
                        add_nodes=1,
                        dropout=p_drop,
                        z_dropout=z_drop,
                        positive_decoder=True,
                        gmt_paths=pathway_file,
                        regularizer='l1',
                        use_cuda=True,
                        reg_kwargs={'lr':LR, 'lambda1':l, 'use_gpu':True})
        model.train_vega(n_epochs=N_EPOCHS, lr=LR, train_size=0.8, use_gpu=True)
        # Save
        model.save('.'+local_out+'vega_l1_%s/'%(str(l)), save_adata=True, save_history=True)
    print('DONE', flush=True)
    return

if __name__ == "__main__":
    train_vega()

