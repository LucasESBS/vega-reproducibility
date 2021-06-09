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
from intercode import Intercode
import itertools
import argparse
import time

def train_comparison():
    """ Main """
    train_path = "./data/kang_pbmc.h5ad"
    pathway_file = "./data/reactomes.gmt"
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
    t0 = time.time()
    model.train_vega(n_epochs=N_EPOCHS, lr=LR, train_size=1., use_gpu=True)
    t1 = time.time()
    t = t1 - t0
    with open('.'+local_out+'vega_time.txt', 'w') as f:
        f.write('Time training:%.10f'%(t))
    model.save('.'+local_out+'vega_model/', save_adata=True, save_history=True)
    # Done
    print('DONE training VEGA in %.5f'%(t), flush=True)
    adata_ae = model.adata.copy()
    adata_ae.varm['I'] = adata.uns['_vega']['mask'][:,:-1]
    adata_ae.uns['terms'] = adata.uns['_vega']['gmv_names'][:-1]
    adata_ae.X = adata_ae.X.A
    print('Training Intercode', flush=True)
    ae_model = Intercode(adata_ae, n_dense=1, n_sparse=None, dropout_rate=0.5, mid_layers_size=800, use_cuda=True)
    lamda0 = 0.1
    lamda1 = 0.93
    lamda3 = 0.57
    n_epochs = len(model.epoch_history['train_loss'])
    t0 = time.time()
    ae_model.train(LR, 128, n_epochs, l2_reg_lambda0=lamda0, lambda1=lamda1, lambda3=lamda3)
    t1 = time.time()
    t = t1 - t0
    with open('.'+local_out+'intercode_time.txt', 'w') as f:
        f.write('Time training:%.10f'%(t))
    ae_model.save('.'+local_out+'intercode_model/')
    print('DONE training Intercode in %.5f'%(t), flush=True)
    return

if __name__ == "__main__":
    train_comparison()

