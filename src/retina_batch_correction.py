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
    train_path = "./data/retina_pp_hg.h5ad"
    pathway_file = "./data/reactomes.gmt"
    LR = 5e-4
    N_EPOCHS=300
    p_drop = 0.2
    z_drop = 0.

    # Set model
    random_seed = 1
    # Out dir
    local_out = '/trained_models/retina/'
    torch.backends.cudnn.enabled = True
    torch.manual_seed(random_seed)

    print('Loading and preprocessing...', flush=True)
    adata = sc.read(train_path)
    adata = adata[adata.obs['Cluster']<=15, :].copy()
    adata_count = adata.copy()
    # Setup Anndata for VEGA
    vega.utils.setup_anndata(adata, batch_key="Batch")
    # Setup Anndata for LinearSCVI
    scvi.data.setup_anndata(adata_count, layer="count", batch_key="Batch")
    # ------ VEGA ---------
    # Init and train VEGA
    print('Initializing VEGA and training...', flush=True)
    model_vega = VEGA(adata, 
                    add_nodes=3,
                    dropout=p_drop,
                    z_dropout=z_drop,
                    positive_decoder=False,
                    gmt_paths=pathway_file,
                    use_cuda=True)

    model_vega.train_vega(n_epochs=N_EPOCHS, lr=LR, train_size=0.8, use_gpu=True, train_patience=20, test_patience=20)
    # Save
    model_vega.save('.'+local_out+'vega_retina/', save_adata=True, save_history=True)
    # ------- LinearSCVI --------------
    model_linear = scvi.model.LinearSCVI(adata_count,
                                        n_hidden=800,
                                        n_latent=adata.uns['_vega']['mask'].shape[1],
                                        n_layers=2,
                                        dropout_rate=p_drop)
    model_linear.train(max_epochs=N_EPOCHS, use_gpu=True, early_stopping=True, plan_kwargs={'lr':LR}, train_size=0.8, check_val_every_n_epoch=1, early_stopping_patience=20)
    model_linear.save('.'+local_out+'lvae_retina/', save_anndata=True)
    # Done
    print('DONE', flush=True)
    return

if __name__ == "__main__":
    train_vega()

