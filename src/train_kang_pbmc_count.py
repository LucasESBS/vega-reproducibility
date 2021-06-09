#!/usr/bin/env python3

import torch
import sys
sys.path.append('./vega/')
sys.path.append('./')
from vega import VegaSCVI
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
    train_path = "./data/kang_all_raw.h5ad"
    pathway_file = "./data/reactomes.gmt"
    LR = 1e-4
    N_EPOCHS=400
    p_drop = 0.1
    z_drop = 0.5

    # Set model
    random_seed = 1
    # Out dir
    local_out = '/trained_models/'
    torch.backends.cudnn.enabled = True
    torch.manual_seed(random_seed)

    # Reduce to top 7000 highly variable genes
    print('Loading and preprocessing...')
    adata = sc.read(train_path)
    sc.pp.filter_genes(adata, min_counts=3)
    adata.layers["counts"] = adata.X.copy() # preserve counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata # freeze the state in `.raw`
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=7000,
        subset=True,
        layer="counts",
        flavor="seurat_v3")
    # Setup Anndata
    scvi.data.setup_anndata(adata, layer="counts")
    # Init VEGA
    print('Initializing model and training...')
    model = VegaSCVI(adata, gmt_paths=pathway_file, dropout_rate=p_drop, gene_likelihood='nb', z_dropout=z_drop, use_cuda=torch.cuda.is_available())
    model.train(max_epochs=N_EPOCHS, plan_kwargs={'lr':1e-4}, train_size=0.8, check_val_every_n_epoch=5)
    # Save model after training
    model.save('.'+local_out+'kang_trained_nb/', save_anndata=True)
    print('DONE', flush=True)
    return

if __name__ == "__main__":
    train_vega()

