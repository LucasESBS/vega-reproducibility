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

def train_vega(n_nodes):
    """ Main """
    train_path = "./data/kang_pbmc.h5ad"
    pathway_file = "./data/reactomes.gmt"
    LR = 1e-4
    N_EPOCHS=500
    p_drop = 0.2
    z_drop = 0.5

    # Set model
    random_seed = 1
    # Out dir
    local_out = '/trained_models/node_experiment/'
    torch.backends.cudnn.enabled = True
    torch.manual_seed(random_seed)

    print('Loading and preprocessing...', flush=True)
    adata = sc.read(train_path)
    print('Tests with %i extra nodes'%(n_nodes), flush=True)
    # ------ VEGA ---------
    # Init and train VEGA
    for k in range(5):
        print('Model number %i'%(k), flush=True)
        vega.utils.setup_anndata(adata)
        print('Initializing VEGA and training...', flush=True)
        model = VEGA(adata, 
                        add_nodes=n_nodes,
                        dropout=p_drop,
                        z_dropout=z_drop,
                        positive_decoder=True,
                        gmt_paths=pathway_file,
                        use_cuda=True)

        model.train_vega(n_epochs=N_EPOCHS, lr=LR, train_size=0.8, use_gpu=True)
        # Save
        model.save('.'+local_out+'vega_%i_nodes_%i/'%(n_nodes, k), save_adata=True, save_history=True)
    print('DONE', flush=True)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--nodes', help='Number of extra latent nodes', required=True)
    args = vars(parser.parse_args())
    n_nodes = int(args['nodes'])
    train_vega(n_nodes)

