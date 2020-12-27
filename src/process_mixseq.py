#!/usr/bin/env python3

""" Code for processing and preparing MIXSseq-data, according to paper """
import os 
import sys
import scanpy as sc
import pandas as pd
import numpy as np

def preprocess_mixseq_data(adata, metric_df):
    """ Preprocess Anndata object according to MIXseq paper.
        Highly variable genes wont be done here, but after DMSO is merged with 
    """
    # Add annot
    adata.obs = metric_df
    # Simple filter first
    sc.pp.filter_cells(adata, min_genes=200)
    #sc.pp.filter_genes(adata, min_cells=3)
    # Keep only "normal" cells (filter low quality ones)
    adata = adata[adata.obs['cell_quality']=='normal', :]
    # Normalization according to paper
    sc.pp.normalize_total(adata, target_sum=1e5)
    sc.pp.log1p(adata)
    #sc.pp.highly_variable_genes(adata, n_top_genes=5000)
    #adata.raw = adata
    #adata = adata[:, adata.var.highly_variable]
    # Done
    return adata

def make_data_exp3(in_dirname, out_dirname, n_top_genes=5000):
    """ Create all anndata datasets for exp3 (only 24hr treatment).
        DMSO cells are included in every dataset as a reference.
    """
    if not in_dirname.endswith('/'):
        in_dirname += '/'
    if not out_dirname.endswith('/'):
        out_dirname += '/'
    list_f = os.listdir(in_dirname)
    list_f = [f for f in list_f if ('24hr_expt3' in f) and (os.path.isdir(in_dirname+f))]
    ref_dir = 'DMSO_24hr_expt3'
    print('Reading '+ref_dir)
    adata_dmso = sc.read_10x_mtx(in_dirname+ref_dir)
    metric_dmso = pd.read_csv(in_dirname+ref_dir+'/classifications.csv', index_col=0)
    adata_dmso = preprocess_mixseq_data(adata_dmso, metric_dmso)
    adata_dmso.obs['condition'] = 'DMSO'
    # Iterate over all drugs
    for drug_dir in list_f:
        if drug_dir == ref_dir:
            continue
        else:
            print('Reading '+drug_dir)
            adata_drug = sc.read_10x_mtx(in_dirname+drug_dir)
            metric_drug = pd.read_csv(in_dirname+drug_dir+'/classifications.csv', index_col=0)
            adata_drug = preprocess_mixseq_data(adata_drug, metric_drug)
            drug_name = drug_dir.split('_')[0]
            adata_drug.obs['condition'] = drug_name
            adata_f = adata_dmso.copy()
            adata_f = adata_f.concatenate(adata_drug)
            # highly variable genes after merging
            if n_top_genes:
                sc.pp.highly_variable_genes(adata_f, n_top_genes=5000)
                adata_f.raw = adata_f
                adata_f = adata_f[:, adata_f.var.highly_variable] 
            out_f = '_'.join([drug_name, 'DMSO', '24hr', 'expt3'])
            adata_f.write(out_dirname+out_f+'.h5ad')
    return
   
 
def main():
    make_data_exp3('../data/mixseq_datasets/', '../data/mixseq_datasets/data_expt3_allgenes/', n_top_genes=None)
    return

if __name__ == "__main__":
    main()
