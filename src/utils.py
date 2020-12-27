from collections import OrderedDict
import torch
import numpy as np
import scanpy as sc
from scipy import sparse


def dict_to_gmt(dict_obj, path_gmt, sep='\t', second_col=True):
    """ Write dictionary to gmt format """
    with open(path_gmt, 'w') as f:
        for k,v in dict_obj.items():
            if second_col:
                to_write = sep.join([k,'SECOND_COL'] + v)+'\n'
            else:
                to_write = sep.join([k] + v) + '\n'
            f.write(to_write)
    return
         

def read_gmt(fname, sep='\t', min_g=0, max_g=5000):
    """
    Read GMT file into dictionary of pathway:genes.
    min_g and max_g are optional gene set size filters.
    """
    dict_pathway = OrderedDict()
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            val = line.split(sep)
            if min_g <= len(val[2:]) <= max_g:
                dict_pathway[val[0]] = val[2:]
    return dict_pathway

def create_pathway_mask(feature_list, dict_pathway, add_missing=True, fully_connected=True, to_tensor=False):
    """ Creates a mask of shape [genes,pathways] where (i,j) = 1 if gene i is in pathway j, 0 else.
    Expects a list of genes and pathway dict.
    If add_missing is True or an int, input genes that are not part of pathways are all connected to a "placeholder" pathway.
    If fully_connected is True, all input genes are connected to the placeholder units.
    Note: dict_pathway should be an Ordered dict so that the ordering can be later interpreted."""
    assert type(dict_pathway) == OrderedDict
    p_mask = np.zeros((len(feature_list), len(dict_pathway)))
    for j, k in enumerate(dict_pathway.keys()):
        for i in range(p_mask.shape[0]):
            if feature_list[i] in dict_pathway[k]:
                p_mask[i,j] = 1.
    if add_missing:
        n = 1 if type(add_missing)==bool else add_missing
        # Get non connected genes
        if not fully_connected:
            idx_0 = np.where(np.sum(p_mask, axis=1)==0)
            vec = np.zeros((p_mask.shape[0],n))
            vec[idx_0,:] = 1.
        else:
            vec = np.ones((p_mask.shape[0], n))
        p_mask = np.hstack((p_mask, vec))
    if to_tensor:
        p_mask = torch.Tensor(p_mask)
    return p_mask

def shuffle_mask(x):
    """ shuffle mask, ie each pathway is assigned random genes while
    keeping the same numbers of genes"""
    ix_i = np.random.sample(x.shape).argsort(axis=0)
    ix_j = np.tile(np.arange(x.shape[1]), (x.shape[0], 1))
    return x[ix_i, ix_j]

def filter_pathways(pathway_list, pathway_mask, top_k=1000):
    """ Filter pathway by size """
    print('Retaining top ',top_k,' pathways')
    idx_sorted = np.argsort(np.sum(pathway_mask, axis=0))[::-1][:top_k]
    pathway_mask_filtered = pathway_mask[:,idx_sorted]
    pathway_list_filtered = list(np.array(pathway_list)[idx_sorted])
    return pathway_list_filtered, pathway_mask_filtered

def prepare_anndata(anndata, batch_size, shuffle=False):
    """
    Load Anndata object into pytorch data loader.
    """
    # Add shuffling here
    if sparse.issparse(anndata.X):
        data = anndata.X.A
    else:
        data = anndata.X
    data = torch.Tensor(data)
    my_dataloader = torch.utils.data.DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    return my_dataloader

def balance_populations(adata, ct_key='cell_type', condition_key='condition'):
    """ Balance cell population within condition for unbias sampling and delta estimation """
    ct_names = adata.obs[ct_key].unique()
    ct_counts = adata.obs[ct_key].value_counts()
    max_val = np.max(ct_counts)
    data = []
    label = []
    condition = []
    for ct in ct_names:
        tmp = adata.copy()[adata.obs[ct_key] == ct]
        idx = np.random.choice(range(len(tmp)), max_val)
        if sparse.issparse(tmp.X):
            tmp_X = tmp.X.A[idx]
        else:
            tmp_X = tmp.X[idx]
        data.append(tmp_X)
        label.append(np.repeat(ct, max_val))
        condition.append(np.repeat(np.unique(tmp.obs[condition_key]), max_val))
    balanced_adata = sc.AnnData(np.concatenate(data))
    balanced_adata.obs[ct_key] = np.concatenate(label)
    balanced_adata.obs[condition_key] = np.concatenate(condition)
    return balanced_adata

def extend_mask(pathway_mask, max_unit=5, p_unit=0.3, normalize_reduct=True):
    """ Extend the pathway mask to increase the number of hidden unit representing each pathway.
        The formula to extend is inspired by DCell and is: max(max_unit, p_unit*n_genes).
    """
    final_mask = None
    reduce_scheme = []
    for p in range(pathway_mask.shape[1]):
        p_vec = pathway_mask[:,p]
        n = np.sum(p_vec)
        n_unit = max([max_unit, int(p_unit*n)])
        to_add = np.tile(p_vec.reshape(-1,1), (1,n_unit))
        reduce_scheme.append(n_unit)
        if final_mask is None:
            final_mask = to_add
        else:
            final_mask = np.hstack((final_mask, to_add))
    # Make reduct mat
    z = final_mask.shape[1]
    reduct_mat = np.zeros((z,pathway_mask.shape[1]))
    i = 0
    for j in range(len(reduce_scheme)):
        s = reduce_scheme[j]
        if normalize_reduct:
            reduct_mat[i:i+s,j] = 1./s
        else:
            reduct_mat[i:i+s,j] = 1
        i+=s
    return final_mask, reduct_mat
       

def preprocess_adata(adata, n_top_genes=5000):
    """ Simple (default) sc preprocessing function before autoencoders """
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    return adata

 
class ClassificationDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, targets):
        'Initialization'
        self.targets = targets
        self.data = data

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.targets)

    def __getitem__(self, index):
        'Generates samples of data'
        # Load data and get label
        X = self.data[index]
        y = self.targets[index]

        return X, y.long()

class UnsupervisedDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, targets=None):
        'Initialization'
        self.targets = targets
        self.data = data

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates samples of data'
        # Load data
        X = self.data[index]
        return X    
