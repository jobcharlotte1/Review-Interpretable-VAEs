import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import anndata as ad
import scipy.sparse as sp
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler


def preprocess_adata(adata, n_top_genes=5000):
    """ Simple (default) sc preprocessing function before autoencoders """
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    return adata


def create_vanilla_data(adata, test_size, random_state):
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    indices = np.arange(adata.n_obs)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=test_size,
        random_state=random_state
    )
    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]
    
    adata_train = adata[train_idx].copy()
    adata_val = adata[val_idx].copy()
    adata_test = adata[test_idx].copy()
    
    # Convert to PyTorch datasets
    train_data = torch.utils.data.TensorDataset(torch.FloatTensor(X_train))
    val_data = torch.utils.data.TensorDataset(torch.FloatTensor(X_val))
    test_data = torch.utils.data.TensorDataset(torch.FloatTensor(X_test))
    
    return X_train, X_val, X_test, adata_train, adata_val, adata_test, train_data, val_data, test_data