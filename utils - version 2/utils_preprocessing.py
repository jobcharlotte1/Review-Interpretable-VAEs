import scanpy as sc
import pandas as pd
import numpy as np
import sys
import vega
import os
from anndata import AnnData
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

sys.path.append('/home/BS94_SUR/phD/review/models reproductibility/VEGA/vega-reproducibility/src')
import vanilla_vae
import train_vanilla_vae_suppFig1
import utils
from utils import *
from learning_utils import *
from vanilla_vae import VanillaVAE


def preprocess_anndata(adata: AnnData,
                        n_top_genes:int = 5000,
                        copy: bool = False,
                       filter_sc: bool = True,
                       scale: bool = True,
                      ):
    """
    Simple (default) Scanpy preprocessing function before autoencoders.

    Parameters
    ----------
        adata
            Scanpy single-cell object
        n_top_genes
            Number of highly variable genes to retain
        copy
            Return a copy or in place
    Returns
    -------
        adata
            Preprocessed Anndata object
    """
    if copy:
        adata = adata.copy()
    
    if filter_sc:    
        vega.utils.setup_anndata(adata)
    
    if scale:
        sc.pp.scale(adata)
    
    if copy:
        return adata
    

def extract_x_y_from_adata(adata: AnnData, labels: pd.Series):
    X = pd.DataFrame(adata.X, index=adata.obs.index)
    y = adata.obs[labels]
    return X, y

def split_data(adata: AnnData, column_name: str, train_size: float = 0.8):
    X, y = extract_x_y_from_adata(adata, column_name)
    X_train,  X_test, labels_train,  labels_test = train_test_split(
        X, y, train_size=train_size, random_state=0, stratify=y)
    return X_train,  X_test, labels_train,  labels_test

def extract_index(X: np.ndarray):
    index_df = X.index
    return index_df

def build_adata_from_X(X: np.ndarray, adata: AnnData):
    index_df = extract_index(X)
    adata = adata[adata.obs.index.isin(index_df)]
    return adata, index_df

def encode_y(labels):
    le = preprocessing.LabelEncoder().fit(labels)
    y = torch.Tensor(le.transform(labels))
    return y

def build_vega_dataset(adata_train: AnnData, encoded_y_train):
    if sparse.issparse(adata_train.X):
        train_ds = adata_train.X.A
    else:
        train_ds = adata_train.X

    train_ds = torch.Tensor(train_ds)
    train_ds = UnsupervisedDataset(train_ds, targets=encoded_y_train)
    return train_ds
