import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
import scanpy as sc
from anndata import AnnData

#sys.path.append('/home/user/Review-Interpretable-VAEs/Review-Interpretable-VAEs/cloned_github_models/vega/vega-reproducibility/src')
sys.path.append('/home/BS94_SUR/phD/review/models reproductibility/VEGA/vega-reproducibility/src')
import vega
import utils
from utils import *
from learning_utils import *
from vega_model import VEGA
import torch
import itertools

def load_pathways_vega(load, data_path, pathway_file):
    if load == True:
        adata = sc.read(data_path)
    else:
        adata = data_path
    pathway_dict = read_gmt(pathway_file, min_g=0, max_g=1000)
    pathway_mask = create_pathway_mask(adata.var.index.tolist(), pathway_dict, add_missing=1, fully_connected=True)
    list_pathways = list(pathway_dict.keys()) + ['UNANNOTATED_'+str(k) for k in range(1)]
    return list_pathways

def create_path_embeddings(name_model, name_dataset, split, random_seed, perturbation, source, pathway_selected, path_saved_embeddings):
    if source == 'original':
        path_embeddings = path_saved_embeddings + f'{name_model}_{name_dataset}_embeddings_{split}_original_seed_{random_seed}_trial.txt'
    if source == 'perturbated':
        path_embeddings = path_saved_embeddings + f'{name_model}_{name_dataset}_embeddings_{split}_{pathway_selected}_{perturbation}_seed_{random_seed}.txt'
    return path_embeddings


def load_embeddings(path_embeddings_pathway, list_pathways):
    embeddings_pathway = np.loadtxt(path_embeddings_pathway)
    if list_pathways:
        embeddings_pathway= pd.DataFrame(embeddings_pathway, columns=list_pathways)
    return embeddings_pathway


def access_data_vega(data_path, adata, pathway_file, path_data_description):
    if data_path is not None:
        adata = sc.read(data_path)
    pathway_dict = read_gmt(pathway_file, min_g=0, max_g=1000)
    pathway_mask = create_pathway_mask(adata.var.index.tolist(), pathway_dict, add_missing=1, fully_connected=True)
    list_pathways = load_pathways_vega(False, adata, pathway_file)
    if path_data_description is not None:
        df_genespathways = pd.read_parquet(path_data_description + f'df_pathways_genes_description.parquet')
        overlap_matrix = pd.read_csv(path_data_description + 'overlap_matrix_vega.csv')
    else:
        df_genespathways = None
        overlap_matrix = None
    
    return adata, pathway_dict, pathway_mask, list_pathways, df_genespathways, overlap_matrix


def extract_x_y_from_adata(adata, column_labels_name: pd.Series):
    X = pd.DataFrame(adata.X, index=adata.obs.index)
    y = adata.obs[column_labels_name]
    return X, y

def split_data(X, y, train_size, random_seed):
    X_train,  X_test, labels_train,  labels_test = train_test_split(
        X, y, train_size=train_size, random_state=random_seed, stratify=y)
    return X_train,  X_test, labels_train,  labels_test

def extract_index(X):
    index_df = X.index
    return index_df
    
def build_adata_from_X(adata, index_df):
    adata = adata[adata.obs.index.isin(index_df)]
    return adata, index_df

def encode_y(y):
    le = preprocessing.LabelEncoder().fit(y)
    y_encoded = torch.Tensor(le.transform(y))
    return y_encoded

def build_vega_dataset(adata, y_encoded, pathway_file):
    if sparse.issparse(adata.X):
        data = adata.X.A
    else:
        data = adata.X

    data = torch.Tensor(data)
    data = UnsupervisedDataset(data, targets=y_encoded)

    pathway_dict = read_gmt(pathway_file, min_g=0, max_g=1000)
    pathway_mask = create_pathway_mask(adata.var.index.tolist(), pathway_dict, add_missing=1, fully_connected=True)

    return data, pathway_dict, pathway_mask

def preprocess_adata(adata, n_top_genes=5000):
    """ Simple (default) sc preprocessing function before autoencoders """
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    #sc.pp.normalize_total(adata, target_sum=1e4)
    #sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    return adata


def create_vega_training_data(name_model, preprocess, select_hvg, n_top_genes, random_seed, train_size, column_labels_name, adata, pathway_file):
    if preprocess == True:
        print('Preprocessing adata')
        adata = preprocess_adata(adata, n_top_genes=n_top_genes)
    else:
        adata = adata.copy()
    X, y = extract_x_y_from_adata(adata, column_labels_name)
    X_train,  X_test, labels_train,  labels_test = split_data(X, y, train_size, random_seed)
    y_train = encode_y(labels_train)
    y_test = encode_y(labels_test)
    index_train = extract_index(X_train)
    index_test = extract_index(X_test)
    adata_train, index_train = build_adata_from_X(adata, index_train)
    adata_test, index_test = build_adata_from_X(adata, index_test)

    train_ds, pathway_dict, pathway_mask = build_vega_dataset(adata_train, y_train, pathway_file)
    test_ds, pathway_dict, pathway_mask = build_vega_dataset(adata_test, y_test, pathway_file)
    return adata, adata_train, adata_test, train_ds, test_ds, pathway_dict, pathway_mask
