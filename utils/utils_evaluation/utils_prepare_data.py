name_model = 'OntoVAE'

import sys
import os
from pathlib import Path
import pandas as pd
import scanpy as sc
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import sparse
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import itertools
import argparse
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import anndata as ad
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px  # For color palette
from sklearn.model_selection import train_test_split
import umap
from scipy.sparse import issparse
#import optuna
#from optuna.samplers import TPESampler
from pathlib import Path
import anndata as ad
from anndata import AnnData

if name_model == 'VEGA' or name_model == 'VanillaVAE':
    sys.path.append('/home/BS94_SUR/phD/review/models reproductibility/VEGA/vega-reproducibility/src')
    import vanilla_vae
    import vega
    import train_vanilla_vae_suppFig1
    import utils
    from utils import *
    from learning_utils import *
    from vanilla_vae import VanillaVAE
    from vega_model import VEGA

if name_model == 'pmVAE':
    import sys
    sys.path.append('/home/BS94_SUR/phD/review/models reproductibility/pmvae/')
    import tensorflow as tf
    from pmvae.model import PMVAE
    from pmvae.train import train
    from pmvae.utils import load_annotations

if name_model == 'OntoVAE':
    sys.path.append('/home/BS94_SUR/phD/review/models reproductibility/OntoVAE/cobra-ai')
    from cobra_ai.module.ontobj import *
    from cobra_ai.module.utils import *
    from cobra_ai.model.onto_vae import *
    import onto_vae


class VAE_prepare_dataset:
    def __init__(self,
                 adata: AnnData,
                 column_labels_name: str,
                 random_seed: int,
                 name_model: str,
                 train_size: int,
                 pathway_file: str,
                 preprocess:bool,
                 select_hvg: bool,
                 n_top_genes: int) -> None:
        super(VAE_prepare_dataset, self).__init__()

        self.adata = adata
        self.column_labels_name = column_labels_name
        self.random_seed = random_seed
        self.name_model = name_model
        self.train_size = train_size
        self.pathway_file = pathway_file
        self.preprocess = preprocess
        self.select_hvg = select_hvg
        self.n_top_genes = n_top_genes
        
    def preprocess_data(self, adata, name_model, preprocess, select_hvg, n_top_genes):
        if name_model == 'Vega':
            vega.utils.setup_anndata(adata)
        elif preprocess == True:
            sc.pp.filter_cells(adata, min_genes=200)
            sc.pp.filter_genes(adata, min_cells=3)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            if select_hvg == True:
                sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
                adata.raw = adata
                adata = adata[:, adata.var.highly_variable]
        return adata

    def extract_x_y_from_adata(self, adata: AnnData, column_labels_name: pd.Series):
        X = pd.DataFrame(adata.X, index=adata.obs.index)
        y = adata.obs[column_labels_name]
        return X, y

    def split_data(self, X, y, train_size, random_seed):
        X_train,  X_test, labels_train,  labels_test = train_test_split(
            X, y, train_size=train_size, random_state=random_seed, stratify=y)
        return X_train,  X_test, labels_train,  labels_test
    
    def extract_index(self, X):
        index_df = X.index
        return index_df
        
    def build_adata_from_X(self, adata, index_df):
        adata = adata[adata.obs.index.isin(index_df)]
        return adata, index_df
    
    def encode_y(self, y):
        le = preprocessing.LabelEncoder().fit(y)
        y_encoded = torch.Tensor(le.transform(y))
        return y_encoded

    def build_vega_dataset(self, adata, y_encoded, pathway_file):
        if sparse.issparse(adata.X):
            data = adata.X.A
        else:
            data = adata.X

        data = torch.Tensor(data)
        data = UnsupervisedDataset(data, targets=y_encoded)

        pathway_dict = read_gmt(pathway_file, min_g=0, max_g=1000)
        pathway_mask = create_pathway_mask(adata.var.index.tolist(), pathway_dict, add_missing=1, fully_connected=True)

        return data, pathway_mask
    
    def build_pmVAE_dataset(self, adata, pathway_file):
        adata.varm['annotations'] = load_annotations(
            pathway_file,
            adata.var_names,
            min_genes=13
        )
        pathway_mask = adata.varm['annotations'].astype(bool).T
        return adata, pathway_mask

    def build_OntoVAE_dataset(self, adata, pathway_file):
        ontobj = Ontobj()
        ontobj.load(pathway_file)
        genes = ontobj.extract_genes()
        adata = adata.copy()
        adata = setup_anndata_ontovae(adata, ontobj)
        return adata
