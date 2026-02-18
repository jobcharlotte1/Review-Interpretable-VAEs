import anndata
import numpy as np
import pandas as pd
import tensorflow as tf
import scanpy as sc
from matplotlib import pyplot as plt
import anndata as ad
from anndata import AnnData
from scipy import sparse

from pathlib import Path
from sklearn.model_selection import train_test_split

import sys
sys.path.append('/home/BS94_SUR/phD/review/models reproductibility/pmvae/')

from pmvae.model import PMVAE
from pmvae.train import train
from pmvae.utils import load_annotations


def build_overlap_matrix_pmVAE(pathway_mask, adata, list_pathways):
    pathway_dict = {
        pathway: pathway_mask.columns[row.values].tolist()
        for pathway, row in pathway_mask.iterrows()
    }

    nb_genes = len(adata.var)
    all_results = []  # accumulate results across all pathways

    for pathway_selected in list_pathways:
        list_genes_to_perturbate = pathway_mask.columns[(pathway_mask.loc[pathway_selected] == 1) ].tolist()
        genes1 = [gene for gene in list_genes_to_perturbate if gene in adata.var_names]

        for pathway_compared, genes in pathway_dict.items():
            genes2 = [gene for gene in genes if gene in adata.var_names]
            intersection = set(genes1).intersection(set(genes2))

            #if intersection:  # only keep if non-empty
            all_results.append({
                "Pathway Selected": pathway_selected,   # which pathway we started with
                "Nb Genes Pathway Selected": len(genes1),
                "Compared Pathway": pathway_compared,            # the pathway we are comparing against
                "Nb Genes Compared Pathway": len(genes2),
                "Genes Overlap": len(intersection),
                "Commun Genes": list(intersection)
            })

    # Convert to a single dataframe
    overlap_matrix = pd.DataFrame(all_results)
    overlap_matrix['Overlap Proportion'] = overlap_matrix ['Genes Overlap'] / nb_genes * 100

    # Sort for convenience (optional)
    overlap_matrix  = overlap_matrix .sort_values(
        by=["Pathway Selected", "Genes Overlap"],
        ascending=[True, False]
    ).reset_index(drop=True)
    
    return overlap_matrix 



def access_data_pmVAE(data_path, adata, min_genes, pathway_file, path_data_description):
    if data_path is not None:
        adata = sc.read(data_path)
    adata.varm['annotations'] = load_annotations(
        pathway_file,
        adata.var_names,
        min_genes=min_genes
    )
    pathway_mask = adata.varm['annotations'].astype(bool).T
    list_pathways = list(adata.varm['annotations'].columns)
    df_genespathways = pd.read_parquet(path_data_description + f'df_pathways_genes_description.parquet')
    #overlap_matrix = pd.read_csv(path_data_description + 'overlap_matrix_pmVAE.csv')
    overlap_matrix = build_overlap_matrix_pmVAE(pathway_mask, adata, list_pathways)
    
    return adata, pathway_mask, list_pathways, df_genespathways, overlap_matrix


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

def pmvae_train_test_split(adata, test_size, shuffle, random_state):
    if isinstance(adata, AnnData):
        X = adata.X
    else:
        X = adata
    if sparse.issparse(X):
        X = X.toarray()
        
    idx = np.arange(adata.n_obs)

    idx_train, idx_test = train_test_split(
        idx,
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state
    )

    X_train = X[idx_train]
    X_test = X[idx_test]
    
    return X_train, X_test, idx_train, idx_test

def get_train_set(trainset, batch_size):
    trainset = tf.data.Dataset.from_tensor_slices(trainset)
    trainset = trainset.shuffle(5 * batch_size).batch(batch_size)
    return trainset

def get_grouped_embeddings_pmvae(n, embeddings, list_pathways):
    new_cols = {}
    for idx, i in enumerate(range(0, embeddings.shape[1], n)):
        group = embeddings.iloc[:, i:i+n]  
        new_cols[list_pathways[idx]] = np.linalg.norm(group, axis=1)  # row-wise norm

    embeddings_groupped = pd.DataFrame(new_cols)
    return embeddings_groupped
