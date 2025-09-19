import sys
import vega
import os
sys.path.append('/home/BS94_SUR/phD/review/models reproductibility/VEGA/vega-reproducibility/src')
import vanilla_vae
from vanilla_vae import VanillaVAE

import pandas as pd
import scanpy as sc
import numpy as np
from scipy import sparse
from scipy.sparse import issparse
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import mean_squared_error
import umap
import torch
from scipy.stats import pearsonr
import umap

def access_to_latent(model, model_type, adata, col_name):
    
    X = adata.X
    y = adata.obs[col_name]
    
    if model_type == 'vanilla_vae':
        z = model.to_latent(torch.tensor(X.A if issparse(X) else X, device=model.dev, dtype=torch.float)).detach().cpu().numpy()
    
    return z

def reconstruct_data(data_tensor, model, device):
    model.eval()  # evaluation mode

    reconstruct_results = []

    with torch.no_grad():
        reconstructed, _, _ = model(data_tensor)
        reconstruct_results.append(reconstructed.cpu().numpy())

    reconstruct_results = np.concatenate(reconstruct_results, axis=0)
    return reconstruct_results


def compute_silhouette_score_from_latent(z, y):
    
    asw = silhouette_score(X=z, labels=y, random_state=42)
    
    return asw

def compute_mse_score(X, X_reconstructed):
    mse = mean_squared_error(X, X_reconstructed)
    return mse

def compute_pearson_score(X, X_reconstructed, mean):
    if mean == True:
        corr, _ = pearsonr(X, X_reconstructed)
        
    else:
        correlations = []
        for i in range(X.shape[0]):
            corr, _ = pearsonr(X[i, :], X_reconstructed[i, :])
            correlations.append(corr)
        corr = np.mean(correlations)
    return corr

def apply_umap_array(z):
    reducer = umap.UMAP(random_state=42, min_dist=0.5, n_neighbors=15)
    embedding = reducer.fit_transform(z)
    return embedding
        
