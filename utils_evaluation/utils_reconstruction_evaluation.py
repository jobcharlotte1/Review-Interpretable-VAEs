import torch 
import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np
from sklearn.metrics import mean_squared_error
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import umap
import umap.plot
from anndata import AnnData


class VAE_reconstruction():
    def __init__(self, X_reconstructed: np.array, adata: AnnData) -> None:
        self.X_reconstructed = X_reconstructed
        self.adata = adata

    def compute_pearson_score(self):
        X = self.adata.X
        X = X.toarray() if hasattr(X, "toarray") else X
        correlations = [
            pearsonr(X[i, :], self.X_reconstructed[i, :])[0]
            for i in range(X.shape[0])
        ]
        return np.mean(correlations)

    def compute_mse_score(self):
        X = self.adata.X
        X = X.toarray() if hasattr(X, "toarray") else X
        return mean_squared_error(X, self.X_reconstructed)
