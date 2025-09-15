import torch 
import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import seaborn as sns
from matplotlib import pyplot as plt
import umap
import umap.plot
from anndata import AnnData
from pathlib import Path

class VAE_clustering_evaluation():
    def __init__(self,
                 X_embedding: np.array,
                 adata: AnnData,
                 name_labels: pd.Series,
                 clustering_method: str,
                 val_resolution: int,
                 dataset_name: str,
                 model_type: str,
                 split:str,
                 trial:int,
                 path_save_fig: str,
                 ) -> None:
        super(VAE_clustering_evaluation, self).__init__()

        self.X_embedding = X_embedding
        self.adata = adata
        self.name_labels = name_labels
        self.clustering_method = clustering_method
        self.val_resolution = val_resolution
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.path_save_fig = path_save_fig
        self.split = split
        self.trial = trial 

    def extract_x_y_from_adata(self, adata: AnnData, name_labels: pd.Series):
        X = pd.DataFrame(adata.X, index=adata.obs.index)
        y = adata.obs[name_labels]
        return X, y
    
    def compute_silhouette_score_from_latent(self, X_embedding, y):
        asw = silhouette_score(X=X_embedding, labels=y)
        return asw

    def build_adata_latent(self, X_embedding, adata, name_labels):
        adata_latent = AnnData(self.X_embedding)
        adata_latent.obs[name_labels] = self.adata.obs[name_labels].values
        return adata_latent

    def apply_clustering_algo(self, adata, name_labels, clustering_method, val_resolution):
        true_labels = adata.obs[name_labels].values
        num_clusters = len(np.unique(true_labels))
        sc.pp.neighbors(adata, n_neighbors=num_clusters, use_rep="X")

        if clustering_method == 'Louvain':
            sc.tl.louvain(adata, resolution=val_resolution)
            clusters = adata.obs["louvain"].astype(int).values
        elif clustering_method == 'Leiden':
            sc.tl.leiden(adata, resolution=val_resolution)
            clusters = adata.obs["leiden"].astype(int).values
        else:
            raise ValueError(f"Unsupported clustering method: {clustering_method}")

        return clusters, true_labels

    def apply_clustering_metrics(self, true_labels, clusters):
        ari = adjusted_rand_score(true_labels, clusters)
        nmi = normalized_mutual_info_score(true_labels, clusters)
        return ari, nmi

    def plot_umap_orig(self, embedding, true_labels, dataset_name, model_type):
        mapper = umap.UMAP(random_state=42).fit(np.nan_to_num(np.array(embedding)))
        umap.plot.points(mapper, labels=true_labels, color_key_cmap='Paired', show_legend=True)
        title_orig = f'{model_type} - {self.split} - Trial {self.trial} (true labels)'
        plt.title(title_orig)
        plt.savefig(self.path_save_fig + f'/{model_type}_{dataset_name}_{self.split}_{self.trial}_umap_original.png')
        plt.show()

    def plot_umap_cluster(self, embedding, clusters, dataset_name, model_type, ari, nmi):
        mapper = umap.UMAP(random_state=42).fit(np.nan_to_num(np.array(embedding)))
        umap.plot.points(mapper, labels=clusters, color_key_cmap='Paired', show_legend=True)
        title_clusters = f'{model_type} ({self.clustering_method}) - {self.split} - Trial {self.trial}\nARI: {ari}, NMI: {nmi}'
        plt.title(title_clusters)
        plt.savefig(self.path_save_fig + f'/{model_type}_{dataset_name}_{self.split}_{self.trial}_umap_clusters.png')
        plt.show()
