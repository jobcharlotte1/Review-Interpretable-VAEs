import torch 
import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import mean_squared_error
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import umap
import umap.plot

class VAEexperiment():
    def __init__(self,
                 vae_model,
                 model_type: str,
                 adata: AnnData,
                 params: dict) -> None:
        super(VAEexperiment, self).__init__()

        self.model = vae_model
        self.model_type = model_type
        self.model = vae_model
        self.params = params

    def extract_x_y_from_adata(adata: AnnData, labels: pd.Series):
        X = pd.DataFrame(adata.X, index=adata.obs.index)
        y = adata.obs[labels]
        return X, y

    def give_embeddings(X_tensor, model, model_type):
        if model_type == 'VEGA':
            embeddings = model.to_latent(X_tensor.to(dev)).detach().cpu().numpy()
        if model_type == 'OntoVAE':
            embeddings = model.to_latent(adata)
        return embeddings

    def reconstruct_data(model, embeddings, model_type):
        if model_type == 'VEGA':
            X_reconstructed = model.decode(torch.Tensor(embeddings).to(dev))
        if model_type == 'OntoVAE':
            X_reconstructed = model._run_batches(adata, 'rec', False)
        return X_reconstructed

    def compute_silhouette_score_from_latent(embeddings, y):
        asw = silhouette_score(X=embeddings, labels=y, random_state=42)
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
    
    def build_adata_latent(embedding, name_labels, adata, resolution):
        adata_latent = ad.AnnData(embedding)
        adata_latent.obs[name_labels] = adata.obs[name_labels].values
        return adata_latent
    
    def apply_clustering_algo(adata, name_labels, clustering_method):
        true_labels = adata.obs[name_labels].values
        num_clusters = len(np.unique(true_labels))
        sc.pp.neighbors(adata, n_neighbors=num_clusters, use_rep="X") 

        if clustering_method == 'Louvain':
            sc.tl.louvain(adata, resolution=resolution)
            clusters = adata.obs["louvain"].astype(int).values
        if clustering_method == 'Leiden':
            sc.tl.leiden(adata, resolution=resolution)
            clusters = adata.obs["leiden"].astype(int).values
            
        return clusters, true_labels
        
        
    def apply_clustering_metrics(true_labels, clusters):
        ari = adjusted_rand_score(true_labels, clusters)
        nmi = normalized_mutual_info_score(true_labels, clusters)
        return ari, nmi
    
    def plot_umap(embedding, labels, title):
        mapper = umap.UMAP().fit(np.nan_to_num(embedding.to_numpy()))
        umap.plot.points(mapper, labels=labels.values, color_key_cmap='Paired',show_legend=True)
        plt.title(title)
        umap.plot.plt.show()

            

        