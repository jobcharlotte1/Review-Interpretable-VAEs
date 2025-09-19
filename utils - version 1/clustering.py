import pickle
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_losses(epoch_losses, epoch_kl_losses, epoch_mse_losses):
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_losses, label="Total Loss", color="blue")
    plt.plot(epoch_mse_losses, label="Reconstruction Loss (MSE)", color="red")
    plt.plot(epoch_kl_losses, label="KL Divergence Loss", color="green")
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("VAE Training Loss Evolution")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def extract_latent_results(adata, dataloader, model, latent_dim):
    
    latent_results = []
    sample_val, _ = adata.X.shape
    for i in range(sample_val):
        data_tem = dataloader.dataset[i][0]
        output = model.encoder(data_tem.reshape(1, -1).to(device))
        latent_results.append(output[0].cpu().detach().numpy())

    latent_results = np.array(latent_results)
    latent_results = latent_results.reshape(sample_val, latent_dim)
    
    return latent_results


#def build_adata_latent(adata, latent_results, column_true_labels):
    
 #   true_labels = adata.obs[column_true_labels].values
  #  adata_latent = ad.AnnData(latent_results)
   # adata_latent.obs["true_labels"] = true_labels
    
    #return adata_latent


def apply_reducuction_algo(array_to_reduce, true_labels_or_clusters, clustering_method, method):

    if method == 'UMAP': 
        reduced_model = umap.UMAP(n_components=2, random_state=42)
    if method == 'TSNE':
        reduced_model = TSNE(n_components=2, random_state=42)

    reduced_latent = reduced_model.fit_transform(array_to_reduce)

    df = pd.DataFrame(reduced_latent, columns=["dim1", "dim2"])
    
    if clustering_method == None:
        df["true_labels"] = true_labels_or_clusters
    else:
        df[f"{clustering_method}_cluster"] = true_labels_or_clusters
    
    return df

def plot_reduced_data(df, x_name, y_name, color_name, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x_name, y=y_name, hue=color_name, data=df, palette="tab10", alpha=0.7)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    

def apply_clustering_algo(true_labels, adata, clustering_method):
    num_clusters = len(np.unique(true_labels))
    sc.pp.neighbors(adata, n_neighbors=num_clusters, use_rep="X") 

    if clustering_method == 'Louvain':
        sc.tl.louvain(adata, resolution=1.0)
        clusters = adata.obs["louvain"].astype(int).values
    if clustering_method == 'Leiden':
        sc.tl.leiden(adata, resolution=1.0)
        clusters = adata.obs["leiden"].astype(int).values
        
    return clusters
    
def apply_clustering_metrics(true_labels, clusters):
    ari = adjusted_rand_score(true_labels, clusters)
    nmi = normalized_mutual_info_score(true_labels, clusters)
    
    return ari, nmi