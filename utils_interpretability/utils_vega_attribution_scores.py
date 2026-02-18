import torch
import torch.nn as nn
import torch.optim as optim
import captum
from captum.attr import LRP
from captum.attr import NeuronIntegratedGradients
import shap
from typing import Optional, Union, Tuple
from captum.attr import Saliency, IntegratedGradients, DeepLift, InputXGradient
from tqdm import tqdm
from matplotlib.patches import Patch
import pandas as pd 
import numpy as np
from scipy.stats import gaussian_kde
import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Attribution scores

def compute_gene_attributions_captum(
    model,
    adata,
    methods,  # Liste des méthodes ou None pour toutes
    layer_name,  # 'mean' ou 'logvar' pour l'espace latent
    target_neuron,  # Si None, calcule pour tous les neurones
    batch_size,
    device
):
    """
    Calcule l'importance des gènes pour les neurones de l'espace latent avec Captum.
    Calcule TOUTES les méthodes spécifiées en un seul appel.
    
    Returns:
        DataFrame avec colonnes: latent_neuron, gene, importance_saliency, 
        importance_input_x_gradient, importance_integrated_gradients, importance_deeplift
    """
    # Méthodes disponibles
    available_methods = ["saliency", "input_x_gradient", "integrated_gradients", "deeplift"]
    
    # Si methods est None, utiliser toutes les méthodes
    if methods is None:
        methods = available_methods
    else:
        # Valider les méthodes
        for method in methods:
            if method not in available_methods:
                raise ValueError(f"Méthode inconnue: {method}. Méthodes disponibles: {available_methods}")
    
    model.to(device)
    model.eval()
    
    # Préparer les données
    if hasattr(adata, 'X'):
        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
    else:
        raise ValueError("adata doit contenir adata.X")
    
    X_tensor = torch.FloatTensor(X).to(device)
    gene_names = adata.var_names if hasattr(adata, 'var_names') else [f"Gene_{i}" for i in range(X.shape[1])]
    
    # Wrapper pour extraire l'espace latent
    class LatentExtractor(torch.nn.Module):
        def __init__(self, base_model, layer_name):
            super().__init__()
            self.base_model = base_model
            self.layer_name = layer_name
        
        def forward(self, x):
            encoded = self.base_model.encoder(x)
            if self.layer_name == "mean":
                return self.base_model.mean(encoded)
            elif self.layer_name == "mu":
                return self.base_model.mu(encoded)
            else:
                return self.base_model.logvar(encoded)
    
    wrapped_model = LatentExtractor(model, layer_name)
    wrapped_model.eval()
    
    # Déterminer le nombre de neurones latents
    with torch.no_grad():
        sample_output = wrapped_model(X_tensor[:1])
        n_latent = sample_output.shape[1]
    
    # Initialiser toutes les méthodes d'attribution
    attribution_methods = {}
    for method in methods:
        if method == "saliency":
            attribution_methods[method] = Saliency(wrapped_model)
        elif method == "input_x_gradient":
            attribution_methods[method] = InputXGradient(wrapped_model)
        elif method == "integrated_gradients":
            attribution_methods[method] = IntegratedGradients(wrapped_model)
        elif method == "deeplift":
            attribution_methods[method] = DeepLift(wrapped_model)
    
    # Déterminer les neurones à traiter
    neurons_to_process = [target_neuron] if target_neuron is not None else range(n_latent)
    
    # Stocker les résultats pour chaque méthode
    results_by_method = {method: [] for method in methods}
    
    # Pour chaque neurone latent
    for neuron_idx in tqdm(neurons_to_process, desc="Calcul des attributions par neurone"):
        
        # Pour chaque méthode
        for method_name, attribution_method in attribution_methods.items():
            neuron_attributions = []
            
            # Traiter par batches
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                
                # Calculer les attributions pour ce neurone
                if method_name == "integrated_gradients":
                    # Baseline = zéros pour RNA-seq
                    baseline = torch.zeros_like(batch)
                    attr = attribution_method.attribute(batch, baselines=baseline, target=neuron_idx)
                else:
                    attr = attribution_method.attribute(batch, target=neuron_idx)
                
                neuron_attributions.append(attr.detach().cpu().numpy())
            
            # Concaténer tous les batches
            neuron_attributions = np.concatenate(neuron_attributions, axis=0)
            
            # Moyenne sur toutes les cellules
            mean_attr = np.mean(np.abs(neuron_attributions), axis=0)
            
            # Stocker pour cette méthode
            results_by_method[method_name].append({
                'neuron_idx': neuron_idx,
                'mean_attr': mean_attr
            })
    
    # Construire le DataFrame final
    all_results = []
    
    for gene_idx, gene_name in enumerate(gene_names):
        for neuron_idx in neurons_to_process:
            row = {
                'latent_neuron': neuron_idx,
                'gene': gene_name
            }
            
            # Ajouter l'importance pour chaque méthode
            for method_name in methods:
                # Trouver les résultats pour ce neurone
                neuron_results = [r for r in results_by_method[method_name] if r['neuron_idx'] == neuron_idx][0]
                row[f'importance_{method_name}'] = neuron_results['mean_attr'][gene_idx]
            
            all_results.append(row)
    
    return pd.DataFrame(all_results)


def get_top_genes_per_neuron(
    attribution_df,
    top_k=20,
    save_path=None
):
    """
    Extrait les top K gènes les plus importants pour chaque neurone
    et pour chaque méthode d'attribution.

    Retourne un DataFrame long :
        latent_neuron | method | gene | importance
    """
    
    # détecter dynamiquement quelles méthodes sont présentes
    method_cols = [c for c in attribution_df.columns if c.startswith("importance_")]
    methods = [c.replace("importance_", "") for c in method_cols]

    results = []

    for neuron in sorted(attribution_df["latent_neuron"].unique()):

        df_n = attribution_df[attribution_df["latent_neuron"] == neuron]

        for method in methods:
            col = f"importance_{method}"

            # top K gènes pour cette méthode et ce neurone
            top_genes = df_n.nlargest(top_k, col)[["gene", col]]

            for _, row in top_genes.iterrows():
                results.append({
                    "latent_neuron": neuron,
                    "method": method,
                    "gene": row["gene"],
                    "importance": row[col]
                })

    result_df = pd.DataFrame(results)

    if save_path:
        result_df.to_csv(save_path, index=False)
        print(f"Top gènes enregistrés dans : {save_path}")

    return result_df


def visualize_top_genes(
    attribution_df,
    neuron_idx,
    true_genes=None,
    top_k=20
):
    """
    Visualize top K genes for a latent neuron.
    Highlight genes that are in true_genes list/array.
    """

    # Detect importance methods dynamically
    method_cols = [c for c in attribution_df.columns if c.startswith("importance_")]
    methods = [c.replace("importance_", "") for c in method_cols]
    n_methods = len(methods)

    # Filter for the given neuron
    df_n = attribution_df[attribution_df["latent_neuron"] == neuron_idx].copy()
    df_n['gene_str'] = df_n['gene'].astype(str).str.strip()

    # Flatten true_genes into a set of strings
    true_genes_set = set()
    if true_genes is not None:
        for g in true_genes:
            # g could be a np.ndarray
            if isinstance(g, (np.ndarray, list)):
                true_genes_set.update([str(x).strip() for x in g])
            else:
                true_genes_set.add(str(g).strip())

    # Prepare subplots
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 8), sharey=True)
    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        col = f"importance_{method}"

        # Top K genes by importance
        top_genes = df_n.nlargest(top_k, col).copy()

        # Colors: highlight true genes
        colors = ['orange' if g in true_genes_set else 'skyblue' for g in top_genes['gene_str']]

        # Reverse both top_genes and colors for barh (largest on top)
        top_genes = top_genes.iloc[::-1]
        colors = colors[::-1]

        # Plot horizontal bars
        ax.barh(top_genes['gene_str'], top_genes[col], color=colors)
        ax.set_title(f"Top {top_k} genes - {method}")
        ax.set_xlabel("Importance")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Gene")

    # Legend
    handles = [Patch(color='orange', label='True gene'),
               Patch(color='skyblue', label='Other gene')]
    axes[-1].legend(handles=handles, loc='lower right')

    plt.suptitle(f"Top genes for latent neuron {neuron_idx}", fontsize=16)
    plt.tight_layout()
    plt.show()

    
