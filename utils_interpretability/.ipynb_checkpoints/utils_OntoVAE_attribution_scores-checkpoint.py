import torch
import torch.nn as nn
import torch.optim as optim
from anndata import AnnData
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
import torch
from anndata import AnnData
import scanpy as sc

from cobra_ai.module.utils import FastTensorDataLoader


# Attribution scores

def average_attr(array_attr, nb_neurons):
    return np.array(np.split(array_attr, array_attr.shape[1]/nb_neurons, axis=1)).mean(axis=2).T

def compute_gene_to_neuron_attribution(
    model,
    adata: Optional[AnnData] = None,
    layer_index: int = 0,
    batch_size: int = 128,
    aggregate_cells: bool = True,
    methods: str = 'all',
    n_steps: int = 50  # for integrated gradients
):
    """
    Compute attribution of each input gene to each neuron in a decoder layer using multiple methods.
    """
    
    model.eval()
    device = model.device
    
    # Use model's adata if not provided
    if adata is None:
        adata = model.adata
    
    # Validate layer index
    n_decoder_layers = len(model.decoder.decoder) - 1  # -1 for reconstruction layer
    if layer_index < 0 or layer_index >= n_decoder_layers:
        raise ValueError(
            f"layer_index must be between 0 and {n_decoder_layers-1}, got {layer_index}"
        )
    
    # Parse methods
    available_methods = ['saliency', 'input_x_gradient', 'deeplift', 'integrated_gradients']
    if methods == 'all':
        methods_to_use = available_methods
    elif isinstance(methods, str):
        if methods not in available_methods:
            raise ValueError(f"Unknown method: {methods}. Choose from {available_methods}")
        methods_to_use = [methods]
    else:
        methods_to_use = methods
        for m in methods_to_use:
            if m not in available_methods:
                raise ValueError(f"Unknown method: {m}. Choose from {available_methods}")
    
    # Get gene names
    gene_names = adata.uns['_ontovae']['genes']
    n_genes = len(gene_names)
    
    # Prepare data
    covs = model._cov_tensor(adata)
    from cobra_ai.module.utils import FastTensorDataLoader
    
    dataloader = FastTensorDataLoader(
        torch.tensor(adata.X.todense(), dtype=torch.float32),
        covs,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Store results for each method
    method_results = {method: [] for method in methods_to_use}
    
    # Create a wrapper class to extract layer activations
    class LayerExtractor(torch.nn.Module):
        def __init__(self, model, layer_index, cat_list):
            super().__init__()
            self.model = model
            self.layer_index = layer_index
            self.cat_list = cat_list
            self.activation = None
            
        def forward(self, x):
            # Store activations
            activation = {}
            
            def get_activation_hook(name):
                def hook(module, input, output):
                    activation[name] = output
                return hook
            
            # Register hook
            hook = self.model.decoder.decoder[self.layer_index][0].register_forward_hook(
                get_activation_hook('target_layer')
            )
            
            # Forward pass
            _, _, _, _ = self.model.forward(x, self.cat_list)
            
            # Get activation
            self.activation = activation['target_layer']
            
            # Remove hook
            hook.remove()
            
            return self.activation
    
    # Process each batch
    for minibatch in dataloader:
        x_batch = minibatch[0].to(device)
        cat_list = torch.split(minibatch[1].T.to(device), 1)
        
        # Get number of neurons by doing a forward pass
        with torch.no_grad():
            wrapper = LayerExtractor(model, layer_index, cat_list)
            sample_output = wrapper(x_batch[:1])
            n_neurons = sample_output.shape[1]
        
        # Compute attributions for each neuron using each method
        batch_attributions = {method: [] for method in methods_to_use}
        
        for neuron_idx in range(n_neurons):
            # Create a wrapper that outputs only this neuron
            class NeuronWrapper(torch.nn.Module):
                def __init__(self, model, layer_index, cat_list, neuron_idx):
                    super().__init__()
                    self.model = model
                    self.layer_index = layer_index
                    self.cat_list = cat_list
                    self.neuron_idx = neuron_idx
                    
                def forward(self, x):
                    activation = {}
                    
                    def get_activation_hook(name):
                        def hook(module, input, output):
                            activation[name] = output
                        return hook
                    
                    hook = self.model.decoder.decoder[self.layer_index][0].register_forward_hook(
                        get_activation_hook('target_layer')
                    )
                    
                    _, _, _, _ = self.model.forward(x, self.cat_list)
                    
                    hook.remove()
                    
                    # Return only the target neuron's activation
                    return activation['target_layer'][:, self.neuron_idx:self.neuron_idx+1]
            
            neuron_model = NeuronWrapper(model, layer_index, cat_list, neuron_idx)
            neuron_model.eval()
            
            # Baseline for DeepLift and Integrated Gradients (zeros)
            baseline = torch.zeros_like(x_batch)
            
            # Compute attributions for each method
            for method in methods_to_use:
                if method == 'saliency':
                    attr_method = Saliency(neuron_model)
                    attr = attr_method.attribute(x_batch, abs=False)
                    
                elif method == 'input_x_gradient':
                    attr_method = InputXGradient(neuron_model)
                    attr = attr_method.attribute(x_batch)
                    
                elif method == 'deeplift':
                    attr_method = DeepLift(neuron_model)
                    attr = attr_method.attribute(x_batch, baselines=baseline)
                    
                elif method == 'integrated_gradients':
                    attr_method = IntegratedGradients(neuron_model)
                    attr = attr_method.attribute(x_batch, baselines=baseline, n_steps=n_steps)
                
                # Convert to numpy and store
                attr_np = attr.detach().cpu().numpy()
                batch_attributions[method].append(attr_np)
        
        # Stack attributions for all neurons
        # Shape: (n_neurons, batch_size, n_genes) -> (batch_size, n_genes, n_neurons)
        for method in methods_to_use:
            batch_attr = np.stack(batch_attributions[method], axis=0)
            batch_attr = np.transpose(batch_attr, (1, 2, 0))
            method_results[method].append(batch_attr)
    
    # Concatenate all batches and aggregate if needed
    final_attributions = {}
    for method in methods_to_use:
        attributions = np.vstack(method_results[method])  # (n_cells, n_genes, n_neurons)
        
        if aggregate_cells:
            attributions = attributions.mean(axis=0)  # (n_genes, n_neurons)
        
        final_attributions[method] = attributions
    
    result = {
        'attributions': final_attributions,
        'gene_names': gene_names,
        'layer_index': layer_index,
        'n_neurons': n_neurons,
        'n_genes': n_genes,
        'methods_used': methods_to_use
    }
    
    return result


def _compute_single_layer_attribution_worker(layer_idx, model_path, adata_path, batch_size, aggregate_cells, methods, n_steps, device_str):
    """
    Helper function for parallel computation of a single layer.
    Uses file paths instead of objects to avoid CUDA serialization issues.
    """
    device = 'cpu'

    raise NotImplementedError("This approach needs model serialization")


def _compute_single_layer_simple(args):
    """
    Simplified worker function that computes attribution for a single layer.
    Receives all necessary data as arguments to avoid serialization issues.
    """
    (layer_idx, X_data, covs_data, gene_names, masks, layer_dims, 
     batch_size, aggregate_cells, methods, n_steps, model_params) = args
    
    pass


def compute_gene_to_all_neurons_attribution(
    model,
    adata,
    batch_size: int = 128,
    aggregate_cells: bool = True,
    methods: str = 'all',
    n_steps: int = 50,
    parallel: bool = False,  # Changed default to False due to CUDA issues
    n_jobs: Optional[int] = None
):
    """
    Compute attribution of each gene to each neuron in ALL decoder layers using multiple methods.
    
    """
    
    n_decoder_layers = len(model.decoder.decoder) - 1  # -1 for reconstruction layer
    
    if adata is None:
        adata = model.adata
    
    # Check if using CUDA
    using_cuda = next(model.parameters()).is_cuda
    if parallel and using_cuda:
        print("WARNING: Parallel processing with CUDA can cause errors.")
        print("Switching to sequential processing. To use parallel processing,")
        print("move model to CPU first: model_cpu = model.cpu()")
        parallel = False
    
    # Determine number of jobs
    if n_jobs is None:
        n_jobs = os.cpu_count()
    elif n_jobs < 1:
        n_jobs = 1
    
    # Sequential processing (RECOMMENDED for GPU)
    if not parallel or n_jobs == 1:
        results = {}
        print(f"Computing {n_decoder_layers} layers sequentially...")
        for layer_idx in tqdm(range(n_decoder_layers), desc="Computing layers"):
            results[layer_idx] = compute_gene_to_neuron_attribution(
                model=model,
                adata=adata,
                layer_index=layer_idx,
                batch_size=batch_size,
                aggregate_cells=aggregate_cells,
                methods=methods,
                n_steps=n_steps
            )
        return results
    
    # Parallel processing (ONLY for CPU models)
    print(f"Computing {n_decoder_layers} layers in parallel using {n_jobs} workers...")
    print("NOTE: Model must be on CPU for parallel processing to work correctly.")
    
    # Ensure model is on CPU for multiprocessing
    original_device = model.device
    model = model.cpu()
    
    # Create a context with 'spawn' method
    ctx = get_context('spawn')
    
    # Function to compute single layer (needs to be picklable)
    def compute_layer(layer_idx):
        return compute_gene_to_neuron_attribution(
            model=model,
            adata=adata,
            layer_index=layer_idx,
            batch_size=batch_size,
            aggregate_cells=aggregate_cells,
            methods=methods,
            n_steps=n_steps
        )
    
    # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid CUDA issues
    from concurrent.futures import ThreadPoolExecutor
    
    results = {}
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all tasks
        futures = {executor.submit(compute_layer, i): i for i in range(n_decoder_layers)}
        
        # Collect results with progress bar
        from tqdm import tqdm
        for future in tqdm(futures, total=n_decoder_layers, desc="Computing layers"):
            layer_idx = futures[future]
            results[layer_idx] = future.result()
    
    # Move model back to original device if needed
    if str(original_device) != 'cpu':
        model.to(original_device)
    
    return results


def get_top_genes_per_neuron(
    attribution_result: dict,
    method: str = 'saliency',
    top_k: int = 10,
    use_abs: bool = True
):
    """
    Get the top contributing genes for each neuron using a specific attribution method.

    """
    
    if method not in attribution_result['attributions']:
        raise ValueError(
            f"Method '{method}' not found. Available methods: {list(attribution_result['attributions'].keys())}"
        )
    
    attributions = attribution_result['attributions'][method]  # (n_genes, n_neurons)
    gene_names = attribution_result['gene_names']
    n_neurons = attribution_result['n_neurons']
    
    top_genes_dict = {}
    
    for neuron_idx in range(n_neurons):
        neuron_attr = attributions[:, neuron_idx]
        
        if use_abs:
            # Sort by absolute value
            top_indices = np.argsort(np.abs(neuron_attr))[-top_k:][::-1]
        else:
            # Sort by actual value (highest first)
            top_indices = np.argsort(neuron_attr)[-top_k:][::-1]
        
        top_genes = [(gene_names[i], neuron_attr[i]) for i in top_indices]
        top_genes_dict[neuron_idx] = top_genes
    
    return top_genes_dict


def get_top_neurons_per_gene(
    attribution_result: dict,
    method: str = 'saliency',
    top_k: int = 10,
    use_abs: bool = True
):
    """
    Get the top neurons influenced by each gene using a specific attribution method.
    """
    
    if method not in attribution_result['attributions']:
        raise ValueError(
            f"Method '{method}' not found. Available methods: {list(attribution_result['attributions'].keys())}"
        )
    
    attributions = attribution_result['attributions'][method]  # (n_genes, n_neurons)
    gene_names = attribution_result['gene_names']
    n_genes = attribution_result['n_genes']
    
    top_neurons_dict = {}
    
    for gene_idx in range(n_genes):
        gene_attr = attributions[gene_idx, :]
        
        if use_abs:
            # Sort by absolute value
            top_indices = np.argsort(np.abs(gene_attr))[-top_k:][::-1]
        else:
            # Sort by actual value (highest first)
            top_indices = np.argsort(gene_attr)[-top_k:][::-1]
        
        top_neurons = [(int(i), gene_attr[i]) for i in top_indices]
        top_neurons_dict[gene_names[gene_idx]] = top_neurons
    
    return top_neurons_dict


def visualize_gene_neuron_heatmap(
    attribution_result: dict,
    method: str = 'saliency',
    top_genes: int = 50,
    figsize: tuple = (12, 10),
    save_path: Optional[str] = None
):
    """
    Create a heatmap visualization of gene-to-neuron attributions for a specific method.
    """
    
    attributions = attribution_result['attributions'][method]  # (n_genes, n_neurons)
    gene_names = attribution_result['gene_names']
    
    # Select top genes by total absolute attribution
    gene_importance = np.abs(attributions).sum(axis=1)
    top_gene_indices = np.argsort(gene_importance)[-top_genes:][::-1]
    
    # Subset data
    plot_data = attributions[top_gene_indices, :]
    plot_genes = [gene_names[i] for i in top_gene_indices]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        plot_data,
        xticklabels=[f"N{i}" for i in range(attributions.shape[1])],
        yticklabels=plot_genes,
        cmap='RdBu_r',
        center=0,
        cbar_kws={'label': 'Attribution Score'},
        ax=ax
    )
    
    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Gene')
    ax.set_title(f'Gene-to-Neuron Attribution - {method.replace("_", " ").title()} (Layer {attribution_result["layer_index"]})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig


def compare_attribution_methods(
    attribution_result: dict,
    gene_name: str,
    neuron_idx: Optional[int] = None,
    figsize: tuple = (10, 6)
):
    """
    Compare different attribution methods for a specific gene or neuron.
    """
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return
    
    gene_names = attribution_result['gene_names']
    if gene_name not in gene_names:
        raise ValueError(f"Gene '{gene_name}' not found in gene list")
    
    gene_idx = gene_names.index(gene_name)
    methods = attribution_result['methods_used']
    
    if neuron_idx is not None:
        # Compare specific neuron across methods
        scores = []
        for method in methods:
            attr = attribution_result['attributions'][method]
            scores.append(attr[gene_idx, neuron_idx])
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(methods, scores)
        ax.set_xlabel('Attribution Method')
        ax.set_ylabel('Attribution Score')
        ax.set_title(f'{gene_name} → Neuron {neuron_idx}')
        ax.set_xticklabels([m.replace('_', '\n') for m in methods], rotation=0)
        plt.tight_layout()
        plt.show()
        
    else:
        # Show top neurons for each method
        fig, axes = plt.subplots(1, len(methods), figsize=(5*len(methods), 6))
        if len(methods) == 1:
            axes = [axes]
        
        for idx, method in enumerate(methods):
            attr = attribution_result['attributions'][method]
            gene_attr = attr[gene_idx, :]
            
            # Get top 10 neurons
            top_10 = np.argsort(np.abs(gene_attr))[-10:][::-1]
            
            axes[idx].barh(range(10), gene_attr[top_10])
            axes[idx].set_yticks(range(10))
            axes[idx].set_yticklabels([f'N{i}' for i in top_10])
            axes[idx].set_xlabel('Attribution Score')
            axes[idx].set_title(method.replace('_', ' ').title())
            axes[idx].invert_yaxis()
        
        fig.suptitle(f'Top Neurons for {gene_name} (Different Methods)')
        plt.tight_layout()
        plt.show()
    
    return fig
