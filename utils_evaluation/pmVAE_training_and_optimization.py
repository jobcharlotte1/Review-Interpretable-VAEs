import anndata
import numpy as np
import pandas as pd
import tensorflow as tf
import scanpy as sc
from matplotlib import pyplot as plt
import anndata as ad

from pathlib import Path
from sklearn.model_selection import train_test_split

import sys
sys.path.append('/home/BS94_SUR/phD/review/models reproductibility/pmvae/')

from pmvae.model import PMVAE
from pmvae.train import train
from pmvae.utils import load_annotations


def build_pmVAE_dataset(adata, pathway_file):
    adata.varm['annotations'] = load_annotations(
        pathway_file,
        adata.var_names,
        min_genes=13
    )
    pathway_mask = adata.varm['annotations'].astype(bool).T
    return adata, pathway_mask

def preprocess_adata(adata, n_top_genes=5000):
    """ Simple (default) sc preprocessing function before autoencoders """
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    return adata

def train_test_split(adata, test_size, suffle, random_state, batch_size):
    trainset, testset = train_test_split(
        data.X,
        test_size,
        shuffle,
        random_state,
        
    )
    trainset = tf.data.Dataset.from_tensor_slices(trainset)
    trainset = trainset.shuffle(5 * batch_size).batch(batch_size)
    return trainset, testset


def pmvae_training_data(adata, n_top_genes, pathway_file):
    adata = preprocess_adata(adata, n_top_genes=5000)
    adata, pathway_mask = build_pmVAE_dataset(adata, pathway_file)
    return adata, pathway_mask

def grid_search_pmVAE(
    betas,
    batch_sizes,
    lrs,
    epochs_list,
    pathway_mask,
    device,
    test_size,
    suffle, 
    random_state,
    save_path="grid_search_results.csv"
):
    results = []

    # If file exists, resume and append to it
    if os.path.exists(save_path):
        print(f"Resuming from existing file: {save_path}")
        results_df = pd.read_csv(save_path)
        # Convert existing rows to dicts
        results = results_df.to_dict(orient="records")

    for beta, batch_size, lr, n_epochs in itertools.product(betas, batch_sizes, lrs, epochs_list):

        print(f"\n=== Running config: beta={beta}, batch={batch_size}, lr={lr}, epochs={n_epochs} ===")

        # ---- Create data loaders ----
        trainset, testset = train_test_split(adata, test_size, shuffle, random_state, batch_size)

        model = PMVAE(
            membership_mask=pathway_mask.values,
            module_latent_dim=4,
            hidden_layers=[12],
            add_auxiliary_module=True,
            beta=beta,
            kernel_initializer='he_uniform',
            bias_initializer='zero',
            activation='elu',
            terms=pathway_mask.index
        )

        opt = tf.keras.optimizers.Adam(learning_rate=lr)



        # ---- Train model ----
        history = train(model, opt, trainset, testset, nepochs=n_epochs)

        # Get validation loss or last element
        final_loss = history["valid_loss"][-1] 

        print(f"Final validation loss: {final_loss:.4f}")

        # ---- Save record for this run ----
        result = {
            "beta": beta,
            "batch_size": batch_size,
            "lr": lr,
            "n_epochs": n_epochs,
            "val_loss": final_loss,
        }
        results.append(result)

        # ---- Save DataFrame to disk at each iteration ----
        df = pd.DataFrame(results)
        df.to_csv(save_path, index=False)
        print(f"Saved progress to {save_path}")

    # Return final dataframe and best config
    df = pd.DataFrame(results)
    best_row = df.loc[df['val_loss'].idxmin()].to_dict()

    return df, best_row


