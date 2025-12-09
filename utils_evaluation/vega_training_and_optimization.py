import sys
import os
from pathlib import Path

sys.path.append('/home/user/Review-Interpretable-VAEs/Review-Interpretable-VAEs/utils_evaluation')
import utils_train_models
from utils_train_models import *
from utils_load_files_embeddings import *

sys.path.append('/home/user/Review-Interpretable-VAEs/Review-Interpretable-VAEs/cloned_github_models/vega/vega-reproducibility/src')
#import vanilla_vae
import vega
import train_vanilla_vae_suppFig1
import utils
from utils import *
from learning_utils import *
#from vanilla_vae import VanillaVAE
from vega_model import VEGA
import torch
import itertools
import pandas as pd
import os

def access_data_vega(data_path, adata, pathway_file, path_data_description):
    if data_path is not None:
        adata = sc.read(data_path)
    pathway_dict = read_gmt(pathway_file, min_g=0, max_g=1000)
    pathway_mask = create_pathway_mask(adata.var.index.tolist(), pathway_dict, add_missing=1, fully_connected=True)
    list_pathways = load_pathways_vega(False, adata, pathway_file)
    if path_data_description is not None:
        df_genespathways = pd.read_parquet(path_data_description + f'df_pathways_genes_description.parquet')
        overlap_matrix = pd.read_csv(path_data_description + 'overlap_matrix_vega.csv')
    else:
        df_genespathways = None
        overlap_matrix = None
    
    return adata, pathway_dict, pathway_mask, list_pathways, df_genespathways, overlap_matrix


def extract_x_y_from_adata(adata: AnnData, column_labels_name: pd.Series):
    X = pd.DataFrame(adata.X, index=adata.obs.index)
    y = adata.obs[column_labels_name]
    return X, y

def split_data(X, y, train_size, random_seed):
    X_train,  X_test, labels_train,  labels_test = train_test_split(
        X, y, train_size=train_size, random_state=random_seed, stratify=y)
    return X_train,  X_test, labels_train,  labels_test

def extract_index(X):
    index_df = X.index
    return index_df
    
def build_adata_from_X(adata, index_df):
    adata = adata[adata.obs.index.isin(index_df)]
    return adata, index_df

def encode_y(y):
    le = preprocessing.LabelEncoder().fit(y)
    y_encoded = torch.Tensor(le.transform(y))
    return y_encoded

def build_vega_dataset(adata, y_encoded, pathway_file):
    if sparse.issparse(adata.X):
        data = adata.X.A
    else:
        data = adata.X

    data = torch.Tensor(data)
    data = UnsupervisedDataset(data, targets=y_encoded)

    pathway_dict = read_gmt(pathway_file, min_g=0, max_g=1000)
    pathway_mask = create_pathway_mask(adata.var.index.tolist(), pathway_dict, add_missing=1, fully_connected=True)

    return data, pathway_dict, pathway_mask

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


def create_vega_training_data(name_model, preprocess, select_hvg, n_top_genes, random_seed, train_size, column_labels_name, adata, pathway_file):
    adata = preprocess_adata(adata, n_top_genes=n_top_genes)
    X, y = extract_x_y_from_adata(adata, column_labels_name)
    X_train,  X_test, labels_train,  labels_test = split_data(X, y, train_size, random_seed)
    y_train = encode_y(labels_train)
    y_test = encode_y(labels_test)
    index_train = extract_index(X_train)
    index_test = extract_index(X_test)
    adata_train, index_train = build_adata_from_X(adata, index_train)
    adata_test, index_test = build_adata_from_X(adata, index_test)

    train_ds, pathway_dict, pathway_mask = build_vega_dataset(adata_train, y_train, pathway_file)
    test_ds, pathway_dict, pathway_mask = build_vega_dataset(adata_test, y_test, pathway_file)
    return adata, adata_train, adata_test, train_ds, test_ds, pathway_dict, pathway_mask

def grid_search_VEGA(
    betas,
    batch_sizes,
    lrs,
    epochs_list,
    pathway_mask,
    device,
    p_drop,
    train_ds,
    test_ds,
    train_p,
    test_p,
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
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)

        # ---- Model parameters ----
        dict_params = {
            "pathway_mask": pathway_mask,
            "n_pathways": pathway_mask.shape[1],
            "n_genes": pathway_mask.shape[0],
            "device": device,
            "beta": beta,
            "save_path": False,
            "dropout": p_drop,
            "pos_dec": True,
        }

        # ---- Instantiate model ----
        model = VEGA(**dict_params).to(device)

        # ---- Train model ----
        hist = model.train_model(
            train_loader,
            lr,
            n_epochs,
            train_p,
            test_p,
            test_loader,
            save_model=False
        )

        # Get validation loss or last element
        final_loss = hist["valid_loss"][-1] 

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
