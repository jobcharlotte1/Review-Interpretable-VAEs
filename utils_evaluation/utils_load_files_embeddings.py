import pandas as pd
import numpy as np
import sys
import os
import argparse

sys.path.append('/home/BS94_SUR/phD/review/models reproductibility/VEGA/vega-reproducibility/src')
import vanilla_vae
import vega
import train_vanilla_vae_suppFig1
import utils
from utils import *
from learning_utils import *
from vanilla_vae import VanillaVAE
from vega_model import VEGA

def load_pathways_vega(load, data_path, pathway_file):
    if load == True:
        adata = sc.read(data_path)
    else:
        adata = data_path
    pathway_dict = read_gmt(pathway_file, min_g=0, max_g=1000)
    pathway_mask = create_pathway_mask(adata.var.index.tolist(), pathway_dict, add_missing=1, fully_connected=True)
    list_pathways = list(pathway_dict.keys()) + ['UNANNOTATED_'+str(k) for k in range(1)]
    return list_pathways

def create_path_embeddings(name_model, name_dataset, split, random_seed, perturbation, source, pathway_selected, path_saved_embeddings):
    if source == 'original':
        path_embeddings = path_saved_embeddings + f'{name_model}_{name_dataset}_embeddings_{split}_original_seed_{random_seed}_trial.txt'
    if source == 'perturbated':
        path_embeddings = path_saved_embeddings + f'{name_model}_{name_dataset}_embeddings_{split}_{pathway_selected}_{perturbation}_seed_{random_seed}.txt'
    return path_embeddings


def load_embeddings(path_embeddings_pathway, list_pathways):
    embeddings_pathway = np.loadtxt(path_embeddings_pathway)
    if list_pathways:
        embeddings_pathway= pd.DataFrame(embeddings_pathway, columns=list_pathways)
    return embeddings_pathway


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Vega pathways and embeddings")
    parser.add_argument("function", choices=["load_pathways_vega", "create_path_embeddings", "load_embeddings"])

    # Arguments for pathways
    parser.add_argument("data_path", help="Path to AnnData (.h5ad) file")
    parser.add_argument("pathway_file", help="Path to pathway file for Vega dataset")

    # Arguments for embeddings
    parser.add_argument("--name_dataset")
    parser.add_argument("--split")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--name_model")
    parser.add_argument("--source", choices=["original", "perturbated"], default="original")
    parser.add_argument("--perturbation", default=None)
    parser.add_argument("--pathway_selected", default=None)
    parser.add_argument("--path_saved_embeddings", default="path_saved_embeddings")
    parser.add_argument("--list_pathways", action="store_true", help="Add pathway names as column headers")

    args = parser.parse_args()
    if args.function == "load_pathways_vega":
        list_pathways = load_pathways_vega(args.data_path, args.pathway_file)
        print("Number of pathways:", len(list_pathways))
    else:
        list_pathways = load_pathways_vega(args.data_path, args.pathway_file)
        print("Number of pathways:", len(list_pathways))
        path_embeddings = create_path_embeddings(args, args.source, args.pathway_selected, args.path_saved_embeddings)
        print("Embedding path:", path_embeddings)
        embeddings_df = load_embeddings(path_embeddings, args.list_pathways)
        print("Embeddings shape:", embeddings_df.shape)
        
def load_parquet_repository(path_repository):
    parquet_files = [os.path.join(path_repository, f) for f in os.listdir(path_repository) if f.endswith(".parquet")]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {path_repository}")
        
    df_list = [pd.read_parquet(file) for file in parquet_files]
    big_df = pd.concat(df_list, axis=0, ignore_index=True)
    
    return big_df