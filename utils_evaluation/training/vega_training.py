import sys
import os
from pathlib import Path

sys.path.append('/home/user/Review-Interpretable-VAEs/Review-Interpretable-VAEs/utils_evaluation')
import vega_training_and_optimization
from vega_training_and_optimization import grid_search_VEGA, access_data_vega, create_vega_training_data

import scanpy as sc
import pandas as pd
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pathway_file = '/home/user/Review-Interpretable-VAEs/Review-Interpretable-VAEs/cloned_github_models/vega/vega-reproducibility/data/reactomes.gmt'
path_data_description = 'gs://bs94_sur-axe-workspace-prod-a05a/Vega/data info/'

#Load data
metadata = pd.read_csv('gs://bs94_sur-axe-workspace-prod-a05a/datasets/PBMC_8K_CellMetainfo_table (1).tsv', sep="\t")
adata = sc.read_10x_h5('/home/user/Review-Interpretable-VAEs/Review-Interpretable-VAEs/data/datasets/PBMC_8K_expression.h5', genome='GRCh38', gex_only=False)
adata.obs = metadata

#prepare data
adata, pathway_dict, pathway_mask, list_pathways, df_genespathways, overlap_matrix = access_data_vega(None, adata, pathway_file, path_data_description)

print(adata.shape)
print(pathway_mask.shape[1])
print(pathway_mask.shape[0])

train_size = 0.75
column_labels_name = 'Celltype (major-lineage)'
name_model = 'Vega'
preprocess = 'Vega'
select_hvg = False
n_top_genes = 2000
random_seed = 42

adata, adata_train, adata_test, train_ds, test_ds, pathway_dict, pathway_mask = create_vega_training_data(name_model, preprocess, select_hvg, n_top_genes, random_seed, train_size, column_labels_name, adata, pathway_file)
print(adata.shape)
print(adata_train.shape)
print(adata_test.shape)
print(pathway_mask.shape[1])
print(pathway_mask.shape[0])


#training
betas = np.logspace(-3, 0, num=10)
batch_sizes = [64, 128, 256]
lrs = np.logspace(-5, 0, num=10)
epochs_list = [800]
p_drop = 0.5
train_p=25
test_p=25
save_path = '/home/user/Review-Interpretable-VAEs/Review-Interpretable-VAEs/data/optimization_results/grid_search_results.csv'



df, best_row = grid_search_VEGA(
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
    save_path
)