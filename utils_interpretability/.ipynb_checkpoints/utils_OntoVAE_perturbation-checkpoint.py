import pandas as pd
import numpy as np
import argparse
import os
import sys
import gc
import re
from threading import Lock



def safe_filename(name: str) -> str:
    name = "_".join(name.split())
    name = re.sub(r'[\\/:"*?<>|]+', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')    
    return name


def compute_scores_OntoVAE(pathway_selected, split, embeddings_original, embeddings_perturbated, list_pathways, name_model, path_to_save):
    #print(pathway_selected)
    rows = []
    for pathway in list_pathways:
        rows.append(pd.DataFrame({
        'model': [name_model] * len(embeddings_original), 
        'cell_id': [f'cell_{i}' for i in range(len(embeddings_original))],
        'perturbation': [0] * len(embeddings_original),
        'pathway perturbated': [pathway_selected] * len(embeddings_original),
        'pathway observed': [pathway] * len(embeddings_original), 
        'original neuron activation': embeddings_original[pathway],
        'perturbated neuron activation': embeddings_perturbated[pathway],
        'magnitude perturbation': abs(embeddings_original[pathway] - embeddings_perturbated[pathway]),
        'reduction_score': abs(embeddings_original[pathway]) - abs(embeddings_perturbated[pathway]),
        'reduction_rate': ((abs(embeddings_original[pathway]) - abs(embeddings_perturbated[pathway]))/(abs(embeddings_original[pathway]))),
        'reduction_z-score': (abs(embeddings_original[pathway]) - abs(embeddings_perturbated[pathway]))/(embeddings_original[pathway].std()),
        
        }))
                    
    df = pd.concat(rows, axis=0, ignore_index=True)
    
    pathway_selected_2 = safe_filename(pathway_selected)
    
    if path_to_save:
        os.makedirs(path_to_save, exist_ok=True)   
        file_path = os.path.join(path_to_save, f'{name_model}_reduction_scores_{pathway_selected_2}_{split}_perturbation.parquet')
        df.to_parquet(file_path, engine='pyarrow')
    return df


def compute_one_df_scores_perturbation_OntoVAE(list_pathways, name_dataset, split, pathway_selected, perturbation, name_model, type_file, path_to_save_embeddings_original, path_to_save_embeddings_perturbated, path_to_save_reduction_scores):
    
    pathway_selected_2 = safe_filename(pathway_selected)
    
    if type_file == 'layers':
        path_embeddings_original = os.path.join(path_to_save_embeddings_original, f'{name_model}_{name_dataset}_layers_embeddings_{split}_original.parquet')
        embeddings_original = pd.read_parquet(path_embeddings_original, engine="pyarrow")
        path_embeddings_perturbated = os.path.join(path_to_save_embeddings_perturbated, f'{name_model}_{name_dataset}_layers_embeddings_{split}_{pathway_selected_2}_{perturbation}.parquet')
        embeddings_perturbated = pd.read_parquet(path_embeddings_perturbated, engine="pyarrow")
    else:
        path_embeddings_original = os.path.join(path_to_save_embeddings_original, f'{name_model}_{name_dataset}_embeddings_{split}_original.parquet')
        embeddings_original = pd.read_parquet(path_embeddings_original, engine="pyarrow")
        path_embeddings_perturbated = os.path.join(path_to_save_embeddings_perturbated, f'{name_model}_{name_dataset}_embeddings_{split}_{pathway_selected_2}_{perturbation}.parquet')
        embeddings_perturbated = pd.read_parquet(path_embeddings_perturbated, engine="pyarrow")

    df = compute_scores_OntoVAE(pathway_selected, split, embeddings_original, embeddings_perturbated, list_pathways, name_model, path_to_save_reduction_scores)
    return df

