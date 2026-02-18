import pandas as pd
import numpy as np
import argparse
import os


def compute_reduction_scores(rows, pathway_selected, embeddings_original, embeddings_pathway_perturbated, list_pathways, name_model, path_to_save):
    
    print(pathway_selected)
    
    df_reduction_scores = pd.DataFrame(columns=list_pathways)
    for pathway in list_pathways:
        df_reduction_scores[pathway] = ((abs(embeddings_original[pathway]) - abs(embeddings_pathway_perturbated[pathway]))/abs(embeddings_original[pathway])) * 100
        rows.append(pd.DataFrame({
            'cell_id': [f'cell_{i}' for i in range(len(df_reduction_scores[pathway_selected]))],
            'perturbation': [0] * len(df_reduction_scores[pathway_selected]),
            'pathway perturbated': [pathway_selected] * len(df_reduction_scores[pathway_selected]),
            'pathway observed': [pathway] * len(df_reduction_scores[pathway_selected]), 
            'reduction score': df_reduction_scores[pathway]
        }))
    df = pd.concat(rows, axis=0, ignore_index=True) 
    
    os.makedirs(path_to_save, exist_ok=True)   
    file_path = os.path.join(path_to_save, f'{name_model}_reduction_scores_{pathway_selected}_perturbation.parquet')
    df.to_parquet(file_path, engine='pyarrow')
    
    return df

def compute_reduction_scores2(rows, pathway_selected, embeddings_original, embeddings_pathway_perturbated, list_pathways, name_model, path_to_save):
    
    print(pathway_selected)
    
    df_reduction_scores = pd.DataFrame(columns=list_pathways)
    for pathway in list_pathways:
        df_reduction_scores[pathway] = (abs(embeddings_original[pathway] - embeddings_pathway_perturbated[pathway])/abs(embeddings_original[pathway])) * 100
        rows.append(pd.DataFrame({
            'cell_id': [f'cell_{i}' for i in range(len(df_reduction_scores[pathway_selected]))],
            'perturbation': [0] * len(df_reduction_scores[pathway_selected]),
            'pathway perturbated': [pathway_selected] * len(df_reduction_scores[pathway_selected]),
            'pathway observed': [pathway] * len(df_reduction_scores[pathway_selected]), 
            'reduction score': df_reduction_scores[pathway]
        }))
    df = pd.concat(rows, axis=0, ignore_index=True) 
    
    os.makedirs(path_to_save, exist_ok=True)   
    file_path = os.path.join(path_to_save, f'{name_model}_reduction_scores_{pathway_selected}_perturbation.parquet')
    df.to_parquet(file_path, engine='pyarrow')
    
    return df

def compute_distances(rows, pathway_selected, embeddings_original, embeddings_perturbated, list_pathways, name_model, path_to_save):  
    print(pathway_selected)
    for pathway in list_pathways:
        rows.append(pd.DataFrame({
        'cell_id': [f'cell_{i}' for i in range(len(embeddings_original))],
        'perturbation': [0] * len(embeddings_original),
        'pathway perturbated': [pathway_selected] * len(embeddings_original),
        'pathway observed': [pathway] * len(embeddings_original), 
        'original neuron activation': embeddings_original[pathway],
        'perturbated neuron activation': embeddings_perturbated[pathway],
        'delta': (embeddings_original[pathway] - embeddings_perturbated[pathway]),
        'magnitude': abs(embeddings_original[pathway] - embeddings_perturbated[pathway]),
        'score 1': ((abs(embeddings_original[pathway]) - abs(embeddings_perturbated[pathway]))/abs(embeddings_original[pathway])) * 100,
        'score 2':  (abs(embeddings_original[pathway] - embeddings_perturbated[pathway])/abs(embeddings_original[pathway])) * 100

    }))
    df = pd.concat(rows, axis=0, ignore_index=True) 
    
    os.makedirs(path_to_save, exist_ok=True)   
    file_path = os.path.join(path_to_save, f'{name_model}_reduction_scores_{pathway_selected}_perturbation.parquet')
    df.to_parquet(file_path, engine='pyarrow')
    
    return df

def compute_scores(rows, pathway_selected, embeddings_original, embeddings_perturbated, list_pathways, name_model, save_file, path_to_save):
    print(pathway_selected)
    for pathway in list_pathways:
        rows.append(pd.DataFrame({
        'cell_id': [f'cell_{i}' for i in range(len(embeddings_original))],
        'perturbation': [0] * len(embeddings_original),
        'pathway perturbated': [pathway_selected] * len(embeddings_original),
        'pathway observed': [pathway] * len(embeddings_original), 
        'original neuron activation': embeddings_original[pathway],
        'perturbated neuron activation': embeddings_perturbated[pathway],
        'magnitude perturbation': abs(embeddings_original[pathway] - embeddings_perturbated[pathway]),
        'abs difference perturbation': abs(embeddings_original[pathway]) - abs(embeddings_perturbated[pathway]),
        'reduction_score': ((abs(embeddings_original[pathway]) - abs(embeddings_perturbated[pathway]))/(abs(embeddings_original[pathway]))),
        }))
                    
    df = pd.concat(rows, axis=0, ignore_index=True)
    
    if save_file == True:
        os.makedirs(path_to_save, exist_ok=True)   
        file_path = os.path.join(path_to_save, f'{name_model}_reduction_scores_{pathway_selected}_perturbation.parquet')
        df.to_parquet(file_path, engine='pyarrow')
    return df

    
    
