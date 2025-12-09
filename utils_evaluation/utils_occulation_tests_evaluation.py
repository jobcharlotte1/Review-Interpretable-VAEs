import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from joblib import Parallel, delayed
import os
from pathlib import Path


def overlap_score(list_genes_pathway1, list_genes_pathway2, list_genes_in_adata):
    set_intersection_with_adata1 = set(list_genes_in_adata).intersection(set(list_genes_pathway1))
    set_intersection_with_adata2 = set(list_genes_in_adata).intersection(set(list_genes_pathway2))
    intersection = set_intersection_with_adata1.intersection(set_intersection_with_adata2)
    overlap = len(intersection)/len(set_intersection_with_adata2 )

    return overlap 

def reduction_score(activation_score_1, activation_score_2):
    return abs(activation_score_1) - abs(activation_score_2)

def reduction_rate(activation_score_1, activation_score_2):
    return (abs(activation_score_1) - abs(activation_score_2))/abs(activation_score_1)

def reduction_zscore(activation_score_1, activation_score_2, overall_activ_score):
    return abs(activation_score_1) - abs(activation_score_2)/np.std(overall_activ_score)

def overall_reduction_score(nb_cells, vector_activation_score_1, vector_activation_score_2):
    return 1/nb_cells * np.sum(reduction_score(vector_activation_score_1, vector_activation_score_2))

def overall_reduction_rate(nb_cells, vector_activation_score_1, vector_activation_score_2):
    return 1/nb_cells * np.sum(reduction_rate(vector_activation_score_1, vector_activation_score_2))

def overall_reduction_zscore(nb_cells, vector_activation_score_1, vector_activation_score_2, overall_activ_score):
    return 1/nb_cells * np.sum(reduction_zscore(vector_activation_score_1, vector_activation_score_2, overall_activ_score))


def proba_impact_pathway_perturbation(df_scores, overlap_matrix, pathway_observed, pathway_compared, name_reduction_metric, name_activation_column, name_overlap_compared_pathway, name_overlap_column, activ_threshold, overlap_threshold):
    overlap_score = overlap_matrix[(overlap_matrix['Pathway Selected'] == pathway_observed) & (overlap_matrix[name_overlap_compared_pathway] == pathway_compared)][name_overlap_column].values[0]
    #print(overlap_score)
    if overlap_score >= overlap_threshold:
        print(overlap_score)
        return None
    
    df_selected = df_scores[
        (df_scores['pathway perturbated'] == pathway_observed) & 
        (df_scores['pathway observed'] == pathway_compared)
    ]
    
    df_selected = df_selected[abs(df_selected[name_activation_column]) > activ_threshold]
    
    if df_selected.empty:
        return None 
    
    observed_value = df_scores[
        (df_scores['pathway observed'] == pathway_observed) & 
        (df_scores['pathway perturbated'] == pathway_observed)
    ][name_reduction_metric].values[0]
    
    count_pathway_p_over_pathway_q = (observed_value > df_selected[name_reduction_metric]).sum()
    count_threshold_requirements = len(df_selected)
    
    proba = count_pathway_p_over_pathway_q / count_threshold_requirements
    
    return proba


def overall_proba_impact_pathway_perturbation(
    pathway_observed,
    list_pathways,
    df_scores,
    overlap_matrix,
    name_reduction_metric,
    name_activation_column,
    name_overlap_compared_pathway,
    name_overlap_column,
    activ_threshold,
    overlap_threshold,
    n_cpus=n_cpus
):
    """
    Compute probabilities for all comparisons of one pathway in parallel.
    """

    def run_one(pathway_compared):
        proba = proba_impact_pathway_perturbation(
            df_scores=df_scores,
            overlap_matrix=overlap_matrix,
            pathway_observed=pathway_observed,
            pathway_compared=pathway_compared,
            name_reduction_metric=name_reduction_metric,
            name_activation_column=name_activation_column,
            name_overlap_compared_pathway=name_overlap_compared_pathway,
            name_overlap_column=name_overlap_column,
            activ_threshold=activ_threshold,
            overlap_threshold=overlap_threshold
        )
        if proba is None:
            return None, None
        record = {
            'Pathway Perturbated': pathway_observed,
            'Pathway Compared': pathway_compared,
            f'{name_reduction_metric} Probability': proba
        }
        return proba, record

    # Parallel execution over all pathway comparisons
    results = Parallel(n_jobs=n_cpus)(
        delayed(run_one)(pathway_compared) for pathway_compared in tqdm(
            list_pathways, desc=f"Comparison {pathway_observed} vs {pathway_compared}", ncols=100
        )
    )

    # Collect results
    probas_for_pathway = [p for p, r in results if p is not None]
    records = [r for p, r in results if r is not None]

    if len(probas_for_pathway) == 0:
        return None, records

    proba_pathway = sum(probas_for_pathway) / len(probas_for_pathway)
    return proba_pathway, records


def compute_metrics_for_pathway(
    pathway_perturbated,
    list_pathways,
    path_to_save_reduction_scores,
    overlap_matrix,
    name_activation_column,
    name_overlap_compared_pathway,
    name_overlap_column,
    activ_threshold,
    overlap_threshold,
    n_cpus
):
    """
    Compute all three reduction metrics for one pathway_perturbated.
    """
    # Load scores
    df_scores = pd.read_parquet(
        os.path.join(path_to_save_reduction_scores, f'vega_reduction_scores_{pathway_perturbated}_perturbation.parquet')
    )
    df_scores["std_activation"] = df_scores.groupby("pathway observed")["original neuron activation"].transform("std")
    df_scores["reduction z-score"] = df_scores['abs difference perturbation'] / df_scores["std_activation"]

    results = {}
    all_records = []

    # Compute the three metrics
    for name_reduction_metric in ['abs difference perturbation', 'reduction_score', 'reduction z-score']:
        proba, records = overall_proba_impact_pathway_perturbation(
            pathway_observed=pathway_perturbated,
            list_pathways=list_pathways[:-1], 
            df_scores=df_scores,
            overlap_matrix=overlap_matrix,
            name_reduction_metric=name_reduction_metric,
            name_activation_column=name_activation_column,
            name_overlap_compared_pathway=name_overlap_compared_pathway,
            name_overlap_column=name_overlap_column,
            activ_threshold=activ_threshold,
            overlap_threshold=overlap_threshold,
            n_cpus=n_cpus 
        )
        results['activation_threshold'] = activ_threshold
        results['overlap_threshold'] = overlap_threshold
        # Save metric results
        if name_reduction_metric == 'abs difference perturbation':
            results['Reduction Score Probability'] = proba
        elif name_reduction_metric == 'reduction_score':
            results['Reduction Rate Probability'] = proba
        elif name_reduction_metric == 'reduction z-score':
            results['Reduction Z-Score Probability'] = proba
        

        all_records.extend(records)

    return pathway_perturbated, results, all_records


def compute_all_pathways_parallel(
    list_pathways_perturbated,
    list_pathways_compared,
    path_to_save_reduction_scores,
    overlap_matrix,
    name_activation_column,
    name_overlap_compared_pathway,
    name_overlap_column,
    activ_threshold,
    overlap_threshold,
    path_to_save_final_df,
    n_cpus=n_cpus
):
    pathway_probs = []
    all_records = []

    results = Parallel(n_jobs=n_cpus)(
        delayed(compute_metrics_for_pathway)(
            pathway_perturbated,
            list_pathways_compared,
            path_to_save_reduction_scores,
            overlap_matrix,
            name_activation_column,
            name_overlap_compared_pathway,
            name_overlap_column,
            activ_threshold,
            overlap_threshold,
            n_cpus
        )
        for pathway_perturbated in tqdm(list_pathways_perturbated, desc="Processing pathways", ncols=100)
    )

    # Collect results
    for pathway_perturbated, metrics, records in results:
        pathway_probs.append({'Pathway Perturbated': pathway_perturbated, **metrics})
        all_records.extend(records)

    # Convert to DataFrame
    df_probas = pd.DataFrame(pathway_probs)
    df_all_pairs = pd.DataFrame(all_records).groupby(['Pathway Perturbated', 'Pathway Compared']).first().reset_index()

    # Compute overall row
    metric_names = ['Reduction Score Probability', 'Reduction Rate Probability', 'Reduction Z-Score Probability']
    overall_values = {col: df_probas[col].dropna().mean() for col in metric_names}

    df_probas = pd.concat([df_probas, pd.DataFrame([{'Pathway Perturbated': 'Overall', 'activation_threshold': activ_threshold, 'overlap_threshold': overlap_threshold, **overall_values}])])
    
    df_probas.to_csv(path_to_save_final_df+f'df_probas_all_pathways_activt_{activ_threshold}_overlapt_{overlap_threshold}.csv')
    df_all_pairs.to_csv(path_to_save_final_df+f'df_all_pairs_all_pathways_activt_{activ_threshold}_overlapt_{overlap_threshold}.csv')

    return df_probas, df_all_pairs
