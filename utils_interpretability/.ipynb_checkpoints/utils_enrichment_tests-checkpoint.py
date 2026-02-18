import sys
import os
import gseapy as gp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Gene set enrichment
path_gene_sets_vega = '/home/BS94_SUR/phD/review/models reproductibility/VEGA/vega-reproducibility/data/reactomes.gmt'

def run_gsea(gene_list, gene_sets, top_n=10, organism='Human'):
    """
    Perform gene set enrichment analysis (ORA) using gseapy.enrichr.
    
    Returns:
    - pandas DataFrame with top pathways and enrichment statistics
    """
    
    if len(gene_list) == 0:
        raise ValueError("gene_list is empty!")
    
    # Perform enrichment analysis
    enr = gp.enrichr(
        gene_list=gene_list,
        gene_sets=gene_sets,
        organism=organism,
        outdir=None,  # do not save files, return results in memory
        cutoff=0.05   # only pathways with adjusted p < 0.05
    )
    
    # Get results
    if enr.results.empty:
        print("No significant pathways found.")
        return pd.DataFrame()
    
    # Sort by adjusted p-value and take top_n
    top_results = enr.results.sort_values('Adjusted P-value').head(top_n)
    
    return top_results


def compute_score_gsea(gene_sets, top_genes_df, top_n, method_name):
    count_pathways = 0
    unique_neurons = top_genes_df['latent_neuron'].unique()
    valid_neurons = 0
    for neuron_idx in unique_neurons:
        #print(neuron_idx)
        neuron_data = top_genes_df[
            (top_genes_df['latent_neuron'] == neuron_idx) & 
            (top_genes_df['method'] == method_name) 
        ]
        if neuron_data.empty:
            print(f"Warning: No data found for neuron {neuron_idx} with method {method}")
            continue
        top_genes = neuron_data['gene'].tolist()
        pathway_name = neuron_data['pathway_name'].unique()
        #print(pathway_name)
        #print(pathway_name[0])
        if len(top_genes) == 0:
            print(f"Warning: No genes found for neuron {neuron_idx}")
            continue
        if len(top_genes) != 0:
            valid_neurons +=1
            #print(valid_neurons)
        try:
            top_pathways_df = run_gsea(top_genes, gene_sets=gene_sets, top_n=top_n)
            #print(top_pathways_df)
            
            if not top_pathways_df.empty:
                top_pathways = top_pathways_df['Term'].tolist()
                #print(top_pathways)
                
                if pathway_name[0] in top_pathways:
                    count_pathways += 1
                    #print(count_pathways)
        except Exception as e:
            print(f"Error running GSEA for neuron {neuron_idx}: {e}")
            continue
    
    # Return score based on valid neurons only
    if valid_neurons == 0:
        print("Warning: No valid neurons found!")
        return 0.0
    
    return count_pathways / valid_neurons

