import os
import sys
import scanpy as sc
import scipy.sparse as sp
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
import anndata as ad
import re
import math
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('/home/BS94_SUR/phD/review/models reproductibility/OntoVAE/cobra-ai')
from cobra_ai.module.ontobj import *
from cobra_ai.module.utils import *
from cobra_ai.model.onto_vae import *
from cobra_ai.model.cobra import *
from cobra_ai.module.autotune import *

def build_overlap_matrix_OntoVAE(df_mask, adata, list_pathways):
    pathway_dict = {
        pathway: df_mask.index[df_mask[pathway].astype(bool)].tolist()
        for pathway in df_mask.columns
    }
    all_results = []  # accumulate results across all pathways

    for pathway_selected in list_pathways:
        list_genes_to_perturbate = df_mask.loc[(df_mask[pathway_selected] == 1)].index.tolist()
        genes1 = [gene for gene in list_genes_to_perturbate if gene in adata.var_names]

        for pathway_compared, genes in pathway_dict.items():
            genes2 = [gene for gene in genes if gene in adata.var_names]
            intersection = set(genes1).intersection(set(genes2))


            all_results.append({
                "Pathway Selected": pathway_selected,   # which pathway we started with
                "Nb Genes Pathway Selected": len(genes1),
                "Compared Pathway": pathway_compared,            # the pathway we are comparing against
                "Nb Genes Compared Pathway": len(genes2),
                "Genes Overlap": len(intersection),
                "Commun Genes": list(intersection),
                "Overlap Proportion": len(intersection)/len(genes1) if len(genes1) != 0 else 0
            })

    # Convert to a single dataframe
    overlap_matrix = pd.DataFrame(all_results)

    # Sort for convenience (optional)
    overlap_matrix  = overlap_matrix .sort_values(
        by=["Pathway Selected", "Genes Overlap"],
        ascending=[True, False]
    ).reset_index(drop=True)
    
    return overlap_matrix


def access_data_OntoVAE(data_path, adata, pathway_file, path_data_description):
    if data_path is not None:
        adata = sc.read(data_path)
    ontobj = Ontobj()
    ontobj.load(pathway_file)
    genes = ontobj.extract_genes()
    adata = adata.copy()
    adata = setup_anndata_ontovae(adata, ontobj)
    ontologie = adata.uns['_ontovae']
    list_genes_ontologie = ontologie['genes']
    list_pathways = ontologie['annot']['Name']
    df_genespathways = pd.read_parquet(path_data_description + f'df_pathways_genes_description.parquet')
    #overlap_matrix = pd.read_csv(path_data_description + 'overlap_matrix_OntoVAE.csv')
    df_mask = pd.DataFrame(ontologie['masks'][len(ontologie['masks'])-1], columns=ontologie['annot']['Name'], index=ontologie['genes'])
    overlap_matrix = build_overlap_matrix_OntoVAE(df_mask, adata, list_pathways)
    
    return adata, ontobj, ontologie, list_genes_ontologie, list_pathways, df_genespathways, overlap_matrix

def preprocess_adata(adata, n_top_genes=5000):
    """ Simple (default) sc preprocessing function before autoencoders """
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    #sc.pp.normalize_total(adata, target_sum=1e4)
    #sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    return adata

def safe_filename(name: str) -> str:
    name = "_".join(name.split())
    name = re.sub(r'[\\/:"*?<>|]+', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')    
    return name


def plot_interval_subplots(
    df,
    intervals,
    nrows=2,
    ncols=4,
    bw_adjust=1,
    figsize=(20, 8)
):
    """
    Plot one KDE per interval in a grid with the same x-axis range.

    Parameters
    ----------
    df : pandas.DataFrame
    intervals : list of tuples
        Each tuple is (start_col_index, end_col_index)
    nrows : int
        Number of subplot rows
    ncols : int
        Number of subplot columns
    bw_adjust : float
        KDE bandwidth adjustment
    figsize : tuple
        Overall figure size
    """

    # First, compute global min/max across all intervals
    all_values = []
    for start, end in intervals:
        vals = df.iloc[:, start:end].values.flatten()
        vals = vals[~np.isnan(vals)]
        if np.var(vals) != 0:  # skip zero variance
            all_values.append(vals)
    if len(all_values) == 0:
        print("All intervals have zero variance")
        return

    all_values = np.concatenate(all_values)
    xmin, xmax = all_values.min(), all_values.max()

    # Create subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=figsize,
                             squeeze=False)

    axes_flat = axes.flatten()

    for i, (ax, (start, end)) in enumerate(zip(axes_flat, intervals)):
        subset = df.iloc[:, start:end]
        values = subset.values.flatten()
        values = values[~np.isnan(values)]

        if np.var(values) == 0:
            ax.text(0.5, 0.5, "Zero variance", ha='center', va='center', transform=ax.transAxes)
        else:
            sns.kdeplot(values, ax=ax, bw_adjust=bw_adjust)

        ax.set_title(f"Depth {i}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_xlim(xmin, xmax)  # same x-axis

    # Hide any unused subplots
    for j in range(i+1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    plt.show()
