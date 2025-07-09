import pandas as pd
import scanpy as sc
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import sparse

def get_gene_names(adata, labels):
    genes_names = adata.var[labels].values
    return genes_names

def get_expression_matrix(adata, sparse):
    if sparse == True:
        df_expression = pd.DataFrame.sparse.from_spmatrix(
            adata.X,
            index=adata.obs_names,
            columns=adata.var_names
        ).sparse.to_dense()
    else:
        df_expression = pd.DataFrame(
            adata.X,
            index=adata.obs_names,
            columns=adata.var_names
        )
    return df_expression

def get_top_genes(df_expression, genes_names):
    genes_sums = np.array(df_expression.sum(axis=0)).flatten()
    genes_df = pd.DataFrame({'gene': genes_names, 'total_expr': genes_sums})
    genes_df = genes_df.sort_values(by="total_expr", ascending=False)
    genes_df = genes_df.reset_index(drop=False)
    return genes_df

def plot_genes_distribution(genes_df, df_expression, n_genes):
    fig, ax = plt.subplots(3, int(n_genes/3), figsize=(8, 6))
    ax = ax.flatten()

    for i in range(n_genes):
        idx = genes_df.iloc[i, :]['index']
        gene = genes_df.iloc[i, :]['gene']
        sns.kdeplot(df_expression[gene], shade=True, color="rebeccapurple", ax=ax[i])
        ax[i].set_title(f" {gene} Distribution")

    plt.tight_layout()
    plt.show()