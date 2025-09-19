import scanpy as sc
import anndata as ad
import pandas as pd


def read_dscigm_datasets(path_folder, dataset_name, col, name_column):
    dataset = pd.read_csv(path_folder + '/' + dataset_name + '_HIGHPRE.csv')
    dataset = dataset.drop(columns=['Unnamed: 0'])
    
    dataset_meta = pd.read_csv(path_folder + '/' + dataset_name + '_cell_anno.csv')
    
    adata = ad.AnnData(X=dataset, obs=dataset_meta)
    
    if col == True:
        column = dataset_name + '@colData$cell_type1'
    else:
        column = name_column
    adata.obs = adata.obs.rename(columns={column: 'Cell Type'})
    
    return adata

def convert_anndata_in_df(adata):
    df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    return df


def preprocessing_anndata(adata, select_hvg, nb_hvg):
    
    sc.pp.filter_cells(adata, min_genes=200)  
    sc.pp.filter_genes(adata, min_cells=3)    
    sc.pp.normalize_total(adata, target_sum=1, inplace=False)
    sc.pp.log1p(adata)
    #sc.pp.scale(adata, zero_center=True)
    
    if select_hvg == True:
        sc.pp.highly_variable_genes(adata, n_top_genes=nb_hvg)
        adata = adata[:, adata.var['highly_variable']]
    
    return adata

def apply_clustering_anndata(adata):
    sc.pp.pca(adata, n_comps=30)
    sc.pp.neighbors(adata, n_pcs=15, metric='cosine')
    sc.tl.louvain(adata)
    sc.tl.leiden(adata)
    sc.tl.umap(adata)
    return adata
    
def plot_umap_adata(adata, column_name):
    return sc.pl.umap(adata, color=column_name, frameon=False, size=20)

def write_adata(path_where_to_write, dataset_name, batch_size, num_epochs, lr):
    adata.write(path_where_to_write + "/" + f"adata_latent_{dataset_name}_{batch_size}_{num_epochs}_{lr}.h5ad")