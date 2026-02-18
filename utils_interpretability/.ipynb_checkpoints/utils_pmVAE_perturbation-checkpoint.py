import sys
import vega
import os
from pathlib import Path


def get_genes_in_pathway_pmVAE(pathway_selected, pathway_mask):
    genes_in_adata = pathway_mask.columns[
        (pathway_mask.loc[pathway_selected] == 1) 
    ].tolist()
    other_genes = pathway_mask.columns[
        (pathway_mask.loc[pathway_selected] == 0) 
    ].tolist()

    return genes_in_adata, other_genes


def pmVAE_perturbation(model, name_dataset, data, module_latent_dim, pathway_selected, list_pathways, list_genes, latent_names, pathway_mask, perturbation, split, path_to_save_embeddings, path_to_save_reconstructed):
    #select list of genes to perturb involved in pathway_name
    print(pathway_selected)
    genes_in_adata, other_genes = get_genes_in_pathway_pmVAE(pathway_selected, pathway_mask)

    df_data_perturbated = pd.DataFrame(data, columns=list_genes)
    df_data_perturbated.loc[:, genes_in_adata]  = 0

    outputs = model.call(df_data_perturbated.values)
    
    embeddings = outputs.z.numpy()
    embeddings = pd.DataFrame(embeddings, columns=latent_names)
    embeddings_groupped = get_grouped_embeddings_pmvae(module_latent_dim, embeddings, list_pathways)
    
    X_reconstructed = outputs.global_recon.numpy()
    
    n = f'{split}_{pathway_selected}_{perturbation}'
    
    if path_to_save_embeddings: 
        file_path = os.path.join(path_to_save_embeddings, f'pmVAE_{name_dataset}_embeddings_{n}.parquet')
        embedding_groupped.to_parquet(file_path, engine='pyarrow')
    if path_to_save_reconstructed:
        np.savetxt(Path(path_to_save_reconstructed + f'/pmVAE_{name_dataset}_reconstruction_{n}.txt'), X_reconstructed.cpu().detach().numpy())
    
    return embeddings, X_reconstructed