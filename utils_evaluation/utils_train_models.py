import sys
import os
from pathlib import Path

name_model = 'VEGA'

if name_model == 'VEGA' or name_model == 'VanillaVAE':
    sys.path.append('/home/user/Review-Interpretable-VAEs/Review-Interpretable-VAEs/cloned_github_models/vega/vega-reproducibility/src')
    import vanilla_vae
    import vega
    import train_vanilla_vae_suppFig1
    import utils
    from utils import *
    from learning_utils import *
    from vanilla_vae import VanillaVAE
    from vega_model import VEGA

if name_model == 'pmVAE':
    import sys
    sys.path.append('/home/BS94_SUR/phD/review/models reproductibility/pmvae/')
    import tensorflow as tf
    from pmvae.model import PMVAE
    from pmvae.train import train
    from pmvae.utils import load_annotations

if name_model == 'OntoVAE':
    sys.path.append('/home/BS94_SUR/phD/review/models reproductibility/OntoVAE/cobra-ai')
    from cobra_ai.module.ontobj import *
    from cobra_ai.module.utils import *
    from cobra_ai.model.onto_vae import *
    import onto_vae
    
if name_model == 'ExpiMap':
    sys.path.append('/home/BS94_SUR/phD/review/models reproductibility/Expimap/expiMap_reproducibility')
    import scarches as sca
    from scarches.utils import add_annotations


import pandas as pd
import scanpy as sc
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import sparse
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import itertools
import argparse
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import anndata as ad
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px  # For color palette
from sklearn.model_selection import train_test_split
#import umap
from scipy.sparse import issparse
#import optuna
#from optuna.samplers import TPESampler
from pathlib import Path
import anndata as ad
from anndata import AnnData

if name_model != 'pmVAE':
    import torch
    from torchinfo import summary


# Ignore warnings in this tutorial
import warnings
warnings.filterwarnings('ignore')


class VAE_prepare_dataset:
    def __init__(self,
                 adata: AnnData,
                 column_labels_name: str,
                 random_seed: int,
                 name_model: str,
                 train_size: int,
                 pathway_file: str,
                 preprocess:bool,
                 select_hvg: bool,
                 n_top_genes: int) -> None:
        super(VAE_prepare_dataset, self).__init__()

        self.adata = adata
        self.column_labels_name = column_labels_name
        self.random_seed = random_seed
        self.name_model = name_model
        self.train_size = train_size
        self.pathway_file = pathway_file
        self.preprocess = preprocess
        self.select_hvg = select_hvg
        self.n_top_genes = n_top_genes
        
    def preprocess_data(self, adata, name_model, preprocess, select_hvg, n_top_genes):
        if name_model == 'Vega':
            vega.utils.setup_anndata(adata)
        elif preprocess == True:
            sc.pp.filter_cells(adata, min_genes=200)
            sc.pp.filter_genes(adata, min_cells=3)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            if select_hvg == True:
                sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
                adata.raw = adata
                adata = adata[:, adata.var.highly_variable]
        return adata

    def extract_x_y_from_adata(self, adata: AnnData, column_labels_name: pd.Series):
        X = pd.DataFrame(adata.X, index=adata.obs.index)
        y = adata.obs[column_labels_name]
        return X, y

    def split_data(self, X, y, train_size, random_seed):
        X_train,  X_test, labels_train,  labels_test = train_test_split(
            X, y, train_size=train_size, random_state=random_seed, stratify=y)
        return X_train,  X_test, labels_train,  labels_test
    
    def extract_index(self, X):
        index_df = X.index
        return index_df
        
    def build_adata_from_X(self, adata, index_df):
        adata = adata[adata.obs.index.isin(index_df)]
        return adata, index_df
    
    def encode_y(self, y):
        le = preprocessing.LabelEncoder().fit(y)
        y_encoded = torch.Tensor(le.transform(y))
        return y_encoded

    def build_vega_dataset(self, adata, y_encoded, pathway_file):
        if sparse.issparse(adata.X):
            data = adata.X.A
        else:
            data = adata.X

        data = torch.Tensor(data)
        data = UnsupervisedDataset(data, targets=y_encoded)

        pathway_dict = read_gmt(pathway_file, min_g=0, max_g=1000)
        pathway_mask = create_pathway_mask(adata.var.index.tolist(), pathway_dict, add_missing=1, fully_connected=True)

        return data, pathway_dict, pathway_mask
    
    def build_pmVAE_dataset(self, adata, pathway_file):
        adata.varm['annotations'] = load_annotations(
            pathway_file,
            adata.var_names,
            min_genes=13
        )
        pathway_mask = adata.varm['annotations'].astype(bool).T
        return adata, pathway_mask

    def build_OntoVAE_dataset(self, adata, pathway_file):
        ontobj = Ontobj()
        ontobj.load(pathway_file)
        genes = ontobj.extract_genes()
        adata = adata.copy()
        adata = setup_anndata_ontovae(adata, ontobj)
        return adata

    def build_ExpiMap_dataset(self, adata, pathway_file):
        add_annotations(adata, pathway_file, min_genes=12)
        print(adata.shape)
        adata._inplace_subset_var(adata.varm['I'].sum(1)>0)
        print(adata.shape)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        #sc.pp.highly_variable_genes(
         #   adata,
          #  n_top_genes=2000,
            #batch_key="batch",
           # subset=True)
        print(adata.shape)
        select_terms = adata.varm['I'].sum(0)>12
        adata.uns['terms'] = np.array(adata.uns['terms'])[select_terms].tolist()
        adata.varm['I'] = adata.varm['I'][:, select_terms]
        adata._inplace_subset_var(adata.varm['I'].sum(1)>0)
        return adata



class Vega_train_multiple_times:
    def __init__(self,
                 adata_train: AnnData,
                 adata_val: AnnData,
                 adata_test: AnnData,
                 name_model: str,
                 name_dataset: str,
                 n: int, 
                 train_data,
                 val_data,
                 test_data,
                 n_epochs: int,
                 lr:int,
                 pathway_mask,
                 batch_size:int,
                 beta: int,
                 dropout: int,
                 train_p,
                 test_p,
                 pos_dec: bool,
                 dev: str,
                 save_path: bool,
                 path_to_save_embeddings: str,
                 path_to_save_reconstructed: str
                        ) -> None:
        super(Vega_train_multiple_times, self).__init__()

        self.adata_train = adata_train
        self.adata_val = adata_val
        self.adata_test = adata_test
        self.name_model = name_model
        self.name_dataset = name_dataset
        self.n = n
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.n_epochs = n_epochs
        self.lr = lr
        self.pathway_mask = pathway_mask
        self.batch_size = batch_size
        self.beta = beta
        self.dropout = dropout
        self.train_p = train_p
        self.test_p = test_p
        self.pos_dec = pos_dec
        self.dev = dev
        self.save_path = save_path
        self.path_to_save_embeddings = path_to_save_embeddings
        self.path_to_save_reconstructed = path_to_save_reconstructed

    def build_data_loader(self, train_data, val_data, test_data, batch_size, name_model):
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

        return train_loader, val_loader, test_loader

    def train_VEGA(self, adata_train, adata_val, adata_test, train_loader, val_loader, n, n_epochs, lr, pathway_mask, dev, beta, save_path, dropout, pos_dec, train_p, test_p, path_to_save_embeddings, path_to_save_reconstructed, name_dataset):
        dict_params = {'pathway_mask': pathway_mask, 'n_pathways':pathway_mask.shape[1], 'n_genes':pathway_mask.shape[0], 'device':dev, 'beta':beta, 'save_path':save_path,  'dropout':dropout, 'pos_dec':pos_dec}

        model = VEGA(**dict_params).to(dev)
        hist = model.train_model(train_loader, lr, n_epochs, train_p, test_p, val_loader, save_model=True)

        embedding_train = model.to_latent(torch.Tensor(adata_train.X).to(dev))
        np.savetxt(Path(path_to_save_embeddings + f'/vega_{name_dataset}_embeddings_train_{n}_trial.txt'), embedding_train.cpu().detach().numpy())
        X_reconstructed_train = model.decode(torch.Tensor(embedding_train).to(dev))
        np.savetxt(Path(path_to_save_reconstructed + f'/vega_{name_dataset}_reconstruction_train_{n}_trial.txt'), X_reconstructed_train.cpu().detach().numpy())

        embedding_val = model.to_latent(torch.Tensor(adata_val.X).to(dev))
        np.savetxt(Path(path_to_save_embeddings + f'/vega_{name_dataset}_embeddings_val_{n}_trial.txt'), embedding_val.cpu().detach().numpy())
        X_reconstructed_val = model.decode(torch.Tensor(embedding_val).to(dev))
        np.savetxt(Path(path_to_save_reconstructed + f'/vega_{name_dataset}_reconstruction_val_{n}_trial.txt'), X_reconstructed_val.cpu().detach().numpy())

        embedding_test = model.to_latent(torch.Tensor(adata_test.X).to(dev))
        np.savetxt(Path(path_to_save_embeddings + f'/vega_{name_dataset}_embeddings_test_{n}_trial.txt'), embedding_test.cpu().detach().numpy())
        X_reconstructed_test = model.decode(torch.Tensor(embedding_test).to(dev))
        np.savetxt(Path(path_to_save_reconstructed + f'/vega_{name_dataset}_reconstruction_test_{n}_trial.txt'), X_reconstructed_test.cpu().detach().numpy())

        
        return hist
        


class VanillaVAE_train_multiple_times:
    def __init__(self,
                 adata_train: AnnData,
                 adata_val: AnnData,
                 adata_test: AnnData,
                 name_model: str,
                 name_dataset: str,
                 n: int,
                 train_data,
                 val_data,
                 test_data,
                 n_epochs: int,
                 lr:int,
                 pathway_mask,
                 batch_size:int,
                 beta: int,
                 dropout: int,
                 train_p,
                 test_p,
                 dev: str,
                 init_w: bool,
                 path_model: str,
                 path_to_save_embeddings: str,
                 path_to_save_reconstructed: str
                        ) -> None:
        super(VanillaVAE_train_multiple_times, self).__init__()

        self.adata_train = adata_train
        self.adata_val = adata_val
        self.adata_test = adata_test
        self.name_model = name_model
        self.name_dataset = name_dataset
        self.n = n
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.n_epochs = n_epochs
        self.lr = lr
        self.pathway_mask = pathway_mask
        self.batch_size = batch_size
        self.beta = beta
        self.dropout = dropout
        self.train_p = train_p
        self.test_p = test_p
        self.dev = dev
        self.init_w = init_w
        self.path_model = path_model
        self.path_to_save_embeddings = path_to_save_embeddings
        self.path_to_save_reconstructed = path_to_save_reconstructed

    def build_data_loader(self, train_data, val_data, test_data, batch_size):
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
        return train_loader, val_loader, test_loader

    def train_Vanilla(self, adata_train, adata_val, adata_test, train_loader, val_loader, n, n_epochs, lr, pathway_mask, dev, beta, path_model, dropout, init_w, train_p, test_p, path_to_save_embeddings, path_to_save_reconstructed, name_dataset):
        dict_params = {'n_latent':pathway_mask.shape[1], 'n_genes':pathway_mask.shape[0], 'device':dev, 'beta':beta, 'path_model': path_model,  'dropout':dropout, 'init_w': init_w}
            
        model = VanillaVAE(**dict_params).to(dev)
        hist = model.train_model(train_loader, lr, n_epochs, train_p, test_p, val_loader, save_model=False)

        embedding_train = model.to_latent(torch.Tensor(adata_train.X).to(dev))
        np.savetxt(Path(path_to_save_embeddings + f'/vanilla_{name_dataset}_embeddings_train_{n}_trial.txt'), embedding_train.cpu().detach().numpy())
        X_reconstructed_train = model.decode(torch.Tensor(embedding_train).to(dev))
        np.savetxt(Path(path_to_save_reconstructed + f'/vanilla_{name_dataset}_reconstruction_train_{n}_trial.txt'), X_reconstructed_train.cpu().detach().numpy())

        embedding_val = model.to_latent(torch.Tensor(adata_val.X).to(dev))
        np.savetxt(Path(path_to_save_embeddings + f'/vanilla_{name_dataset}_embeddings_val_{n}_trial.txt'), embedding_val.cpu().detach().numpy())
        X_reconstructed_val = model.decode(torch.Tensor(embedding_val).to(dev))
        np.savetxt(Path(path_to_save_reconstructed + f'/vanilla_{name_dataset}_reconstruction_val_{n}_trial.txt'), X_reconstructed_val.cpu().detach().numpy())

        embedding_test = model.to_latent(torch.Tensor(adata_test.X).to(dev))
        np.savetxt(Path(path_to_save_embeddings + f'/vanilla_{name_dataset}_embeddings_test_{n}_trial.txt'), embedding_test.cpu().detach().numpy())
        X_reconstructed_test = model.decode(torch.Tensor(embedding_test).to(dev))
        np.savetxt(Path(path_to_save_reconstructed + f'/vanilla_{name_dataset}_reconstruction_test_{n}_trial.txt'), X_reconstructed_test.cpu().detach().numpy())
        
        return hist
        



class pmVAE_train_multiple_times:
    def __init__(self,
                 name_model: str,
                 name_dataset: str,
                 n: int,
                 train_data,
                 val_data,
                 test_data,
                 n_epochs: int,
                 lr:int,
                 pathway_mask,
                 batch_size:int,
                 beta: int,
                 module_latent_dim: int,
                 hidden_layers: int,
                 add_auxialiary_module:bool,
                 path_to_save_embeddings: str,
                 path_to_save_reconstructed: str
                        ) -> None:
        super(pmVAE_train_multiple_times, self).__init__()

        self.name_model = name_model
        self.name_dataset = name_dataset
        self.n = n
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.n_epochs = n_epochs
        self.lr = lr
        self.pathway_mask = pathway_mask
        self.batch_size = batch_size
        self.beta = beta
        self.module_latent_dim = module_latent_dim
        self.hidden_layers = hidden_layers
        self.add_auxiliary_module = add_auxialiary_module
        self.path_to_save_embeddings = path_to_save_embeddings
        self.path_to_save_reconstructed = path_to_save_reconstructed

    def build_data_loader(self, train_data, batch_size):
        train_loader = tf.data.Dataset.from_tensor_slices(train_data)
        train_loader = train_loader.shuffle(5 * batch_size).batch(batch_size)
        return train_loader

    def train_pmVAE(self, train_loader, train_data, val_data, test_data, n_epochs, lr, pathway_mask, beta, module_latent_dim , hidden_layers, add_auxialiary_module, n, path_to_save_embeddings, path_to_save_reconstructed, name_dataset):
        
        model = PMVAE(
        membership_mask=pathway_mask.values,
        module_latent_dim=module_latent_dim,
        hidden_layers=hidden_layers,
        add_auxiliary_module=add_auxialiary_module,
        beta=beta,
        kernel_initializer='he_uniform',
        bias_initializer='zero',
        activation='elu',
        terms=pathway_mask.index
        )

        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        hist = train(model, opt, train_loader, val_data, n_epochs)

        outputs_train = model.call(train_data)
        embedding_train = outputs_train.z.numpy()
        np.savetxt(Path(path_to_save_embeddings + f'/pmVAE_{name_dataset}_embeddings_train_{n}_trial.txt'), embedding_train)
        X_reconstructed_train = outputs_train.global_recon.numpy()
        np.savetxt(Path(path_to_save_reconstructed + f'/pmVAE_{name_dataset}_reconstruction_train_{n}_trial.txt'), X_reconstructed_train)

        outputs_val = model.call(val_data)
        embedding_val = outputs_val.z.numpy()
        np.savetxt(Path(path_to_save_embeddings + f'/pmVAE_{name_dataset}_embeddings_val_{n}_trial.txt'), embedding_val)
        X_reconstructed_val = outputs_val.global_recon.numpy()
        np.savetxt(Path(path_to_save_reconstructed + f'/pmVAE_{name_dataset}_reconstruction_val_{n}_trial.txt'), X_reconstructed_val)

        outputs_test = model.call(test_data)
        embedding_test = outputs_test.z.numpy()
        np.savetxt(Path(path_to_save_embeddings + f'/pmVAE_{name_dataset}_embeddings_test_{n}_trial.txt'), embedding_test)
        X_reconstructed_test = outputs_test.global_recon.numpy()
        np.savetxt(Path(path_to_save_reconstructed + f'/pmVAE_{name_dataset}_reconstruction_test_{n}_trial.txt'), X_reconstructed_test)
        
        return hist
        


class OntoVAE_train_multiple_times:
    def __init__(self,
                 adata_train: AnnData,
                 adata_test: AnnData,
                 name_model: str,
                 name_dataset: str,
                 train_size:float,
                 n:int,
                 n_epochs: int,
                 lr:int,
                 batch_size:int,
                 latent_dim: int,
                 hidden_layers: int,
                 z_dropout: float,
                 kl_coeff: float,
                 use_batch_norm_dec:bool,
                 use_activation_dec:bool,
                 path_model: str,
                 path_to_save_embeddings: str,
                 path_to_save_reconstructed: str,
                 dev:str,
                 random_seed:int
                        ) -> None:
        super(OntoVAE_train_multiple_times, self).__init__()

        self.adata_train = adata_train
        self.adata_test = adata_test
        self.name_model = name_model
        self.name_dataset = name_dataset
        self.train_size = train_size
        self.n = n
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.z_dropout = z_dropout
        self.kl_coeff = kl_coeff
        self.use_batch_norm_dec = use_batch_norm_dec
        self.use_activation_dec = use_activation_dec
        self.path_model = path_model
        self.path_to_save_embeddings = path_to_save_embeddings
        self.path_to_save_reconstructed = path_to_save_reconstructed
        self.dev = dev
        self.random_seed = random_seed

    def train_OntoVAE(self, adata_train, adata_test, train_size, n_epochs, lr, batch_size, latent_dim , hidden_layers, z_dropout, kl_coeff, use_batch_norm_dec, use_activation_dec, n, path_to_save_embeddings, path_to_save_reconstructed, name_dataset, path_model, dev, random_seed):
        
        model = OntoVAE(adata_train, latent_dim=latent_dim, z_drop=z_dropout, hidden_layers_enc=hidden_layers, use_batch_norm_dec=use_batch_norm_dec, use_activation_dec=use_activation_dec)
        # train the model
        model.train_model(modelpath=path_model+'test',  
                            train_size=train_size, 
                            seed=random_seed,
                            lr=lr,                                 
                            kl_coeff=kl_coeff,                           
                            batch_size=batch_size,                          
                            epochs=n_epochs) 

        embedding_train = model.to_latent(adata_train)
        np.savetxt(Path(path_to_save_embeddings + f'/OntoVAE_{name_dataset}_embeddings_train_{n}_trial.txt'), embedding_train)
        X_reconstructed_train = model._run_batches(adata_train, 'rec', False)
        np.savetxt(Path(path_to_save_reconstructed + f'/OntoVAE_{name_dataset}_reconstruction_train_{n}_trial.txt'), X_reconstructed_train)

        embedding_test = model.to_latent(adata_test)
        np.savetxt(Path(path_to_save_embeddings + f'/OntoVAE_{name_dataset}_embeddings_test_{n}_trial.txt'), embedding_test)
        X_reconstructed_test = model._run_batches(adata_test, 'rec', False)
        np.savetxt(Path(path_to_save_reconstructed + f'/OntoVAE_{name_dataset}_reconstruction_test_{n}_trial.txt'), X_reconstructed_test)

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()    
    


class ExpiMap_train_multiple_times:
    def __init__(self,
                 adata_train: AnnData,
                 adata_test: AnnData,
                 name_model: str,
                 name_dataset: str,
                 n:int,
                 condition_key:str,
                 conditions:str,
                 hidden_layer_sizes:list,
                 recon_loss:str,
                 mask:str,
                 early_stopping_kwargs:dict,
                 n_epochs: int,
                 alpha_epoch_anneal:int,
                 alpha:int,
                 omega:int,
                 alpha_kl:int,
                 weight_decay:float,
                 use_early_stoppping:bool,
                 use_stratify_sampling: bool,
                 random_seed:int,
                 path_to_save_embeddings: str,
                 path_to_save_reconstructed: str,
                 dev:str
                        ) -> None:
        super(ExpiMap_train_multiple_times, self).__init__()

        self.adata_train = adata_train
        self.adata_test = adata_test
        self.name_model = name_model
        self.name_dataset = name_dataset
        self.n = n
        self.condition_key = condition_key
        self.conditions = conditions
        self.hidden_layer_sizes = hidden_layer_sizes
        self.recon_loss = recon_loss
        self.mask = mask
        self.early_stopping_kwargs = early_stopping_kwargs
        self.n_epochs = n_epochs
        self.alpha_epoch_anneal = alpha_epoch_anneal
        self.alpha = alpha
        self.omega = omega
        self.alpha_kl = alpha_kl
        self.weight_decay = weight_decay
        self.use_early_stopping = use_early_stoppping
        self.use_stratify_sampling = use_stratify_sampling
        self.random_seed = random_seed
        self.path_to_save_embeddings = path_to_save_embeddings
        self.path_to_save_reconstructed = path_to_save_reconstructed
        self.dev = dev

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def train_ExpiMap(self, name_model, name_dataset, n, adata_train, adata_test, condition_key, conditions, hidden_layer_sizes, recon_loss, mask, n_epochs, alpha_epoch_anneal, alpha, omega, alpha_kl, weight_decay,
        early_stopping_kwargs, use_early_stopping, use_stratified_sampling, path_to_save_embeddings, path_to_save_reconstructed, dev, random_seed):
        
        model_expimap = sca.models.EXPIMAP(
            adata=adata_train,
            condition_key=condition_key,
            conditions=conditions,
            hidden_layer_sizes=hidden_layer_sizes,
            #use_mmd=False,
            recon_loss=recon_loss,
            mask=adata_train.varm['I'].T,
            #use_decoder_relu=False,
            #mmd_instead_kl=False
        )
        # train the model
        model_expimap.train(n_epochs=n_epochs, 
            alpha_epoch_anneal=alpha_epoch_anneal, 
            alpha=alpha, 
            omega=omega,
            alpha_kl=alpha_kl,
            weight_decay=weight_decay, 
            early_stopping_kwargs=early_stopping_kwargs,
            use_early_stopping=use_early_stopping,
            use_stratified_sampling=use_stratified_sampling,
            seed=random_seed) 
        
        mu_train = model_expimap.model.encoder(torch.Tensor(adata_train.X).to(dev))[0]
        logvar_test = model_expimap.model.encoder(torch.Tensor(adata_train.X).to(dev))[1]
        embedding_train = reparameterize(mu_train, logvar_train)
        np.savetxt(Path(path_to_save_embeddings + f'/ExpiMap_{name_dataset}_embeddings_train_{n}_trial.txt'), embedding_train)

        X_reconstructed_train = model_expimap.model.decoder(embedding_train.to(dev))[0].cpu().detach()
        np.savetxt(Path(path_to_save_reconstructed + f'/ExpiMap_{name_dataset}_reconstruction_train_{n}_trial.txt'), X_reconstructed_train)

        mu_test = model_expimap.model.encoder(torch.Tensor(adata_test.X).to(dev))[0]
        logvar_test = model_expimap.model.encoder(torch.Tensor(adata_test.X).to(dev))[1]
        embedding_test = reparameterize(mu_test, logvar_test)
        np.savetxt(Path(path_to_save_embeddings + f'/ExpiMap_{name_dataset}_embeddings_test_{n}_trial.txt'), embedding_test)
        X_reconstructed_test = model_expimap.model.decoder(embedding_test.to(dev))[0].cpu().detach()
        np.savetxt(Path(path_to_save_reconstructed + f'/ExpiMap_{name_dataset}_reconstruction_test_{n}_trial.txt'), X_reconstructed_test)

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()    
    


            





            










            






