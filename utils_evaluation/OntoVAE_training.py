import os
import sys
import scanpy as sc
import scipy.sparse as sp
from sklearn import preprocessing
import anndata as ad
import torch
import itertools
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
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
import umap
from scipy.sparse import issparse
import anndata as ad
from anndata import AnnData
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split


sys.path.append('/home/BS94_SUR/phD/review/models reproductibility/OntoVAE/cobra-ai')
from cobra_ai.module.ontobj import *
from cobra_ai.module.utils import *
from cobra_ai.model.onto_vae import *
from cobra_ai.model.cobra import *
from cobra_ai.module.autotune import *

sys.path.append('/home/BS94_SUR/phD/review/utils/utils_evaluation/')
import OntoVAE_utils
from OntoVAE_utils import *
import utils_evaluation_models
from utils_evaluation_models import *
import OntoVAE_code_version2
from OntoVAE_code_version2 import *



class OntoVAE_train_multiple_times:
    def __init__(self,
                 adata: AnnData,
                 device:str,
                 name_model: str,
                 name_dataset: str,
                 n_training: int, 
                 n_epochs: int,
                 lr: float,
                 batch_size: int,
                 kl_coeff: float,
                 test_size: float,
                 path_to_save_embeddings: str,
                 path_to_save_reconstructed: str,
                 path_to_save_models: str) -> None:

        self.adata = adata
        self.device = device
        self.name_model = name_model
        self.name_dataset = name_dataset
        self.n_training = n_training
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.kl_coeff = kl_coeff
        self.test_size = test_size
        self.path_to_save_embeddings = path_to_save_embeddings
        self.path_to_save_reconstructed = path_to_save_reconstructed
        self.path_to_save_models = path_to_save_models

    def train_n_times(self, 
                      random_seed_list,
                      column_labels_name,
                      save_path_results,
                      path_save_fig,
                      compute_mse_score,
                      compute_pearson_score,
                      build_adata_latent,
                      apply_clustering_algo,
                      apply_clustering_metrics,
                      resolution_value,
                      plot_umap_orig_and_clusters):
        
        results = []
        
        for n in range(self.n_training):
            random_seed = random_seed_list[n]
            print(f'Training {n} - seed {random_seed}')
            
            model_dir = f'{self.path_to_save_models}/ontovae_{self.name_dataset}_training_{n}_seed_{random_seed}'
            os.makedirs(model_dir, exist_ok=True)
            
            # Split train/test puis train/val
            indices = np.arange(self.adata.n_obs)
            idx_train_full, idx_test = train_test_split(
                indices, test_size=self.test_size, random_state=random_seed, shuffle=True
            )
            idx_train2, idx_val = train_test_split(
                idx_train_full, test_size=self.test_size, random_state=random_seed, shuffle=True
            )
            
            adata_train = self.adata[idx_train_full].copy()
            adata_val = self.adata[idx_val].copy()
            adata_test = self.adata[idx_test].copy()
            
            # Initialiser et entraîner le modèle OntoVAE
            model = OntoVAE2(adata_train, self.device)

            model.train_model(
                model_dir,
                seed=random_seed,
                lr=self.lr,
                kl_coeff=self.kl_coeff,
                batch_size=self.batch_size,
                epochs=self.n_epochs
            )
            
            model = OntoVAE2.load(adata_train, model_dir)
            
            embedding_train = model.to_latent(adata_train)
            embedding_train_df = pd.DataFrame(embedding_train, index=adata_train.obs.index)
            embedding_train_df.to_csv(
                f'{self.path_to_save_embeddings}/ontovae_{self.name_dataset}_embeddings_train_{n}_seed_{random_seed}.txt'
            )
            
            X_reconstructed_train = model._run_batches(adata_train, 'rec', False)
            if isinstance(X_reconstructed_train, torch.Tensor):
                X_reconstructed_train_array = X_reconstructed_train.cpu().detach().numpy()
            else:
                X_reconstructed_train_array = X_reconstructed_train
            
            pd.DataFrame(X_reconstructed_train_array).to_csv(
                f'{self.path_to_save_reconstructed}/ontovae_{self.name_dataset}_reconstruction_train_{n}_seed_{random_seed}.txt'
            )
            
            embedding_val = model.to_latent(adata_val)
            embedding_val_df = pd.DataFrame(embedding_val, index=adata_val.obs.index)
            embedding_val_df.to_csv(
                f'{self.path_to_save_embeddings}/ontovae_{self.name_dataset}_embeddings_val_{n}_seed_{random_seed}.txt'
            )
            
            X_reconstructed_val = model._run_batches(adata_val, 'rec', False)
            if isinstance(X_reconstructed_val, torch.Tensor):
                X_reconstructed_val_array = X_reconstructed_val.cpu().detach().numpy()
            else:
                X_reconstructed_val_array = X_reconstructed_val
            
            pd.DataFrame(X_reconstructed_val_array).to_csv(
                f'{self.path_to_save_reconstructed}/ontovae_{self.name_dataset}_reconstruction_val_{n}_seed_{random_seed}.txt'
            )
            
            embedding_test = model.to_latent(adata_test)
            embedding_test_df = pd.DataFrame(embedding_test, index=adata_test.obs.index)
            embedding_test_df.to_csv(
                f'{self.path_to_save_embeddings}/ontovae_{self.name_dataset}_embeddings_test_{n}_seed_{random_seed}.txt'
            )
            
            X_reconstructed_test = model._run_batches(adata_test, 'rec', False)
            if isinstance(X_reconstructed_test, torch.Tensor):
                X_reconstructed_test_array = X_reconstructed_test.cpu().detach().numpy()
            else:
                X_reconstructed_test_array = X_reconstructed_test
            
            pd.DataFrame(X_reconstructed_test_array).to_csv(
                f'{self.path_to_save_reconstructed}/ontovae_{self.name_dataset}_reconstruction_test_{n}_seed_{random_seed}.txt'
            )
            
            mse_score_train = compute_mse_score(adata_train, X_reconstructed_train_array)
            mse_score_val = compute_mse_score(adata_val, X_reconstructed_val_array)
            mse_score_test = compute_mse_score(adata_test, X_reconstructed_test_array)
            
            corr_train = compute_pearson_score(adata_train, X_reconstructed_train_array)
            corr_val = compute_pearson_score(adata_val, X_reconstructed_val_array)
            corr_test = compute_pearson_score(adata_test, X_reconstructed_test_array)
            
            adata_latent_train = build_adata_latent(embedding_train_df, adata_train, column_labels_name)
            adata_latent_val = build_adata_latent(embedding_val_df, adata_val, column_labels_name)
            adata_latent_test = build_adata_latent(embedding_test_df, adata_test, column_labels_name)
            
            clusters_train, true_labels_train = apply_clustering_algo(adata_latent_train, column_labels_name, 'Leiden', resolution_value)
            clusters_val, true_labels_val = apply_clustering_algo(adata_latent_val, column_labels_name, 'Leiden', resolution_value)
            clusters_test, true_labels_test = apply_clustering_algo(adata_latent_test, column_labels_name, 'Leiden', resolution_value)
            
            ari_train, nmi_train = apply_clustering_metrics(true_labels_train, clusters_train)
            ari_val, nmi_val = apply_clustering_metrics(true_labels_val, clusters_val)
            ari_test, nmi_test = apply_clustering_metrics(true_labels_test, clusters_test)
            
            
            plot_umap_orig_and_clusters(embedding_train, true_labels_train, self.name_dataset, self.name_model,
                                       clusters_train, ari_train, nmi_train, 'Train', n, 'Leiden', path_save_fig)
            plot_umap_orig_and_clusters(embedding_val, true_labels_val, self.name_dataset, self.name_model,
                                       clusters_val, ari_val, nmi_val, 'Val', n, 'Leiden', path_save_fig)
            plot_umap_orig_and_clusters(embedding_test, true_labels_test, self.name_dataset, self.name_model,
                                       clusters_test, ari_test, nmi_test, 'Test', n, 'Leiden', path_save_fig)
            
            accuracy_train_rf, precision_train_rf, recall_train_rf, f1_train_rf, roc_auc_train_rf, accuracy_test_rf, precision_test_rf, recall_test_rf, f1_test_rf, roc_auc_test_rf = apply_classification_algo2(adata_latent_train, adata_latent_test, column_labels_name, 'random_forest', test_size=0.1)
                
            accuracy_train_xg, precision_train_xg, recall_train_xg, f1_train_xg, roc_auc_train_xg, accuracy_test_xg, precision_test_xg, recall_test_xg, f1_test_xg, roc_auc_test_xg = apply_classification_algo2(adata_latent_train, adata_latent_test, column_labels_name, 'xgboost', test_size=0.1)
            

            results.append({
                'model_name': self.name_model,
                'dataset_name': self.name_dataset,
                'id_training': n,
                'random_seed': random_seed,
                'kl_coeff': self.kl_coeff,
                'learning_rate': self.lr,
                'batch_size': self.batch_size,
                'n_epochs': self.n_epochs,
                'mse_score_train': mse_score_train,
                'mse_score_val': mse_score_val,
                'mse_score_test': mse_score_test,
                'corr_train': corr_train,
                'corr_val': corr_val,
                'corr_test': corr_test,
                'ari_train': ari_train,
                'nmi_train': nmi_train,
                'ari_val': ari_val,
                'nmi_val': nmi_val,
                'ari_test': ari_test,
                'nmi_test': nmi_test,
                'nmi_test': nmi_test,
                'accuracy_train_rf': accuracy_train_rf,
                'precision_train_rf': precision_train_rf,
                'recall_train_rf': recall_train_rf,
                'f1_train_rf': f1_train_rf,
                'roc_auc_train_rf': roc_auc_train_rf,
                'accuracy_test_rf': accuracy_test_rf,
                'precision_test_rf': precision_test_rf,
                'recall_test_rf': recall_test_rf,
                'f1_test_rf': f1_test_rf,
                'roc_auc_test_rf': roc_auc_test_rf,
                'accuracy_train_xg': accuracy_train_xg,
                'precision_train_xg': precision_train_xg,
                'recall_train_xg': recall_train_xg,
                'f1_train_xg': f1_train_xg,
                'roc_auc_train_xg': roc_auc_train_xg,
                'accuracy_test_xg': accuracy_test_xg,
                'precision_test_xg': precision_test_xg,
                'recall_test_xg': recall_test_xg,
                'f1_test_xg': f1_test_xg,
                'roc_auc_test_xg': roc_auc_test_xg
            })
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(
            f'{save_path_results}/{self.name_model}_{self.name_dataset}_{self.n_training}_training_results.csv',
            index=False
        )
        print(f"Results saved to {save_path_results}/{self.name_model}_{self.name_dataset}_{self.n_training}_training_results.csv")
        
        return results_df
    
    
    
class OntoVAE_nfold_cross_validation:
    def __init__(self,
                 adata: AnnData,
                 device:str,
                 name_model: str,
                 name_dataset: str,
                 n_folds: int, 
                 n_epochs: int,
                 lr: float,
                 batch_size: int,
                 kl_coeff: float,
                 test_size: float,
                 path_to_save_embeddings: str,
                 path_to_save_reconstructed: str,
                 path_to_save_models: str) -> None:

        self.adata = adata
        self.device = device
        self.name_model = name_model
        self.name_dataset = name_dataset
        self.n_folds = n_folds
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.kl_coeff = kl_coeff
        self.test_size = test_size
        self.path_to_save_embeddings = path_to_save_embeddings
        self.path_to_save_reconstructed = path_to_save_reconstructed
        self.path_to_save_models = path_to_save_models

    def cross_validate_OntoVAE(self, 
                      random_seed,
                      column_labels_name,
                      save_path_results,
                      path_save_fig,
                      compute_mse_score,
                      compute_pearson_score,
                      build_adata_latent,
                      apply_clustering_algo,
                      apply_clustering_metrics,
                      resolution_value,
                      plot_umap_orig_and_clusters):
        
        results = []
        
        labels = self.adata.obs[column_labels_name].values
        
        # Initialize StratifiedKFold cross-validator
        skfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=random_seed)
        
        # Get indices for splitting
        indices = np.arange(self.adata.n_obs)
        
        for fold, (train_val_idx, test_idx) in enumerate(skfold.split(indices, labels)):
            print(f'Fold {fold + 1}/{self.n_folds}')
            
            # Split data into train+val and test sets
            adata_train_val = self.adata[train_val_idx].copy()
            adata_test = self.adata[test_idx].copy()
            
            train_val_labels = adata_train_val.obs[column_labels_name].values
            
            train_idx_relative, val_idx_relative = train_test_split(
                np.arange(len(train_val_idx)),
                test_size=0.1,
                random_state=random_seed,
                stratify=train_val_labels
            )
            
            adata_train = adata_train_val[train_idx_relative].copy()
            adata_val = adata_train_val[val_idx_relative].copy()
            
            model_dir = f'{self.path_to_save_models}/ontovae_{self.name_dataset}_fold_{fold}'
            os.makedirs(model_dir, exist_ok=True)
            
            
            # Initialiser et entraîner le modèle OntoVAE
            model = OntoVAE2(adata_train, self.device)

            model.train_model(
                model_dir,
                seed=random_seed,
                lr=self.lr,
                kl_coeff=self.kl_coeff,
                batch_size=self.batch_size,
                epochs=self.n_epochs
            )
            
            model = OntoVAE2.load(adata_train, model_dir, self.device)
            
            embedding_train = model.to_latent(adata_train)
            embedding_train_df = pd.DataFrame(embedding_train, index=adata_train.obs.index)
            embedding_train_df.to_csv(
                f'{self.path_to_save_embeddings}/ontovae_{self.name_dataset}_embeddings_train_fold_{fold}.txt'
            )
            
            X_reconstructed_train = model._run_batches(adata_train, 'rec', False)
            if isinstance(X_reconstructed_train, torch.Tensor):
                X_reconstructed_train_array = X_reconstructed_train.cpu().detach().numpy()
            else:
                X_reconstructed_train_array = X_reconstructed_train
            
            pd.DataFrame(X_reconstructed_train_array).to_csv(
                f'{self.path_to_save_reconstructed}/ontovae_{self.name_dataset}_reconstruction_train_fold_{fold}.txt'
            )
            
            embedding_val = model.to_latent(adata_val)
            embedding_val_df = pd.DataFrame(embedding_val, index=adata_val.obs.index)
            embedding_val_df.to_csv(
                f'{self.path_to_save_embeddings}/ontovae_{self.name_dataset}_embeddings_val_fold_{fold}.txt'
            )
            
            X_reconstructed_val = model._run_batches(adata_val, 'rec', False)
            if isinstance(X_reconstructed_val, torch.Tensor):
                X_reconstructed_val_array = X_reconstructed_val.cpu().detach().numpy()
            else:
                X_reconstructed_val_array = X_reconstructed_val
            
            pd.DataFrame(X_reconstructed_val_array).to_csv(
                f'{self.path_to_save_reconstructed}/ontovae_{self.name_dataset}_reconstruction_val_fold_{fold}.txt'
            )
            
            embedding_test = model.to_latent(adata_test)
            embedding_test_df = pd.DataFrame(embedding_test, index=adata_test.obs.index)
            embedding_test_df.to_csv(
                f'{self.path_to_save_embeddings}/ontovae_{self.name_dataset}_embeddings_test_fold_{fold}.txt'
            )
            
            X_reconstructed_test = model._run_batches(adata_test, 'rec', False)
            if isinstance(X_reconstructed_test, torch.Tensor):
                X_reconstructed_test_array = X_reconstructed_test.cpu().detach().numpy()
            else:
                X_reconstructed_test_array = X_reconstructed_test
            
            pd.DataFrame(X_reconstructed_test_array).to_csv(
                f'{self.path_to_save_reconstructed}/ontovae_{self.name_dataset}_reconstruction_test_fold_{fold}.txt'
            )
            
            mse_score_train = compute_mse_score(adata_train, X_reconstructed_train_array)
            mse_score_val = compute_mse_score(adata_val, X_reconstructed_val_array)
            mse_score_test = compute_mse_score(adata_test, X_reconstructed_test_array)
            
            corr_train = compute_pearson_score(adata_train, X_reconstructed_train_array)
            corr_val = compute_pearson_score(adata_val, X_reconstructed_val_array)
            corr_test = compute_pearson_score(adata_test, X_reconstructed_test_array)
            
            adata_latent_train = build_adata_latent(embedding_train_df, adata_train, column_labels_name)
            adata_latent_val = build_adata_latent(embedding_val_df, adata_val, column_labels_name)
            adata_latent_test = build_adata_latent(embedding_test_df, adata_test, column_labels_name)
            
            clusters_train, true_labels_train = apply_clustering_algo(adata_latent_train, column_labels_name, 'Leiden', resolution_value)
            clusters_val, true_labels_val = apply_clustering_algo(adata_latent_val, column_labels_name, 'Leiden', resolution_value)
            clusters_test, true_labels_test = apply_clustering_algo(adata_latent_test, column_labels_name, 'Leiden', resolution_value)
            
            ari_train, nmi_train = apply_clustering_metrics(true_labels_train, clusters_train)
            ari_val, nmi_val = apply_clustering_metrics(true_labels_val, clusters_val)
            ari_test, nmi_test = apply_clustering_metrics(true_labels_test, clusters_test)
            
            
            plot_umap_orig_and_clusters(embedding_train, true_labels_train, self.name_dataset, self.name_model,
                                       clusters_train, ari_train, nmi_train, 'Train', fold, 'Leiden', path_save_fig)
            plot_umap_orig_and_clusters(embedding_val, true_labels_val, self.name_dataset, self.name_model,
                                       clusters_val, ari_val, nmi_val, 'Val', fold, 'Leiden', path_save_fig)
            plot_umap_orig_and_clusters(embedding_test, true_labels_test, self.name_dataset, self.name_model,
                                       clusters_test, ari_test, nmi_test, 'Test', fold, 'Leiden', path_save_fig)
            
            accuracy_train_rf, precision_train_rf, recall_train_rf, f1_train_rf, roc_auc_train_rf, accuracy_val_rf, precision_val_rf, recall_val_rf, f1_val_rf, roc_auc_val_rf, accuracy_test_rf, precision_test_rf, recall_test_rf, f1_test_rf, roc_auc_test_rf = apply_classification_algo2(adata_latent_train, adata_latent_val, adata_latent_test, column_labels_name, 'random_forest', test_size=0.1)
            accuracy_train_xg, precision_train_xg, recall_train_xg, f1_train_xg, roc_auc_train_xg, accuracy_val_xg, precision_val_xg, recall_val_xg, f1_val_xg, roc_auc_val_xg, accuracy_test_xg, precision_test_xg, recall_test_xg, f1_test_xg, roc_auc_test_xg = apply_classification_algo2(adata_latent_train, adata_latent_val, adata_latent_test, column_labels_name, 'xgboost', test_size=0.1)


            results.append({
                'model_name': self.name_model,
                'dataset_name': self.name_dataset,
                'fold': fold + 1,
                'random_seed': random_seed,
                'kl_coeff': self.kl_coeff,
                'learning_rate': self.lr,
                'batch_size': self.batch_size,
                'n_epochs': self.n_epochs,
                'mse_score_train': mse_score_train,
                'mse_score_val': mse_score_val,
                'mse_score_test': mse_score_test,
                'corr_train': corr_train,
                'corr_val': corr_val,
                'corr_test': corr_test,
                'ari_train': ari_train,
                'nmi_train': nmi_train,
                'ari_val': ari_val,
                'nmi_val': nmi_val,
                'ari_test': ari_test,
                'nmi_test': nmi_test,
                'nmi_test': nmi_test,
                'accuracy_train_rf': accuracy_train_rf,
                'precision_train_rf': precision_train_rf,
                'recall_train_rf': recall_train_rf,
                'f1_train_rf': f1_train_rf,
                'roc_auc_train_rf': roc_auc_train_rf,
                'accuracy_val_rf': accuracy_val_rf,
                'precision_val_rf': precision_val_rf,
                'recall_val_rf': recall_val_rf,
                'f1_val_rf': f1_val_rf,
                'roc_auc_val_rf': roc_auc_val_rf,
                'accuracy_test_rf': accuracy_test_rf,
                'precision_test_rf': precision_test_rf,
                'recall_test_rf': recall_test_rf,
                'f1_test_rf': f1_test_rf,
                'roc_auc_test_rf': roc_auc_test_rf,
                'accuracy_train_xg': accuracy_train_xg,
                'precision_train_xg': precision_train_xg,
                'recall_train_xg': recall_train_xg,
                'f1_train_xg': f1_train_xg,
                'roc_auc_train_xg': roc_auc_train_xg,
                'accuracy_val_xg': accuracy_val_xg,
                'precision_val_xg': precision_val_xg,
                'recall_val_xg': recall_val_xg,
                'f1_val_xg': f1_val_xg,
                'roc_auc_val_xg': roc_auc_val_xg,
                'accuracy_test_xg': accuracy_test_xg,
                'precision_test_xg': precision_test_xg,
                'recall_test_xg': recall_test_xg,
                'f1_test_xg': f1_test_xg,
                'roc_auc_test_xg': roc_auc_test_xg
            })
        
        results_df = pd.DataFrame(results)
        
        # Add summary statistics
        print("\n=== Cross-Validation Summary ===")
        metrics_to_summarize = ['ari_test', 'nmi_test', 'mse_score_test', 'corr_test', 
                               'accuracy_test_rf', 'f1_test_rf', 'accuracy_test_xg', 'f1_test_xg']
        for metric in metrics_to_summarize:
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
            
            
        results_df.to_csv(
            f'{save_path_results}/{self.name_model}_{self.name_dataset}_{self.n_folds}fold_cv_results.csv',
            index=False
        )
        print(f"Results saved to {save_path_results}/{self.name_model}_{self.name_dataset}_{self.n_folds}fold_cv_results.csv")
        
        return results_df