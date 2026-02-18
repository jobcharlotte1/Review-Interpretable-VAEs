import anndata
import numpy as np
import pandas as pd
import tensorflow as tf
import scanpy as sc
from matplotlib import pyplot as plt
import anndata as ad
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

import sys
sys.path.append('/home/BS94_SUR/phD/review/models reproductibility/pmvae/')
from pmvae.model import PMVAE
from pmvae.train import train
from pmvae.utils import load_annotations

sys.path.append('/home/BS94_SUR/phD/review/utils/utils_evaluation')
import pmVAE_utils
from pmVAE_utils import *
import utils_evaluation_models
from utils_evaluation_models import *


def random_search_pmVAE(
    adata,
    pathway_mask,
    test_size=0.1,
    shuffle=True,
    random_state=42,
    n_epochs=1200,
    save_path='pmvae_random_search_results.csv',
    n_iterations=50
):
    """
    Perform random search for pmVAE hyperparameter optimization.
    
    Parameters:
    -----------
    adata : AnnData
        Preprocessed AnnData object with pathway annotations
    pathway_mask : pd.DataFrame
        Boolean mask indicating gene-pathway memberships
    test_size : float
        Proportion of data to use for validation
    shuffle : bool
        Whether to shuffle data before splitting
    random_state : int
        Random seed for reproducibility
    save_path : str
        Path where to save the results CSV file
    n_iterations : int
        Number of random search iterations
    module_latent_dim : int
        Latent dimension for each pathway module
    hidden_layers : list
        Hidden layer sizes for encoder/decoder
    add_auxiliary_module : bool
        Whether to add auxiliary module for unannotated genes
    
    Returns:
    --------
    pd.DataFrame : DataFrame with hyperparameters and corresponding losses
    dict : Best hyperparameters configuration
    """
    
    # Set random seed
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    
    # Define hyperparameter search spaces
    param_distributions = {
        'module_latent_dim': ([2, 4, 6, 8], 'choice'),
        'hidden_layers': ([8, 12, 16], 'choice'),
        'beta': (1e-9, 1e-0, 'log'),  # (min, max, scale)
        'learning_rate': (1e-5, 1e-2, 'log'),
        'batch_size': ([32, 64, 128, 256, 512], 'choice'),
    }
    
    # Initialize results storage
    results = []
    
    # Check if file exists and load previous results
    try:
        existing_df = pd.read_csv(save_path)
        results = existing_df.to_dict('records')
        print(f"✅ Loaded {len(results)} existing results from {save_path}")
        start_iteration = len(results)
    except FileNotFoundError:
        print(f"📝 No existing results found. Starting fresh.")
        start_iteration = 0
    
    print(f"\n{'='*70}")
    print(f" Starting pmVAE Random Search with {n_iterations} iterations")
    print(f"{'='*70}\n")
    
    for iteration in range(start_iteration, start_iteration + n_iterations):
        print(f"\n{'─'*70}")
        print(f" Iteration {iteration + 1}/{start_iteration + n_iterations}")
        print(f"{'─'*70}")
        
        # Sample hyperparameters
        if param_distributions['module_latent_dim'][1] == 'choice':
            module_latent_dim = int(np.random.choice(param_distributions['module_latent_dim'][0]))

        if param_distributions['hidden_layers'][1] == 'choice':
            hidden_layers = int(np.random.choice(param_distributions['hidden_layers'][0]))

        if param_distributions['beta'][2] == 'log':
            beta = 10 ** np.random.uniform(
                np.log10(param_distributions['beta'][0]),
                np.log10(param_distributions['beta'][1])
            )
        
        if param_distributions['learning_rate'][2] == 'log':
            lr = 10 ** np.random.uniform(
                np.log10(param_distributions['learning_rate'][0]),
                np.log10(param_distributions['learning_rate'][1])
            )
        
        if param_distributions['batch_size'][1] == 'choice':
            batch_size = int(np.random.choice(param_distributions['batch_size'][0]))
       
         # Print current hyperparameters
        print(f" Hyperparameters:")
        print(f"   • module_latent_dim: {module_latent_dim}")
        print(f"   • hidden_layers: {hidden_layers:.2e}")
        print(f"   • beta: {beta:.2e}")
        print(f"   • learning_rate: {lr:.2e}")
        print(f"   • batch_size: {batch_size}")

        try:
            # Create train/test split
            trainset, testset, idx_train, idx_test = pmvae_train_test_split(adata, test_size, shuffle, random_state)
            trainset, valset, idx_train, idx_val = pmvae_train_test_split(trainset, test_size, shuffle, random_state)
            trainloader = get_train_set(trainset, batch_size)

            # Create model with current hyperparameters
            model = PMVAE(
                membership_mask=pathway_mask.values,
                module_latent_dim=module_latent_dim,
                hidden_layers=hidden_layers,
                add_auxiliary_module=True,
                beta=beta,
                kernel_initializer='he_uniform',
                bias_initializer='zero',
                activation='elu',
                terms=pathway_mask.index
            )
            
            # Create optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            
            print(f"\n Starting training...")
            
            # Train model
            history = train(model, optimizer, trainloader, valset, n_epochs)
            
            # Get final losses
            final_valid_loss = history["test-loss"][-1]
            final_train_loss = history["train-loss"][-1]
            
            # Get best validation loss
            best_valid_loss = min(history["test-loss"])
            best_epoch = np.argmin(history["train-loss"]) + 1
            
            print(f"\n✅ Training completed!")
            print(f"   • Final validation loss: {final_valid_loss:.4f}")
            print(f"   • Final training loss: {final_train_loss:.4f}")
            print(f"   • Best validation loss: {best_valid_loss:.4f} (epoch {best_epoch})")
            
            # Store results
            results.append({
                'iteration': iteration + 1,
                'beta': beta,
                'learning_rate': lr,
                'batch_size': batch_size,
                'n_epochs': n_epochs,
                'module_latent_dim': module_latent_dim,
                'hidden_layers': hidden_layers,
                'add_auxiliary_module': True,
                'final_valid_loss': float(final_valid_loss),
                'final_train_loss': float(final_train_loss),
                'best_valid_loss': float(best_valid_loss),
                'best_epoch': int(best_epoch),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'status': 'success'
            })
            
        except Exception as e:
            print(f"\n Error during training: {str(e)}")
            
            # Store error results
            results.append({
                'iteration': iteration + 1,
                'beta': beta,
                'learning_rate': lr,
                'batch_size': batch_size,
                'n_epochs': n_epochs,
                'module_latent_dim': module_latent_dim,
                'hidden_layers': hidden_layers,
                'add_auxiliary_module': True,
                'final_valid_loss': np.nan,
                'final_train_loss': np.nan,
                'best_valid_loss': np.nan,
                'best_epoch': 0,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'status': 'failed',
                'error': str(e)
            })
        
        # Save results immediately after each iteration
        results_df = pd.DataFrame(results)
        results_df.to_csv(save_path, index=False)
        print(f"💾 Results saved to {save_path}")
        
        # Clear memory
        tf.keras.backend.clear_session()
    
    # Create final DataFrame and sort by best validation loss
    results_df = pd.DataFrame(results)
    successful_runs = results_df[results_df['status'] == 'success'].copy()
    
    if len(successful_runs) > 0:
        successful_runs = successful_runs.sort_values('best_valid_loss', ascending=True)
        
        print(f"\n{'='*70}")
        print(f" Random search completed!")
        print(f"{'='*70}")
        print(f"✓ Successful runs: {len(successful_runs)}/{len(results_df)}")
        
        if len(successful_runs) > 0:
            print(f"\nBest hyperparameters (lowest validation loss):")
            print(f"{'─'*70}")
            best_row = successful_runs.iloc[0]
            print(f"   • beta: {best_row['beta']:.2e}")
            print(f"   • learning_rate: {best_row['learning_rate']:.2e}")
            print(f"   • batch_size: {best_row['batch_size']}")
            print(f"   • n_epochs: {best_row['n_epochs']}")
            print(f"   • kernel_initializer: {best_row['kernel_initializer']}")
            print(f"   • activation: {best_row['activation']}")
            print(f"   • best_valid_loss: {best_row['best_valid_loss']:.4f}")
            print(f"   • best_epoch: {best_row['best_epoch']}")
            
            # Return best config as dict
            best_config = best_row.to_dict()
            print(f"\nTop 5 configurations:")
            print(f"{'─'*70}")
            top5 = successful_runs.head(5)[['beta', 'learning_rate', 'batch_size', 
                                            'n_epochs', 'best_valid_loss']]
            print(top5.to_string(index=False))
            
            return results_df, best_config
    else:
        print(f"\nNo successful runs completed!")
        return results_df, None

    
class PMVAE_train_multiple_times:
    
    def __init__(self,
                 adata,
                 pathway_mask,
                 name_model: str,
                 name_dataset: str,
                 n_training: int, 
                 n_epochs: int,
                 lr: float,
                 batch_size: int,
                 beta: float,
                 module_latent_dim: int,
                 hidden_layers: list,
                 add_auxiliary_module: bool,
                 path_to_save_embeddings: str,
                 path_to_save_reconstructed: str):

        self.adata = adata
        self.pathway_mask = pathway_mask
        self.name_model = name_model
        self.name_dataset = name_dataset
        self.n_training = n_training
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.beta = beta
        self.module_latent_dim = module_latent_dim
        self.hidden_layers = hidden_layers
        self.add_auxiliary_module = add_auxiliary_module
        self.path_to_save_embeddings = path_to_save_embeddings
        self.path_to_save_reconstructed = path_to_save_reconstructed

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
                      value_resolution,
                      plot_umap_orig_and_clusters):
        
        results = []
        
        for n in range(self.n_training):
            random_seed = random_seed_list[n]
            print(f'Training {n} - seed {random_seed}')
            
            # Création des splits train/val/test avec pmVAE
            trainset, testset, idx_train, idx_test = pmvae_train_test_split(self.adata, 0.1, True, random_seed)
            adata_train_ = self.adata[idx_train].copy()
            adata_test = self.adata[idx_test].copy()
            trainset, valset, idx_train_new, idx_val = pmvae_train_test_split(adata_train_, 0.1, True, random_seed)
            adata_train = adata_train_[idx_train_new].copy()
            adata_val = adata_train_[idx_val].copy()
            
            # Créer le trainloader
            trainloader = get_train_set(trainset, self.batch_size)
            
            # Initialiser le modèle pmVAE
            model = PMVAE(
                membership_mask=self.pathway_mask.values,
                module_latent_dim=self.module_latent_dim,
                hidden_layers=self.hidden_layers,
                add_auxiliary_module=self.add_auxiliary_module,
                beta=self.beta,
                kernel_initializer='glorot_uniform',
                bias_initializer='zero',
                activation='elu',
                terms=self.pathway_mask.index
            )
            
            # Optimiseur
            opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
            
            # Entraînement
            hist = train(model, opt, trainloader, valset, self.n_epochs)
            
            latent_names = model.latent_space_names()
            
            outputs_train = model.call(trainset)
            embedding_train = outputs_train.z.numpy()
            embedding_train = pd.DataFrame(embedding_train, columns=latent_names)
            embedding_train.to_csv(
                f'{self.path_to_save_embeddings}/pmvae_{self.name_dataset}_embeddings_train_{n}_seed_{random_seed}.txt'
            )
            
            X_reconstructed_train = outputs_train.global_recon.numpy()
            pd.DataFrame(X_reconstructed_train).to_csv(
                f'{self.path_to_save_reconstructed}/pmvae_{self.name_dataset}_reconstruction_train_{n}_seed_{random_seed}.txt'
            )  

            outputs_val = model.call(valset)
            embedding_val = outputs_val.z.numpy()
            embedding_val = pd.DataFrame(embedding_val, columns=latent_names)
            embedding_val.to_csv(
                f'{self.path_to_save_embeddings}/pmvae_{self.name_dataset}_embeddings_val_{n}_seed_{random_seed}.txt'
            )
            
            X_reconstructed_val = outputs_val.global_recon.numpy()
            pd.DataFrame(X_reconstructed_val).to_csv(
                f'{self.path_to_save_reconstructed}/pmvae_{self.name_dataset}_reconstruction_val_{n}_seed_{random_seed}.txt'
            )
            
            outputs_test = model.call(testset)
            embedding_test = outputs_test.z.numpy()
            embedding_test = pd.DataFrame(embedding_test, columns=latent_names)
            embedding_test.to_csv(
                f'{self.path_to_save_embeddings}/pmvae_{self.name_dataset}_embeddings_test_{n}_seed_{random_seed}.txt'
            )
            
            X_reconstructed_test = outputs_test.global_recon.numpy()
            pd.DataFrame(X_reconstructed_test).to_csv(
                f'{self.path_to_save_reconstructed}/pmvae_{self.name_dataset}_reconstruction_test_{n}_seed_{random_seed}.txt'
            )
            
            
            mse_score_train = compute_mse_score(adata_train, X_reconstructed_train)
            mse_score_val = compute_mse_score(adata_val, X_reconstructed_val)
            mse_score_test = compute_mse_score(adata_test, X_reconstructed_test)
            
            corr_train = compute_pearson_score(adata_train, X_reconstructed_train)
            corr_val = compute_pearson_score(adata_val, X_reconstructed_val)
            corr_test = compute_pearson_score(adata_test, X_reconstructed_test)
            
            adata_latent_train = build_adata_latent(embedding_train, adata_train, column_labels_name)
            adata_latent_val = build_adata_latent(embedding_val, adata_val, column_labels_name)
            adata_latent_test = build_adata_latent(embedding_test, adata_test, column_labels_name)
            
            clusters_train, true_labels_train = apply_clustering_algo(adata_latent_train, column_labels_name, 'Leiden', value_resolution)
            clusters_val, true_labels_val = apply_clustering_algo(adata_latent_val, column_labels_name, 'Leiden', value_resolution)
            clusters_test, true_labels_test = apply_clustering_algo(adata_latent_test, column_labels_name, 'Leiden', value_resolution)
            
            ari_train, nmi_train = apply_clustering_metrics(true_labels_train, clusters_train)
            ari_val, nmi_val = apply_clustering_metrics(true_labels_val, clusters_val)
            ari_test, nmi_test = apply_clustering_metrics(true_labels_test, clusters_test)
            
            plot_umap_orig_and_clusters(embedding_train, true_labels_train, self.name_dataset, self.name_model, 
                                       clusters_train, ari_train, nmi_train, 'Train', n, 'Leiden', path_save_fig)
            plot_umap_orig_and_clusters(embedding_val, true_labels_val, self.name_dataset, self.name_model, 
                                       clusters_val, ari_val, nmi_val, 'Val', n, 'Leiden', path_save_fig)
            plot_umap_orig_and_clusters(embedding_test, true_labels_test, self.name_dataset, self.name_model, 
                                       clusters_test, ari_test, nmi_test, 'Test', n, 'Leiden', path_save_fig)
            
            accuracy_train_rf, precision_train_rf, recall_train_rf, f1_train_rf, roc_auc_train_rf, accuracy_val_rf, precision_val_rf, recall_val_rf, f1_val_rf, roc_auc_val_rf, accuracy_test_rf, precision_test_rf, recall_test_rf, f1_test_rf, roc_auc_test_rf = apply_classification_algo2(adata_latent_train, adata_latent_val, adata_latent_test, column_labels_name, 'random_forest', test_size=0.1)
            accuracy_train_xg, precision_train_xg, recall_train_xg, f1_train_xg, roc_auc_train_xg, accuracy_val_xg, precision_val_xg, recall_val_xg, f1_val_xg, roc_auc_val_xg, accuracy_test_xg, precision_test_xg, recall_test_xg, f1_test_xg, roc_auc_test_xg = apply_classification_algo2(adata_latent_train, adata_latent_val, adata_latent_test, column_labels_name, 'xgboost', test_size=0.1)
    
            results.append({
                'model_name': self.name_model,
                'dataset_name': self.name_dataset,
                'id_training': n,
                'random_seed': random_seed,
                'beta': self.beta,
                'learning_rate': self.lr,
                'batch_size': self.batch_size,
                'module_latent_dim': self.module_latent_dim,
                'add_auxiliary_module': self.add_auxiliary_module,
                'hidden_layers': str(self.hidden_layers),
                'final_train_loss': hist['loss'][-1] if 'loss' in hist else None,
                'final_valid_loss': hist['val_loss'][-1] if 'val_loss' in hist else None,
                'n_epochs_trained': len(hist.get('loss', [])),
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
        results_df.to_csv(f'{save_path_results}/{self.name_model}_{self.name_dataset}_{self.n_training}_training_results.csv', index=False)
        
        return results_df

    
    
class PMVAE_kfold_cross_validation:
    
    def __init__(self,
                 adata,
                 pathway_mask,
                 name_model: str,
                 name_dataset: str,
                 n_folds: int,  # Changed from n_training to n_folds
                 n_epochs: int,
                 lr: float,
                 batch_size: int,
                 beta: float,
                 module_latent_dim: int,
                 hidden_layers: list,
                 add_auxiliary_module: bool,
                 path_to_save_embeddings: str,
                 path_to_save_reconstructed: str):

        self.adata = adata
        self.pathway_mask = pathway_mask
        self.name_model = name_model
        self.name_dataset = name_dataset
        self.n_folds = n_folds
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.beta = beta
        self.module_latent_dim = module_latent_dim
        self.hidden_layers = hidden_layers
        self.add_auxiliary_module = add_auxiliary_module
        self.path_to_save_embeddings = path_to_save_embeddings
        self.path_to_save_reconstructed = path_to_save_reconstructed

    def cross_validate(self, 
                      random_seed,
                      column_labels_name,
                      save_path_results,
                      path_save_fig,
                      compute_mse_score,
                      compute_pearson_score,
                      build_adata_latent,
                      apply_clustering_algo,
                      apply_clustering_metrics,
                      value_resolution,
                      plot_umap_orig_and_clusters):
        """
        Perform stratified k-fold cross-validation on PMVAE model
        
        Args:
            random_seed: seed for reproducibility of fold splitting
            column_labels_name: name of the column in adata.obs to use for stratification
        """
        
        results = []
        
        # Get labels for stratification
        labels = self.adata.obs[column_labels_name].values
        
        # Initialize StratifiedKFold cross-validator
        skfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=random_seed)
        
        # Get data matrix
        if hasattr(self.adata.X, 'A'):  # sparse matrix
            X_full = self.adata.X.A
        else:
            X_full = self.adata.X
        
        # Get indices for splitting
        indices = np.arange(self.adata.n_obs)
        
        # Perform stratified k-fold cross-validation
        for fold, (train_val_idx, test_idx) in enumerate(skfold.split(indices, labels)):
            print(f'Fold {fold + 1}/{self.n_folds}')
            
            # Split into train+val and test
            X_train_val = X_full[train_val_idx]
            X_test = X_full[test_idx]
            adata_test = self.adata[test_idx].copy()
            
            # Get labels for train_val set for further stratification
            train_val_labels = labels[train_val_idx]
            
            # Further stratified split of train_val into train and validation
            train_idx_relative, val_idx_relative = train_test_split(
                np.arange(len(train_val_idx)),
                test_size=0.1,
                random_state=random_seed,
                stratify=train_val_labels
            )
            
            X_train = X_train_val[train_idx_relative]
            X_val = X_train_val[val_idx_relative]
            
            adata_train = self.adata[train_val_idx[train_idx_relative]].copy()
            adata_val = self.adata[train_val_idx[val_idx_relative]].copy()
            
            # Convert to TensorFlow tensors for pmVAE
            trainset = tf.constant(X_train, dtype=tf.float32)
            valset = tf.constant(X_val, dtype=tf.float32)
            testset = tf.constant(X_test, dtype=tf.float32)
            
            # Create trainloader
            trainloader = get_train_set(trainset, self.batch_size)
            
            # Initialize pmVAE model for this fold
            model = PMVAE(
                membership_mask=self.pathway_mask.values,
                module_latent_dim=self.module_latent_dim,
                hidden_layers=self.hidden_layers,
                add_auxiliary_module=self.add_auxiliary_module,
                beta=self.beta,
                kernel_initializer='glorot_uniform',
                bias_initializer='zero',
                activation='elu',
                terms=self.pathway_mask.index
            )
            
            # Optimizer
            opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
            
            # Train model
            hist = train(model, opt, trainloader, valset, self.n_epochs)
            
            latent_names = model.latent_space_names()
            
            # Get embeddings and reconstructions for train set
            outputs_train = model.call(trainset)
            embedding_train = outputs_train.z.numpy()
            embedding_train = pd.DataFrame(embedding_train, columns=latent_names)
            embedding_train.to_csv(
                f'{self.path_to_save_embeddings}/pmvae_{self.name_dataset}_embeddings_train_fold_{fold}.txt'
            )
            
            X_reconstructed_train = outputs_train.global_recon.numpy()
            pd.DataFrame(X_reconstructed_train).to_csv(
                f'{self.path_to_save_reconstructed}/pmvae_{self.name_dataset}_reconstruction_train_fold_{fold}.txt'
            )  

            # Get embeddings and reconstructions for validation set
            outputs_val = model.call(valset)
            embedding_val = outputs_val.z.numpy()
            embedding_val = pd.DataFrame(embedding_val, columns=latent_names)
            embedding_val.to_csv(
                f'{self.path_to_save_embeddings}/pmvae_{self.name_dataset}_embeddings_val_fold_{fold}.txt'
            )
            
            X_reconstructed_val = outputs_val.global_recon.numpy()
            pd.DataFrame(X_reconstructed_val).to_csv(
                f'{self.path_to_save_reconstructed}/pmvae_{self.name_dataset}_reconstruction_val_fold_{fold}.txt'
            )
            
            # Get embeddings and reconstructions for test set
            outputs_test = model.call(testset)
            embedding_test = outputs_test.z.numpy()
            embedding_test = pd.DataFrame(embedding_test, columns=latent_names)
            embedding_test.to_csv(
                f'{self.path_to_save_embeddings}/pmvae_{self.name_dataset}_embeddings_test_fold_{fold}.txt'
            )
            
            X_reconstructed_test = outputs_test.global_recon.numpy()
            pd.DataFrame(X_reconstructed_test).to_csv(
                f'{self.path_to_save_reconstructed}/pmvae_{self.name_dataset}_reconstruction_test_fold_{fold}.txt'
            )
            
            # Compute reconstruction metrics
            mse_score_train = compute_mse_score(adata_train, X_reconstructed_train)
            mse_score_val = compute_mse_score(adata_val, X_reconstructed_val)
            mse_score_test = compute_mse_score(adata_test, X_reconstructed_test)
            
            corr_train = compute_pearson_score(adata_train, X_reconstructed_train)
            corr_val = compute_pearson_score(adata_val, X_reconstructed_val)
            corr_test = compute_pearson_score(adata_test, X_reconstructed_test)
            
            # Build latent space AnnData objects
            adata_latent_train = build_adata_latent(embedding_train, adata_train, column_labels_name)
            adata_latent_val = build_adata_latent(embedding_val, adata_val, column_labels_name)
            adata_latent_test = build_adata_latent(embedding_test, adata_test, column_labels_name)
            
            # Apply clustering
            clusters_train, true_labels_train = apply_clustering_algo(
                adata_latent_train, column_labels_name, 'Leiden', value_resolution
            )
            clusters_val, true_labels_val = apply_clustering_algo(
                adata_latent_val, column_labels_name, 'Leiden', value_resolution
            )
            clusters_test, true_labels_test = apply_clustering_algo(
                adata_latent_test, column_labels_name, 'Leiden', value_resolution
            )
            
            # Compute clustering metrics
            ari_train, nmi_train = apply_clustering_metrics(true_labels_train, clusters_train)
            ari_val, nmi_val = apply_clustering_metrics(true_labels_val, clusters_val)
            ari_test, nmi_test = apply_clustering_metrics(true_labels_test, clusters_test)
            
            # Plot results
            plot_umap_orig_and_clusters(
                embedding_train, true_labels_train, self.name_dataset, self.name_model, 
                clusters_train, ari_train, nmi_train, 'Train', fold, 'Leiden', path_save_fig
            )
            plot_umap_orig_and_clusters(
                embedding_val, true_labels_val, self.name_dataset, self.name_model, 
                clusters_val, ari_val, nmi_val, 'Val', fold, 'Leiden', path_save_fig
            )
            plot_umap_orig_and_clusters(
                embedding_test, true_labels_test, self.name_dataset, self.name_model, 
                clusters_test, ari_test, nmi_test, 'Test', fold, 'Leiden', path_save_fig
            )
            
            # Apply classification algorithms
            accuracy_train_rf, precision_train_rf, recall_train_rf, f1_train_rf, roc_auc_train_rf, accuracy_val_rf, precision_val_rf, recall_val_rf, f1_val_rf, roc_auc_val_rf, accuracy_test_rf, precision_test_rf, recall_test_rf, f1_test_rf, roc_auc_test_rf = apply_classification_algo2(adata_latent_train, adata_latent_val, adata_latent_test, column_labels_name, 'random_forest', test_size=0.1)
            accuracy_train_xg, precision_train_xg, recall_train_xg, f1_train_xg, roc_auc_train_xg, accuracy_val_xg, precision_val_xg, recall_val_xg, f1_val_xg, roc_auc_val_xg, accuracy_test_xg, precision_test_xg, recall_test_xg, f1_test_xg, roc_auc_test_xg = apply_classification_algo2(adata_latent_train, adata_latent_val, adata_latent_test, column_labels_name, 'xgboost', test_size=0.1)
            
            results.append({
                'model_name': self.name_model,
                'dataset_name': self.name_dataset,
                'fold': fold + 1,
                'random_seed': random_seed,
                'beta': self.beta,
                'learning_rate': self.lr,
                'batch_size': self.batch_size,
                'module_latent_dim': self.module_latent_dim,
                'add_auxiliary_module': self.add_auxiliary_module,
                'hidden_layers': str(self.hidden_layers),
                'final_train_loss': hist['loss'][-1] if 'loss' in hist else None,
                'final_valid_loss': hist['val_loss'][-1] if 'val_loss' in hist else None,
                'n_epochs_trained': len(hist.get('loss', [])),
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
        
        # Create DataFrame with all results
        results_df = pd.DataFrame(results)
        
        # Print summary statistics
        print("\n=== K-Fold Cross-Validation Summary ===")
        metrics_to_summarize = [
            'ari_test', 'nmi_test', 'mse_score_test', 'corr_test', 
            'accuracy_test_rf', 'f1_test_rf', 'roc_auc_test_rf',
            'accuracy_test_xg', 'f1_test_xg', 'roc_auc_test_xg'
        ]
        for metric in metrics_to_summarize:
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Save results
        results_df.to_csv(
            f'{save_path_results}/{self.name_model}_{self.name_dataset}_{self.n_folds}fold_cv_results.csv', 
            index=False
        )
        print(f"\nResults saved to {save_path_results}/{self.name_model}_{self.name_dataset}_{self.n_folds}fold_cv_results.csv")
        
        return results_df