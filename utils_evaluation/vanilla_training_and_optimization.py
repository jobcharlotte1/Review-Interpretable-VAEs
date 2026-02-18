import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.append('/home/BS94_SUR/phD/review/utils/utils_evaluation/')
import vanilla_utils
from vanilla_utils import *
import utils_evaluation_models
from utils_evaluation_models import *

# ============================================================================
# VANILLA VAE FOR scRNA-seq
# ============================================================================

class EarlyStopping:
    """Simple EarlyStopping class."""
    def __init__(self, patience=10, verbose=False, mode='min'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == 'min' and score >= self.best_score) or (self.mode == 'max' and score <= self.best_score):
            self.counter += 1
            #if self.verbose:
                #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class VanillaVAE(nn.Module):
    def __init__(self, n_genes, n_latent, hidden_layers=[800, 800], dropout=0.3, beta=0.05,
                 init_w=False, device=None, path_model="trained_vae.pt"):
        super(VanillaVAE, self).__init__()

        self.n_genes = n_genes
        self.n_latent = n_latent
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.beta = beta
        self.init_w = init_w
        self.dev = device or torch.device('cpu')
        self.path_model = path_model

        # Build encoder
        encoder_layers = []
        input_dim = n_genes
        for h_dim in hidden_layers:
            encoder_layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent layers
        self.mean = nn.Sequential(nn.Linear(hidden_layers[-1], n_latent), nn.Dropout(dropout))
        self.logvar = nn.Sequential(nn.Linear(hidden_layers[-1], n_latent), nn.Dropout(dropout))

        # Build decoder (mirror of encoder)
        decoder_layers = []
        input_dim = n_latent
        for h_dim in reversed(hidden_layers):
            decoder_layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_layers[0], n_genes))
        self.decoder = nn.Sequential(*decoder_layers)

        if self.init_w:
            self.encoder.apply(self._init_weights)
            self.decoder.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)

    # --------------------- VAE Core ---------------------
    def encode(self, X):
        y = self.encoder(X)
        mu, logvar = self.mean(y), self.logvar(y)
        z = self.sample_latent(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def sample_latent(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std).to(self.dev)
        return eps.mul_(std).add_(mu)

    def forward(self, X):
        z, mu, logvar = self.encode(X)
        X_rec = self.decode(z)
        return X_rec, mu, logvar

    def vae_loss(self, y_pred, y_true, mu, logvar):
        """
        Returns total loss, mse loss, and kl divergence loss separately
        """
        mse = F.mse_loss(y_pred, y_true, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = torch.mean(mse + self.beta * kld)
        return total_loss, mse.item(), kld.item()

    def to_latent(self, X):
        y = self.encoder(X)
        mu, logvar = self.mean(y), self.logvar(y)
        return self.sample_latent(mu, logvar)

    def _average_latent(self, X):
        z = self.to_latent(X)
        return z.mean(0)

    # --------------------- Training & Testing ---------------------
    def train_model(self, train_loader, learning_rate=1e-3, n_epochs=100, train_patience=10,
                    test_patience=10, test_loader=None, save_model=True):
        epoch_hist = {'train_loss': [], 'valid_loss': []}
        epoch_hist_mse = {'train_loss': [], 'valid_loss': []}
        epoch_hist_kld = {'train_loss': [], 'valid_loss': []}
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=5e-4)
        train_ES = EarlyStopping(patience=train_patience, verbose=True, mode='min')
        valid_ES = EarlyStopping(patience=test_patience, verbose=True, mode='min') if test_loader else None

        for epoch in range(n_epochs):
            self.train()
            train_loss = 0
            train_mse = 0
            train_kld = 0
            
            n_train_samples = 0
            for batch in train_loader:
                x_train = batch[0].to(self.dev)
                n_train_samples += x_train.size(0)
                optimizer.zero_grad()
                x_rec, mu, logvar = self.forward(x_train)
                loss, mse, kld = self.vae_loss(x_rec, x_train, mu, logvar)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_mse += mse
                train_kld += kld

            #epoch_loss = train_loss / (len(train_loader) * train_loader.batch_size)
            #epoch_mse = train_mse / (len(train_loader) * train_loader.batch_size)
            #epoch_kld = train_kld / (len(train_loader) * train_loader.batch_size)
            epoch_loss = train_loss / n_train_samples
            epoch_mse = train_mse / n_train_samples
            epoch_kld = train_kld / n_train_samples
            epoch_hist['train_loss'].append(epoch_loss)
            epoch_hist_mse['train_loss'].append(epoch_mse)
            epoch_hist_kld['train_loss'].append(epoch_kld)
            train_ES(epoch_loss)

            if test_loader:
                test_dict = self.test_model(test_loader)
                test_loss, test_mse, test_kld = test_dict['total_loss'], test_dict['mse'], test_dict['kld']
                epoch_hist['valid_loss'].append(test_loss)
                epoch_hist_mse['valid_loss'].append(test_mse)
                epoch_hist_kld['valid_loss'].append(test_kld)
                valid_ES(test_loss)
                print(f"[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f} (MSE: {epoch_mse:.4f}, KLD: {epoch_kld:.4f}) | "
                      f"Valid Loss: {test_loss:.4f} (MSE: {test_mse:.4f}, KLD: {test_kld:.4f})")
                if valid_ES.early_stop or train_ES.early_stop:
                    print(f"[Epoch {epoch+1}] Early stopping triggered.")
                    break
            else:
                print(f"[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f} (MSE: {epoch_mse:.4f}, KLD: {epoch_kld:.4f})")
                if train_ES.early_stop:
                    print(f"[Epoch {epoch+1}] Early stopping triggered.")
                    break

        if save_model:
            print(f"Saving model to {self.path_model}")
            torch.save(self.state_dict(), self.path_model)

        return epoch_hist, epoch_hist_mse, epoch_hist_kld

    def test_model(self, loader):
        self.eval()
        loss_total = 0
        mse_total = 0
        kld_total = 0
        n_samples = 0
        with torch.no_grad():
            for batch in loader:
                data = batch[0].to(self.dev)
                n_samples += data.size(0)
                recon, mu, logvar = self.forward(data)
                total_loss, mse, kld = self.vae_loss(recon, data, mu, logvar)
                loss_total += total_loss.item()
                mse_total += mse
                kld_total += kld

        #n_samples = len(loader) * loader.batch_size
        return {
            'total_loss': loss_total / n_samples,
            'mse': mse_total / n_samples,
            'kld': kld_total / n_samples
        }


# ============================================================================
# HYPERPARAMETER + ARCHITECTURE OPTIMIZATION
# ============================================================================

def optimize_vae_architecture(
    train_data,
    test_data,
    device,
    save_path='vae_optimization_results.csv',
    n_iterations=50,
    n_epochs=500,
    early_stopping_patience=25,
    random_seed=42
):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Define search spaces
    search_space = {
        'n_layers': [2, 3, 4],
        'layer_size_base': [128, 256, 512, 800, 1000],
        'latent_dim': [50, 75, 100, 150, 200, 300, 400, 500, 600],
        'batch_size': [32, 64, 128, 256, 512],
        'learning_rate': (1e-5, 1e-2, 'log'),
        'beta': (1e-8, 1e1, 'log'),
        'dropout': (0.0, 0.5, 'uniform')
    }
    
    # Determine n_genes from sample batch
    sample_batch = next(iter(torch.utils.data.DataLoader(train_data, batch_size=1)))
    x = sample_batch[0]
    n_genes = x.shape[1]
    
    # Load existing results
    try:
        existing_df = pd.read_csv(save_path)
        results = existing_df.to_dict('records')
        start_iteration = len(results)
        print(f"Loaded {len(results)} existing results from {save_path}")
    except FileNotFoundError:
        results = []
        start_iteration = 0
        print("Starting fresh optimization")
    
    for iteration in range(start_iteration, start_iteration + n_iterations):
        print(f"\nIteration {iteration + 1}/{start_iteration + n_iterations}")
        
        # Sample architecture
        n_layers = int(np.random.choice(search_space['n_layers']))
        layer_size_base = int(np.random.choice(search_space['layer_size_base']))
        latent_dim = int(np.random.choice(search_space['latent_dim']))
        hidden_layers = [max(latent_dim*2, layer_size_base // (2**i)) for i in range(n_layers)]
        
        # Sample hyperparameters
        batch_size = int(np.random.choice(search_space['batch_size']))
        lr = 10 ** np.random.uniform(np.log10(search_space['learning_rate'][0]),
                                    np.log10(search_space['learning_rate'][1]))
        beta = 10 ** np.random.uniform(np.log10(search_space['beta'][0]),
                                      np.log10(search_space['beta'][1]))
        dropout = np.random.uniform(search_space['dropout'][0], search_space['dropout'][1])
        
        # Print configuration
        print(f"Hidden layers: {hidden_layers}, Latent dim: {latent_dim}")
        print(f"Batch size: {batch_size}, LR: {lr:.2e}, Beta: {beta:.2e}, Dropout: {dropout:.3f}")
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)
        
        try:
            # Instantiate model
            model = VanillaVAE(
                n_genes=n_genes,
                n_latent=latent_dim,
                hidden_layers=hidden_layers,
                dropout=dropout,
                beta=beta,
                device=device
            ).to(device)
            
            n_params = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {n_params:,}")
            
            # Train model
            history, history_mse, history_kld = model.train_model(
                train_loader=train_loader,
                test_loader=test_loader,
                learning_rate=lr,
                n_epochs=n_epochs,
                train_patience=early_stopping_patience,
                test_patience=early_stopping_patience,
                save_model=False
            )
            
            # Extract metrics
            final_valid_loss = history['valid_loss'][-1] if len(history['valid_loss'])>0 else np.nan
            final_valid_mse = history_mse['valid_loss'][-1] if len(history_mse['valid_loss'])>0 else np.nan
            final_valid_kld = history_kld['valid_loss'][-1] if len(history_kld['valid_loss'])>0 else np.nan
            
            result = {
                'iteration': iteration + 1,
                'n_layers': n_layers,
                'hidden_layers': str(hidden_layers),
                'latent_dim': latent_dim,
                'n_parameters': n_params,
                'batch_size': batch_size,
                'learning_rate': lr,
                'beta': beta,
                'dropout': dropout,
                'final_valid_loss': final_valid_loss,
                'final_valid_recon': final_valid_mse,
                'final_valid_kld': final_valid_kld,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'status': 'success'
            }
            
        except Exception as e:
            print(f"Error during training: {e}")
            result = {
                'iteration': iteration + 1,
                'n_layers': n_layers,
                'hidden_layers': str(hidden_layers),
                'latent_dim': latent_dim,
                'n_parameters': 0,
                'batch_size': batch_size,
                'learning_rate': lr,
                'beta': beta,
                'dropout': dropout,
                'final_valid_loss': np.nan,
                'final_valid_recon': np.nan,
                'final_valid_kld': np.nan,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'status': f'error: {e}'
            }
        
        results.append(result)
        pd.DataFrame(results).to_csv(save_path, index=False)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return pd.DataFrame(results)




class VanillaVAE_train_multiple_times:
    
    def __init__(self,
                 adata,
                 name_model: str,
                 name_dataset: str,
                 n_training: int, 
                 n_epochs: int,
                 lr: float,
                 batch_size: int,
                 beta: float,
                 dropout: float,
                 n_latent: int,
                 hidden_layers: list,
                 train_patience: int,
                 test_patience: int,
                 dev: str,
                 path_to_save_embeddings: str,
                 path_to_save_reconstructed: str):

        self.adata = adata
        self.name_model = name_model
        self.name_dataset = name_dataset
        self.n_training = n_training
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.beta = beta
        self.dropout = dropout
        self.n_latent = n_latent
        self.hidden_layers = hidden_layers
        self.train_patience = train_patience
        self.test_patience = test_patience
        self.dev = dev
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
                      resolution_value,
                      plot_umap_orig_and_clusters):
        
        results = []
        
        for n in range(self.n_training):
            random_seed = random_seed_list[n]
            print(f'Training {n} - seed {random_seed}')
            
            X_train, X_val, X_test, adata_train, adata_val, adata_test, train_data, val_data, test_data = create_vanilla_data(self.adata, 0.1, random_seed)
            
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
            
            model = VanillaVAE(
                n_genes=self.adata.n_vars,
                n_latent=self.n_latent,
                hidden_layers=self.hidden_layers,
                dropout=self.dropout,
                beta=self.beta,
                init_w=True,
                device=self.dev,
                path_model="vae_model.pt"
            ).to(self.dev)
            
            history_loss, history_mse, history_kld = model.train_model(
                train_loader=train_loader,
                learning_rate=self.lr,
                n_epochs=self.n_epochs,
                train_patience=self.train_patience,
                test_patience=self.test_patience,
                test_loader=val_loader,
                save_model=False
            )
            
            embedding_train = model.to_latent(torch.Tensor(X_train).to(self.dev))
            embedding_train_array = embedding_train.cpu().detach().numpy()
            pd.DataFrame(embedding_train_array).to_csv(f'{self.path_to_save_embeddings}/vanillavae_{self.name_dataset}_embeddings_train_{n}_seed_{random_seed}.txt')
            
            X_reconstructed_train = model.decode(torch.Tensor(embedding_train).to(self.dev))
            X_reconstructed_train_array = X_reconstructed_train.cpu().detach().numpy()
            pd.DataFrame(X_reconstructed_train_array).to_csv(f'{self.path_to_save_reconstructed}/vanillavae_{self.name_dataset}_reconstruction_train_{n}_seed_{random_seed}.txt')
            
            embedding_val = model.to_latent(torch.Tensor(X_val).to(self.dev))
            embedding_val_array = embedding_val.cpu().detach().numpy()
            pd.DataFrame(embedding_val_array).to_csv(f'{self.path_to_save_embeddings}/vanillavae_{self.name_dataset}_embeddings_val_{n}_seed_{random_seed}.txt')
            
            X_reconstructed_val = model.decode(torch.Tensor(embedding_val).to(self.dev))
            X_reconstructed_val_array = X_reconstructed_val.cpu().detach().numpy()
            pd.DataFrame(X_reconstructed_val_array).to_csv(f'{self.path_to_save_reconstructed}/vanillavae_{self.name_dataset}_reconstruction_val_{n}_seed_{random_seed}.txt')
            
            embedding_test = model.to_latent(torch.Tensor(X_test).to(self.dev))
            embedding_test_array = embedding_test.cpu().detach().numpy()
            pd.DataFrame(embedding_test_array).to_csv(f'{self.path_to_save_embeddings}/vanillavae_{self.name_dataset}_embeddings_test_{n}_seed_{random_seed}.txt')
            
            X_reconstructed_test = model.decode(torch.Tensor(embedding_test).to(self.dev))
            X_reconstructed_test_array = X_reconstructed_test.cpu().detach().numpy()
            pd.DataFrame(X_reconstructed_test_array).to_csv(f'{self.path_to_save_reconstructed}/vanillavae_{self.name_dataset}_reconstruction_test_{n}_seed_{random_seed}.txt')
            
            mse_score_train = compute_mse_score(adata_train, X_reconstructed_train_array)
            mse_score_val = compute_mse_score(adata_val, X_reconstructed_val_array)
            mse_score_test = compute_mse_score(adata_test, X_reconstructed_test_array)
            
            corr_train = compute_pearson_score(adata_train, X_reconstructed_train_array)
            corr_val = compute_pearson_score(adata_val, X_reconstructed_val_array)
            corr_test = compute_pearson_score(adata_test, X_reconstructed_test_array)
            
            adata_latent_train = build_adata_latent(embedding_train_array, adata_train, column_labels_name)
            adata_latent_val = build_adata_latent(embedding_val_array, adata_val, column_labels_name)
            adata_latent_test = build_adata_latent(embedding_test_array, adata_test, column_labels_name)
            
            clusters_train, true_labels_train = apply_clustering_algo(adata_latent_train, column_labels_name, 'Leiden', resolution_value)
            clusters_val, true_labels_val = apply_clustering_algo(adata_latent_val, column_labels_name, 'Leiden', resolution_value)
            clusters_test, true_labels_test = apply_clustering_algo(adata_latent_test, column_labels_name, 'Leiden', resolution_value)
            
            ari_train, nmi_train = apply_clustering_metrics(true_labels_train, clusters_train)
            ari_val, nmi_val = apply_clustering_metrics(true_labels_val, clusters_val)
            ari_test, nmi_test = apply_clustering_metrics(true_labels_test, clusters_test)
            
            plot_umap_orig_and_clusters(embedding_train_array, true_labels_train, self.name_dataset, self.name_model, 
                                       clusters_train, ari_train, nmi_train, 'Train', n, 'Leiden', path_save_fig)
            plot_umap_orig_and_clusters(embedding_val_array, true_labels_val, self.name_dataset, self.name_model, 
                                       clusters_val, ari_val, nmi_val, 'Val', n, 'Leiden', path_save_fig)
            plot_umap_orig_and_clusters(embedding_test_array, true_labels_test, self.name_dataset, self.name_model, 
                                       clusters_test, ari_test, nmi_test, 'Test', n, 'Leiden', path_save_fig)
            
            accuracy_train_rf, precision_train_rf, recall_train_rf, f1_train_rf, roc_auc_train_rf, accuracy_test_rf, precision_test_rf, recall_test_rf, f1_test_rf, roc_auc_test_rf = apply_classification_algo2(adata_latent_train, adata_latent_test, column_labels_name, 'random_forest', test_size=0.1)
            accuracy_train_xg, precision_train_xg, recall_train_xg, f1_train_xg, roc_auc_train_xg, accuracy_test_xg, precision_test_xg, recall_test_xg, f1_test_xg, roc_auc_test_xg = apply_classification_algo2(adata_latent_train, adata_latent_test, column_labels_name, 'xgboost', test_size=0.1)

            
            results.append({
                'model_name': self.name_model,
                'dataset_name': self.name_dataset,
                'id_training': n,
                'random_seed': random_seed,
                'beta': self.beta,
                'learning_rate': self.lr,
                'batch_size': self.batch_size,
                'dropout': self.dropout,
                'n_latent': self.n_latent,
                'hidden_layers': str(self.hidden_layers),
                'final_train_loss': history_loss["train_loss"][-1],
                'final_mse_train_loss': history_mse["train_loss"][-1],
                'final_kld_train_loss': history_kld["train_loss"][-1],
                'final_valid_loss': history_loss["valid_loss"][-1],
                'final_mse_valid_loss': history_mse["valid_loss"][-1],
                'final_kld_valid_loss': history_kld["valid_loss"][-1],
                'n_epochs_trained': len(history_loss["train_loss"]),
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
        results_df.to_csv(f'{save_path_results}/{self.name_model}_{self.name_dataset}_{self.n_training}_training_results.csv', index=False)
        
        return results_df
    
    
    
class VanillaVAE_kfold_cross_validation:
    
    def __init__(self,
                 adata,
                 name_model: str,
                 name_dataset: str,
                 n_folds: int,  # Changed from n_training to n_folds
                 n_epochs: int,
                 lr: float,
                 batch_size: int,
                 beta: float,
                 dropout: float,
                 n_latent: int,
                 hidden_layers: list,
                 train_patience: int,
                 test_patience: int,
                 dev: str,
                 path_to_save_embeddings: str,
                 path_to_save_reconstructed: str):

        self.adata = adata
        self.name_model = name_model
        self.name_dataset = name_dataset
        self.n_folds = n_folds
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.beta = beta
        self.dropout = dropout
        self.n_latent = n_latent
        self.hidden_layers = hidden_layers
        self.train_patience = train_patience
        self.test_patience = test_patience
        self.dev = dev
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
                      resolution_value,
                      plot_umap_orig_and_clusters):
        """
        Perform stratified k-fold cross-validation on VanillaVAE model
        
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
                test_size=0.2,
                random_state=random_seed,
                stratify=train_val_labels
            )
            
            X_train = X_train_val[train_idx_relative]
            X_val = X_train_val[val_idx_relative]
            
            adata_train = self.adata[train_val_idx[train_idx_relative]].copy()
            adata_val = self.adata[train_val_idx[val_idx_relative]].copy()
            
            # Create data loaders
            train_data = torch.utils.data.TensorDataset(torch.FloatTensor(X_train))
            val_data = torch.utils.data.TensorDataset(torch.FloatTensor(X_val))
            test_data = torch.utils.data.TensorDataset(torch.FloatTensor(X_test))
            
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
            
            # Initialize model for this fold
            model = VanillaVAE(
                n_genes=self.adata.n_vars,
                n_latent=self.n_latent,
                hidden_layers=self.hidden_layers,
                dropout=self.dropout,
                beta=self.beta,
                init_w=True,
                device=self.dev,
                path_model=f"vae_model_fold_{fold}.pt"
            ).to(self.dev)
            
            # Train model
            history_loss, history_mse, history_kld = model.train_model(
                train_loader=train_loader,
                learning_rate=self.lr,
                n_epochs=self.n_epochs,
                train_patience=self.train_patience,
                test_patience=self.test_patience,
                test_loader=val_loader,
                save_model=False
            )
            
            # Get embeddings and reconstructions for train set
            embedding_train = model.to_latent(torch.Tensor(X_train).to(self.dev))
            embedding_train_array = embedding_train.cpu().detach().numpy()
            pd.DataFrame(embedding_train_array).to_csv(
                f'{self.path_to_save_embeddings}/vanillavae_{self.name_dataset}_embeddings_train_fold_{fold}.txt'
            )
            
            X_reconstructed_train = model.decode(torch.Tensor(embedding_train).to(self.dev))
            X_reconstructed_train_array = X_reconstructed_train.cpu().detach().numpy()
            pd.DataFrame(X_reconstructed_train_array).to_csv(
                f'{self.path_to_save_reconstructed}/vanillavae_{self.name_dataset}_reconstruction_train_fold_{fold}.txt'
            )
            
            # Get embeddings and reconstructions for validation set
            embedding_val = model.to_latent(torch.Tensor(X_val).to(self.dev))
            embedding_val_array = embedding_val.cpu().detach().numpy()
            pd.DataFrame(embedding_val_array).to_csv(
                f'{self.path_to_save_embeddings}/vanillavae_{self.name_dataset}_embeddings_val_fold_{fold}.txt'
            )
            
            X_reconstructed_val = model.decode(torch.Tensor(embedding_val).to(self.dev))
            X_reconstructed_val_array = X_reconstructed_val.cpu().detach().numpy()
            pd.DataFrame(X_reconstructed_val_array).to_csv(
                f'{self.path_to_save_reconstructed}/vanillavae_{self.name_dataset}_reconstruction_val_fold_{fold}.txt'
            )
            
            # Get embeddings and reconstructions for test set
            embedding_test = model.to_latent(torch.Tensor(X_test).to(self.dev))
            embedding_test_array = embedding_test.cpu().detach().numpy()
            pd.DataFrame(embedding_test_array).to_csv(
                f'{self.path_to_save_embeddings}/vanillavae_{self.name_dataset}_embeddings_test_fold_{fold}.txt'
            )
            
            X_reconstructed_test = model.decode(torch.Tensor(embedding_test).to(self.dev))
            X_reconstructed_test_array = X_reconstructed_test.cpu().detach().numpy()
            pd.DataFrame(X_reconstructed_test_array).to_csv(
                f'{self.path_to_save_reconstructed}/vanillavae_{self.name_dataset}_reconstruction_test_fold_{fold}.txt'
            )
            
            # Compute reconstruction metrics
            mse_score_train = compute_mse_score(adata_train, X_reconstructed_train_array)
            mse_score_val = compute_mse_score(adata_val, X_reconstructed_val_array)
            mse_score_test = compute_mse_score(adata_test, X_reconstructed_test_array)
            
            corr_train = compute_pearson_score(adata_train, X_reconstructed_train_array)
            corr_val = compute_pearson_score(adata_val, X_reconstructed_val_array)
            corr_test = compute_pearson_score(adata_test, X_reconstructed_test_array)
            
            # Build latent space AnnData objects
            adata_latent_train = build_adata_latent(embedding_train_array, adata_train, column_labels_name)
            adata_latent_val = build_adata_latent(embedding_val_array, adata_val, column_labels_name)
            adata_latent_test = build_adata_latent(embedding_test_array, adata_test, column_labels_name)
            
            # Apply clustering
            clusters_train, true_labels_train = apply_clustering_algo(
                adata_latent_train, column_labels_name, 'Leiden', resolution_value
            )
            clusters_val, true_labels_val = apply_clustering_algo(
                adata_latent_val, column_labels_name, 'Leiden', resolution_value
            )
            clusters_test, true_labels_test = apply_clustering_algo(
                adata_latent_test, column_labels_name, 'Leiden', resolution_value
            )
            
            # Compute clustering metrics
            ari_train, nmi_train = apply_clustering_metrics(true_labels_train, clusters_train)
            ari_val, nmi_val = apply_clustering_metrics(true_labels_val, clusters_val)
            ari_test, nmi_test = apply_clustering_metrics(true_labels_test, clusters_test)
            
            # Plot results
            plot_umap_orig_and_clusters(
                embedding_train_array, true_labels_train, self.name_dataset, self.name_model, 
                clusters_train, ari_train, nmi_train, 'Train', fold, 'Leiden', path_save_fig
            )
            plot_umap_orig_and_clusters(
                embedding_val_array, true_labels_val, self.name_dataset, self.name_model, 
                clusters_val, ari_val, nmi_val, 'Val', fold, 'Leiden', path_save_fig
            )
            plot_umap_orig_and_clusters(
                embedding_test_array, true_labels_test, self.name_dataset, self.name_model, 
                clusters_test, ari_test, nmi_test, 'Test', fold, 'Leiden', path_save_fig
            )
            
            # Apply classification algorithms
            accuracy_train_rf, precision_train_rf, recall_train_rf, f1_train_rf, roc_auc_train_rf, accuracy_val_rf, precision_val_rf, recall_val_rf, f1_val_rf, roc_auc_val_rf, accuracy_test_rf, precision_test_rf, recall_test_rf, f1_test_rf, roc_auc_test_rf = apply_classification_algo2(adata_latent_train, adata_latent_val, adata_latent_test, column_labels_name, 'random_forest', test_size=0.1)
            accuracy_train_xg, precision_train_xg, recall_train_xg, f1_train_xg, roc_auc_train_xg, accuracy_val_xg, precision_val_xg, recall_val_xg, f1_val_xg, roc_auc_val_xg, accuracy_test_xg, precision_test_xg, recall_test_xg, f1_test_xg, roc_auc_test_xg = apply_classification_algo2(adata_latent_train, adata_latent_val, adata_latent_test, column_labels_name, 'xgboost', test_size=0.1)

            # Store results for this fold
            results.append({
                'model_name': self.name_model,
                'dataset_name': self.name_dataset,
                'fold': fold + 1,
                'random_seed': random_seed,
                'beta': self.beta,
                'learning_rate': self.lr,
                'batch_size': self.batch_size,
                'dropout': self.dropout,
                'n_latent': self.n_latent,
                'hidden_layers': str(self.hidden_layers),
                'final_train_loss': history_loss["train_loss"][-1],
                'final_mse_train_loss': history_mse["train_loss"][-1],
                'final_kld_train_loss': history_kld["train_loss"][-1],
                'final_valid_loss': history_loss["valid_loss"][-1],
                'final_mse_valid_loss': history_mse["valid_loss"][-1],
                'final_kld_valid_loss': history_kld["valid_loss"][-1],
                'n_epochs_trained': len(history_loss["train_loss"]),
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