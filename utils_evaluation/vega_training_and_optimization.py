import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.append('/home/BS94_SUR/phD/review/utils/utils_evaluation/')
import utils_train_models
from utils_train_models import *
from utils_load_files_embeddings import *

sys.path.append('/home/BS94_SUR/phD/review/models reproductibility/VEGA/vega-reproducibility/src')
#import vanilla_vae
import vega
import train_vanilla_vae_suppFig1
import utils
from utils import *
from learning_utils import *
from customized_linear import CustomizedLinear
import torch.nn.functional as F
from torch import nn, optim
from scipy import sparse
#from vanilla_vae import VanillaVAE
from vega_model import VEGA
import torch
import itertools
import pandas as pd
import os
import scipy.sparse as sp
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler


class VEGA2(torch.nn.Module):
    def __init__(self, pathway_mask, positive_decoder=False, **kwargs):
        """ Constructor for class VEGA (VAE Enhanced by Gene Annotations). """
        super(VEGA2, self).__init__()
        self.pathway_mask = pathway_mask
        self.n_pathways = self.pathway_mask.shape[1]
        self.n_genes = self.pathway_mask.shape[0]
        self.dev = kwargs.get('device', torch.device('cpu'))
        self.beta = kwargs.get("beta", 0.01)
        self.save_path = kwargs.get('path_model', "trained_vae.pt")
        self.dropout = kwargs.get('dropout', 0.2)
        self.pos_dec = positive_decoder
        print(self.dropout)
        self.encoder = nn.Sequential(nn.Linear(self.n_genes, 800),
                                    nn.BatchNorm1d(800),
                                    nn.ReLU(),
                                    nn.Dropout(self.dropout),
                                    nn.Linear(800,800),
                                    nn.BatchNorm1d(800),
                                    nn.ReLU(),
                                    nn.Dropout(self.dropout))
        self.mean = nn.Sequential(nn.Linear(800, self.n_pathways), 
                                    nn.Dropout(self.dropout))
        self.logvar = nn.Sequential(nn.Linear(800, self.n_pathways), 
                                    nn.Dropout(self.dropout))
        self.decoder = CustomizedLinear(self.pathway_mask.T)
        # Constraining decoder or not
        if self.pos_dec:
            print('Constraining decoder to positive weights', flush=True)
            self.decoder.reset_params_pos()
            self.decoder.weight.data *= self.decoder.mask        


    def encode(self, X):
        """ Encode data """
        y = self.encoder(X)
        mu, logvar = self.mean(y), self.logvar(y)
        z = self.sample_latent(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        """ Decode data """
        X_rec = self.decoder(z)
        return X_rec
    
    def sample_latent(self, mu, logvar):
        """ Sample latent space with reparametrization trick. First convert to std, sample normal(0,1) and get Z."""
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(self.dev)
        eps = eps.mul_(std).add_(mu)
        return eps

    def to_latent(self, X):
        """ Same as encode, but only returns z (no mu and logvar) """
        y = self.encoder(X)
        mu, logvar = self.mean(y), self.logvar(y)
        z = self.sample_latent(mu, logvar)
        return z
 
    def _average_latent(self, X):
        """ """
        z = self.to_latent(X)
        mean_z = z.mean(0)
        return mean_z

    def bayesian_diff_exp(self, adata1, adata2, n_samples=2000, use_permutations=True, n_permutations=1000, random_seed=False):
        """ Run Bayesian differential expression in latent space.
            Returns Bayes factor of all factors. 
        """
        self.eval()
        # Set seed for reproducibility
        if random_seed:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        epsilon = 1e-12
        # Sample cell from each condition
        idx1 = np.random.choice(np.arange(len(adata1)), n_samples)
        idx2 = np.random.choice(np.arange(len(adata2)), n_samples)
        # To latent
        z1 = self.to_latent(torch.Tensor(adata1[idx1,:].X)).detach().numpy()
        z2 = self.to_latent(torch.Tensor(adata2[idx2,:].X)).detach().numpy()
        # Compare samples by using number of permutations - if 0, just pairwise comparison
        # This estimates the double integral in the posterior of the hypothesis
        if use_permutations:
            z1, z2 = self._scale_sampling(z1, z2, n_perm=n_permutations)
        p_h1 = np.mean(z1 > z2, axis=0)
        p_h2 = 1.0 - p_h1
        mad = np.abs(np.mean(z1 - z2, axis=0))
        bf = np.log(p_h1 + epsilon) - np.log(p_h2 + epsilon) 
        # Wrap results
        res = {'p_h1':p_h1,
                'p_h2':p_h2,
                'bayes_factor': bf,
                'mad':mad}
        return res

    def _scale_sampling(self, arr1, arr2, n_perm=1000):
        """ Use permutation to better estimate double integral (create more pair comparisons)
            Inspired by scVI (Lopez et al., 2018) """
        u, v = (np.random.choice(arr1.shape[0], size=n_perm), np.random.choice(arr2.shape[0], size=n_perm))
        scaled1 = arr1[u]
        scaled2 = arr2[v]
        return scaled1, scaled2

    def forward(self, X):
        """ Forward pass through full network"""
        z, mu, logvar = self.encode(X)
        X_rec = self.decode(z)
        return X_rec, mu, logvar

    def vae_loss(self, y_pred, y_true, mu, logvar):
        """ Custom loss for VAE """
        kld = -0.5 * torch.sum(1. + logvar - mu.pow(2) - logvar.exp(), )
        mse = F.mse_loss(y_pred, y_true, reduction="sum")
        return torch.mean(mse + self.beta*kld), mse, kld

    def train_model(self, train_loader, learning_rate, n_epochs, train_patience, test_patience, test_loader=False, save_model=True):
        """ Train VAE """
        epoch_hist = {}
        epoch_hist['train_loss'] = []
        epoch_hist['valid_loss'] = []
        epoch_mse_hist = {}
        epoch_mse_hist['train_loss'] = []
        epoch_mse_hist['valid_loss'] = []
        epoch_kld_hist = {}
        epoch_kld_hist['train_loss'] = []
        epoch_kld_hist['valid_loss'] = []
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=5e-4)
        train_ES = EarlyStopping(patience=train_patience, verbose=True, mode='train')
        if test_loader:
            valid_ES = EarlyStopping(patience=test_patience, verbose=True, mode='valid')
        clipper = WeightClipper(frequency=1)
        # Train
        for epoch in range(n_epochs):
            loss_value = 0
            mse_loss_value = 0
            kld_loss_value = 0
            self.train()
            for x_train in train_loader:
                x_train = x_train.to(self.dev)
                optimizer.zero_grad()
                x_rec, mu, logvar = self.forward(x_train)
                loss, mse, kld = self.vae_loss(x_rec, x_train, mu, logvar)
                loss_value += loss.item()
                mse_loss_value += mse.item()
                kld_loss_value += kld.item()
                loss.backward()
                optimizer.step()
                if self.pos_dec:
                    self.decoder.apply(clipper)
            # Get epoch loss
            epoch_loss = loss_value / (len(train_loader) * train_loader.batch_size)
            epoch_mse_loss = mse_loss_value / (len(train_loader) * train_loader.batch_size)
            epoch_kld_loss = kld_loss_value / (len(train_loader) * train_loader.batch_size)
            epoch_hist['train_loss'].append(epoch_loss)
            epoch_mse_hist['train_loss'].append(epoch_mse_loss)
            epoch_kld_hist['train_loss'].append(epoch_kld_loss)
            train_ES(epoch_loss)
            # Eval
            if test_loader:
                self.eval()
                test_dict, mse_test_dict, kld_test_dict = self.test_model(test_loader)
                test_loss, mse_test_loss, kld_test_loss = test_dict['loss'], mse_test_dict['loss'], kld_test_dict['loss']
                epoch_hist['valid_loss'].append(test_loss)
                epoch_mse_hist['valid_loss'].append(mse_test_loss)
                epoch_kld_hist['valid_loss'].append(kld_test_loss)
                valid_ES(test_loss)
                print('[Epoch %d] | train_loss: %.3f, train_mse: %.3f, train_kld: %.3f | test_loss: %.3f, test_mse: %.3f, test_kld: %.3f |'%(epoch+1, epoch_loss, epoch_mse_loss, epoch_kld_loss, test_loss, mse_test_loss, kld_test_loss), flush=True)
                if valid_ES.early_stop or train_ES.early_stop:
                    print('[Epoch %d] Early stopping' % (epoch+1), flush=True)
                    break
            else:
                print('[Epoch %d] | loss: %.3f |' % (epoch + 1, epoch_loss), flush=True)
                if train_ES.early_stop:
                    print('[Epoch %d] Early stopping' % (epoch+1), flush=True)
                    break
        # Save model
        if save_model:
            print('Saving model to ...', self.save_path)
            torch.save(self.state_dict(), self.save_path)

        return epoch_hist, epoch_mse_hist, epoch_kld_hist

    def test_model(self, loader):
        """Test model on input loader."""
        test_dict = {}
        mse_test_dict = {}
        kld_test_dict = {}
        total_loss = 0.0
        total_mse = 0.0
        total_kld = 0.0
        self.eval()
        with torch.no_grad():
            for data in loader:
                data = data.to(self.dev)
                reconstruct_X, mu, logvar = self.forward(data)
                loss, mse, kld = self.vae_loss(reconstruct_X, data, mu, logvar)
                total_loss += loss.item()
                total_mse += mse.item()
                total_kld += kld.item()
        test_dict['loss'] = total_loss/(len(loader)*loader.batch_size)
        mse_test_dict['loss'] = total_mse/(len(loader)*loader.batch_size)
        kld_test_dict['loss'] = total_kld/(len(loader)*loader.batch_size)
        return test_dict, mse_test_dict, kld_test_dict
    
    

def access_data_vega(data_path, adata, pathway_file, path_data_description):
    if data_path is not None:
        adata = sc.read(data_path)
    pathway_dict = read_gmt(pathway_file, min_g=0, max_g=1000)
    pathway_mask = create_pathway_mask(adata.var.index.tolist(), pathway_dict, add_missing=1, fully_connected=True)
    list_pathways = load_pathways_vega(False, adata, pathway_file)
    if path_data_description is not None:
        df_genespathways = pd.read_parquet(path_data_description + f'df_pathways_genes_description.parquet')
        overlap_matrix = pd.read_csv(path_data_description + 'overlap_matrix_vega.csv')
    else:
        df_genespathways = None
        overlap_matrix = None
    
    return adata, pathway_dict, pathway_mask, list_pathways, df_genespathways, overlap_matrix


def extract_x_y_from_adata(adata: AnnData, column_labels_name: pd.Series):
    if sparse.issparse(adata.X):
        data = adata.X.A
    else:
        data = adata.X
    X = pd.DataFrame(data, index=adata.obs.index)
    y = adata.obs[column_labels_name]
    return X, y

def split_data(X, y, train_size, random_seed):
    X_train,  X_test, labels_train,  labels_test = train_test_split(
        X, y, train_size=train_size, random_state=random_seed, stratify=y)
    return X_train,  X_test, labels_train,  labels_test

def extract_index(X):
    index_df = X.index
    return index_df
    
def build_adata_from_X(adata, index_df):
    adata = adata[adata.obs.index.isin(index_df)]
    return adata, index_df

def encode_y(y):
    le = preprocessing.LabelEncoder().fit(y)
    y_encoded = torch.Tensor(le.transform(y))
    return y_encoded

def build_vega_dataset(adata, y_encoded, pathway_file):
    if sparse.issparse(adata.X):
        data = adata.X.A
    else:
        data = adata.X

    data = torch.Tensor(data)
    data = UnsupervisedDataset(data, targets=y_encoded)

    pathway_dict = read_gmt(pathway_file, min_g=0, max_g=1000)
    pathway_mask = create_pathway_mask(adata.var.index.tolist(), pathway_dict, add_missing=1, fully_connected=True)

    return data, pathway_dict, pathway_mask

def preprocess_adata(adata, n_top_genes=5000):
    """ Simple (default) sc preprocessing function before autoencoders """
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    #sc.pp.normalize_total(adata, target_sum=1e4)
    #sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    #scaler = MinMaxScaler()
    #X = adata.X
    #if not sp.issparse(X):
    #    X = sp.csr_matrix(X)
    #scaler = MaxAbsScaler()
    #adata.X = scaler.fit_transform(X)
    return adata


def create_vega_training_data(name_model, preprocess, select_hvg, n_top_genes, random_seed, train_size, column_labels_name, adata, pathway_file):
    if preprocess == True:
        print('Preprocessing adata')
        adata = preprocess_adata(adata, n_top_genes=n_top_genes)
    else:
        adata = adata.copy()
    X, y = extract_x_y_from_adata(adata, column_labels_name)
    X_train,  X_test, labels_train,  labels_test = split_data(X, y, train_size, random_seed)
    y_train = encode_y(labels_train)
    y_test = encode_y(labels_test)
    index_train = extract_index(X_train)
    index_test = extract_index(X_test)
    adata_train, index_train = build_adata_from_X(adata, index_train)
    adata_test, index_test = build_adata_from_X(adata, index_test)

    train_ds, pathway_dict, pathway_mask = build_vega_dataset(adata_train, y_train, pathway_file)
    test_ds, pathway_dict, pathway_mask = build_vega_dataset(adata_test, y_test, pathway_file)
    return adata, adata_train, adata_test, train_ds, test_ds, pathway_dict, pathway_mask


def grid_search_VEGA(
    betas,
    batch_sizes,
    lrs,
    epochs_list,
    pathway_mask,
    device,
    p_drop,
    train_ds,
    test_ds,
    train_p,
    test_p,
    save_path="grid_search_results.csv"
):
    results = []

    # If file exists, resume and append to it
    if os.path.exists(save_path):
        print(f"Resuming from existing file: {save_path}")
        results_df = pd.read_csv(save_path)
        # Convert existing rows to dicts
        results = results_df.to_dict(orient="records")

    for beta, batch_size, lr, n_epochs in itertools.product(betas, batch_sizes, lrs, epochs_list):

        print(f"\n=== Running config: beta={beta}, batch={batch_size}, lr={lr}, epochs={n_epochs} ===")

        # ---- Create data loaders ----
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)

        # ---- Model parameters ----
        dict_params = {
            "pathway_mask": pathway_mask,
            "n_pathways": pathway_mask.shape[1],
            "n_genes": pathway_mask.shape[0],
            "device": device,
            "beta": beta,
            "save_path": False,
            "dropout": p_drop,
            "pos_dec": True,
        }

        # ---- Instantiate model ----
        model = VEGA(**dict_params).to(device)

        # ---- Train model ----
        hist = model.train_model(
            train_loader,
            lr,
            n_epochs,
            train_p,
            test_p,
            test_loader,
            save_model=False
        )

        # Get validation loss or last element
        final_loss = hist["valid_loss"][-1] 

        print(f"Final validation loss: {final_loss:.4f}")

        # ---- Save record for this run ----
        result = {
            "beta": beta,
            "batch_size": batch_size,
            "lr": lr,
            "n_epochs": n_epochs,
            "val_loss": final_loss,
        }
        results.append(result)

        # ---- Save DataFrame to disk at each iteration ----
        df = pd.DataFrame(results)
        df.to_csv(save_path, index=False)
        print(f"Saved progress to {save_path}")

    # Return final dataframe and best config
    df = pd.DataFrame(results)
    best_row = df.loc[df['val_loss'].idxmin()].to_dict()

    return df, best_row


def random_search_hyperparameters(
    train_data,
    test_data,
    pathway_mask,
    device,
    save_path='random_hyperparameter_search_results.csv',
    n_iterations=50,
    n_epochs=2000,
    train_p=25,
    test_p=25,
    random_seed=42
):
    """
    Perform random search for hyperparameter optimization.
    
    Parameters:
    -----------
    train_data : torch.utils.data.Dataset
        Training dataset
    test_data : torch.utils.data.Dataset
        Test dataset
    pathway_mask : torch.Tensor
        Pathway mask for VEGA model
    device : torch.device
        Device to run training on
    save_path : str
        Path where to save the results CSV file (results saved after each iteration)
    n_iterations : int
        Number of random search iterations
    n_epochs : int
        Number of epochs per training
    train_p : int
        Training patience for early stopping
    test_p : int
        Test patience for early stopping
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame : DataFrame with hyperparameters and corresponding losses
    """
    
    # Set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Define hyperparameter search spaces
    param_distributions = {
        'beta': (1e-9, 1e-3, 'log'),  # (min, max, scale)
        'learning_rate': (1e-5, 1e-2, 'log'),
        'batch_size': ([32, 64, 128, 256, 512], 'choice'),
        'dropout': (0.0, 1, 'uniform')
    }
    
    # Initialize results storage
    results = []
    
    # Check if file exists and load previous results
    try:
        existing_df = pd.read_csv(save_path)
        results = existing_df.to_dict('records')
        print(f"Loaded {len(results)} existing results from {save_path}")
        start_iteration = len(results)
    except FileNotFoundError:
        print(f"No existing results found. Starting fresh.")
        start_iteration = 0
    
    print(f"Starting random search with {n_iterations} iterations...")
    print("="*70)
    
    for iteration in range(start_iteration, start_iteration + n_iterations):
        print(f"\nIteration {iteration + 1}/{start_iteration + n_iterations}")
        print("-"*70)
        
        # Sample hyperparameters
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
        
        if param_distributions['dropout'][2] == 'uniform':
            dropout = np.random.uniform(
                param_distributions['dropout'][0],
                param_distributions['dropout'][1]
            )
        
        # Print current hyperparameters
        print(f"beta: {beta:.2e}")
        print(f"learning_rate: {lr:.2e}")
        print(f"batch_size: {batch_size}")
        print(f"dropout: {dropout:.3f}")
        
        # Create data loaders with current batch size
        train_loader = torch.utils.data.DataLoader(
            train_data, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True
        )
        
        # Create model with current hyperparameters
        dict_params = {
            "pathway_mask": pathway_mask,
            "n_pathways": pathway_mask.shape[1],
            "n_genes": pathway_mask.shape[0],
            "device": device,
            "beta": beta,
            "save_path": False,
            "dropout": dropout,
            "pos_dec": True,
        }
        
        try:
            # Initialize model
            model = VEGA2(**dict_params).to(device)
            
            # Train model
            hist, mse_hist, kld_hist = model.train_model(
                train_loader,
                lr,
                n_epochs,
                train_p,
                test_p,
                test_loader,
                save_model=False
            )
            
            # Get final validation loss
            final_loss = hist["valid_loss"][-1]
            final_train_loss = hist["train_loss"][-1]
            final_mse_loss = mse_hist["valid_loss"][-1]
            final_mse_train_loss = mse_hist["train_loss"][-1]
            final_kld_loss = kld_hist["valid_loss"][-1]
            final_kld_train_loss = kld_hist["train_loss"][-1]
            n_epochs_trained = len(hist["train_loss"])
            
            print(f"Final validation loss: {final_loss:.4f}")
            print(f"Final training loss: {final_train_loss:.4f}")
            print(f"Final validation mse loss: {final_mse_loss:.4f}")
            print(f"Final training mse loss: {final_mse_train_loss:.4f}")
            print(f"Final validation kld loss: {final_kld_loss:.4f}")
            print(f"Final training kld loss: {final_kld_train_loss:.4f}")
            print(f"Epochs trained: {n_epochs_trained}")
            
            # Store results
            results.append({
                'iteration': iteration + 1,
                'beta': beta,
                'learning_rate': lr,
                'batch_size': batch_size,
                'dropout': dropout,
                'final_valid_loss': final_loss,
                'final_train_loss': final_train_loss,
                'final_mse_valid_loss': final_mse_loss,
                'final_mse_train_loss': final_mse_train_loss,
                'final_kld_valid_loss': final_kld_loss,
                'final_kld_train_loss': final_kld_train_loss,
                'n_epochs_trained': n_epochs_trained,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Save results immediately after each iteration
            results_df = pd.DataFrame(results)
            results_df.to_csv(save_path, index=False)
            print(f"Results saved to {save_path}")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            results.append({
                'iteration': iteration + 1,
                'beta': beta,
                'learning_rate': lr,
                'batch_size': batch_size,
                'dropout': dropout,
                'final_valid_loss': np.nan,
                'final_train_loss': np.nan,
                'n_epochs_trained': 0,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'error': str(e)
            })
            
            # Save results even on error
            results_df = pd.DataFrame(results)
            results_df.to_csv(save_path, index=False)
            print(f"Results saved to {save_path}")
        
        # Clear memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by validation loss
    results_df = results_df.sort_values('final_valid_loss', ascending=True)
    
    print("\n" + "="*70)
    print("Random search completed!")
    print(f"\nBest hyperparameters (lowest validation loss):")
    print("-"*70)
    best_idx = results_df['final_valid_loss'].idxmin()
    print(results_df.loc[best_idx, ['beta', 'learning_rate',
                                     'batch_size', 'dropout', 'final_valid_loss']])
    
    return results_df
