import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scanpy as sc
from torch.utils.data import DataLoader, TensorDataset
import time
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import optuna
import anndata as ad

# Load and preprocess the AnnData matrix
def prepare_dataloader_adata(adata, batch_size):
    # Convert the AnnData matrix to a tensor
    X = torch.tensor(adata.X, dtype=torch.float32)  # Use .toarray() if sparse
    dataset = TensorDataset(X)
    data_loader =  DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader

def prepare_dataloader(data, batch_size, data_type):
    data = torch.Tensor(data)
    dataset = TensorDataset(data, data) 
        
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader

        
def objective(trial):
    batch_size = trial.suggest_int('batch_size', 8, 256, 8)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    num_epochs = trial.suggest_int('num_epochs', 10, 200, 10)
    dataloader = prepare_dataloader(adata, batch_size=batch_size)
    vae = VAE(input_dim=adata.n_vars, hidden_dim=hidden_dim, latent_dim=latent_dim)  # Adjust input_dim to match gene count
    model, epoch_losses, epoch_kl_losses, epoch_mse_losses = train_vae(vae, dataloader, num_epochs=num_epochs, learning_rate=lr)
    return epoch_losses[-1]


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        
        self.latent_dim = latent_dim 
        
        # Encoder
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, latent_dim*2)
        self.kl = 0
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # Sample from standard normal distribution
        sample = mu + eps * std
        return sample  # Ensuring Gaussian prior
    
    def forward(self, x):
        x = F.leaky_relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, self.latent_dim)
        
        mu = x[:, 0, :]  # first dimension for mu
        log_var = x[:, 1, :]  # second dimension for log_var
        z = self.reparameterize(mu, log_var)
        
        self.kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return mu, log_var, z
    
    
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Decoder, self).__init__()
    
    # Decoder
        self.dec1 = nn.Linear(latent_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        x = F.leaky_relu(self.dec1(x))
        x = F.softplus(self.dec2(x)) 
        return x

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim)
        
    def forward(self, x):
        z = self.encoder(x)[2]
        return self.decoder(z)
    


def train_vae(model, dataloader, num_epochs, learning_rate, device="cuda"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = LRScheduler(optimizer)
    model.train()
    
    # Store loss values for plotting
    epoch_losses = []
    epoch_kl_losses = []
    epoch_mse_losses = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_kl = 0.0
        total_mse = 0.0

        for i, data in enumerate(dataloader):
            data = data[0].to(device)
            data = data.to(device)
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            data_predict = model(data)

            kl_loss = model.encoder.kl
            criterion = nn.MSELoss(reduction='sum')
            mse_loss = criterion(data_predict, data)

            loss = mse_loss + kl_loss

            running_loss += loss.item()
            total_kl += kl_loss.item()
            total_mse += mse_loss.item()

            loss.backward()
            optimizer.step()

        train_loss = running_loss/len(dataloader.dataset)
        kl_loss = total_kl / len(dataloader.dataset)
        mse_loss = total_mse/ len(dataloader.dataset)
        
        epoch_losses.append(train_loss)
        epoch_kl_losses.append(kl_loss)
        epoch_mse_losses.append(mse_loss)

        print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Recon Loss: {mse_loss:.4f}, KL Loss: {kl_loss:.4f}")

    return model, epoch_losses, epoch_kl_losses, epoch_mse_losses



class LRScheduler ():

    def __init__ (self, optimizer, patience = 10,
    min_lr = 1e-6, factor = 0.5):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
 
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode = 'min',
            patience = self.patience,
            factor = self.factor,
            min_lr = self.min_lr,
            verbose = True
            )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)
        

def fit_vae(model, dataloader, optimizer, beta):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.train()
    running_loss = 0.0
    total_kl = 0.0
    total_mse = 0.0

    for i, data in enumerate(dataloader):
        data = data[0].to(device)
        data = data.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        data_predict = model(data)

        kl_loss = model.encoder.kl
        criterion = nn.MSELoss(reduction='mean')
        mse_loss = criterion(data_predict, data)

        loss = mse_loss + beta * kl_loss

        running_loss += loss.item()
        total_kl += kl_loss.item()
        total_mse += mse_loss.item()

        loss.backward()
        optimizer.step()

    train_loss = running_loss/len(dataloader.dataset)
    train_kl_loss = total_kl / len(dataloader.dataset)
    train_mse_loss = total_mse/ len(dataloader.dataset)

    return train_loss, train_kl_loss, train_mse_loss

def validate_vae(model, dataloader, optimizer, beta):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    running_loss = 0.0
    total_kl = 0.0
    total_mse = 0.0
    for i, data in enumerate(dataloader):
        data, _ = data
        data = data.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        data_predict = model(data)
        
        kl_loss = model.encoder.kl
        criterion = nn.MSELoss(reduction='mean')
        mse_loss = criterion(data_predict, data)
        loss = mse_loss + beta * kl_loss
        
        running_loss += loss.item()
        total_kl += kl_loss.item()
        total_mse += mse_loss.item()

    val_loss = running_loss/len(dataloader.dataset)
    val_kl_loss = total_kl / len(dataloader.dataset)
    val_mse_loss = total_mse/ len(dataloader.dataset)
    
    
    return val_loss, val_kl_loss, val_mse_loss


def train_and_validate_vae(model, train_loader, val_loader, beta, num_epochs, lr, model_type, path_save_loss, path_save_plot, dataset_name, device="cuda"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    early_stopping = EarlyStopping()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = LRScheduler(optimizer)
    
    train_loss = []
    val_loss = []
    mse_train_loss = []
    mse_val_loss = []
    kl_train_loss = []
    kl_val_loss = [] 
    
    start_time = time.time()
    for epoch in range(num_epochs):
        epochStartTime = time.time()
        print(f"Epoch {epoch+1} of {num_epochs}")
        
        train_epoch_loss, train_epoch_kl_loss, train_epoch_mse_loss = fit_vae(model, train_loader, optimizer, beta)
        val_epoch_loss, val_epoch_kl_loss, val_epoch_mse_loss = validate_vae(model, val_loader, optimizer, beta)
        
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        
        mse_train_loss.append(train_epoch_mse_loss)
        mse_val_loss.append(val_epoch_mse_loss)
        
        kl_train_loss.append(train_epoch_kl_loss)
        kl_val_loss.append(val_epoch_kl_loss)
        
        lr_scheduler(val_epoch_loss)
        
        early_stopping(val_epoch_loss)
        if early_stopping.early_stop:
            break
            
        epochEndTime = time.time()
        print(f"Train Loss: {train_epoch_loss:.4f}, Train MSE Loss: {train_epoch_mse_loss:.4f}, Train KL Loss {train_epoch_kl_loss:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f}, Val MSE Loss: {val_epoch_mse_loss:.4f}, Val KL Loss: {val_epoch_kl_loss:.4f}")
        print(f"Epoch time: {epochEndTime-epochStartTime:.2f} sec")
        
    run_time = time.time() - start_time
    print(f"Run time: {(run_time/60):.2f} mins")
    #torch.save(model.state_dict(), args.outputpath + "/autoencoder_models/" + model + ".pth")
    #torch.save(model.encoder.state_dict(), args.outputpath + "/encoder_models/" + model + "_encoder.pth")   
    
    train_test_dict = {}
    train_test_dict['train_loss'] = train_loss
    train_test_dict['val_loss'] = val_loss
    
    detailed_loss = {}
    detailed_loss['recon_train'] = mse_train_loss
    detailed_loss['recon_val'] = mse_val_loss
    detailed_loss['kl_train'] = kl_train_loss
    detailed_loss['kl_val'] = kl_val_loss
    
    df_loss = pd.DataFrame(train_test_dict)
    df_loss_detailed = pd.DataFrame(detailed_loss) 
    
    df_loss.to_csv(path_save_loss+ "/"  + model_type + "/" + f"/training_test_loss_{dataset_name}" + '.csv')
    df_loss_detailed.to_csv(path_save_loss + "/" + model_type + "/"  + f"/recon_kl_loss_{dataset_name}" + '.csv')
    
    # Loss plot
    fig = plt.figure()
    ax = plt.axes()
    #x_axis = np.linspace(1, epochs, num = epochs)
    x_axis = np.linspace(1, epoch+1, num=epoch+1)
    ax.set(xlabel='Epoch', ylabel="Loss")
    plt.xticks(np.arange(1, epoch+1, step=5))
    plt.plot(x_axis, train_loss, color='cyan', label="Train loss")
    plt.plot(x_axis, val_loss, color='orange', label="Validate loss")
    plt.legend()
    plt.savefig(path_save_plot + "/" + model_type + "/" + f"{dataset_name}_loss_to_epoch" + '.png')
    
    return model, train_test_dict, detailed_loss, df_loss, df_loss_detailed, epoch


class EarlyStopping():

    def __init__(self, patience = 10, min_delta = 0):
        self.patience = patience
        self.min_delta = min_delta 
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self,  val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss -  val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        
def extract_latent_dim(X_test, test_dataloader, model, latent_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('X_test Size:' + str(X_test.shape))
    sample_val, _ = X_test.shape
    latent_results = []
    for i in range(sample_val):
        data_tem = test_dataloader.dataset[i][0]
        output = model.encoder(data_tem.reshape(1, -1).to(device))
        output = output[2]
        latent_results.append(output.cpu().detach().numpy())
    latent_results = np.array(latent_results)
    latent_results = latent_results.reshape(sample_val, latent_dim)
    print('Latent Result Size: ' + str(latent_results.shape))
    
    return latent_results

def build_adata_latent(y, latent_results, column_name):
    
    adata_latent = ad.AnnData(latent_results)
    adata_latent.obs["true_labels"] = y[column_name].values
    
    return adata_latent


def get_data_reconstructed(X_test, test_dataloader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('X_test Size:' + str(X_test.shape))
    sample_val, _ = X_test.shape
    X_test_reconstructed = []
    for i in range(sample_val):
        data_tem = test_dataloader.dataset[i][0]
        z = model.encoder(data_tem.reshape(1, -1).to(device))[-1]  # Extract last element (z)
        output = model.decoder(z)
        X_test_reconstructed.append(output.cpu().detach().numpy())
    X_test_reconstructed = np.array(X_test_reconstructed).reshape(sample_val, X_test.shape[1])
    
    return X_test_reconstructed



def compute_mse_per_cell_type(X_test, X_test_reconstructed, y_test):
    """
    Computes MSE for each row in X_test for each cell type and stores results in a dictionary.

    Parameters:
    - X_test (pd.DataFrame): Original test dataset
    - X_test_reconstructed (np.array or pd.DataFrame): Reconstructed test dataset
    - y_test (pd.DataFrame): DataFrame containing 'Cell Type' column

    Returns:
    - mse_dict (dict): Dictionary where keys are cell types and values are lists of row-wise MSE values.
    """
    mse_dict = {}
    
    # Ensure X_test_reconstructed is a DataFrame with the same index as X_test
    X_test_reconstructed_df = pd.DataFrame(X_test_reconstructed, index=X_test.index)
    
    # Get unique cell types
    cell_types = y_test['Cell Type'].unique()
    
    for cell_type in cell_types:
        # Get indices corresponding to the cell type
        indices = y_test.loc[y_test['Cell Type'] == cell_type].index
        
        # Filter the original and reconstructed datasets
        X_test_filtered = X_test.loc[indices]
        X_test_reconstructed_filtered = X_test_reconstructed_df.loc[indices]

        # Compute row-wise MSE and store in the dictionary
        mse_values = [
            torch.nn.functional.mse_loss(
                torch.tensor(X_test_filtered.iloc[i].values, dtype=torch.float32),
                torch.tensor(X_test_reconstructed_filtered.iloc[i].values, dtype=torch.float32),
                reduction='mean'
            ).item()
            for i in range(len(X_test_filtered))
        ]
        
        mse_dict[cell_type] = mse_values

    return mse_dict








import plotly.express as px
def plot_mse_distribution(mse_dict):
    """
    Plots the distribution of MSE values for each cell type using Plotly.

    Parameters:
    - mse_dict (dict): Dictionary where keys are cell types and values are lists of MSE values.
    """
    # Convert dictionary to DataFrame
    mse_data = []
    for cell_type, mse_values in mse_dict.items():
        for mse in mse_values:
            mse_data.append({'Cell Type': cell_type, 'MSE': mse})
    
    mse_df = pd.DataFrame(mse_data)

    # Create box plot (you can also use violin plot by changing 'box' to 'violin')
    fig = px.box(mse_df, x="Cell Type", y="MSE", color="Cell Type",  category_orders={'Cell Type': list_cell_types}, title=f"MSE Distribution - {dataset_name} Dataset ")
    
    fig.write_image(f"/home/BS94_SUR/phD/review/plots/mse plots/mse_distribution_{dataset_name}.png") 
    
    # Show plot
    fig.show()
    
import seaborn as sns
import matplotlib.pyplot as plt

def plot_mse_kde_seaborn(mse_dict, bw_adjust=0.5):
    plt.figure(figsize=(16, 6))  # Set figure size

    for cell_type, mse_values in mse_dict.items():
        sns.kdeplot(mse_values, bw_adjust=bw_adjust, label=cell_type, linewidth=2)

    # Formatting
    plt.title(f"MSE KDE Distribution by Cell Type - {dataset_name} Dataset")
    plt.xlabel("MSE")
    plt.ylabel("Density")
    plt.legend(title="Cell Type")
    plt.grid(True)
    
    plt.savefig(f"/home/BS94_SUR/phD/review/plots/mse plots/mse_kde_distribution_{dataset_name}.png", dpi=300, bbox_inches="tight") 

    # Show the plot
    plt.show()

