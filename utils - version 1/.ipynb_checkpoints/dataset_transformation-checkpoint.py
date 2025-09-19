import pandas as pd 
import numpy as np
import os
import optuna


def apply_melt_function(df, columns_to_drop, columns_to_keep, var_name, value_name):
    df_melted = pd.melt(df.drop(columns_to_drop, axis=1), id_vars=columns_to_keep, var_name=var_name, value_name=value_name)
    return df_melted 



def save_results_to_dataframe(model_name, dataset_name, batch_size, learning_rate, latent_dim, 
                              num_epochs, ari_raw_louvain, ari_raw_leiden, nmi_raw_louvain, nmi_raw_leiden, 
                              ari_latent_louvain, ari_latent_leiden, nmi_latent_louvain, nmi_latent_leiden, 
                              file_path):
    # Check if the file already exists
    if os.path.exists(file_path):
        # If the file exists, load the existing dataframe
        df = pd.read_csv(file_path)
    else:
        # If the file doesn't exist, create a new dataframe with the correct columns
        columns = ['model_name', 'dataset_name', 'batch_size', 'learning_rate', 'latent_dim', 'num_epochs',
                   'ari_raw_louvain', 'ari_raw_leiden', 'nmi_raw_louvain', 'nmi_raw_leiden', 
                   'ari_latent_louvain', 'ari_latent_leiden', 'nmi_latent_louvain', 'nmi_latent_leiden']
        df = pd.DataFrame(columns=columns)
    
    # Create a dictionary with the values to add as a new row
    new_row = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'latent_dim': latent_dim,
        'num_epochs': num_epochs,
        'ari_raw_louvain': ari_raw_louvain,
        'ari_raw_leiden': ari_raw_leiden,
        'nmi_raw_louvain': nmi_raw_louvain,
        'nmi_raw_leiden': nmi_raw_leiden,
        'ari_latent_louvain': ari_latent_louvain,
        'ari_latent_leiden': ari_latent_leiden,
        'nmi_latent_louvain': nmi_latent_louvain,
        'nmi_latent_leiden': nmi_latent_leiden
    }
    
    new_row_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_row_df], ignore_index=True)
    
    # Save the updated dataframe to the CSV file
    df.to_csv(file_path, index=False)
    
    return df


def save_hyperparmeters_optimization_results(dataset_name, study, file_path):
        if os.path.exists(file_path):
            df_hyperparameters = pd.read_csv(file_path)

        else:
            df_hyperparameters = pd.DataFrame(
                columns=['Dataset Name', 'Trial', 'Batch Size', 'Learning Rate', 'Num Epochs', 'Final Loss']
            )

        df_hyperparameters_new = pd.DataFrame([
            {
                'Dataset Name': dataset_name,
                'Trial': trial.number,
                'Batch Size': trial.params['batch_size'],
                'Learning Rate': trial.params['lr'],
                'Num Epochs': trial.params['num_epochs'],
                'Final Loss': trial.value
            }
            for trial in study.trials
        ])

        df_hyperparameters = pd.concat([df_hyperparameters, df_hyperparameters_new], ignore_index=True)

        df_hyperparameters.to_csv(file_path, index=False)

        return df_hyperparameters
    
    
def save_classification_results(dataset_name, pred_dict, key, file_path):
        if os.path.exists(file_path):
            df_classification_all_datasets = pd.read_csv(file_path)

        else:
            df_classification_all_datasets = pd.DataFrame(columns=['Dataset Name', 'Data Type', 'Metrics', 'Models', 'Metric'])
            
        dict_classif = {}    
        dict_classif[dataset_name] = {'raw': {}, 'latent': {}}
        dict_classif[dataset_name][key] = pred_dict
        df_classif = pd.DataFrame(dict_classif[dataset_name][key]).reset_index(names=['Metrics'])
        df_classif['Dataset Name'] = list(dict_classif.keys())[0]
        df_classif['Data Type'] = key
        df_classif_melted = pd.melt(df_classif, id_vars=['Dataset Name', 'Data Type', 'Metrics'], var_name='Models', value_name='Metric')


        

        df_classification_all_datasets = pd.concat([df_classification_all_datasets, df_classif_melted], ignore_index=True)

        df_classification_all_datasets.to_csv(file_path, index=False)

        return df_classification_all_datasets
    
    
def save_best_optimization_results(dataset_name, nb_hidden, hidden_dim, latent_dim, batch_size, lr, num_epochs, best_val_loss, file_path):
        if os.path.exists(file_path):
            df_hyperparameters = pd.read_csv(file_path)

        else:
            df_hyperparameters = pd.DataFrame(
                columns=['Dataset Name', 'Nb Hidden Layers', 'Hidden Dim', 'Latent Dim', 'Batch Size', 'Learning Rate', 'Num Epochs', 'Best Val Loss']
            )

        df_hyperparameters_new = pd.DataFrame([
            {
                'Dataset Name': dataset_name,
                'Nb Hidden Layers': nb_hidden,
                'Hidden Dim': hidden_dim,
                'Latent Dim': latent_dim,
                'Batch Size': batch_size,
                'Learning Rate': lr,
                'Num Epochs': num_epochs,
                'Best Val Loss': best_val_loss
            }
        ])

        df_best_optimization_results = pd.concat([df_hyperparameters, df_hyperparameters_new], ignore_index=True)

        df_best_optimization_results.to_csv(file_path, index=False)

        return df_best_optimization_results
    

def save_mse_results(dataset_name, nb_hidden, hidden_dim, latent_dim, batch_size, lr, num_epochs, mean_l2_error, file_path):
        if os.path.exists(file_path):
            df_mse= pd.read_csv(file_path)

        else:
            df_mse = pd.DataFrame(
                columns=['Dataset Name', 'Nb Hidden Layers', 'Hidden Dim', 'Latent Dim', 'Batch Size', 'Learning Rate', 'Num Epochs', 'Mean Reconstruction Error (MSE)']
            )

        df_mse_new = pd.DataFrame([
            {
                'Dataset Name': dataset_name,
                'Nb Hidden Layers': nb_hidden,
                'Hidden Dim': hidden_dim,
                'Latent Dim': latent_dim,
                'Batch Size': batch_size,
                'Learning Rate': lr,
                'Num Epochs': num_epochs,
                'Mean Reconstruction Error (MSE)': mean_l2_error
            }
        ])

        df_mse_results = pd.concat([df_mse, df_mse_new], ignore_index=True)

        df_mse_results.to_csv(file_path, index=False)

        return df_mse_results