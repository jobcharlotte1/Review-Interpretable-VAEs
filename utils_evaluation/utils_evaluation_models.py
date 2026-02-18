import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from anndata import AnnData
from sklearn.metrics import mean_squared_error
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import umap
import umap.plot
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def preprocess_adata(adata, n_top_genes=5000):
    """ Simple (default) sc preprocessing function before autoencoders """
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    return adata


def compute_pearson_score(adata, X_reconstructed):
    X = adata.X
    X = X.toarray() if hasattr(X, "toarray") else X
    correlations = [
        pearsonr(X[i, :], X_reconstructed[i, :])[0]
        for i in range(X.shape[0])
    ]
    return np.mean(correlations)

def compute_mse_score(adata, X_reconstructed):
    X = adata.X
    X = X.toarray() if hasattr(X, "toarray") else X
    return mean_squared_error(X, X_reconstructed)

def build_adata_latent(X_embedding, adata, name_labels):
    adata_latent = AnnData(X_embedding)
    adata_latent.obs[name_labels] = adata.obs[name_labels].values
    return adata_latent

def apply_clustering_algo(adata, name_labels, clustering_method, val_resolution):
    true_labels = adata.obs[name_labels].values
    num_clusters = len(np.unique(true_labels))
    if sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
    sc.pp.neighbors(adata, n_neighbors=num_clusters, use_rep="X")

    if clustering_method == 'Louvain':
        sc.tl.louvain(adata, resolution=val_resolution)
        clusters = adata.obs["louvain"].astype(int).values
    elif clustering_method == 'Leiden':
        sc.tl.leiden(adata, resolution=val_resolution)
        clusters = adata.obs["leiden"].astype(int).values
    else:
        raise ValueError(f"Unsupported clustering method: {clustering_method}")

    return clusters, true_labels

def apply_clustering_metrics(true_labels, clusters):
    ari = adjusted_rand_score(true_labels, clusters)
    nmi = normalized_mutual_info_score(true_labels, clusters)
    return ari, nmi

def plot_umap_orig_and_clusters(embedding, true_labels, dataset_name, model_type, clusters, ari, nmi, split, trial, clustering_method, path_save_fig):
    mapper = umap.UMAP(random_state=42).fit(np.nan_to_num(np.array(embedding)))

    umap.plot.points(mapper, labels=true_labels, color_key_cmap='Paired', show_legend=True)
    title_orig = f'{model_type} - {split} - Trial {trial} (true labels)'
    plt.title(title_orig)
    if path_save_fig is not None:
        plt.savefig(path_save_fig + f'/{model_type}_{dataset_name}_{split}_{trial}_umap_original.png')
    plt.show()

    umap.plot.points(mapper, labels=clusters, color_key_cmap='Paired', show_legend=True)
    title_clusters = f'{model_type} ({clustering_method}) - {split} - Trial {trial}\nARI: {ari}, NMI: {nmi}'
    plt.title(title_clusters)
    if path_save_fig is not None: 
        plt.savefig(path_save_fig + f'/{model_type}_{dataset_name}_{split}_{trial}_umap_clusters.png')
    plt.show()


def extract_x_y_from_adata(adata: AnnData, name_labels: str):
    X = pd.DataFrame(adata.X, index=adata.obs.index)
    y = adata.obs[name_labels]
    return X, y

def extract_x_y_from_adata2(adata: AnnData, name_labels: str):
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    y = adata.obs[name_labels].values
    return X, y

def apply_classifier(X_embedding, y, classifier_name):
    if classifier_name == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif classifier_name == 'xgboost':
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")

    model.fit(X_embedding, y)
    y_pred = model.predict(X_embedding)
    y_proba = model.predict_proba(X_embedding)

    return y_pred, y_proba

def apply_classifier2(X_train, y_train, X_val, y_val, X_test, y_test, classifier_name):
    
    if classifier_name == 'random_forest':
        model = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
            
    elif classifier_name == 'xgboost':
        model = XGBClassifier(
            n_estimators=20,        
            max_depth=5, 
            max_leaves = 10,
            learning_rate=0.05,    
            subsample=0.8,        
            colsample_bytree=0.8,  
            gamma=0,              
            reg_alpha=0.1,    
            reg_lambda=1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            #early_stopping_rounds=20,
            verbose=False
        )
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")

    y_pred_train = model.predict(X_train)
    y_proba_train = model.predict_proba(X_train)
    
    y_pred_val = model.predict(X_val)
    y_proba_val = model.predict_proba(X_val)
    
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)

    return y_pred_train, y_proba_train, y_pred_val, y_proba_val, y_pred_test, y_proba_test

def encode_y(y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    return y_encoded, label_mapping

def compute_classification_metrics(y, y_pred):
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average="macro")
    recall = recall_score(y, y_pred, average="macro")
    f1 = f1_score(y, y_pred, average="macro")

    return accuracy, precision, recall, f1

def compute_roc_auc(y, y_proba):
    roc_auc = roc_auc_score(y, y_proba, multi_class="ovr")
    return roc_auc

def apply_classification_algo(adata, column_labels_name, algo_name):
    X, y = extract_x_y_from_adata(adata, column_labels_name)
    y_encoded, label_mapping = encode_y(y)
    y_pred, y_proba = apply_classifier(X, y_encoded, algo_name)
    accuracy, precision, recall, f1 = compute_classification_metrics(y_encoded, y_pred)
    roc_auc = compute_roc_auc(y_encoded, y_proba)
    
    return accuracy, precision, recall, f1, roc_auc


def apply_classification_algo2(adata_train, adata_val, adata_test, column_labels_name, classifier_name, test_size=0.1):
    X_train, y_train = extract_x_y_from_adata(adata_train, column_labels_name)
    y_encoded_train, label_mapping_train = encode_y(y_train)
    X_val, y_val = extract_x_y_from_adata(adata_val, column_labels_name)
    y_encoded_val, label_mapping_train = encode_y(y_val)
    X_test, y_test = extract_x_y_from_adata(adata_test, column_labels_name)
    y_encoded_test, label_mapping_test = encode_y(y_test)
    
    y_pred_train, y_proba_train, y_pred_val, y_proba_val, y_pred_test, y_proba_test = apply_classifier2(X_train, y_encoded_train, X_val, y_encoded_val, X_test, y_encoded_test, classifier_name)
    
    accuracy_train, precision_train, recall_train, f1_train = compute_classification_metrics(y_encoded_train, y_pred_train)
    roc_auc_train = compute_roc_auc(y_encoded_train, y_proba_train)
    
    accuracy_val, precision_val, recall_val, f1_val = compute_classification_metrics(y_encoded_val, y_pred_val)
    roc_auc_val = compute_roc_auc(y_encoded_val, y_proba_val)
    
    accuracy_test, precision_test, recall_test, f1_test = compute_classification_metrics(y_encoded_test, y_pred_test)
    roc_auc_test = compute_roc_auc(y_encoded_test, y_proba_test)
    
    return accuracy_train, precision_train, recall_train, f1_train, roc_auc_train, accuracy_val, precision_val, recall_val, f1_val, roc_auc_val, accuracy_test, precision_test, recall_test, f1_test, roc_auc_test

