import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from xgboost import XGBClassifier


def extract_x_y_from_adata(adata, labels):
    X = pd.DataFrame(adata.X, index=adata.obs.index)
    y = adata.obs[labels]
    return X, y

def encode_y(y):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    return y, label_mapping

def transpose_label_mapping(label_mapping):
    return {v: k for k, v in label_mapping.items()}

def filter_classes(X, y):
    class_counts = Counter(y)

    # Filter out classes with only 1 sample
    valid_classes = [cls for cls, count in class_counts.items() if count > 1]
    mask = np.isin(y, valid_classes)

    X= X[mask]
    y = y[mask]
    
    return X, y

def apply_classifier(X, y, classifier_name):
    if classifier_name == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return y_pred

def compute_classification_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    
    return accuracy, precision, recall, f1

def compute_roc_auc(y_test, y_prob):
    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    return roc_auc

def map_label_mapping(y, label_mapping):
    y = pd.Series(y).map(label_mapping).values
    return y

def stratify_small_classes(X, y, test_size):
    y_unique, y_counts = np.unique(y, return_counts=True)

    # Compute a per-class minimum test size (at least 1 sample)
    min_test_samples = np.maximum(1, (y_counts * test_size).astype(int))

    # Adjust labels to ensure at least one sample per class in test
    test_indices = []
    train_indices = np.arange(len(y))

    for cls, min_samples in zip(y_unique, min_test_samples):
        cls_indices = np.where(y == cls)[0]
        if len(cls_indices) > 1:  # Only split if more than one sample exists
            np.random.seed(42)
            test_cls_indices = np.random.choice(cls_indices, min_samples, replace=False)
            test_indices.extend(test_cls_indices)

    # Remove selected test indices from train set
    train_indices = np.setdiff1d(train_indices, test_indices)

    # Extract train/test splits
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = pd.DataFrame(y).iloc[train_indices], pd.DataFrame(y).iloc[test_indices]
    
    return X_train, X_test, y_train, y_test

def compute_classification_adata(adata, labels_column, models, test_size):
    
    X, y = extract_x_y_from_adata(adata, labels_column)
    y, label_mapping = encode_y(y)
    
    #X, y= filter_classes(X, y)

    X_train, X_test, y_train, y_test = stratify_small_classes(X, y, test_size)
    
    pred_dict = {}
    for model_name in models:
        pred_dict[model_name] = {}
        if model_name == 'Random Forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred_rf = model.predict(X_test)
            y_prob_rf = model.predict_proba(X_test)
            pred_dict[model_name]['pred'] = y_pred_rf
            accuracy, precision, recall, f1 = compute_classification_metrics(y_test, y_pred_rf)
            roc_auc = compute_roc_auc(y_test, y_prob_rf)
            
            pred_dict[model_name]['accuracy'] = accuracy
            pred_dict[model_name]['precision'] = precision
            pred_dict[model_name]['recall'] = recall
            pred_dict[model_name]['f1'] = f1
            pred_dict[model_name]['roc_auc'] = roc_auc
        if model_name == 'XGBoost':
            model = XGBClassifier(objective="multi:softprob", num_class=int(adata.obs[labels_column].unique().size), n_estimators=100, max_depth=3)
            model.fit(X_train, y_train)
            y_prob_xgb = model.predict_proba(X_test)
            y_pred_xgb = y_prob_xgb.argmax(axis=1)
            accuracy, precision, recall, f1 = compute_classification_metrics(y_test, y_pred_xgb)
            roc_auc = compute_roc_auc(y_test, y_prob_xgb)
            
            pred_dict[model_name]['pred'] = y_pred_xgb
            pred_dict[model_name]['accuracy'] = accuracy
            pred_dict[model_name]['precision'] = precision
            pred_dict[model_name]['recall'] = recall
            pred_dict[model_name]['f1'] = f1
            pred_dict[model_name]['roc_auc'] = roc_auc
            
    
    return pred_dict

