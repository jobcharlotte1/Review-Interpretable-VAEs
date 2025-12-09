import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from anndata import AnnData


class VAE_classification_evaluation:
    def __init__(self,
                 X_embedding: np.array,
                 adata: AnnData,
                 name_labels: str,
                 classifier_name: str) -> None:
        super(VAE_classification_evaluation, self).__init__()

        self.X_embedding = X_embedding
        self.adata = adata
        self.name_labels = name_labels
        self.classifier_name = classifier_name

    def extract_x_y_from_adata(self, adata: AnnData, name_labels: str):
        X = pd.DataFrame(adata.X, index=adata.obs.index)
        y = adata.obs[name_labels]
        return X, y

    def apply_classifier(self, X_embedding, y, classifier_name):
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

    def encode_y(self, y):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        return y_encoded, label_mapping

    def compute_classification_metrics(self, y, y_pred):
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average="macro")
        recall = recall_score(y, y_pred, average="macro")
        f1 = f1_score(y, y_pred, average="macro")

        return accuracy, precision, recall, f1

    def compute_roc_auc(self, y, y_proba):
        roc_auc = roc_auc_score(y, y_proba, multi_class="ovr")
        return roc_auc
