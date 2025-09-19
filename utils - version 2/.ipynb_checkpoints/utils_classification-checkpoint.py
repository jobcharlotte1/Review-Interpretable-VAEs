from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

def apply_classifier(X, y, classifier_name):
    if classifier_name == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif classifier_name == 'xgboost':
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        
    model.fit(X, y)
    y_pred = model.predict(X)
    
    return y_pred

def encode_y(y):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    return y, label_mapping


def compute_classification_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    
    return accuracy, precision, recall, f1

def compute_roc_auc(y_test, y_prob):
    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    return roc_auc