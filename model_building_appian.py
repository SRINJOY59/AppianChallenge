!pip install catboost
!pip install keras

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.utils import to_categorical
import numpy as np
import pandas as pd

! pip install huggingface-hub

! pip install -q xformers

from huggingface_hub import login

login("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")

file_path = r'/content/processed_final_data.csv'
data = pd.read_csv(file_path)
data = data[['text', 'label']]

label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True)
data['text'] = data['text'].astype(str)
X = model.encode(data['text'].tolist())

y = data['label']
y_categorical = to_categorical(y)  # For LSTM and GRU

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict(X_test)

catboost_model = CatBoostClassifier(silent=True)
catboost_model.fit(X_train, y_train)
catboost_preds = catboost_model.predict(X_test)

stacked_features_train = np.column_stack((
    xgb_model.predict(X_train),
    lgb_model.predict(X_train),
    catboost_model.predict(X_train)
))

stacked_features_test = np.column_stack((
    xgb_preds,
    lgb_preds,
    catboost_preds
))

meta_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
meta_model.fit(stacked_features_train, y_train)
y_stack = meta_model.predict(stacked_features_test)

final_acc = accuracy_score(y_test, y_stack)
print(f"Final Ensemble Accuracy: {final_acc:.4f}")

def print_metrics(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"\n{model_name} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    return accuracy, precision, recall, f1

print_metrics(y_test, xgb_preds, "XGBoost")
print_metrics(y_test, lgb_preds, "LightGBM")
print_metrics(y_test, catboost_preds, "CatBoost")

print_metrics(y_test, y_stack, "Meta-Classifier")
