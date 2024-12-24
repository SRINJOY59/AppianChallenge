import os
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

current_dir = os.path.dirname(__file__)

models_dir = os.path.join(current_dir, '..', 'Trained_Base_Models')

def load_models_and_predict(input_text):

    xgb_model = joblib.load(os.path.join(models_dir, 'xgboost_model.joblib'))
    lgb_model = joblib.load(os.path.join(models_dir, 'lightgbm_model.joblib'))
    catboost_model = joblib.load(os.path.join(models_dir, 'catboost_model.joblib'))
    meta_model = joblib.load(os.path.join(models_dir, 'meta_classifier_model.joblib'))
    label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.joblib'))

    sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', trust_remote_code=True)

    encoded_input = sentence_transformer.encode([input_text])

    xgb_pred = xgb_model.predict(encoded_input)
    lgb_pred = lgb_model.predict(encoded_input)
    catboost_pred = catboost_model.predict(encoded_input)

    stacked_features = np.column_stack([xgb_pred, lgb_pred, catboost_pred])
    meta_pred = meta_model.predict(stacked_features)

    proper_label = label_encoder.inverse_transform(meta_pred)
    

    return proper_label[0]


sample_receipt = """
----------------------------------------
              SHOP NAME
           Address: 123 Main St
           Phone: (123) 456-7890
----------------------------------------
Date: 2023-10-01
Time: 14:30
----------------------------------------
Item                Qty     Price
----------------------------------------
Item 1              2       $10.00
Item 2              1       $5.50
Item 3              3       $7.25
----------------------------------------
Subtotal:                     $32.75
Tax (5%):                    $1.64
----------------------------------------
Total:                      $34.39
----------------------------------------
Thank you for shopping with us!
----------------------------------------
"""


input_text = sample_receipt



final_prediction = load_models_and_predict(input_text)
print(f"Final Prediction: {final_prediction}")


