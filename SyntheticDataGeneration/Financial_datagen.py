from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

dataset_name = "yjmsvma/document_classification_dataset_v1"
ds_train = load_dataset(dataset_name, split="train")

df_train = pd.DataFrame(ds_train)

df_train.to_csv(r"Main_Data/Financial_data.csv", index=False)

