from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

dataset_name = "D1st3f/Receipts"
ds_train = load_dataset(dataset_name, split="train")
ds_test = load_dataset(dataset_name, split="test")
ds_val = load_dataset(dataset_name, split="validation")

df_train = pd.DataFrame(ds_train)
df_test = pd.DataFrame(ds_test)
df_val = pd.DataFrame(ds_val)

df_all = pd.concat([df_train, df_test, df_val], ignore_index=True)

df_all["label"] = "receipt"

output_file = r"Main_Data/receipt_combined_data.csv"
df_all.to_csv(output_file, index=False)

print(f"Concatenated financial data saved to {output_file}")

