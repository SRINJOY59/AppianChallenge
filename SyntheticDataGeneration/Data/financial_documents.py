from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Load both train and test splits
ds = load_dataset("dodogeny/receipts-dataset-v1")
df_train = pd.DataFrame(ds['train'])
df_test = pd.DataFrame(ds['test'])

# Combine train and test
df = pd.concat([df_train, df_test], ignore_index=True)

# Shuffle and split the dataset into 80% train and 20% test
train_split, test_split = train_test_split(df, test_size=0.2, random_state=42)

# Save the splits to CSV
train_split.to_csv('train_split_receipts.csv', index=False)
test_split.to_csv('test_split_receipts.csv', index=False)

print("Train Split Shape:", train_split.shape)
print("Test Split Shape:", test_split.shape)
print("CSV files created: 'train_split_receipts.csv' and 'test_split_receipts.csv'")
