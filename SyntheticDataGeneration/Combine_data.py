import os
import pandas as pd
from sklearn.utils import shuffle

main_data_folder = "Main_Data"

dataframes = []

for file_name in os.listdir(main_data_folder):
    if file_name.endswith(".csv"):
        file_path = os.path.join(main_data_folder, file_name)
        
        df = pd.read_csv(file_path)
        
        if "text" in df.columns and "label" in df.columns:
            dataframes.append(df[["text", "label"]])  

merged_df = pd.concat(dataframes, ignore_index=True)

shuffled_df = shuffle(merged_df, random_state=42).reset_index(drop=True)

output_path = os.path.join(main_data_folder, "Final_data.csv")
shuffled_df.to_csv(output_path, index=False)

print(f"Data merged and shuffled. Saved as {output_path}")
