import pandas as pd
import numpy as np

df = pd.read_csv(r"Data\Financial_data.csv")

df_main = df[["category", "text"]]

df_main["label"] = "finance"

df_main = df_main.sample(frac=1).reset_index(drop=True)  

df_main.to_csv("Data/Financial_data_combined.csv", index=False)
