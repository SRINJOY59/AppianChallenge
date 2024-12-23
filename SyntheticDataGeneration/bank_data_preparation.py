import os
import pandas as pd

data_dir = 'Data/'

def clean_data(df):
    df = df[~df['information'].str.startswith('#', na=False)]
    
    df['information'] = df['information'].str.strip()  
    df.dropna(subset=['information'], inplace=True) 
    
    return df

all_cleaned_data = []


for file_name in os.listdir(data_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(data_dir, file_name)
        
        try:
            df = pd.read_csv(file_path)
            
            cleaned_df = clean_data(df)
            
            all_cleaned_data.append(cleaned_df)
        except Exception as e:
            print(f"Error reading or cleaning {file_name}: {e}")

final_cleaned_df = pd.concat(all_cleaned_data, ignore_index=True)

final_cleaned_df = final_cleaned_df.sample(frac=1).reset_index(drop=True)  




final_cleaned_df['label'] = 'bank'

output_file = 'Main_Data/bank_data_combined.csv'
final_cleaned_df.to_csv(output_file, index=False)

print(f"Cleaned data saved to {output_file}")

