import os
import json
import pandas as pd

data_folder = "Data"

text_data = []
labels = []

for file_name in os.listdir(data_folder):
    if file_name.endswith(".json"):
        file_path = os.path.join(data_folder, file_name)
        
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)  
            
            if isinstance(data, list):
                for document in data:  
                    content_list = document.get("content", [])  
                    
                    for elem in content_list:
                        combined_text = elem.get("text", "")
                        text_data.append(combined_text)
                        if "passport" in file_name.lower():
                            labels.append("passport")
                        elif "voter" in file_name.lower():
                            labels.append("votar card")
                        else:
                            labels.append("aadhar")
                    
            

df = pd.DataFrame({
    "text": text_data,
    "identity": labels
})

df["label"] = "identity"

df.to_csv("Main_Data/identity_data_combined.csv", index=False)
