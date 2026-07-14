import pandas as pd
import requests
import json
import re

url = "http://localhost:5000/predict"
data_path = "X_train.csv" 

try:
    print("Reading base row from X_train.csv...")
    df = pd.read_csv(data_path, nrows=1)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    gene_features = df.iloc[0].to_dict()

    print("Starting automatic gene discovery loop with backend server...")
    
    # Run a loop to collect missing genes one batch at a time
    for attempt in range(1, 50):
        payload = {"gene_expression": gene_features}
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            print(f"\n🎉 Success after loop {attempt}! Prediction received:")
            print(json.dumps(response.json(), indent=2))
            break
            
        elif response.status_code == 400 and "Missing" in response.text:
            error_msg = response.json().get("error", "")
            # Find the gene array inside the error message string
            match = re.search(r"First few missing:\s*\[(.*?)\]", error_msg)
            
            if match:
                # Clean up and extract the strings
                missing_str = match.group(1)
                new_genes = [g.strip(" '\"") for g in missing_str.split(",") if g.strip()]
                
                print(f"Loop {attempt}: Found missing genes: {new_genes}")
                
                # Add the newly discovered genes to our dictionary
                for gene in new_genes:
                    gene_features[gene] = 0.0
            else:
                print(f"Could not parse error text anymore: {error_msg}")
                break
        else:
            print(f"\nFailed with unexpected status code: {response.status_code}")
            print(response.text)
            break
            
except FileNotFoundError:
    print(f"\nCould not find '{data_path}' in this folder.")
except Exception as e:
    print(f"An error occurred: {e}")