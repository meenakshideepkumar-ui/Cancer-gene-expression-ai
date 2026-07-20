import os
import joblib

print("Reading exact gene layout from models/all_input_genes.pkl...")

pkl_path = os.path.join("models", "all_input_genes.pkl")

if os.path.exists(pkl_path):
    try:
        # Load the exact array of 16,609 gene names
        all_input_genes = joblib.load(pkl_path)
        
        # Convert to string and format into a CSV row
        header_row = ",".join(all_input_genes)
        # Put a baseline profile value (like 5.5) for each gene
        data_row = ",".join(["5.5"] * len(all_input_genes))
        
        # Save it to your main project folder
        with open("../sample_patient.csv", "w") as f:
            f.write(header_row + "\n" + data_row)
            
        print(f"\nSuccess! 'sample_patient.csv' generated with all {len(all_input_genes)} required genes!")
    except Exception as e:
        print(f"Error reading the pickle file: {e}")
else:
    print(f"Could not find the file at {pkl_path}. Please check if it's named correctly in your models folder.")