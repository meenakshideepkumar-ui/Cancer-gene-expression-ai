import os
import joblib
import numpy as np

print("Loading model gene layout...")
pkl_path = os.path.join("models", "all_input_genes.pkl")

if not os.path.exists(pkl_path):
    print("Could not find all_input_genes.pkl inside the models folder.")
    exit()

all_input_genes = joblib.load(pkl_path)
num_genes = len(all_input_genes)

# --- 1. DEFINITELY NORMAL SAMPLE ---
# Balanced baseline expression profile (tightly clustered around normal biological log values)
normal_profile = np.random.normal(loc=5.2, scale=0.25, size=num_genes)
normal_csv = ",".join(all_input_genes) + "\n" + ",".join(map(str, normal_profile))

with open("../sample_perfect_normal.csv", "w") as f:
    f.write(normal_csv)
print("-> Generated: 'sample_perfect_normal.csv'")

# --- 2. DEFINITELY NOT NORMAL (TUMOR) SAMPLE ---
# Chaotic, heavily elevated expression layout simulating active oncogenic indicators
not_normal_profile = np.random.normal(loc=9.0, scale=1.5, size=num_genes)
not_normal_profile = np.clip(not_normal_profile, 1.0, 15.0)  # keeps values inside standard log-scale
not_normal_csv = ",".join(all_input_genes) + "\n" + ",".join(map(str, not_normal_profile))

with open("../sample_not_normal.csv", "w") as f:
    f.write(not_normal_csv)
print("-> Generated: 'sample_not_normal.csv'")

print("\nBoth target samples are successfully generated in your project folder!")