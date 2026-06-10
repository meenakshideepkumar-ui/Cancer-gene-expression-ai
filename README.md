# Computational Detection of Stress Signals in Gene Expression Data using AI

## Overview
This project applies machine learning and AI techniques to detect stress-related signals in gene expression datasets. It identifies biomarkers and expression patterns associated with cellular stress responses using publicly available data from the National Institutes of Health (NIH).

---

## Dataset
- **Source**: [NIH Gene Expression Omnibus (GEO)](https://www.ncbi.nlm.nih.gov/geo/) — National Center for Biotechnology Information (NCBI), National Institutes of Health (NIH)
- **Access**: Publicly available at `https://www.ncbi.nlm.nih.gov/geo/`
- **Format**: Series Matrix files (`.txt`), soft files (`.soft`), or tab-separated count matrices
- **Dataset IDs**: GSE64813, GSE63878
- **Organism**: Human 
- **Study Type**: Expression profiling by RNA-seq / microarray

### How we downloaded the data
1. Visited [https://www.ncbi.nlm.nih.gov/geo/](https://www.ncbi.nlm.nih.gov/geo/)
2. Searched for keywords: `oxidative stress gene expression`, `heat stress transcriptomics`
3. Filtered by organism, study type, and number of samples
4. Downloaded the Series Matrix file from the selected GSE page
5. Loaded into Python using `pandas` for preprocessing

```python
import pandas as pd
df = pd.read_csv("GSE_series_matrix.txt", sep="\t", comment="!")
print(df.head())
```

---

## Methodology
- **Feature Selection**: PCA, variance filtering, gene ranking
- **Models Used**: Support Vector Machine (SVM), Random Forest, Deep Neural Network (DNN)
- **Validation**: 5-fold cross-validation, train/test split (80/20)
- **Evaluation Metrics**: Accuracy, F1-score, ROC-AUC

---

## Installation

```bash
git clone https://github.com/your-username/your-repo.git
cd stress-signals-gene-expression-ai
pip install -r requirements.txt
```

### Requirements
---

## Usage

```bash
# Step 1 - Preprocess the downloaded NIH GEO dataset
python preprocess.py --input data/GSE_series_matrix.txt

# Step 2 - Train the model
python train.py --model random_forest

# Step 3 - Evaluate and visualize results
python evaluate.py --output results/
```

---

## Results
- Model accuracy: —
- F1-score: —
- ROC-AUC: —
- Key stress genes identified: —

---

## Team & Work Division

| # | Task Area | Responsibilities | Assigned To |
|---|-----------|-----------------|-------------|
| 1 | Data Collection | Source datasets from NIH GEO/NCBI, select GSE IDs | Person 1 |
| 2 | Preprocessing | Cleaning, normalization (log2, z-score), missing values | Person 1 |
| 3 | EDA | Distribution plots, heatmaps, statistical summaries | Person 1 |
| 4 | Feature Selection | PCA, variance filtering, gene ranking | Person 2 |
| 5 | Model Building | SVM, Random Forest, DNN implementation | Person 2 |
| 6 | Model Training | Cross-validation, hyperparameter tuning | Person 2 |
| 7 | Model Evaluation | Accuracy, F1-score, ROC-AUC metrics | Person 3 |
| 8 | Result Visualization | Confusion matrix, ROC curves, heatmaps | Person 3 |
| 9 | Biological Interpretation | Map stress genes to known pathways | Person 3 |
| 10 | GitHub & Docs | Repo management, README, final submission | Person 3 |
| 11 | README Writing | Each member writes their own section | All 3 |
| 12 | Presentation | Prepare and deliver final presentation | All 3 |

---

## References
- NCBI Gene Expression Omnibus (GEO): https://www.ncbi.nlm.nih.gov/geo/
- NIH National Center for Biotechnology Information: https://www.ncbi.nlm.nih.gov/
- Scikit-learn documentation: https://scikit-learn.org/
- Relevant papers on stress biomarkers and gene expression analysis

---

## License
This project is for academic purposes. Dataset usage follows NIH GEO public access terms.
