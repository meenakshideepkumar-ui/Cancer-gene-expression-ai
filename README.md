# Computational Prediction of Cancer from Gene Expression Data using AI

A machine learning pipeline that predicts cancer (or cancer subtype) from patient gene expression data using an Autoencoder + 1D-CNN + XGBoost/Random Forest + SHAP architecture.

## Overview

This project classifies patients as Cancer/Normal (or by cancer subtype) using high-dimensional gene expression data from NIH GEO / TCGA. Given the small sample sizes and 20,000+ gene features typical of this data, the pipeline uses statistical feature selection, an autoencoder for non-linear dimensionality reduction, a 1D-CNN as the primary classifier, an XGBoost/Random Forest baseline for interpretability, and SHAP for explainability — identifying which genes actually drive each prediction.

## Architecture

1. **Data Input** — NIH GEO / TCGA gene expression matrix
2. **Preprocessing** — Log2 normalization, Z-score scaling, missing value handling, low-variance gene removal, SMOTE class balancing
3. **Feature Selection** — Mutual Information / ANOVA F-test (20,000+ genes → top 200–500)
4. **Autoencoder** — Non-linear compression to a 64-dim gene fingerprint
5. **1D-CNN** — Primary classifier
6. **XGBoost / Random Forest** — Baseline classifier + gene importance
7. **SHAP** — Explainability, biomarker gene identification
8. **Evaluation** — Accuracy, F1, ROC-AUC, Confusion Matrix
9. **Output** — Cancer/Normal prediction + top biomarker genes + pathway analysis

## Project Structure

## Dataset

- **Source:** NIH Gene Expression Omnibus (GEO) — https://www.ncbi.nlm.nih.gov/geo/ and/or The Cancer Genome Atlas (TCGA) — https://portal.gdc.cancer.gov/
- **Format:** GEO Series Matrix (.txt, GSE ID) or TCGA RNA-seq counts/FPKM
- **Structure:** Rows = patients, Columns = genes (20,000+), Values = expression levels

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python src/preprocessing.py
python src/feature_selection.py
python src/autoencoder.py
python src/cnn_model.py
python src/rf_xgb_model.py
python src/shap_analysis.py
python src/evaluate.py
```

## Results

_(Fill in after training: accuracy, F1, ROC-AUC for both models, top biomarker genes, key plots)_

## Tech Stack

pandas, numpy, scikit-learn, imbalanced-learn, tensorflow/keras, xgboost, shap, matplotlib, seaborn

## Team

- **[Namitha]** — Data pipeline, preprocessing, feature selection, autoencoder
- **[Meenakshi]** — CNN, Random Forest/XGBoost, SHAP explainability, evaluation

## References

See project architecture document for full list of research papers referenced.

## License

MIT
