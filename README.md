# Computational Prediction of Cancer from Gene Expression Data using AI

A machine learning web application that predicts cancer (or cancer subtype) from patient gene expression data, using an Autoencoder + 1D-CNN + XGBoost/Random Forest + SHAP architecture. Models are trained in Google Colab and served through a Flask/FastAPI backend with a simple web frontend.

## Overview

This project classifies patients as Cancer/Normal (or by cancer subtype) using high-dimensional gene expression data from NIH GEO / TCGA. Given the small sample sizes and 20,000+ gene features typical of this data, the pipeline uses statistical feature selection, an autoencoder for non-linear dimensionality reduction, a 1D-CNN as the primary classifier, an XGBoost/Random Forest baseline for interpretability, and SHAP for explainability — identifying which genes actually drive each prediction. The trained models are then wrapped in a web app so a user can input a gene expression profile and get a live prediction.

## Architecture

1. **Data Input** — NIH GEO / TCGA gene expression matrix
2. **Preprocessing** — Log2 normalization, Z-score scaling, missing value handling, low-variance gene removal, SMOTE class balancing
3. **Feature Selection** — Mutual Information / ANOVA F-test (20,000+ genes → top 200–500)
4. **Autoencoder** — Non-linear compression to a 64-dim gene fingerprint
5. **1D-CNN** — Primary classifier
6. **XGBoost / Random Forest** — Baseline classifier + gene importance
7. **SHAP** — Explainability, biomarker gene identification
8. **Evaluation** — Accuracy, F1, ROC-AUC, Confusion Matrix
9. **Output** — Cancer/Normal prediction + top biomarker genes + pathway analysis, served via a web app

## Project Structure

## Dataset

- **Source:** NIH Gene Expression Omnibus (GEO) — https://www.ncbi.nlm.nih.gov/geo/ and/or The Cancer Genome Atlas (TCGA) — https://portal.gdc.cancer.gov/
- **Format:** GEO Series Matrix (.txt, GSE ID) or TCGA RNA-seq counts/FPKM
- **Structure:** Rows = patients, Columns = genes (20,000+), Values = expression levels

## Training (Google Colab)

Model training is done in Google Colab to use free GPU acceleration rather than a local machine.

1. Open the notebooks in `notebooks/` in Google Colab (in order, 01 → 07).
2. Mount Google Drive or clone this repo inside Colab to access/save data:
```python
   from google.colab import drive
   drive.mount('/content/drive')
```
   or
```python
   !git clone https://github.com/<your-username>/<repo-name>.git
```
3. Run preprocessing → feature selection → autoencoder → CNN / XGBoost → SHAP → evaluation.
4. Save trained artifacts at the end of training:
```python
   cnn_model.save("cnn_model.h5")
   encoder.save("encoder_model.h5")
   joblib.dump(xgb_model, "xgb_model.pkl")
   joblib.dump(scaler, "scaler.pkl")
   joblib.dump(selected_genes, "selected_genes.pkl")
```
5. Download these files (or save to Drive) and place them in the local `models/` folder.
6. Commit and push the trained artifacts (see note on large files below) along with the notebooks.

## Running the Web App Locally

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r backend/requirements.txt

cd backend
python app.py
```

Then open the frontend (`frontend/index.html`) in a browser, or serve it via the Flask app if integrated.

## Deployment

- **Backend:** Render / Railway / PythonAnywhere (free tier)
- **Frontend:** Served from the same backend, or deployed separately on Vercel/Netlify
- **Large model files:** If any `.h5`/`.pkl` file exceeds GitHub's limits, use [Git LFS](https://git-lfs.github.com/) or host the file externally (Google Drive / Hugging Face Hub) and download it at app startup.

## Results

_(Fill in after training: accuracy, F1, ROC-AUC for both models, top biomarker genes, key plots)_

## Tech Stack

**ML/Training:** pandas, numpy, scikit-learn, imbalanced-learn, tensorflow/keras, xgboost, shap, matplotlib, seaborn, Google Colab (GPU)
**Backend:** Flask / FastAPI
**Frontend:** HTML, CSS, JavaScript (or React)
**Deployment:** Render / Railway / Vercel

## Team

- **[Name A]** — Data pipeline, preprocessing, feature selection, autoencoder (Colab)
- **[Name B]** — CNN, Random Forest/XGBoost, SHAP explainability, evaluation (Colab), backend API, frontend integration

## References

1. Machine Learning Methods for Cancer Classification Using Gene Expression Data: A Review — PMC9952758 / arXiv:2301.12222
2. Feature Selection in Cancer Classification: Utilizing Explainable Artificial Intelligence to Uncover Influential Genes in Machine Learning Models — *AI* (MDPI), 6(1):2, 2024
3. Cancer Classification from Gene Expression Using Ensemble Learning with an Influential Feature Selection Technique — *BioMedInformatics* (MDPI), 4(2):70, 2024
4. Precision Cancer Classification and Biomarker Identification from mRNA Gene Expression via Dimensionality Reduction and Explainable AI — arXiv:2410.07260, 2024
5. Unsupervised Feature Selection for Tumor Profiles using Autoencoders and Kernel Methods — arXiv:2007.06106
6. Deep Learning Techniques for Cancer Classification Using Microarray Gene Expression Data — *Frontiers in Physiology*, 2022
7. Integrating Deep Learning and SHAP for Breast Cancer Classification and Biomarker Discovery Using Gene Expression Data — *IEEE Access* (in press), 2024/2025
8. Transformer-Based Representation Learning for Robust Gene Expression Modeling and Cancer Prognosis — arXiv:2504.09704, 2025 (suggested future extension beyond CNN)

## License

MIT
