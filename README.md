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

├── notebooks/                  # Colab notebooks used for training
│   ├── 01_preprocessing.ipynb
│   ├── 02_feature_selection.ipynb
│   ├── 03_autoencoder.ipynb
│   ├── 04_cnn_model.ipynb
│   ├── 05_rf_xgb_model.ipynb
│   ├── 06_shap_analysis.ipynb
│   └── 07_evaluation.ipynb
├── src/                         # Reusable pipeline code (mirrors notebook logic)
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── autoencoder.py
│   ├── cnn_model.py
│   ├── rf_xgb_model.py
│   ├── shap_analysis.py
│   └── evaluate.py
├── models/                      # Trained model artifacts downloaded from Colab
│   ├── encoder_model.h5
│   ├── cnn_model.h5
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   └── selected_genes.pkl
├── backend/                     # Flask/FastAPI app serving the trained models
│   ├── app.py
│   ├── predict.py               # /predict endpoint — CNN pipeline (Person A)
│   ├── explain.py               # /explain endpoint — SHAP pipeline (Person B)
│   └── requirements.txt
├── frontend/                    # Web UI (HTML/CSS/JS or React)
│   ├── index.html
│   ├── style.css
│   └── script.js
├── results/                     # Plots, metrics, SHAP outputs (from training)
├── requirements.txt
├── README.md
└── .gitignore

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

**[Meenakshi] — Prediction Track**
- ML: Preprocessing, feature selection, autoencoder, 1D-CNN classifier (Colab)
- Backend: `/predict` API endpoint (model inference pipeline, deployment)
- Frontend: Prediction results UI (input form, confidence display)

**[Namitha] — Explainability Track**
- ML: XGBoost/Random Forest classifier, SHAP analysis, model evaluation (Colab)
- Backend: `/explain` API endpoint (SHAP inference pipeline, deployment)
- Frontend: SHAP/biomarker visualization UI, overall UI layout

## Work Division

This project is split into two end-to-end tracks:

- **Prediction Track:** preprocessing → feature selection → autoencoder → 1D-CNN → `/predict` API → results UI
- **Explainability Track:** XGBoost/Random Forest → SHAP → evaluation → `/explain` API → SHAP visualization UI

Both tracks share the same preprocessed feature set and dataset, and are integrated behind a common frontend.

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
