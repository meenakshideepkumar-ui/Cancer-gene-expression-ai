"""
Backend API for Cancer Prediction from Gene Expression Data
Prediction Track (Person A) — /predict endpoint

Inference pipeline (must exactly match the training pipeline in the notebooks):
  1. Raw gene expression values (all genes present after dataset alignment)
  2. Log2 normalization (only if input looks like raw/linear-scale values)
  3. VarianceThreshold filter (variance_selector.pkl)
  4. Z-score scaling (scaler.pkl)
  5. Reduce to the selected genes (selected_genes.pkl)
  6. Autoencoder encoder -> 64-dim fingerprint (encoder_model.h5)
  7. 1D-CNN -> Sigmoid probability (cnn_model.h5)
  8. Threshold at 0.5 -> label via label_mapping.pkl
"""

import os
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras

app = Flask(__name__)
CORS(app)  # allows your frontend (different origin) to call this API

# ---------------------------------------------------------------------------
# Load all model artifacts ONCE at startup (not per-request — that would be slow)
# ---------------------------------------------------------------------------
MODEL_DIR = os.environ.get("MODEL_DIR", "models")

print("Loading model artifacts from:", MODEL_DIR)

scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
variance_selector = joblib.load(os.path.join(MODEL_DIR, "variance_selector.pkl"))
selected_genes = joblib.load(os.path.join(MODEL_DIR, "selected_genes.pkl"))
label_mapping = joblib.load(os.path.join(MODEL_DIR, "label_mapping.pkl"))  # {0: "normal", 1: "tumor"}

# The full input gene list, in the exact order variance_selector was fit on.
# NOTE: this file must be saved from notebook 01 (see note at the bottom of this file).
all_input_genes = joblib.load(os.path.join(MODEL_DIR, "all_input_genes.pkl"))

encoder_model = keras.models.load_model(os.path.join(MODEL_DIR, "encoder_model.h5"))
cnn_model = keras.models.load_model(os.path.join(MODEL_DIR, "cnn_model.h5"))

print(f"Loaded. Expecting {len(all_input_genes)} input genes, "
      f"{len(selected_genes)} selected genes, label mapping: {label_mapping}")


# ---------------------------------------------------------------------------
# Preprocessing helper — mirrors the training notebooks exactly
# ---------------------------------------------------------------------------
def preprocess_input(gene_dict):
    """
    gene_dict: a dict like {"1007_s_at": 5.2, "1053_at": 0.1, ...}
    Must contain a value for every gene in all_input_genes (missing genes -> error).
    Returns a (1, N) numpy array ready for the encoder.
    """
    missing = [g for g in all_input_genes if g not in gene_dict]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} required gene(s) in input. "
            f"First few missing: {missing[:5]}"
        )

    # Build the vector in the exact same gene order used during training
    raw_values = np.array([[gene_dict[g] for g in all_input_genes]], dtype=float)

    # Step 2: Log2 normalize only if this looks like raw/linear-scale data
    if raw_values.max() > 50:
        raw_values = np.log2(np.clip(raw_values, a_min=0, a_max=None) + 1)

    # Step 3: Variance threshold filter (same genes that survived during training)
    filtered = variance_selector.transform(raw_values)

    # Step 4: Z-score scale (using the scaler fit during training)
    scaled = scaler.transform(filtered)

    # Step 5: Reduce to selected genes only
    # `filtered`/`scaled` columns correspond to variance_selector.get_support() order;
    # we need to map selected_genes (subset of that) to their positions here.
    kept_gene_names = [g for g, keep in zip(all_input_genes, variance_selector.get_support()) if keep]
    gene_index = {g: i for i, g in enumerate(kept_gene_names)}

    try:
        selected_idx = [gene_index[g] for g in selected_genes]
    except KeyError as e:
        raise ValueError(f"Selected gene {e} not found after variance filtering — "
                          f"check that all_input_genes.pkl matches the training run.")

    reduced = scaled[:, selected_idx]  # shape (1, len(selected_genes))
    return reduced


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected JSON body:
    {
        "gene_expression": {
            "1007_s_at": 5.2,
            "1053_at": 0.1,
            ...
        }
    }
    """
    data = request.get_json(silent=True)
    if not data or "gene_expression" not in data:
        return jsonify({"error": "Request must include a 'gene_expression' object"}), 400

    gene_dict = data["gene_expression"]
    if not isinstance(gene_dict, dict):
        return jsonify({"error": "'gene_expression' must be an object of {gene_id: value}"}), 400

    try:
        reduced = preprocess_input(gene_dict)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Step 6: Autoencoder encoder -> 64-dim fingerprint
    fingerprint = encoder_model.predict(reduced, verbose=0)  # shape (1, 64)

    # Step 7: CNN expects (samples, 64, 1)
    cnn_input = fingerprint.reshape(1, fingerprint.shape[1], 1)
    probability = float(cnn_model.predict(cnn_input, verbose=0).ravel()[0])

    # Step 8: Threshold + label
    predicted_class = 1 if probability >= 0.5 else 0
    predicted_label = label_mapping[predicted_class]
    confidence = probability if predicted_class == 1 else (1 - probability)

    return jsonify({
        "prediction": predicted_label,
        "tumor_probability": round(probability, 4),
        "confidence": round(confidence, 4)
    })


if __name__ == "__main__":
    # debug=True is fine for local development only — turn off before deploying
    app.run(debug=True, host="0.0.0.0", port=5000)
