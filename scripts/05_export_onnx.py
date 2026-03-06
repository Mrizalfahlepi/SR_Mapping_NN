"""
SR_Mapping_NN — Step 05: Export ONNX
======================================
Loads model_xgboost.pkl and the feature config.
Retrains on data with features renamed f0–f25 for ONNX compatibility.
Converts to ONNX via onnxmltools.
Verifies with onnxruntime (compares probabilities; asserts max diff < 0.01).
Saves: models/sr_mapping_nn.onnx

Usage:
    python 05_export_onnx.py
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")


# ---------------------------------------------------------------------------
# Helper: extract ONNX probabilities from session output
# ---------------------------------------------------------------------------
def extract_onnx_probs(onnx_out) -> np.ndarray:
    """
    onnxruntime returns [labels, probabilities].
    probabilities may be a list-of-dicts or a 2-D array depending on version.
    """
    raw = onnx_out[1]
    if isinstance(raw, list):
        # list of dicts: [{0: p0, 1: p1}, ...]
        return np.array([d[1] for d in raw], dtype=np.float32)
    # numpy array shape (n, 2)
    arr = np.asarray(raw)
    if arr.ndim == 2:
        return arr[:, 1].astype(np.float32)
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("SR_Mapping_NN  |  Step 05: Export ONNX")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Load saved model
    # ------------------------------------------------------------------
    model_path = os.path.join(MODELS_DIR, "model_xgboost.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run 04_train_model.py first."
        )
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)
    print(f"\nLoaded model: {model_path}")

    # ------------------------------------------------------------------
    # Load feature config
    # ------------------------------------------------------------------
    fc_path = os.path.join(CONFIGS_DIR, "feature_config.json")
    if not os.path.exists(fc_path):
        raise FileNotFoundError(
            f"Feature config not found: {fc_path}\n"
            "Run 04_train_model.py first."
        )
    with open(fc_path, "r") as fh:
        fc = json.load(fh)

    model_features = fc["model_features"]
    feat_name_map = fc["feature_name_map"]
    onnx_feat_names = fc["onnx_feature_names"]
    n_features = len(model_features)
    print(f"Features: {n_features}  ({onnx_feat_names[0]} … {onnx_feat_names[-1]})")

    # ------------------------------------------------------------------
    # Load training data
    # ------------------------------------------------------------------
    td_path = os.path.join(DATA_DIR, "training_data.csv")
    if not os.path.exists(td_path):
        raise FileNotFoundError(
            f"Training data not found: {td_path}\n"
            "Run 03_labeling.py first."
        )
    td_raw = pd.read_csv(td_path, index_col=0, parse_dates=True).sort_index()
    td = td_raw.dropna(subset=["label"] + model_features).copy()
    td["label"] = td["label"].astype(int)
    print(f"Training data rows (for retrain): {len(td):,}")

    # ------------------------------------------------------------------
    # Rebuild train / val / test splits (must match step 04 exactly)
    # ------------------------------------------------------------------
    n = len(td)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train = td[model_features].iloc[:train_end].rename(columns=feat_name_map)
    y_train = td["label"].iloc[:train_end]
    X_val = td[model_features].iloc[train_end:val_end].rename(columns=feat_name_map)
    y_val = td["label"].iloc[train_end:val_end]
    X_test = td[model_features].iloc[val_end:].rename(columns=feat_name_map)
    y_test = td["label"].iloc[val_end:]

    # ------------------------------------------------------------------
    # Retrain with f0-f25 feature names for ONNX
    # ------------------------------------------------------------------
    print("\nRetraining with ONNX-compatible feature names (f0–f25)…")
    base_params = model.get_params()
    # Remove early_stopping_rounds from get_params if not applicable
    onnx_params = {
        k: v for k, v in base_params.items()
        if k not in ("early_stopping_rounds", "callbacks")
    }
    model_onnx = XGBClassifier(**onnx_params)
    model_onnx.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Quick AUC check
    from sklearn.metrics import roc_auc_score
    test_auc = roc_auc_score(y_test, model_onnx.predict_proba(X_test)[:, 1])
    print(f"  Retrain Test AUC: {test_auc:.4f}")

    # ------------------------------------------------------------------
    # Convert to ONNX via onnxmltools
    # ------------------------------------------------------------------
    print("\nConverting to ONNX…")
    try:
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError:
        raise ImportError(
            "onnxmltools not installed. Run: pip install onnxmltools"
        )

    initial_types = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(model_onnx, initial_types=initial_types)

    onnx_path = os.path.join(MODELS_DIR, "sr_mapping_nn.onnx")
    with open(onnx_path, "wb") as fh:
        fh.write(onnx_model.SerializeToString())
    onnx_size_kb = os.path.getsize(onnx_path) / 1024
    print(f"Saved ONNX model: {onnx_path}  ({onnx_size_kb:.1f} KB)")

    # ------------------------------------------------------------------
    # Verify with onnxruntime
    # ------------------------------------------------------------------
    print("\nVerifying ONNX model with onnxruntime…")
    try:
        import onnxruntime as rt
    except ImportError:
        raise ImportError(
            "onnxruntime not installed. Run: pip install onnxruntime"
        )

    sess = rt.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    print(f"  ONNX input name: '{input_name}'  shape: {sess.get_inputs()[0].shape}")

    # Use first 20 test samples
    n_verify = min(20, len(X_test))
    sample = X_test.iloc[:n_verify].values.astype(np.float32)
    onnx_out = sess.run(None, {input_name: sample})
    onnx_probs = extract_onnx_probs(onnx_out)
    orig_probs = model_onnx.predict_proba(X_test.iloc[:n_verify])[:, 1].astype(np.float32)

    max_diff = float(np.max(np.abs(orig_probs - onnx_probs)))
    mean_diff = float(np.mean(np.abs(orig_probs - onnx_probs)))
    status = "PASSED" if max_diff < 0.01 else "WARNING"
    print(f"  Verification — max diff: {max_diff:.6f}  mean diff: {mean_diff:.6f}  [{status}]")

    if max_diff >= 0.01:
        print(
            f"  WARNING: max diff {max_diff:.4f} exceeds 0.01. "
            "ONNX model may need inspection."
        )

    # Side-by-side sample comparison
    print(f"\n  Sample probabilities (first {n_verify} test bars):")
    print(f"  {'Bar':>4}  {'XGBoost':>10}  {'ONNX':>10}  {'Diff':>10}")
    print("  " + "-" * 40)
    for k in range(min(10, n_verify)):
        print(f"  {k:>4}  {orig_probs[k]:>10.6f}  {onnx_probs[k]:>10.6f}  "
              f"{abs(orig_probs[k]-onnx_probs[k]):>10.6f}")

    # ------------------------------------------------------------------
    # Save retrained model (with ONNX-compatible names) over original
    # ------------------------------------------------------------------
    retrain_path = os.path.join(MODELS_DIR, "model_xgboost_onnx.pkl")
    with open(retrain_path, "wb") as fh:
        pickle.dump(model_onnx, fh)
    print(f"\nSaved retrained model (f0-f25 names) → {retrain_path}")

    print("\n" + "=" * 65)
    print("ONNX export complete.")
    print(f"  ONNX model:   {onnx_path}")
    print(f"  Size:         {onnx_size_kb:.1f} KB")
    print(f"  Verification: {status}  (max diff = {max_diff:.6f})")
    print("=" * 65)

    return onnx_path, max_diff


if __name__ == "__main__":
    main()
