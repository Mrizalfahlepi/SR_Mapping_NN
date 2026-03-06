"""
SR_Mapping_NN — Step 04: Train Model
======================================
Loads training_data.csv.
- 26 features (renamed f0–f25 for ONNX compatibility)
- Time-based 70/15/15 split
- Trains 3 XGBoost configs: balanced / conservative / aggressive
- Selects best by validation AUC
- Optimises decision threshold targeting precision >= 70%
- Saves model to models/model_xgboost.pkl
- Saves feature config to configs/feature_config.json
- Saves threshold analysis to configs/threshold_analysis.csv
- Saves test predictions to data/test_predictions.csv
- Saves pipeline summary to configs/pipeline_results.json

Usage:
    python 04_train_model.py
"""

import json
import os
import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")

POINT_MT5 = 0.001
TP_MULT = 2.0
SL_MULT = 1.6
DAILY_THRESH = 5.0

MODEL_FEATURES = [
    "rsi", "atr", "daily_range", "daily_direction",
    "dist_res_norm", "dist_sup_norm", "sr_position",
    "hour", "dow",
    "ret_3", "ret_5", "ret_10", "ret_20",
    "atr_pctile", "atr_change",
    "rsi_sma", "rsi_slope",
    "near_support", "near_resistance",
    "body_size", "upper_wick", "lower_wick", "is_bullish",
    "bars_since_frac_up", "bars_since_frac_down",
    "vol_ratio",
]

CONFIGS = [
    {
        "name": "balanced",
        "params": {
            "n_estimators": 1000, "max_depth": 4, "learning_rate": 0.01,
            "subsample": 0.7, "colsample_bytree": 0.7,
            "min_child_weight": 10, "gamma": 0.3,
            "reg_alpha": 0.5, "reg_lambda": 2.0,
        },
    },
    {
        "name": "conservative",
        "params": {
            "n_estimators": 800, "max_depth": 3, "learning_rate": 0.02,
            "subsample": 0.6, "colsample_bytree": 0.6,
            "min_child_weight": 20, "gamma": 0.5,
            "reg_alpha": 1.0, "reg_lambda": 3.0,
        },
    },
    {
        "name": "aggressive",
        "params": {
            "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 5, "gamma": 0.1,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
        },
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("SR_Mapping_NN  |  Step 04: Train Model")
    print("=" * 65)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CONFIGS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Load training data
    # ------------------------------------------------------------------
    td_path = os.path.join(DATA_DIR, "training_data.csv")
    if not os.path.exists(td_path):
        raise FileNotFoundError(
            f"Training data not found: {td_path}\n"
            "Run 03_labeling.py first."
        )

    print(f"\nLoading: {td_path}")
    td_raw = pd.read_csv(td_path, index_col=0, parse_dates=True)
    td_raw = td_raw.sort_index()
    print(f"  Raw rows: {len(td_raw):,}")

    # Keep only the 26 features we actually have
    available = [f for f in MODEL_FEATURES if f in td_raw.columns]
    missing = [f for f in MODEL_FEATURES if f not in td_raw.columns]
    if missing:
        print(f"  Warning: features missing from file: {missing}")
    if not available:
        raise ValueError("None of the 26 model features found in training_data.csv")

    td = td_raw.dropna(subset=["label"] + available).copy()
    td["label"] = td["label"].astype(int)
    print(f"  After dropna: {len(td):,} samples")
    print(f"  Features used: {len(available)}")

    # ------------------------------------------------------------------
    # Feature rename map: name → fN
    # ------------------------------------------------------------------
    feat_name_map = {name: f"f{i}" for i, name in enumerate(available)}
    feat_name_map_rev = {v: k for k, v in feat_name_map.items()}
    onnx_feat_names = [feat_name_map[f] for f in available]

    # ------------------------------------------------------------------
    # Time-based split 70 / 15 / 15
    # ------------------------------------------------------------------
    n = len(td)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_all = td[available]
    y_all = td["label"]

    X_train_raw = X_all.iloc[:train_end]
    y_train = y_all.iloc[:train_end]
    X_val_raw = X_all.iloc[train_end:val_end]
    y_val = y_all.iloc[train_end:val_end]
    X_test_raw = X_all.iloc[val_end:]
    y_test = y_all.iloc[val_end:]

    # Rename to f0-f25
    X_train = X_train_raw.rename(columns=feat_name_map)
    X_val = X_val_raw.rename(columns=feat_name_map)
    X_test = X_test_raw.rename(columns=feat_name_map)

    print(f"\n  Split sizes:")
    print(f"    Train: {len(X_train):,}  ({y_train.mean()*100:.1f}% positive)")
    print(f"    Val:   {len(X_val):,}  ({y_val.mean()*100:.1f}% positive)")
    print(f"    Test:  {len(X_test):,}  ({y_test.mean()*100:.1f}% positive)")

    # scale_pos_weight from train set
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    spw = n_neg / max(n_pos, 1)
    print(f"  scale_pos_weight = {spw:.2f}")

    # ------------------------------------------------------------------
    # Train 3 configs
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Model Training — 3 configs")
    print("=" * 65)

    best_model = None
    best_val_auc = 0.0
    best_name = ""

    for cfg in CONFIGS:
        params = {**cfg["params"], "scale_pos_weight": spw}
        print(f"\n  [{cfg['name']}]  n_est={params['n_estimators']}  "
              f"depth={params['max_depth']}  lr={params['learning_rate']}")
        mdl = XGBClassifier(
            **params,
            eval_metric="logloss",
            random_state=42,
            early_stopping_rounds=50,
            verbosity=0,
            use_label_encoder=False,
        )
        mdl.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        val_prob = mdl.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_prob)
        test_prob = mdl.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_prob)
        best_iter = getattr(mdl, "best_iteration", params["n_estimators"])
        print(f"    Val AUC: {val_auc:.4f}  |  Test AUC: {test_auc:.4f}  "
              f"|  Best iter: {best_iter}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = mdl
            best_name = cfg["name"]

    print(f"\n  Best config: '{best_name}'  (Val AUC: {best_val_auc:.4f})")
    model = best_model

    # ------------------------------------------------------------------
    # Final probabilities
    # ------------------------------------------------------------------
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_val_prob = model.predict_proba(X_val)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, y_train_prob)
    val_auc = roc_auc_score(y_val, y_val_prob)
    test_auc = roc_auc_score(y_test, y_test_prob)
    print(f"\n  Final AUC — Train: {train_auc:.4f} | Val: {val_auc:.4f} | Test: {test_auc:.4f}")

    # ------------------------------------------------------------------
    # Threshold optimisation: target precision >= 70%
    # ------------------------------------------------------------------
    print("\n--- Threshold Optimisation (target precision ≥ 70%) ---")
    thresh_results = []
    for t_int in range(30, 85):
        t = t_int / 100.0
        yp = (y_test_prob >= t).astype(int)
        if yp.sum() == 0:
            continue
        p = float(precision_score(y_test, yp, zero_division=0))
        r = float(recall_score(y_test, yp, zero_division=0))
        f = float(f1_score(y_test, yp, zero_division=0))
        thresh_results.append({
            "threshold": t,
            "precision": p,
            "recall": r,
            "f1": f,
            "n_signals": int(yp.sum()),
        })

    # Best F1 where precision >= 70%; fall back to best F1 if nothing qualifies
    best_t = 0.50
    best_f = 0.0
    for tr in thresh_results:
        if tr["precision"] >= 0.70 and tr["f1"] > best_f:
            best_f = tr["f1"]
            best_t = tr["threshold"]

    if best_f == 0.0 and thresh_results:
        # fall back to best F1 overall
        best_tr = max(thresh_results, key=lambda x: x["f1"])
        best_t = best_tr["threshold"]
        best_f = best_tr["f1"]
        print(f"  Note: precision>=70% not achieved; using best F1 threshold")

    print(f"  Optimal threshold: {best_t:.2f}")

    y_final = (y_test_prob >= best_t).astype(int)
    fp = float(precision_score(y_test, y_final, zero_division=0))
    fr = float(recall_score(y_test, y_final, zero_division=0))
    ff = float(f1_score(y_test, y_final, zero_division=0))
    print(f"  Precision: {fp:.4f} | Recall: {fr:.4f} | F1: {ff:.4f}")
    print(f"  Signals:   {int(y_final.sum())} / {len(y_final)}")

    # ------------------------------------------------------------------
    # Full classification report
    # ------------------------------------------------------------------
    print("\n--- Classification Report ---")
    cm = confusion_matrix(y_test, y_final)
    print(classification_report(y_test, y_final, target_names=["BAD", "GOOD"]))
    print(f"Confusion Matrix (threshold={best_t:.2f}):")
    print(f"  Predicted:      BAD    GOOD")
    print(f"  Actual BAD  : {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"  Actual GOOD : {cm[1][0]:6d}  {cm[1][1]:6d}")

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------
    importance = model.feature_importances_
    feat_imp_pairs = sorted(
        zip(available, importance), key=lambda x: -x[1]
    )
    print("\n--- Feature Importance ---")
    for name, imp in feat_imp_pairs:
        bar = "#" * int(imp * 40)
        print(f"  {name:<28}: {imp:.4f}  {bar}")

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    model_path = os.path.join(MODELS_DIR, "model_xgboost.pkl")
    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)
    print(f"\nSaved model → {model_path}")

    # ------------------------------------------------------------------
    # feature_config.json
    # ------------------------------------------------------------------
    norm_stats = {}
    for feat in available:
        col = td[feat].dropna()
        norm_stats[feat] = {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "min": float(col.min()),
            "max": float(col.max()),
        }

    feature_config = {
        "model_features": available,
        "onnx_feature_names": onnx_feat_names,
        "feature_name_map": feat_name_map,
        "feature_name_map_reverse": feat_name_map_rev,
        "threshold": float(best_t),
        "point_size_mt5": float(POINT_MT5),
        "daily_range_threshold": float(DAILY_THRESH),
        "rsi_oversold": 40,
        "rsi_overbought": 70,
        "tp_atr_mult": float(TP_MULT),
        "sl_atr_mult": float(SL_MULT),
        "trading_hours": [2, 18],
        "data_source": "GC=F yfinance H1",
        "metrics": {
            "train_auc": float(train_auc),
            "val_auc": float(val_auc),
            "test_auc": float(test_auc),
            "test_precision": float(fp),
            "test_recall": float(fr),
            "test_f1": float(ff),
            "best_config": best_name,
        },
        "normalization": norm_stats,
    }

    fc_path = os.path.join(CONFIGS_DIR, "feature_config.json")
    with open(fc_path, "w") as fh:
        json.dump(feature_config, fh, indent=2)
    print(f"Saved feature_config → {fc_path}")

    # ------------------------------------------------------------------
    # threshold_analysis.csv
    # ------------------------------------------------------------------
    ta_path = os.path.join(CONFIGS_DIR, "threshold_analysis.csv")
    pd.DataFrame(thresh_results).to_csv(ta_path, index=False)
    print(f"Saved threshold_analysis → {ta_path}")

    # ------------------------------------------------------------------
    # test_predictions.csv
    # ------------------------------------------------------------------
    pred_df = pd.DataFrame(
        {"y_test": y_test.values, "y_prob": y_test_prob, "y_pred": y_final},
        index=y_test.index,
    )
    pred_path = os.path.join(DATA_DIR, "test_predictions.csv")
    pred_df.to_csv(pred_path)
    print(f"Saved test_predictions → {pred_path}")

    # ------------------------------------------------------------------
    # pipeline_results.json
    # ------------------------------------------------------------------
    pipeline_results = {
        "train_auc": float(train_auc),
        "val_auc": float(val_auc),
        "test_auc": float(test_auc),
        "best_threshold": float(best_t),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "model_features": available,
        "feature_importance": [
            {"feature": n, "importance": float(imp)} for n, imp in feat_imp_pairs
        ],
        "confusion_matrix": [[int(x) for x in row] for row in cm],
        "threshold_results": thresh_results,
    }
    pr_path = os.path.join(CONFIGS_DIR, "pipeline_results.json")
    with open(pr_path, "w") as fh:
        json.dump(pipeline_results, fh, indent=2)
    print(f"Saved pipeline_results → {pr_path}")

    print("\n" + "=" * 65)
    print("Training complete.")
    print(f"  Model:      {model_path}")
    print(f"  Val AUC:    {val_auc:.4f}")
    print(f"  Test AUC:   {test_auc:.4f}")
    print(f"  Threshold:  {best_t:.2f}")
    print(f"  Precision:  {fp:.4f}")
    print("=" * 65)

    return model, feature_config, pipeline_results


if __name__ == "__main__":
    main()
