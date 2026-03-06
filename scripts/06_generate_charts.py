"""
SR_Mapping_NN — Step 06: Generate Charts
==========================================
Produces 4 publication-quality charts:

  1. charts/feature_importance.png  — horizontal bar chart (sorted)
  2. charts/confusion_matrix.png    — 2×2 heatmap with counts + percentages
  3. charts/precision_recall_curve.png — PR curve with optimal threshold marked
  4. charts/equity_curve.png        — Original EA vs EA+NN equity lines

Data sources:
  configs/pipeline_results.json   → feature importance, confusion matrix, thresholds
  data/test_predictions.csv       → y_test, y_prob for PR curve
  data/equity_curves.csv          → equity_original, equity_nn
  configs/threshold_analysis.csv  → threshold sweep data

Usage:
    python 06_generate_charts.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from sklearn.metrics import precision_recall_curve, average_precision_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CHARTS_DIR = os.path.join(PROJECT_ROOT, "charts")
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DPI = 150
STYLE = "seaborn-v0_8-whitegrid"

# Colour palette
COL_GOLD = "#D4A017"
COL_RED = "#C0392B"
COL_BLUE = "#2980B9"
COL_GREEN = "#27AE60"
COL_DARK = "#2C3E50"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def apply_style():
    try:
        plt.style.use(STYLE)
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass  # use default


def save_fig(fig, filename):
    path = os.path.join(CHARTS_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved {filename}  ({size_kb:.0f} KB)")
    return path


# ---------------------------------------------------------------------------
# Chart 1: Feature Importance
# ---------------------------------------------------------------------------
def chart_feature_importance(pipeline_results: dict):
    fi_list = pipeline_results.get("feature_importance", [])
    if not fi_list:
        print("  Skipping feature_importance — no data")
        return

    names = [d["feature"] for d in fi_list]
    values = [d["importance"] for d in fi_list]

    # Sort ascending for horizontal bar (bottom = least important)
    order = np.argsort(values)
    names_sorted = [names[i] for i in order]
    values_sorted = [values[i] for i in order]

    n = len(names_sorted)
    fig_height = max(5, n * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    colors = [COL_GOLD if v > np.median(values_sorted) else "#BDC3C7"
              for v in values_sorted]
    bars = ax.barh(range(n), values_sorted, color=colors, edgecolor="white",
                   linewidth=0.5)

    # Annotate bars
    for bar, val in zip(bars, values_sorted):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=8, color=COL_DARK)

    ax.set_yticks(range(n))
    ax.set_yticklabels(names_sorted, fontsize=9)
    ax.set_xlabel("Feature Importance (XGBoost gain)", fontsize=11)
    ax.set_title("Feature Importance — SR_Mapping_NN", fontsize=13,
                 fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.set_xlim(0, max(values_sorted) * 1.15)
    fig.tight_layout()

    return save_fig(fig, "feature_importance.png")


# ---------------------------------------------------------------------------
# Chart 2: Confusion Matrix
# ---------------------------------------------------------------------------
def chart_confusion_matrix(pipeline_results: dict, threshold: float):
    cm_raw = pipeline_results.get("confusion_matrix")
    if cm_raw is None:
        print("  Skipping confusion_matrix — no data")
        return

    cm = np.array(cm_raw)
    total = cm.sum()
    cm_pct = cm / max(total, 1) * 100

    labels = np.array([
        [f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)" for j in range(2)]
        for i in range(2)
    ])

    fig, ax = plt.subplots(figsize=(6, 5))

    if HAS_SEABORN:
        sns.heatmap(
            cm, annot=labels, fmt="", cmap="Blues",
            xticklabels=["Predicted BAD", "Predicted GOOD"],
            yticklabels=["Actual BAD", "Actual GOOD"],
            linewidths=1, linecolor="white",
            annot_kws={"size": 12, "weight": "bold"},
            ax=ax,
        )
    else:
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, labels[i, j], ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Predicted BAD", "Predicted GOOD"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Actual BAD", "Actual GOOD"])

    ax.set_title(f"Confusion Matrix  (threshold = {threshold:.2f})",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()

    return save_fig(fig, "confusion_matrix.png")


# ---------------------------------------------------------------------------
# Chart 3: Precision-Recall Curve
# ---------------------------------------------------------------------------
def chart_precision_recall(pred_df: pd.DataFrame, threshold: float,
                            threshold_df: pd.DataFrame | None = None):
    if "y_test" not in pred_df.columns or "y_prob" not in pred_df.columns:
        print("  Skipping precision_recall_curve — missing columns")
        return

    y_true = pred_df["y_test"].values
    y_prob = pred_df["y_prob"].values

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    # Find index nearest to optimal threshold
    if len(thresholds) > 0:
        idx = int(np.argmin(np.abs(thresholds - threshold)))
        opt_precision = precision[idx]
        opt_recall = recall[idx]
    else:
        opt_precision = opt_recall = None

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color=COL_BLUE, lw=2,
            label=f"PR curve  (AP = {ap:.3f})")
    ax.axhline(y=0.70, color=COL_RED, linestyle="--", lw=1.2, alpha=0.7,
               label="Precision = 0.70 target")

    if opt_precision is not None:
        ax.scatter([opt_recall], [opt_precision], s=120, zorder=5,
                   color=COL_GOLD, edgecolor=COL_DARK, linewidth=1.5,
                   label=f"Optimal threshold = {threshold:.2f}\n"
                         f"(P={opt_precision:.2f}, R={opt_recall:.2f})")

    # Baseline (random)
    baseline = y_true.mean()
    ax.axhline(y=baseline, color="#95A5A6", linestyle=":", lw=1,
               label=f"Baseline ({baseline:.2f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — SR_Mapping_NN", fontsize=13,
                 fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    fig.tight_layout()

    return save_fig(fig, "precision_recall_curve.png")


# ---------------------------------------------------------------------------
# Chart 4: Equity Curve
# ---------------------------------------------------------------------------
def chart_equity_curve(equity_df: pd.DataFrame):
    if equity_df is None or equity_df.empty:
        print("  Skipping equity_curve — no data")
        return

    required = {"equity_original", "equity_nn"}
    missing = required - set(equity_df.columns)
    if missing:
        print(f"  Skipping equity_curve — missing columns: {missing}")
        return

    eq_orig = equity_df["equity_original"].values
    eq_nn = equity_df["equity_nn"].values
    x = np.arange(len(eq_orig))

    # Metrics
    ret_orig = (eq_orig[-1] / eq_orig[0] - 1) * 100
    ret_nn = (eq_nn[-1] / eq_nn[0] - 1) * 100
    dd_orig = _max_drawdown(eq_orig)
    dd_nn = _max_drawdown(eq_nn)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(x, eq_orig, color="#95A5A6", lw=1.5, alpha=0.9,
            label=f"Original EA  "
                  f"${eq_orig[-1]:,.0f}  ({ret_orig:+.1f}%)  DD={dd_orig:.1f}%")
    ax.plot(x, eq_nn, color=COL_GOLD, lw=2.2,
            label=f"EA + NN Filter  "
                  f"${eq_nn[-1]:,.0f}  ({ret_nn:+.1f}%)  DD={dd_nn:.1f}%")

    # Starting equity line
    ax.axhline(y=eq_orig[0], color=COL_DARK, linestyle=":", lw=0.8, alpha=0.5)

    ax.set_xlabel("Trade Bar (test period)", fontsize=12)
    ax.set_ylabel("Account Equity ($)", fontsize=12)
    ax.set_title("Equity Curve — Original EA vs EA + NN Filter",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"${v:,.0f}"
    ))
    fig.tight_layout()

    return save_fig(fig, "equity_curve.png")


def _max_drawdown(equity: np.ndarray) -> float:
    """Maximum drawdown as percentage."""
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-9) * 100
    return float(abs(dd.min()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("SR_Mapping_NN  |  Step 06: Generate Charts")
    print("=" * 65)

    os.makedirs(CHARTS_DIR, exist_ok=True)
    apply_style()

    # ------------------------------------------------------------------
    # Load data files
    # ------------------------------------------------------------------
    pr_path = os.path.join(CONFIGS_DIR, "pipeline_results.json")
    pred_path = os.path.join(DATA_DIR, "test_predictions.csv")
    eq_path = os.path.join(DATA_DIR, "equity_curves.csv")
    ta_path = os.path.join(CONFIGS_DIR, "threshold_analysis.csv")
    fc_path = os.path.join(CONFIGS_DIR, "feature_config.json")

    # Pipeline results
    if not os.path.exists(pr_path):
        raise FileNotFoundError(
            f"pipeline_results.json not found: {pr_path}\n"
            "Run 04_train_model.py first."
        )
    with open(pr_path) as fh:
        pipeline_results = json.load(fh)

    # Threshold
    threshold = pipeline_results.get("best_threshold", 0.50)
    if os.path.exists(fc_path):
        with open(fc_path) as fh:
            fc = json.load(fh)
        threshold = fc.get("threshold", threshold)

    print(f"\nUsing threshold: {threshold:.2f}")

    # Test predictions
    pred_df = None
    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path, index_col=0, parse_dates=True)
        print(f"Loaded test_predictions: {len(pred_df):,} rows")
    else:
        print(f"Warning: {pred_path} not found — PR curve will be skipped.")

    # Equity curves
    equity_df = None
    if os.path.exists(eq_path):
        equity_df = pd.read_csv(eq_path)
        print(f"Loaded equity_curves: {len(equity_df):,} rows")
    else:
        print(f"Warning: {eq_path} not found — equity curve will be skipped.")

    # Threshold analysis
    thresh_df = None
    if os.path.exists(ta_path):
        thresh_df = pd.read_csv(ta_path)

    # ------------------------------------------------------------------
    # Generate charts
    # ------------------------------------------------------------------
    print("\nGenerating charts…")

    # 1. Feature importance
    chart_feature_importance(pipeline_results)

    # 2. Confusion matrix
    chart_confusion_matrix(pipeline_results, threshold)

    # 3. Precision-Recall curve
    if pred_df is not None:
        chart_precision_recall(pred_df, threshold, thresh_df)
    else:
        print("  Skipping precision_recall_curve (no test_predictions.csv)")

    # 4. Equity curve
    if equity_df is not None:
        chart_equity_curve(equity_df)
    else:
        print("  Skipping equity_curve (no equity_curves.csv)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    chart_files = [f for f in os.listdir(CHARTS_DIR) if f.endswith(".png")]
    print(f"\n{'=' * 65}")
    print(f"Charts saved to: {CHARTS_DIR}")
    for f in sorted(chart_files):
        p = os.path.join(CHARTS_DIR, f)
        print(f"  {f:<40} {os.path.getsize(p):>8,} bytes")
    print("=" * 65)

    return CHARTS_DIR


if __name__ == "__main__":
    main()
