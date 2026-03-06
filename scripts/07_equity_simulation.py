"""
SR_Mapping_NN — Step 07: Equity Simulation
============================================
Loads test_predictions.csv and training_data.csv.
Simulates trade-by-trade equity for:
  - Original EA  : all trades where daily_direction != 0 (no NN filter)
  - EA + NN      : only trades where NN confidence >= threshold

Trade sizing: fixed 0.1 lot, $1 per point per 0.1 lot (approximate gold).
  Win PnL  =  ATR × TP_MULT × lot_factor
  Loss PnL = -ATR × SL_MULT × lot_factor

Starting equity: $10,000

Saves: data/equity_curves.csv
Prints comparison statistics.

Usage:
    python 07_equity_simulation.py
"""

import json
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")

TP_MULT = 2.0
SL_MULT = 1.6
LOT_SIZE = 0.1
POINT_FACTOR = 10.0   # $10 per $1 gold move per 0.1 lot (approx)
LOT_FACTOR = LOT_SIZE * POINT_FACTOR   # = 1.0
START_EQUITY = 10_000.0


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------
def compute_stats(equity: list, trade_results: list, label: str) -> dict:
    """Compute summary statistics for one equity curve."""
    arr = np.array(equity)
    n_trades = len(trade_results)
    n_wins = int(sum(r > 0 for r in trade_results))
    n_losses = n_trades - n_wins

    win_rate = n_wins / max(n_trades, 1) * 100
    total_return = (arr[-1] / arr[0] - 1) * 100

    # Max drawdown
    peak = np.maximum.accumulate(arr)
    dd_arr = (arr - peak) / np.maximum(peak, 1e-9) * 100
    max_dd = float(abs(dd_arr.min()))

    # Profit factor
    gross_profit = sum(r for r in trade_results if r > 0)
    gross_loss = abs(sum(r for r in trade_results if r < 0))
    pf = gross_profit / max(gross_loss, 1e-9)

    # Avg win / avg loss
    wins = [r for r in trade_results if r > 0]
    losses = [r for r in trade_results if r < 0]
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 0.0
    rr = avg_win / max(avg_loss, 1e-9)

    return {
        "label": label,
        "n_trades": n_trades,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "win_rate_pct": win_rate,
        "start_equity": arr[0],
        "end_equity": arr[-1],
        "total_return_pct": total_return,
        "max_drawdown_pct": max_dd,
        "profit_factor": pf,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "risk_reward": rr,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("SR_Mapping_NN  |  Step 07: Equity Simulation")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Load files
    # ------------------------------------------------------------------
    pred_path = os.path.join(DATA_DIR, "test_predictions.csv")
    td_path = os.path.join(DATA_DIR, "training_data.csv")
    fc_path = os.path.join(CONFIGS_DIR, "feature_config.json")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"test_predictions.csv not found: {pred_path}\n"
            "Run 04_train_model.py first."
        )
    if not os.path.exists(td_path):
        raise FileNotFoundError(
            f"training_data.csv not found: {td_path}\n"
            "Run 03_labeling.py first."
        )

    pred_df = pd.read_csv(pred_path, index_col=0, parse_dates=True)
    pred_df = pred_df.sort_index()
    print(f"\nLoaded test_predictions: {len(pred_df):,} rows")
    print(f"  Columns: {list(pred_df.columns)}")

    td_raw = pd.read_csv(td_path, index_col=0, parse_dates=True)
    td_raw = td_raw.sort_index()

    # Load threshold from feature config (or fall back to default)
    threshold = 0.50
    if os.path.exists(fc_path):
        with open(fc_path) as fh:
            fc = json.load(fh)
        threshold = float(fc.get("threshold", threshold))
    print(f"NN confidence threshold: {threshold:.2f}")

    # ------------------------------------------------------------------
    # Align test data
    # ------------------------------------------------------------------
    # The test set is the last 15% of labelled rows in training_data.
    labelled_rows = td_raw.dropna(subset=["label"])
    n = len(labelled_rows)
    val_end = int(n * 0.85)
    test_data = labelled_rows.iloc[val_end:].copy()

    # Merge with NN predictions
    test_data = test_data.join(pred_df[["y_prob", "y_pred"]], how="left")

    # If join fails (index mismatch), attempt index-position alignment
    if test_data["y_prob"].isna().all():
        print("  Warning: index mismatch between test_data and pred_df — "
              "aligning by position")
        min_len = min(len(test_data), len(pred_df))
        test_data = test_data.iloc[:min_len].copy()
        test_data["y_prob"] = pred_df["y_prob"].values[:min_len]
        test_data["y_pred"] = pred_df["y_pred"].values[:min_len]

    test_data["nn_pass"] = (test_data["y_prob"] >= threshold).fillna(False).astype(int)
    print(f"Test bars: {len(test_data):,}")
    print(f"NN-pass bars: {int(test_data['nn_pass'].sum()):,}")

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    equity_original = [START_EQUITY]
    equity_nn = [START_EQUITY]
    trades_original = []
    trades_nn = []

    for _, row in test_data.iterrows():
        direction = int(row.get("daily_direction", 0))
        label = int(row.get("label", np.nan)) if not pd.isna(row.get("label")) else None
        atr = row.get("atr", np.nan)
        nn_pass = int(row.get("nn_pass", 0))

        if direction == 0 or label is None or np.isnan(atr) or atr <= 0:
            # No trade — carry equity forward
            equity_original.append(equity_original[-1])
            equity_nn.append(equity_nn[-1])
            continue

        # PnL
        if label == 1:
            pnl = atr * TP_MULT * LOT_FACTOR
        else:
            pnl = -atr * SL_MULT * LOT_FACTOR

        # Original EA: all directional trades
        equity_original.append(equity_original[-1] + pnl)
        trades_original.append(pnl)

        # NN-filtered
        if nn_pass:
            equity_nn.append(equity_nn[-1] + pnl)
            trades_nn.append(pnl)
        else:
            equity_nn.append(equity_nn[-1])

    # ------------------------------------------------------------------
    # Save equity curves
    # ------------------------------------------------------------------
    max_len = max(len(equity_original), len(equity_nn))
    # Pad shorter series with its last value
    while len(equity_original) < max_len:
        equity_original.append(equity_original[-1])
    while len(equity_nn) < max_len:
        equity_nn.append(equity_nn[-1])

    eq_df = pd.DataFrame({
        "equity_original": equity_original,
        "equity_nn": equity_nn,
    })
    eq_path = os.path.join(DATA_DIR, "equity_curves.csv")
    eq_df.to_csv(eq_path, index=False)
    print(f"\nSaved equity_curves.csv ({len(eq_df):,} rows) → {eq_path}")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    stats_orig = compute_stats(equity_original, trades_original, "Original EA")
    stats_nn = compute_stats(equity_nn, trades_nn, "EA + NN Filter")

    print("\n" + "=" * 65)
    print("Equity Simulation Results")
    print("=" * 65)

    headers = [
        ("Trades",         "n_trades",         "d"),
        ("Wins",           "n_wins",            "d"),
        ("Win Rate",       "win_rate_pct",      ".1f%"),
        ("Start Equity",   "start_equity",      "$.0f"),
        ("End Equity",     "end_equity",        "$.0f"),
        ("Total Return",   "total_return_pct",  "+.1f%"),
        ("Max Drawdown",   "max_drawdown_pct",  ".1f%"),
        ("Profit Factor",  "profit_factor",     ".2f"),
        ("Avg Win ($)",    "avg_win",           "$.2f"),
        ("Avg Loss ($)",   "avg_loss",          "$.2f"),
        ("Risk/Reward",    "risk_reward",       ".2f"),
    ]

    print(f"\n  {'Metric':<22}  {'Original EA':>14}  {'EA + NN':>14}")
    print("  " + "-" * 54)

    def fmt(val, spec):
        if spec.endswith("%"):
            inner = spec[:-1]
            if inner == "d":
                return f"{val:.0f}%"
            return f"{val:{inner}}%"
        if spec.startswith("$"):
            inner = spec[1:]
            return f"${val:{inner}}"
        return f"{val:{spec}}"

    for label, key, spec in headers:
        o = fmt(stats_orig[key], spec)
        n = fmt(stats_nn[key], spec)
        print(f"  {label:<22}  {o:>14}  {n:>14}")

    print("\n  Equity trajectory:")
    print(f"    Original EA : ${equity_original[0]:,.0f}  →  ${equity_original[-1]:,.0f}  "
          f"({(equity_original[-1]/equity_original[0]-1)*100:+.1f}%)")
    print(f"    EA + NN     : ${equity_nn[0]:,.0f}  →  ${equity_nn[-1]:,.0f}  "
          f"({(equity_nn[-1]/equity_nn[0]-1)*100:+.1f}%)")

    print("=" * 65)
    return eq_df, stats_orig, stats_nn


if __name__ == "__main__":
    main()
