"""
SR_Mapping_NN — Step 03: Labeling
===================================
Loads features.csv and labels each eligible bar:
  - eligible: daily_direction != 0 AND 2 <= hour < 18
  - TP = entry ± ATR * 2.0
  - SL = entry ∓ ATR * 1.6
  - Scan forward up to 48 bars; label=1 if TP hit first, 0 if SL hit first
  - Timeout: use floating P/L vs entry to determine 1/0
  - Also computes MFE, MAE, hold_time (minutes)

Saves: data/training_data.csv

Usage:
    python 03_labeling.py
"""

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

TP_MULT = 2.0
SL_MULT = 1.6
TIMEOUT = 48       # bars
HOUR_START = 2
HOUR_END = 18


# ---------------------------------------------------------------------------
# Core labeling loop
# ---------------------------------------------------------------------------
def label_bars(df: pd.DataFrame):
    """
    Iterate over each bar in df and produce label, MFE, MAE, hold_time.
    Only bars where daily_direction != 0 and HOUR_START <= hour < HOUR_END
    are eligible. All others get NaN.

    Returns four numpy arrays: labels, mfe, mae, hold_time_minutes.
    """
    n = len(df)
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atrs = df["atr"].values
    directions = df["daily_direction"].values.astype(float)
    hours = df["hour"].values

    labels = np.full(n, np.nan)
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    hold_time = np.full(n, np.nan)

    eligible = 0
    labelled = 0

    for i in range(n - TIMEOUT):
        d = directions[i]
        if d == 0:
            continue
        h = hours[i]
        if h < HOUR_START or h >= HOUR_END:
            continue
        a = atrs[i]
        if np.isnan(a) or a <= 0:
            continue

        eligible += 1
        entry = closes[i]
        tp_dist = a * TP_MULT
        sl_dist = a * SL_MULT

        if d == 1:   # bullish
            tp_price = entry + tp_dist
            sl_price = entry - sl_dist
        else:        # bearish
            tp_price = entry - tp_dist
            sl_price = entry + sl_dist

        max_profit = 0.0
        max_loss = 0.0
        result = None
        hold = TIMEOUT

        for j in range(i + 1, min(i + TIMEOUT + 1, n)):
            # floating profit and loss from perspective of direction
            if d == 1:
                profit = highs[j] - entry
                loss = entry - lows[j]
            else:
                profit = entry - lows[j]
                loss = highs[j] - entry

            max_profit = max(max_profit, profit)
            max_loss = max(max_loss, loss)

            # Check TP / SL hit
            if d == 1:
                if highs[j] >= tp_price:
                    result = 1
                    hold = j - i
                    break
                if lows[j] <= sl_price:
                    result = 0
                    hold = j - i
                    break
            else:
                if lows[j] <= tp_price:
                    result = 1
                    hold = j - i
                    break
                if highs[j] >= sl_price:
                    result = 0
                    hold = j - i
                    break

        # Timeout: use closing price floating P/L
        if result is None:
            idx_final = min(i + TIMEOUT, n - 1)
            final_close = closes[idx_final]
            if d == 1:
                floating_pl = final_close - entry
            else:
                floating_pl = entry - final_close
            result = 1 if floating_pl > 0 else 0

        labels[i] = result
        mfe[i] = max_profit
        mae[i] = max_loss
        hold_time[i] = hold * 60.0   # convert bars→minutes (H1 = 60 min/bar)
        labelled += 1

    return labels, mfe, mae, hold_time, eligible, labelled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("SR_Mapping_NN  |  Step 03: Labeling")
    print("=" * 65)

    features_path = os.path.join(DATA_DIR, "features.csv")
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"Features not found: {features_path}\n"
            "Run 02_feature_engineering.py first."
        )

    print(f"\nLoading features: {features_path}")
    df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    df.index.name = "datetime"
    df = df.sort_index()
    print(f"  {len(df):,} bars  {df.index.min()} → {df.index.max()}")

    # Validate required columns
    required = ["close", "high", "low", "atr", "daily_direction", "hour"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in features.csv: {missing}")

    # ------------------------------------------------------------------
    # Label
    # ------------------------------------------------------------------
    print(f"\nLabeling with TP×{TP_MULT} / SL×{SL_MULT}, "
          f"timeout={TIMEOUT} bars, hours=[{HOUR_START}, {HOUR_END})  …")

    labels, mfe_arr, mae_arr, hold_arr, eligible, labelled = label_bars(df)

    df["label"] = labels
    df["mfe"] = mfe_arr
    df["mae"] = mae_arr
    df["hold_time"] = hold_arr

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    labelled_df = df.dropna(subset=["label"])
    n_total = len(labelled_df)
    n_win = int((labelled_df["label"] == 1).sum())
    n_lose = int((labelled_df["label"] == 0).sum())
    win_rate = n_win / max(n_total, 1) * 100

    print(f"\n--- Label distribution ---")
    print(f"  Eligible bars scanned : {eligible:,}")
    print(f"  Bars labelled         : {labelled:,}")
    print(f"  Label = 1 (WIN)       : {n_win:,}  ({win_rate:.1f}%)")
    print(f"  Label = 0 (LOSE)      : {n_lose:,}  ({100 - win_rate:.1f}%)")

    if n_total > 0:
        avg_mfe = labelled_df["mfe"].mean()
        avg_mae = labelled_df["mae"].mean()
        avg_hold = labelled_df["hold_time"].mean()
        avg_atr = df["atr"].mean()
        print(f"\n  Avg MFE               : ${avg_mfe:.2f}  "
              f"({avg_mfe / avg_atr:.2f}×ATR)")
        print(f"  Avg MAE               : ${avg_mae:.2f}  "
              f"({avg_mae / avg_atr:.2f}×ATR)")
        print(f"  Avg hold time         : {avg_hold:.0f} min  "
              f"({avg_hold/60:.1f} bars)")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = os.path.join(DATA_DIR, "training_data.csv")
    df.to_csv(out_path)
    print(f"\nSaved training_data.csv ({len(df):,} rows) → {out_path}")
    print("=" * 65)
    return df


if __name__ == "__main__":
    main()
