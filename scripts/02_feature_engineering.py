"""
SR_Mapping_NN — Step 02: Feature Engineering
=============================================
Loads H1 and D1 OHLCV data from data/ and computes all 26 model features:

  Williams Fractals, RSI(14), ATR(14), daily open/range/direction,
  dist_res_norm, dist_sup_norm, sr_position, hour, dow,
  ret_3/5/10/20, atr_pctile, atr_change, rsi_sma, rsi_slope,
  near_support, near_resistance, body_size, upper_wick, lower_wick,
  is_bullish, bars_since_frac_up, bars_since_frac_down, vol_ratio

Saves: data/features.csv

Usage:
    python 02_feature_engineering.py
"""

import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DAILY_THRESH = 5.0   # $5 threshold for daily direction
TOL_FRACTION = 0.20  # 20% of ATR defines "near" S/R


# ---------------------------------------------------------------------------
# Williams Fractals
# ---------------------------------------------------------------------------
def compute_williams_fractals(highs: np.ndarray, lows: np.ndarray,
                              lookback: int = 2):
    """
    Classic 5-bar Williams Fractal (lookback=2).
    fup[i]   = high[i] if it is greater than the 2 highs on each side.
    fdown[i] = low[i]  if it is lower  than the 2 lows  on each side.
    Returns (fup, fdown) arrays with NaN where no fractal.
    """
    n = len(highs)
    fup = np.full(n, np.nan)
    fdown = np.full(n, np.nan)
    for i in range(lookback, n - lookback):
        if all(highs[i] > highs[i - j] and highs[i] > highs[i + j]
               for j in range(1, lookback + 1)):
            fup[i] = highs[i]
        if all(lows[i] < lows[i - j] and lows[i] < lows[i + j]
               for j in range(1, lookback + 1)):
            fdown[i] = lows[i]
    return fup, fdown


# ---------------------------------------------------------------------------
# Carry-forward fractals → resistance / support levels
# ---------------------------------------------------------------------------
def carry_forward_fractals(fup: np.ndarray, fdown: np.ndarray):
    n = len(fup)
    res = np.full(n, np.nan)
    sup = np.full(n, np.nan)
    lr, ls = np.nan, np.nan
    for i in range(n):
        if not np.isnan(fup[i]):
            lr = fup[i]
        if not np.isnan(fdown[i]):
            ls = fdown[i]
        res[i] = lr
        sup[i] = ls
    return res, sup


# ---------------------------------------------------------------------------
# Bars since last fractal
# ---------------------------------------------------------------------------
def bars_since(fractal_array: np.ndarray) -> np.ndarray:
    n = len(fractal_array)
    result = np.zeros(n, dtype=float)
    counter = 0
    for i in range(n):
        counter += 1
        if not np.isnan(fractal_array[i]):
            counter = 0
        result[i] = counter
    return result


# ---------------------------------------------------------------------------
# Wilder RSI(14)
# ---------------------------------------------------------------------------
def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# ---------------------------------------------------------------------------
# Wilder ATR(14)
# ---------------------------------------------------------------------------
def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("SR_Mapping_NN  |  Step 02: Feature Engineering")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    h1_path = os.path.join(DATA_DIR, "xauusd_h1.csv")
    d1_path = os.path.join(DATA_DIR, "xauusd_d1.csv")

    if not os.path.exists(h1_path):
        raise FileNotFoundError(
            f"H1 data not found: {h1_path}\n"
            "Run 01_download_data.py first."
        )
    if not os.path.exists(d1_path):
        raise FileNotFoundError(
            f"D1 data not found: {d1_path}\n"
            "Run 01_download_data.py first."
        )

    print(f"\nLoading H1: {h1_path}")
    h1 = pd.read_csv(h1_path, index_col=0, parse_dates=True)
    h1.index.name = "datetime"
    h1 = h1.sort_index()

    print(f"Loading D1: {d1_path}")
    d1 = pd.read_csv(d1_path, index_col=0, parse_dates=True)
    d1.index.name = "datetime"
    d1 = d1.sort_index()

    print(f"  H1: {len(h1):,} bars  {h1.index.min()} → {h1.index.max()}")
    print(f"  D1: {len(d1):,} bars  {d1.index.min()} → {d1.index.max()}")

    df = h1.copy()

    # ------------------------------------------------------------------
    # 1. Williams Fractals → resistance / support
    # ------------------------------------------------------------------
    print("\n--- Computing Williams Fractals (lookback=2) ---")
    fup, fdown = compute_williams_fractals(
        df["high"].values, df["low"].values, lookback=2
    )
    df["resistance"], df["support"] = carry_forward_fractals(fup, fdown)
    print(f"  Fractal UP:   {int(np.sum(~np.isnan(fup))):,}")
    print(f"  Fractal DOWN: {int(np.sum(~np.isnan(fdown))):,}")

    # ------------------------------------------------------------------
    # 2. RSI(14) via Wilder's smoothing
    # ------------------------------------------------------------------
    print("--- Computing RSI(14) ---")
    df["rsi"] = wilder_rsi(df["close"], 14)
    print(f"  RSI range: {df['rsi'].min():.1f} – {df['rsi'].max():.1f}")

    # ------------------------------------------------------------------
    # 3. ATR(14) via Wilder's smoothing
    # ------------------------------------------------------------------
    print("--- Computing ATR(14) ---")
    df["atr"] = wilder_atr(df["high"], df["low"], df["close"], 14)
    print(f"  ATR range: ${df['atr'].min():.2f} – ${df['atr'].max():.2f}")

    # ------------------------------------------------------------------
    # 4. Daily features from D1
    # ------------------------------------------------------------------
    print("--- Computing daily features ---")
    df["_date"] = df.index.date
    d1_open_map = {idx.date(): row["open"] for idx, row in d1.iterrows()}
    df["daily_open"] = df["_date"].map(d1_open_map)
    df["daily_open"] = df["daily_open"].ffill()
    df["daily_range"] = df["close"] - df["daily_open"]
    df["daily_direction"] = np.where(
        df["daily_range"] >= DAILY_THRESH, 1,
        np.where(df["daily_range"] <= -DAILY_THRESH, -1, 0),
    )
    bull = int((df["daily_direction"] == 1).sum())
    bear = int((df["daily_direction"] == -1).sum())
    neutral = int((df["daily_direction"] == 0).sum())
    print(f"  Bullish bars: {bull:,}  Bearish: {bear:,}  Neutral: {neutral:,}")

    # ------------------------------------------------------------------
    # 5. S/R distances (normalised by ATR)
    # ------------------------------------------------------------------
    atr_safe = df["atr"].replace(0, np.nan)
    df["dist_to_res"] = df["resistance"] - df["close"]
    df["dist_to_sup"] = df["close"] - df["support"]
    df["dist_res_norm"] = df["dist_to_res"] / atr_safe
    df["dist_sup_norm"] = df["dist_to_sup"] / atr_safe
    sr_range = df["resistance"] - df["support"]
    df["sr_position"] = np.where(
        sr_range > 0,
        (df["close"] - df["support"]) / sr_range,
        0.5,
    )

    # ------------------------------------------------------------------
    # 6. Time features
    # ------------------------------------------------------------------
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek   # 0=Monday

    # ------------------------------------------------------------------
    # 7. Momentum returns
    # ------------------------------------------------------------------
    for p in [3, 5, 10, 20]:
        df[f"ret_{p}"] = df["close"].pct_change(p) * 100.0

    # ------------------------------------------------------------------
    # 8. Volatility features
    # ------------------------------------------------------------------
    df["atr_pctile"] = df["atr"].rolling(168, min_periods=20).rank(pct=True)
    df["atr_change"] = df["atr"].pct_change(5) * 100.0

    # ------------------------------------------------------------------
    # 9. RSI-derived
    # ------------------------------------------------------------------
    df["rsi_sma"] = df["rsi"].rolling(10).mean()
    df["rsi_slope"] = df["rsi"].diff(3)

    # ------------------------------------------------------------------
    # 10. Near support / resistance (within 20% ATR)
    # ------------------------------------------------------------------
    df["near_support"] = (
        df["dist_to_sup"].abs() <= atr_safe * TOL_FRACTION
    ).astype(int)
    df["near_resistance"] = (
        df["dist_to_res"].abs() <= atr_safe * TOL_FRACTION
    ).astype(int)

    # ------------------------------------------------------------------
    # 11. Candle pattern features (normalised by ATR)
    # ------------------------------------------------------------------
    df["body_size"] = (df["close"] - df["open"]).abs() / atr_safe
    df["upper_wick"] = (
        df["high"] - df[["close", "open"]].max(axis=1)
    ) / atr_safe
    df["lower_wick"] = (
        df[["close", "open"]].min(axis=1) - df["low"]
    ) / atr_safe
    df["is_bullish"] = (df["close"] > df["open"]).astype(int)

    # ------------------------------------------------------------------
    # 12. Bars since last fractal
    # ------------------------------------------------------------------
    df["bars_since_frac_up"] = bars_since(fup)
    df["bars_since_frac_down"] = bars_since(fdown)

    # ------------------------------------------------------------------
    # 13. Volume ratio
    # ------------------------------------------------------------------
    if "volume" in df.columns and df["volume"].sum() > 0:
        vol_sma = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / vol_sma.replace(0, np.nan)
    else:
        df["vol_ratio"] = 0.0

    # ------------------------------------------------------------------
    # Drop helper columns and save
    # ------------------------------------------------------------------
    df.drop(columns=["_date", "dist_to_res", "dist_to_sup"], inplace=True, errors="ignore")

    out_path = os.path.join(DATA_DIR, "features.csv")
    df.to_csv(out_path)

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    model_features = [
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

    print(f"\n--- Feature statistics ({len(model_features)} features) ---")
    print(f"{'Feature':<28}  {'Non-NaN':>8}  {'Mean':>10}  {'Std':>10}")
    print("-" * 62)
    for feat in model_features:
        if feat in df.columns:
            col = df[feat].dropna()
            print(
                f"  {feat:<26}  {len(col):>8,}  "
                f"{col.mean():>10.4f}  {col.std():>10.4f}"
            )
        else:
            print(f"  {feat:<26}  MISSING")

    print(f"\nTotal rows in output: {len(df):,}")
    print(f"Saved features → {out_path}")
    print("=" * 65)
    return df


if __name__ == "__main__":
    main()
