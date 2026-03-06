"""
SR_Mapping_NN — Step 01: Download XAUUSD Data
==============================================
Downloads Gold Futures (GC=F) H1 and D1 data from yfinance.
Handles the 730-day maximum window for intraday data.
Saves to data/ folder relative to project root.

Usage:
    python 01_download_data.py
"""

import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

TICKER = "GC=F"
POINT_MT5 = 0.001  # MT5 XAUUSD point size


# ---------------------------------------------------------------------------
# Helper: safe download with retry
# ---------------------------------------------------------------------------
def safe_download(ticker: str, interval: str, start: str, end: str,
                  max_retries: int = 3) -> pd.DataFrame:
    """Download OHLCV data from yfinance with retries."""
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            if df is not None and len(df) > 0:
                return df
            print(f"    Attempt {attempt}: empty result, retrying…")
        except Exception as exc:
            print(f"    Attempt {attempt} failed: {exc}")
        time.sleep(2 * attempt)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Standardise columns
# ---------------------------------------------------------------------------
def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns, lower-case, drop timezone from index."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    rename = {
        "open": "open", "high": "high", "low": "low",
        "close": "close", "volume": "volume",
        "adj close": "close",  # legacy yf
    }
    df = df.rename(columns=rename)
    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing after rename. Got: {list(df.columns)}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df[required + ["volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "datetime"
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_index()
    return df


# ---------------------------------------------------------------------------
# Download H1
# ---------------------------------------------------------------------------
def download_h1() -> pd.DataFrame:
    """
    yfinance caps intraday (1h) data at ~730 days.
    We fetch in two chunks: [today-730d, today-365d] and [today-365d, today]
    and concatenate, dropping duplicates.
    """
    today = datetime.utcnow().date()
    chunks = []

    windows = [
        (today - timedelta(days=730), today - timedelta(days=365)),
        (today - timedelta(days=365), today + timedelta(days=1)),
    ]

    for (w_start, w_end) in windows:
        start_str = w_start.strftime("%Y-%m-%d")
        end_str = w_end.strftime("%Y-%m-%d")
        print(f"  Downloading H1  {start_str} → {end_str} …")
        df_chunk = safe_download(TICKER, "1h", start_str, end_str)
        if len(df_chunk) == 0:
            print(f"    Warning: no data returned for {start_str}→{end_str}")
            continue
        try:
            df_chunk = clean_ohlcv(df_chunk)
        except ValueError as e:
            print(f"    Warning: {e}")
            continue
        print(f"    Got {len(df_chunk)} bars")
        chunks.append(df_chunk)

    if not chunks:
        raise RuntimeError("H1 download failed — all chunks returned empty.")

    h1 = pd.concat(chunks)
    h1 = h1[~h1.index.duplicated(keep="first")].sort_index()
    return h1


# ---------------------------------------------------------------------------
# Download D1
# ---------------------------------------------------------------------------
def download_d1() -> pd.DataFrame:
    """Daily data — yfinance supports many years for daily bars."""
    today = datetime.utcnow().date()
    start_str = (today - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
    end_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"  Downloading D1  {start_str} → {end_str} …")
    df = safe_download(TICKER, "1d", start_str, end_str)
    if len(df) == 0:
        raise RuntimeError("D1 download failed — empty result.")
    df = clean_ohlcv(df)
    print(f"    Got {len(df)} bars")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("SR_Mapping_NN  |  Step 01: Download Data")
    print("=" * 65)

    os.makedirs(DATA_DIR, exist_ok=True)

    # ------ H1 ------
    print("\n[H1] Gold Futures 1-hour bars")
    h1 = download_h1()
    h1_path = os.path.join(DATA_DIR, "xauusd_h1.csv")
    h1.to_csv(h1_path)
    print(f"  Saved  {len(h1):,} H1 bars  →  {h1_path}")
    print(f"  Range: {h1.index.min()}  →  {h1.index.max()}")
    print(f"  Columns: {list(h1.columns)}")
    print(f"  Close range: ${h1['close'].min():.2f} – ${h1['close'].max():.2f}")
    null_pct = h1.isnull().mean().max() * 100
    print(f"  Max null %: {null_pct:.1f}%")

    # ------ D1 ------
    print("\n[D1] Gold Futures daily bars")
    d1 = download_d1()
    d1_path = os.path.join(DATA_DIR, "xauusd_d1.csv")
    d1.to_csv(d1_path)
    print(f"  Saved  {len(d1):,} D1 bars  →  {d1_path}")
    print(f"  Range: {d1.index.min()}  →  {d1.index.max()}")
    print(f"  Close range: ${d1['close'].min():.2f} – ${d1['close'].max():.2f}")

    # ------ Summary ------
    print("\n" + "=" * 65)
    print("Download complete.")
    print(f"  H1: {len(h1):,} bars  ({h1_path})")
    print(f"  D1: {len(d1):,} bars  ({d1_path})")
    print("=" * 65)

    return h1, d1


if __name__ == "__main__":
    main()
