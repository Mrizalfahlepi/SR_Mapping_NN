# SR_Mapping_NN — Neural Network Entry Filter for XAUUSD

> **XGBoost classifier that filters entry signals for MetaTrader 5 EA.**  
> Replacing grid/martingale with AI-based risk management.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-green)
![ONNX](https://img.shields.io/badge/ONNX-1.20-orange)
![MQL5](https://img.shields.io/badge/MQL5-MetaTrader%205-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Table of Contents

- [About the Project](#about-the-project)
- [System Architecture](#system-architecture)
- [Model Results](#model-results)
- [Repository Structure](#repository-structure)
- [Roadmap & Workflow](#roadmap--workflow)
- [Installation & Usage](#installation--usage)
- [Deployment Guide to MT5](#deployment-guide-to-mt5)
- [Technical Details](#technical-details)
- [Limitations & Important Notes](#limitations--important-notes)
- [License](#license)

---

## About the Project

### Background

The **SR_Mapping_Foundation v7.1** EA has strong entry signals (91.10% win rate, profit factor 2.37) but uses a **10x grid martingale** for risk management, causing:
- Max Equity Drawdown: **40.69%** (dangerous)
- High margin call risk in trending market conditions

### Solution

Building an **XGBoost classifier** that learns from market patterns to filter entries:
- **GOOD entry** = price moves directly toward Take Profit
- **BAD entry** = price will hit Stop Loss

Grid/martingale is **completely removed**. Each trade is protected with individual SL/TP.

### Original EA Backtest (11 months, Dec 2024 — Nov 2025)

| Metric | Value |
|--------|-------|
| Total Trades | 820 |
| Win Rate | 91.10% |
| Profit Factor | 2.37 |
| Net Profit | $57,893 (from $1,000) |
| Max DD | 40.69% (**PROBLEM**) |

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  MARKET DATA (Live)                  │
│          XAUUSD M1/M30/D1 from MetaTrader 5         │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│          SR_Mapping_NN_v1.mq5 (EA in MT5)           │
│                                                      │
│  1. BuildSNR()          → Fractal S/R levels         │
│  2. GetRSI_M1()         → RSI(14) on M1              │
│  3. GetATR_M1_Points()  → ATR(14) on M1              │
│  4. DetectDailyRange()  → Daily direction (+1/0/-1)  │
│  5. Signal Check        → BUY/SELL conditions         │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │        NN FILTER (NEW COMPONENT)             │    │
│  │                                               │    │
│  │  PrepareFeatures() → 26 real-time features   │    │
│  │         ↓                                     │    │
│  │  sr_mapping_nn.onnx → XGBoost inference       │    │
│  │         ↓                                     │    │
│  │  confidence >= 0.51 ?                         │    │
│  │    YES → ExecuteBuy/Sell (with SL/TP)         │    │
│  │    NO  → SKIP (log reason)                    │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  6. ManageSmartTrailing() → Active trailing stop     │
└─────────────────────────────────────────────────────┘
```

---

## Model Results

### Performance on Test Set (Sep — Nov 2025)

| Metric | Original EA | EA + NN Filter |
|--------|-------------|----------------|
| Total Trades | 606 | **29** |
| Win Rate | 51.3% | **72.4%** |
| Precision | — | **72.4%** |
| Signals/day | ~14 | **~0.5** |

### Charts

#### Feature Importance
![Feature Importance](charts/feature_importance.png)

#### Confusion Matrix
![Confusion Matrix](charts/confusion_matrix.png)

#### Precision-Recall Curve
![Precision-Recall Curve](charts/precision_recall_curve.png)

#### Equity Curve Comparison
![Equity Curve](charts/equity_curve.png)

---

## Repository Structure

```
SR_Mapping_NN/
├── README.md                          # Main documentation (this file)
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── scripts/                           # Python scripts
│   ├── 01_download_data.py            # Step 1: Download XAUUSD data
│   ├── 02_feature_engineering.py      # Step 2: Compute all features
│   ├── 03_labeling.py                 # Step 3: Forward-looking labels
│   ├── 04_train_model.py              # Step 4: Train XGBoost
│   ├── 05_export_onnx.py              # Step 5: Export to ONNX
│   ├── 06_generate_charts.py          # Step 6: Generate visualizations
│   ├── 07_equity_simulation.py        # Step 7: Simulate equity curve
│   └── run_full_pipeline.py           # Run all steps at once
│
├── mql5/                              # MetaTrader 5 files
│   ├── SR_Mapping_NN_v1.mq5           # Main EA (ONNX-enabled)
│   └── DEPLOYMENT_GUIDE.md            # MT5 deployment guide
│
├── models/                            # Trained models
│   ├── sr_mapping_nn.onnx             # ONNX model for MT5
│   ├── model_xgboost.pkl              # XGBoost pickle
│   └── model_xgboost.json             # XGBoost JSON (backup)
│
├── configs/                           # Configuration files
│   ├── feature_config.json            # Features + thresholds + normalization
│   ├── pipeline_results.json          # Training results
│   └── threshold_analysis.csv         # Full threshold analysis
│
├── data/                              # Datasets
│   ├── xauusd_h1.csv                  # OHLCV H1 (Oct 2024 — Nov 2025)
│   ├── xauusd_d1.csv                  # OHLCV D1 (Jan 2023 — Nov 2025)
│   ├── training_data.csv              # Training dataset (4,040 samples)
│   ├── test_predictions.csv           # Predictions on test set
│   └── equity_curves.csv              # Equity curve simulation
│
├── charts/                            # Visualizations
│   ├── feature_importance.png          # Feature importance ranking
│   ├── confusion_matrix.png            # Confusion matrix
│   ├── precision_recall_curve.png      # Precision-recall curve
│   └── equity_curve.png               # Equity curve comparison
│
└── docs/                              # Additional documentation
    ├── VALIDATION_REPORT.md            # Full validation report
    ├── EA_LOGIC.md                     # EA logic documentation
    ├── FEATURE_DICTIONARY.md           # Feature explanations
    └── WORKFLOW.md                     # Full workflow diagram
```

---

## Roadmap & Workflow

### Overview: 7-Step Pipeline

```
STEP 1          STEP 2              STEP 3           STEP 4
Download    →   Feature          →  Labeling      →  Training
XAUUSD Data     Engineering         (TP/SL sim)      XGBoost
                                    
STEP 5          STEP 6              STEP 7
ONNX         →  Generate EA      →  Validation &
Export          MQL5 Code           Reporting
```

### Step-by-Step Detail

#### STEP 1: Download Data (`scripts/01_download_data.py`)
```
Input:  Ticker GC=F from yfinance
Output: data/xauusd_h1.csv, data/xauusd_d1.csv

- Download H1 OHLCV (Oct 2024 — Nov 2025): 6,711 bars
- Download D1 OHLCV (Jan 2023 — Nov 2025): 732 bars
- Validation: check missing data, weekend gaps, holidays
```

#### STEP 2: Feature Engineering (`scripts/02_feature_engineering.py`)
```
Input:  data/xauusd_h1.csv, data/xauusd_d1.csv
Output: DataFrame with 26 features per bar

Features computed:
1.  Williams Fractals (lookback=2) → S/R levels
2.  RSI(14) on H1
3.  ATR(14) on H1
4.  Daily Open + Daily Range Direction
5.  Distance to S/R (normalized by ATR)
6.  Momentum (return 3/5/10/20 bars)
7.  Volatility regime (ATR percentile)
8.  Candle patterns (body, wicks, bullish/bearish)
9.  Time features (hour, day_of_week)
10. Bars since last fractal
```

#### STEP 3: Labeling (`scripts/03_labeling.py`)
```
Input:  Feature DataFrame from Step 2
Output: data/training_data.csv (4,040 samples)

For each bar where daily direction != 0:
- TP = entry ± ATR × 2.0
- SL = entry ∓ ATR × 1.6
- Scan forward 48 bars (48 hours)
- Label = 1 if TP hit first (GOOD)
- Label = 0 if SL hit first (BAD)
- Timeout: use floating P/L

Distribution: 45.2% GOOD, 54.8% BAD
```

#### STEP 4: Training (`scripts/04_train_model.py`)
```
Input:  data/training_data.csv
Output: models/model_xgboost.pkl, configs/feature_config.json

- Time-based split: Train 70% / Val 15% / Test 15%
- Try 3 configs: balanced, conservative, aggressive
- Select best config based on Val AUC
- Threshold optimization: target precision >= 70%
- Result: AUC=0.538, Precision=72.4% at threshold=0.51
```

#### STEP 5: ONNX Export (`scripts/05_export_onnx.py`)
```
Input:  models/model_xgboost.pkl
Output: models/sr_mapping_nn.onnx

- Convert XGBoost → ONNX via onnxmltools
- Verification: ONNX vs original predictions (max diff: 0.000000)
- File size: 1.6 KB (very small, fast load in MT5)
```

#### STEP 6: Generate Charts (`scripts/06_generate_charts.py`)
```
Input:  configs/pipeline_results.json, data/test_predictions.csv, etc.
Output: charts/*.png (4 files)

1. feature_importance.png    — Feature ranking bar chart
2. confusion_matrix.png      — Confusion matrix heatmap
3. precision_recall_curve.png — P-R curve + optimal threshold
4. equity_curve.png          — Original vs NN equity comparison
```

#### STEP 7: Validation (`scripts/07_equity_simulation.py`)
```
Input:  data/test_predictions.csv, models/model_xgboost.pkl
Output: data/equity_curves.csv, docs/VALIDATION_REPORT.md

Equity simulation:
- Original EA:  $10,000 → $11,946 (19.5%, 606 trades, WR 51.3%)
- EA + NN:      $10,000 → $10,504 (5.0%,  29 trades,  WR 72.4%)
```

---

## Installation & Usage

### Prerequisites

- Python 3.10+
- MetaTrader 5 (for EA deployment)
- Git

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/Mrizalfahlepi/SR_Mapping_NN.git
cd SR_Mapping_NN

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline
python scripts/run_full_pipeline.py

# Or run step by step:
python scripts/01_download_data.py
python scripts/02_feature_engineering.py
python scripts/03_labeling.py
python scripts/04_train_model.py
python scripts/05_export_onnx.py
python scripts/06_generate_charts.py
python scripts/07_equity_simulation.py
```

### Retrain with Your Own Data

If you have XAUUSD data from your broker (Exness, etc.):

```bash
# 1. Save M30/H1 data to data/xauusd_custom.csv
#    Format: datetime,open,high,low,close,volume

# 2. Edit scripts/01_download_data.py:
#    Set DATA_SOURCE = "custom"
#    Set CUSTOM_FILE = "data/xauusd_custom.csv"

# 3. Run pipeline
python scripts/run_full_pipeline.py
```

---

## Deployment Guide to MT5

### Quick Steps

1. **Copy EA:** `mql5/SR_Mapping_NN_v1.mq5` → `MQL5/Experts/`
2. **Copy Model:** `models/sr_mapping_nn.onnx` → `MQL5/Files/`
3. **Compile** in MetaEditor (F7)
4. **Attach** to XAUUSD chart (M30 or H1)
5. **Set parameters** (see table below)

### Required Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `InpUseNNFilter` | `true` | Enable NN filter |
| `InpConfidenceThresh` | `0.51` | Confidence threshold |
| `InpUseStopLoss` | `true` | **REQUIRED** (grid removed) |
| `InpUseATR_SLTP` | `true` | ATR-based SL/TP |
| `InpSL_ATR_Mult` | `1.6` | SL = ATR x 1.6 |
| `InpTP_ATR_Mult` | `2.0` | TP = ATR x 2.0 |
| `InpUseSmartTrailing` | `true` | Smart trailing active |
| `InpMaxOpenTrades` | `1` | Max 1 position |

### Entry Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `InpRSI_Oversold` | `40` | RSI BUY threshold |
| `InpRSI_Overbought` | `70` | RSI SELL threshold |
| `InpATR_MinPoints` | `200` | Minimum ATR |
| `InpATR_MaxPoints` | `2500` | Maximum ATR |
| `InpSNR_Tolerance` | `50` | S/R tolerance (points) |
| `InpDailyRangeThresh` | `1000` | Daily range threshold |
| `InpTradingHourStart` | `2` | Trading start (hour) |
| `InpTradingHourEnd` | `18` | Trading end (hour) |

Full guide: [mql5/DEPLOYMENT_GUIDE.md](mql5/DEPLOYMENT_GUIDE.md)

---

## Technical Details

### EA Entry Logic (5 Simultaneous Conditions)

```
ALL conditions must be TRUE:
1. IsWithinTradingHours()  → 02:00-18:00 broker time
2. totalPositions < 1      → max 1 trade
3. IsMinBarsElapsed()      → min 5 hours since last trade
4. IsSpreadAcceptable()    → spread <= 400 points
5. IsATR_InRange()         → ATR within 200-2500 points

BUY if:
  - Daily Direction = +1 (bullish)
  - Price near Support (±50 points)
  - RSI <= 40

SELL if:
  - Daily Direction = -1 (bearish)
  - Price near Resistance (±50 points)
  - RSI >= 70
```

### 26 Model Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | `rsi` | RSI(14) |
| 2 | `atr` | ATR(14) in $ |
| 3 | `daily_range` | Close - Daily Open |
| 4 | `daily_direction` | +1 / 0 / -1 |
| 5 | `dist_res_norm` | Distance to Resistance / ATR |
| 6 | `dist_sup_norm` | Distance to Support / ATR |
| 7 | `sr_position` | Position within S/R range (0-1) |
| 8 | `hour` | Hour (0-23) |
| 9 | `dow` | Day (0=Mon, 4=Fri) |
| 10 | `ret_3` | Return 3 bars (%) |
| 11 | `ret_5` | Return 5 bars (%) |
| 12 | `ret_10` | Return 10 bars (%) |
| 13 | `ret_20` | Return 20 bars (%) |
| 14 | `atr_pctile` | ATR percentile (rolling 168) |
| 15 | `atr_change` | ATR change 5 bars (%) |
| 16 | `rsi_sma` | RSI SMA(10) |
| 17 | `rsi_slope` | RSI change 3 bars |
| 18 | `near_support` | 1 if near support |
| 19 | `near_resistance` | 1 if near resistance |
| 20 | `body_size` | Candle body / ATR |
| 21 | `upper_wick` | Upper wick / ATR |
| 22 | `lower_wick` | Lower wick / ATR |
| 23 | `is_bullish` | 1 if bullish candle |
| 24 | `bars_since_frac_up` | Bars since UP fractal |
| 25 | `bars_since_frac_down` | Bars since DOWN fractal |
| 26 | `vol_ratio` | Volume / SMA(20) volume |

### Williams Fractal (S/R Detection)

```python
# Fractal UP (Resistance):
# High[i] > High[i-1] AND High[i] > High[i-2]
# AND High[i] > High[i+1] AND High[i] > High[i+2]

# Fractal DOWN (Support):
# Low[i] < Low[i-1] AND Low[i] < Low[i-2]
# AND Low[i] < Low[i+1] AND Low[i] < Low[i+2]

# Carry forward: if no new fractal, use last known value
```

---

## Limitations & Important Notes

### Data

1. **Data source:** GC=F (Gold Futures) from yfinance, not XAUUSD spot from Exness
2. **Timeframe:** H1 (not M1/M30 like the original EA) due to public data limitations
3. **Period:** 14 months (Oct 2024 — Nov 2025), ideally needs 2-3 years
4. **Spread:** Estimated (not real spread from broker)

### Model

5. **Moderate AUC (0.538):** Model is not a super-predictor, but at high thresholds produces good precision
6. **Low recall (6.8%):** Many good signals are skipped — safety vs opportunity trade-off
7. **Overfitting risk:** Periodic retraining with new data is required

### Deployment

8. **MUST backtest** in Strategy Tester before going live
9. **MUST demo trade** for at least 1 month
10. **Retrain** recommended every 1-3 months with latest data
11. **Feature calibration:** M1 features in EA will differ from H1 training — retrain with Exness data required

### Priority Recommendations

| Priority | Action | Impact |
|----------|--------|--------|
| **P0** | Retrain with Exness M1/M30 data | Significant accuracy improvement |
| **P0** | Backtest + Demo 1 month | Validate before live trading |
| **P1** | Walk-forward cross-validation | Model stability |
| **P1** | Ensemble model (XGB + LightGBM + RF) | Robustness |
| **P2** | Feature expansion (session, news) | More market information |
| **P2** | LSTM/GRU as alternative | Capture temporal patterns |

---

## License

MIT License — See [LICENSE](LICENSE) for details.

**DISCLAIMER:** This project is for educational and research purposes. Forex/gold trading involves high risk. There is no guarantee of profit. Always use strict risk management and never trade with money you cannot afford to lose.

---

*Built with Python, XGBoost, ONNX, and MQL5*  
*By Muhamad Rizal Fahlepi*  
*This project is experimental — profit is not guaranteed.*  
*Happy Trading!*
