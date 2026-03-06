# Panduan Workflow — SR_Mapping_NN

**Versi:** 1.0  
**Tanggal:** 6 Maret 2026  
**Proyek:** SR_Mapping_NN — Pipeline lengkap dari data hingga deployment MT5  

---

## Daftar Isi

1. [Gambaran Umum Pipeline](#1-gambaran-umum-pipeline)
2. [Diagram Alur ASCII — Pipeline Lengkap](#2-diagram-alur-ascii--pipeline-lengkap)
3. [Step-by-Step: Setiap Tahap Pipeline](#3-step-by-step-setiap-tahap-pipeline)
4. [Workflow Retrain](#4-workflow-retrain)
5. [Workflow Deployment (Python ke MT5)](#5-workflow-deployment-python-ke-mt5)
6. [Troubleshooting Masalah Umum](#6-troubleshooting-masalah-umum)

---

## 1. Gambaran Umum Pipeline

Pipeline SR_Mapping_NN terdiri dari 6 tahap utama:

```
[DATA] → [FEATURE ENGINEERING] → [LABELING] → [TRAINING] → [ONNX EXPORT] → [DEPLOYMENT MT5]
```

| Tahap | Script Utama | Output |
|-------|-------------|--------|
| 1. Download Data | (manual / yfinance) | `xauusd_h1.csv`, `xauusd_d1.csv`, `xauusd_m30.csv` |
| 2. Feature Engineering | `pipeline_v3.py` | `xauusd_features.csv`, `training_data.csv` |
| 3. Labeling | `pipeline_v3.py` | `training_data.csv` (dengan kolom `label`) |
| 4. Model Training | `pipeline_v3.py` | `model_xgboost.pkl`, `model_xgboost.json` |
| 5. ONNX Export | `pipeline_v3.py` | `sr_mapping_nn.onnx`, `feature_config.json` |
| 6. Deployment MT5 | Manual | File ONNX di `MQL5/Files/` |

---

## 2. Diagram Alur ASCII — Pipeline Lengkap

```
╔══════════════════════════════════════════════════════════════════════╗
║            SR_MAPPING_NN — PIPELINE DIAGRAM                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌─────────────────┐                                                 ║
║  │  STEP 1: DATA   │                                                 ║
║  │  ACQUISITION    │                                                 ║
║  │                 │                                                 ║
║  │  yfinance GC=F  │                                                 ║
║  │  Oct24–Nov25    │                                                 ║
║  │  6711 bar H1    │                                                 ║
║  └────────┬────────┘                                                 ║
║           │                                                          ║
║           ▼                                                          ║
║  ┌─────────────────────────────┐                                     ║
║  │  STEP 2: FEATURE            │  Input:  xauusd_h1.csv             ║
║  │  ENGINEERING                │          xauusd_d1.csv             ║
║  │                             │          xauusd_m30.csv            ║
║  │  • RSI, ATR (M1 proxy)      │                                     ║
║  │  • S/R distances            │  Output: xauusd_features.csv       ║
║  │  • Daily range/direction    │          (26 kolom fitur)           ║
║  │  • Momentum returns         │                                     ║
║  │  • Candle patterns          │                                     ║
║  │  • Time features            │                                     ║
║  │  • Volume ratio             │                                     ║
║  └────────────┬────────────────┘                                     ║
║               │                                                      ║
║               ▼                                                      ║
║  ┌─────────────────────────────┐                                     ║
║  │  STEP 3: LABELING           │  Input:  xauusd_features.csv        ║
║  │  (Forward TP/SL sim.)       │                                     ║
║  │                             │  Method: Simulasi forward           ║
║  │  TP = ATR × 2.0             │          96 bar M30 (48h)           ║
║  │  SL = ATR × 1.6             │                                     ║
║  │  Timeout = 96 bar M30       │  Output: training_data.csv          ║
║  │                             │          (+ kolom label 0/1)        ║
║  │  label=1: TP hit first      │                                     ║
║  │  label=0: SL hit/timeout    │                                     ║
║  └────────────┬────────────────┘                                     ║
║               │                                                      ║
║               ▼                                                      ║
║  ┌─────────────────────────────────────────────────┐                 ║
║  │  STEP 4: MODEL TRAINING (XGBoost)               │                 ║
║  │                                                 │                 ║
║  │  Train:Val:Test = 2828:606:606 (chronological)  │                 ║
║  │                                                 │                 ║
║  │  3 Konfigurasi diuji:                           │                 ║
║  │  ┌──────────────┬──────────────┬─────────────┐  │                 ║
║  │  │ conservative │   balanced   │  standard   │  │                 ║
║  │  │ depth=3      │  depth=4     │  depth=6    │  │                 ║
║  │  │ lr=0.02      │  lr=0.01     │  lr=0.05    │  │                 ║
║  │  └──────┬───────┴──────────────┴─────────────┘  │                 ║
║  │         │ TERPILIH berdasarkan precision val      │                ║
║  │         ▼                                         │                ║
║  │  Best: conservative                              │                 ║
║  │  Val AUC: 0.550 | Test Precision: 72.4%          │                ║
║  └────────────────────┬────────────────────────────-┘                 ║
║                       │                                              ║
║                       ▼                                              ║
║  ┌─────────────────────────────┐                                     ║
║  │  STEP 5: ONNX EXPORT        │  Input:  model_xgboost.pkl         ║
║  │                             │                                     ║
║  │  XGBoost → ONNX via         │  Output: sr_mapping_nn.onnx        ║
║  │  skl2onnx / onnxmltools     │          feature_config.json       ║
║  │                             │          pipeline_results.json     ║
║  │  Input shape: [1, 26]       │                                     ║
║  │  Output: class + proba      │                                     ║
║  └────────────┬────────────────┘                                     ║
║               │                                                      ║
║               ▼                                                      ║
║  ┌─────────────────────────────┐                                     ║
║  │  STEP 6: DEPLOYMENT MT5     │  Input:  sr_mapping_nn.onnx        ║
║  │                             │                                     ║
║  │  Copy .onnx ke              │  Target: MT5 Data Folder           ║
║  │  MQL5/Files/                │          MQL5/Files/               ║
║  │                             │                                     ║
║  │  Compile SR_Mapping_NN_v1   │  Output: EA berjalan dengan        ║
║  │  Attach ke chart XAUUSD     │          filter NN aktif           ║
║  └─────────────────────────────┘                                     ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 3. Step-by-Step: Setiap Tahap Pipeline

---

### STEP 1: Akuisisi Data

**Apa yang dilakukan:** Download data historis OHLCV dari yfinance untuk simbol GC=F (Gold Futures).

**File Input:** Tidak ada (sumber eksternal)

**File Output:**
- `xauusd_h1.csv` — Data H1 (bar per jam)
- `xauusd_d1.csv` — Data D1 (bar harian)
- `xauusd_m30.csv` — Data M30 (disintesis dari H1)

**Cara Menjalankan:**

```python
import yfinance as yf
import pandas as pd

# Download data GC=F H1
ticker = yf.Ticker("GC=F")
h1 = ticker.history(period="14mo", interval="1h")
h1.columns = h1.columns.str.lower()
h1.to_csv("xauusd_h1.csv")

# Download D1
d1 = ticker.history(period="14mo", interval="1d")
d1.columns = d1.columns.str.lower()
d1.to_csv("xauusd_d1.csv")
```

**Catatan Penting:**
- yfinance memiliki batas historis untuk data intraday (~60 hari untuk 1m, ~730 hari untuk 1h)
- Untuk data lebih panjang, gunakan `start` dan `end` parameter atau sumber data lain (MT5 export, broker API)
- Kolom wajib: `open`, `high`, `low`, `close`, `volume` (lowercase)

---

### STEP 2: Feature Engineering

**Apa yang dilakukan:** Menghitung 26 fitur untuk setiap bar M30 berdasarkan logika EA SR_Mapping_Foundation v7.1.

**File Input:**
- `xauusd_h1.csv`
- `xauusd_d1.csv`
- `xauusd_m30.csv`

**File Output:**
- `xauusd_features.csv` — Dataset lengkap dengan semua 26 fitur
- `training_data.csv` — Subset yang hanya berisi bar dengan sinyal valid

**Cara Menjalankan:**

```bash
cd /home/user/workspace/sr_mapping_nn/
python pipeline_v3.py
```

Step ini dijalankan otomatis sebagai bagian dari `pipeline_v3.py` (tidak ada script terpisah).

**Proses Utama:**
1. Hitung fraktal (simulasi `iFractals` M30, lookback 2000 bar)
2. Hitung RSI(14) dan ATR(14)
3. Hitung semua 26 fitur per bar
4. Filter hanya bar yang memenuhi kondisi entry EA (jam trading, daily direction, dsb.)

---

### STEP 3: Labeling

**Apa yang dilakukan:** Simulasi forward-looking untuk menentukan apakah setiap sinyal entry menghasilkan profit (label=1) atau loss (label=0).

**File Input:** `xauusd_features.csv`

**File Output:** `training_data.csv` (dengan kolom `label`)

**Cara Menjalankan:** Bagian dari `pipeline_v3.py` (otomatis setelah STEP 2)

**Logika Labeling:**

```
Untuk setiap bar dengan sinyal entry:
  Jika direction == +1 (BUY):
    tp = close + ATR * 2.0
    sl = close - ATR * 1.6
    
  Jika direction == -1 (SELL):
    tp = close - ATR * 2.0
    sl = close + ATR * 1.6
  
  Scan 96 bar M30 ke depan (48 jam):
    - Jika high[i] >= tp_buy  → label = 1 (WIN)
    - Jika low[i]  <= sl_buy  → label = 0 (LOSS)
    - Jika 96 bar habis       → label = 0 (TIMEOUT)
```

**Distribusi Label:**
- Train: 42.5% positif (label=1)
- Val: 51.8% positif
- Test: 51.3% positif

---

### STEP 4: Model Training

**Apa yang dilakukan:** Melatih model XGBoost dengan 3 konfigurasi dan memilih konfigurasi dengan precision terbaik pada validation set.

**File Input:** `training_data.csv`

**File Output:**
- `model_xgboost.pkl` — Model XGBoost tersimpan (format pickle)
- `model_xgboost.json` — Model XGBoost tersimpan (format JSON, backup)
- `pipeline_results.json` — Metrik dan hasil evaluasi
- `confusion_matrix.png` — Visualisasi confusion matrix
- `feature_importance.png` — Grafik importance fitur
- `equity_curves.csv` — Perbandingan ekuitas EA original vs EA+NN
- `equity_curve.png` — Grafik kurva ekuitas
- `threshold_analysis.csv` — Analisis berbagai threshold

**Cara Menjalankan:** Bagian dari `pipeline_v3.py`

**Konfigurasi yang Diuji:**

```python
configs = [
    {'name': 'balanced',     'params': {'n_estimators': 1000, 'max_depth': 4, 'learning_rate': 0.01, ...}},
    {'name': 'conservative', 'params': {'n_estimators': 800,  'max_depth': 3, 'learning_rate': 0.02, ...}},
    {'name': 'standard',     'params': {'n_estimators': 500,  'max_depth': 6, 'learning_rate': 0.05, ...}},
]
```

Konfigurasi **conservative** terpilih karena menghasilkan precision tertinggi.

---

### STEP 5: ONNX Export

**Apa yang dilakukan:** Mengkonversi model XGBoost ke format ONNX agar dapat dimuat oleh MetaTrader 5.

**File Input:** `model_xgboost.pkl`

**File Output:**
- `sr_mapping_nn.onnx` — Model dalam format ONNX
- `feature_config.json` — Konfigurasi threshold, normalisasi, dan mapping fitur

**Cara Menjalankan:** Bagian dari `pipeline_v3.py`

**Dependensi Python:**

```bash
pip install skl2onnx onnxmltools onnxruntime xgboost
```

**Validasi ONNX:**

```python
import onnxruntime as rt
import numpy as np

sess = rt.InferenceSession("sr_mapping_nn.onnx")
dummy_input = np.zeros((1, 26), dtype=np.float32)
result = sess.run(None, {"f0": dummy_input})
print("Class:", result[0])
print("Proba:", result[1])
# Proba[0][1] = probabilitas kelas 1 (sinyal baik)
```

---

### STEP 6: Deployment ke MT5

**Apa yang dilakukan:** Menyalin file ONNX ke direktori MT5 dan mengkonfigurasi EA.

**File Input:** `sr_mapping_nn.onnx`

**Cara Menjalankan:**

```
1. Buka MetaTrader 5
2. Tekan [File] → [Open Data Folder]
3. Navigasi ke: MQL5\Files\
4. Salin sr_mapping_nn.onnx ke folder tersebut
5. Di MetaEditor: Buka dan Compile SR_Mapping_NN_v1.mq5
6. Drag EA ke chart XAUUSD (timeframe apapun, EA menggunakan M1/M30/D1 secara internal)
7. Atur parameters:
   - InpUseNNFilter = true
   - InpConfidenceThresh = 0.51
   - InpOnnxModelFile = sr_mapping_nn.onnx
   - InpLotSize = 0.01 (mulai kecil)
```

**Verifikasi di Jurnal MT5:**

```
Setelah EA di-attach, cek tab "Experts" atau "Journal":
✓ "SR_Mapping_NN v1 initialized successfully"
✓ "NN Filter: ENABLED"
✓ "Confidence Threshold: 0.51"
✓ "ONNX model loaded successfully: sr_mapping_nn.onnx"

Jika muncul:
✗ "WARNING: Could not load ONNX model"
→ Cek apakah file .onnx sudah ada di MQL5\Files\
→ Pastikan nama file persis sama (case-sensitive di beberapa OS)
```

---

## 4. Workflow Retrain

### Kapan Harus Retrain?

Pertimbangkan untuk retrain model jika salah satu kondisi berikut terpenuhi:

| Kondisi | Indikator |
|---------|-----------|
| Win rate turun | < 55% selama 30 hari berturut-turut (live/demo) |
| Perubahan kondisi pasar | Volatilitas struktural berubah (contoh: Federal Reserve hawkish/dovish extreme) |
| Data baru tersedia | Setiap 3 bulan, tambahkan data terbaru |
| Pindah broker | Dari yfinance/data lama ke data Exness XAUUSD |
| Model overfitting | AUC validation << AUC training selama periode baru |

### Prosedur Retrain

```
RETRAIN WORKFLOW
────────────────

  ┌──────────────────────────────────┐
  │  1. PERSIAPKAN DATA BARU         │
  │     • Download data terbaru      │
  │     • Gabungkan dengan data lama │
  │     • Validasi tidak ada gap     │
  └─────────────┬────────────────────┘
                │
                ▼
  ┌──────────────────────────────────┐
  │  2. JALANKAN PIPELINE            │
  │     python pipeline_v3.py        │
  │     (otomatis semua step)        │
  └─────────────┬────────────────────┘
                │
                ▼
  ┌──────────────────────────────────┐
  │  3. EVALUASI HASIL               │
  │     • Cek pipeline_results.json  │
  │     • Bandingkan AUC baru vs lama│
  │     • Bandingkan precision test  │
  └─────────────┬────────────────────┘
                │
                ▼
  ┌──────────────────────────────────┐
  │  4. BACKUP FILE LAMA             │
  │     • Rename sr_mapping_nn.onnx  │
  │       → sr_mapping_nn_v1_old.onnx│
  │     • Backup feature_config.json │
  └─────────────┬────────────────────┘
                │
                ▼
  ┌──────────────────────────────────┐
  │  5. DEPLOY MODEL BARU            │
  │     • Salin .onnx baru ke MT5    │
  │     • Update threshold jika perlu│
  │     • Uji di akun demo dulu      │
  └──────────────────────────────────┘
```

### Perintah Retrain Lengkap

```bash
# 1. Aktifkan environment Python
conda activate trading  # atau: source venv/bin/activate

# 2. Install dependensi (pertama kali)
pip install xgboost scikit-learn pandas numpy yfinance skl2onnx onnxruntime

# 3. Update data (opsional — edit bagian download di pipeline)
# Edit pipeline_v3.py bagian STEP 1 jika perlu update data

# 4. Jalankan pipeline penuh
cd /home/user/workspace/sr_mapping_nn/
python pipeline_v3.py

# 5. Periksa output
cat pipeline_results.json | python -m json.tool | grep -E "test_auc|test_precision|n_signals"
```

---

## 5. Workflow Deployment (Dari Python ke MT5)

### Diagram Alur Deployment

```
DEPLOYMENT WORKFLOW
───────────────────

  [Python Workspace]                    [MetaTrader 5]
  ─────────────────                     ───────────────

  sr_mapping_nn.onnx  ──── copy ────►  MQL5\Files\sr_mapping_nn.onnx
  
  SR_Mapping_NN_v1.mq5 ─── copy ───►  MQL5\Experts\SR_Mapping_NN_v1.mq5
                                              │
                                              ▼
                                       [MetaEditor]
                                       Compile (.ex5)
                                              │
                                              ▼
                                       [MetaTrader 5]
                                       Navigator → Expert Advisors
                                       SR_Mapping_NN_v1
                                              │
                                              ▼
                                       Drag ke chart XAUUSD
                                              │
                                              ▼
                                       ┌─────────────────┐
                                       │  MODE DEMO DULU │
                                       │  (minimal 1 bln)│
                                       └────────┬────────┘
                                                │
                                           Evaluasi ✓
                                                │
                                                ▼
                                       ┌─────────────────┐
                                       │   LIVE TRADING  │
                                       │  (lot kecil dulu│
                                       │   0.01)         │
                                       └─────────────────┘
```

### Checklist Deployment

```
□ sr_mapping_nn.onnx ada di: MetaTrader5\MQL5\Files\
□ SR_Mapping_NN_v1.mq5 berhasil dikompilasi (0 errors, 0 warnings)
□ EA ter-attach di chart XAUUSD
□ Jurnal MT5 menampilkan "ONNX model loaded successfully"
□ Auto Trading diaktifkan (tombol hijau di toolbar MT5)
□ Mode Demo aktif sebelum live
□ InpLotSize = 0.01 (mulai konservatif)
□ InpUseStopLoss = true (WAJIB)
□ Magic number dicatat (20250306)
```

---

## 6. Troubleshooting Masalah Umum

### Masalah 1: ONNX Model Gagal Dimuat

**Gejala:**
```
WARNING: Could not load ONNX model 'sr_mapping_nn.onnx'
NN filter will be DISABLED. Error: 5100
```

**Solusi:**
```
1. Verifikasi lokasi file:
   Buka MetaTrader 5 → File → Open Data Folder
   Cek: [DataFolder]\MQL5\Files\sr_mapping_nn.onnx

2. Periksa nama file:
   InpOnnxModelFile harus PERSIS sama dengan nama file
   (case-sensitive: sr_mapping_nn.onnx bukan SR_Mapping_NN.onnx)

3. Periksa integritas file:
   File .onnx tidak boleh corrupt. Ukuran normal: ~500KB–5MB
   python -c "import onnxruntime; sess = onnxruntime.InferenceSession('sr_mapping_nn.onnx'); print('OK')"

4. Coba path lengkap:
   Salin file ke: [DataFolder]\MQL5\Files\ (bukan subfolder)
```

---

### Masalah 2: Pipeline Python Error saat Feature Engineering

**Gejala:**
```
KeyError: 'close' 
IndexError: index out of bounds
```

**Solusi:**
```
1. Pastikan kolom CSV lowercase:
   df.columns = df.columns.str.lower()
   Kolom wajib: open, high, low, close, volume

2. Periksa apakah ada gap data:
   print(df.index.is_monotonic_increasing)  # Harus True
   print(df.isnull().sum())                 # Cek NaN per kolom

3. Periksa format tanggal/index:
   df.index = pd.to_datetime(df.index)
   df = df.sort_index()
```

---

### Masalah 3: Tidak Ada Sinyal Entry Muncul

**Gejala:** EA berjalan tapi tidak ada trade sama sekali.

**Solusi — Cek secara berurutan:**

```
□ Jam trading: Apakah chart MT5 berada di jam 02:00–17:59 server time?
  → Cek InpTradingHourStart/End

□ Daily direction: Apakah pergerakan dari daily open sudah > 1000 points?
  → Di XAUUSD 5-digit, 1000 points = $1.00
  → Gold perlu bergerak > $1.00 dari open hari ini

□ Spread: Spread saat ini > 400 points?
  → Di luar jam ramai, spread bisa tinggi
  → Coba longgarkan InpMaxSpreadPoints = 600

□ ATR terlalu rendah/tinggi:
  → ATR M1 harus 200–2500 points
  → Periksa dengan script: Print("ATR: ", GetATR_M1_Points())
  
□ S/R tidak ditemukan:
  → BuildSNR() bisa gagal jika tidak ada fraktal valid dalam 2000 bar
  → Cek jurnal: g_currentSupport dan g_currentResistance != 0

□ NN terlalu ketat:
  → Coba turunkan InpConfidenceThresh ke 0.50 sementara untuk tes
  → Atau set InpUseNNFilter = false untuk tes rule-based saja
```

---

### Masalah 4: Win Rate Jauh di Bawah 72.4% dalam Live Trading

**Gejala:** Win rate real-time << 72.4% yang ada di laporan validasi.

**Analisis:**

```
Kemungkinan penyebab:
1. Data drift — kondisi pasar berubah sejak periode training (okt24-nov25)
2. Perbedaan broker — GC=F yfinance vs Exness XAUUSD (spread, point size)
3. Overfitting threshold — 0.51 dipilih dari test set, mungkin tidak generalize
4. Slippage — simulasi tidak memperhitungkan slippage dan requote
```

**Solusi:**
```
Jangka pendek:
  → Catat semua trade (confidence, fitur, hasil) ke CSV menggunakan Print() di EA
  → Monitor selama minimal 50–100 trade sebelum membuat kesimpulan

Jangka panjang:
  → Retrain dengan data Exness XAUUSD langsung dari MT5 History
  → Lakukan walk-forward validation sebelum live
```

---

### Masalah 5: Error ONNX saat OnnxRun()

**Gejala:**
```
WARNING: ONNX inference failed. Error: XXXX
```

**Solusi:**
```
1. Verifikasi input shape:
   Model mengharapkan input [1, 26] bertipe float32
   Jika ArraySize(inputData) != 26 → ada bug di PrepareFeatures()

2. Periksa ONNX input name:
   Model mungkin menggunakan nama input berbeda dari "f0"
   Gunakan Python untuk cek:
   import onnxruntime as rt
   sess = rt.InferenceSession("sr_mapping_nn.onnx")
   print([i.name for i in sess.get_inputs()])

3. Cek versi MT5:
   ONNX runtime di MT5 memerlukan build MT5 >= 3370
   Pastikan MetaTrader 5 sudah diupdate ke versi terbaru

4. Fallback behavior:
   Jika ONNX gagal, PredictConfidence() mengembalikan 0.5 (uncertain)
   Sinyal akan DITOLAK jika confidence < 0.51 → tidak ada trade
```

---

### Masalah 6: pipeline_v3.py Gagal Konversi ke ONNX

**Gejala:**
```
ERROR: Cannot convert XGBoost model to ONNX
ImportError: No module named 'skl2onnx'
```

**Solusi:**
```bash
# Install semua dependensi ONNX
pip install skl2onnx onnxmltools onnx onnxruntime

# Jika versi bertabrakan:
pip install --upgrade skl2onnx onnxmltools

# Verifikasi konversi:
python -c "
import xgboost as xgb
import pickle
from skl2onnx import convert_sklearn
print('skl2onnx OK')
"

# Alternatif jika konversi gagal — gunakan model JSON langsung:
# XGBoost memiliki ONNX export eksperimental built-in:
model.save_model('model_xgboost.json')
# Kemudian gunakan onnxmltools untuk convert dari JSON
```

---

### Ringkasan Quick Reference

| Masalah | Cek Pertama | Solusi Cepat |
|---------|------------|-------------|
| ONNX tidak termuat | Lokasi file di `MQL5\Files\` | Salin ulang file .onnx |
| Tidak ada sinyal | Daily direction = 0 | Tunggu pergerakan > 1000 pts dari open |
| Win rate rendah | Data drift / broker mismatch | Retrain dengan data Exness |
| ONNX inference error | Input shape [1, 26] | Verifikasi `PrepareFeatures()` |
| Pipeline Python error | Kolom CSV lowercase | `df.columns = df.columns.str.lower()` |
| Banyak false positive | Threshold terlalu rendah | Naikkan `InpConfidenceThresh` ke 0.52+ |
| Terlalu sedikit trade | Threshold terlalu tinggi | Turunkan ke 0.50 atau nonaktifkan NN filter |

---

*Dokumentasi ini dibuat berdasarkan kode sumber `pipeline_v3.py`, `SR_Mapping_NN_v1.mq5`, `feature_config.json`, dan `pipeline_results.json` di direktori `/home/user/workspace/sr_mapping_nn/`.*
