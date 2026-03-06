# Dokumentasi Logika EA — SR_Mapping_Foundation v7.1 + NN Filter

**File EA:** `SR_Mapping_NN_v1.mq5`  
**Versi:** SR_Mapping_NN v1.0 (berbasis SR_Mapping_Foundation v7.1)  
**Platform:** MetaTrader 5  
**Simbol Target:** XAUUSD  

---

## Daftar Isi

1. [Gambaran Umum Arsitektur](#1-gambaran-umum-arsitektur)
2. [Input Parameters](#2-input-parameters)
3. [BuildSNR — Pemetaan Support/Resistance](#3-buildsnr--pemetaan-supportresistance)
4. [GetRSI_M1_Raw — Indikator RSI](#4-getrsi_m1_raw--indikator-rsi)
5. [GetATR_M1_Points_Raw — Indikator ATR](#5-getatr_m1_points_raw--indikator-atr)
6. [DetectDailyRange — Deteksi Arah Harian](#6-detectdailyrange--deteksi-arah-harian)
7. [Kondisi Entry — 6 Filter Utama](#7-kondisi-entry--6-filter-utama)
8. [Kondisi BUY](#8-kondisi-buy)
9. [Kondisi SELL](#9-kondisi-sell)
10. [Smart Trailing Stop](#10-smart-trailing-stop)
11. [NN Filter — PrepareFeatures & PredictConfidence](#11-nn-filter--preparefeatures--predictconfidence)
12. [Eksekusi Order](#12-eksekusi-order)
13. [Perubahan dari Versi Sebelumnya](#13-perubahan-dari-versi-sebelumnya)
14. [Alur Lengkap OnTick()](#14-alur-lengkap-ontick)

---

## 1. Gambaran Umum Arsitektur

EA SR_Mapping_NN v1 adalah Expert Advisor berbasis dua lapisan keputusan:

```
┌─────────────────────────────────────────────────────────┐
│                  SR_Mapping_NN v1.0                      │
│                                                          │
│  Lapisan 1: SR_Mapping_Foundation v7.1 (Rule-based)     │
│  ─────────────────────────────────────────────────────  │
│  • BuildSNR()       — S/R via iFractals M30             │
│  • DetectDailyRange() — Arah tren harian (D1)           │
│  • GetRSI_M1()      — RSI(14) M1 oversold/overbought    │
│  • GetATR_M1_Points() — ATR(14) M1 filter volatilitas   │
│  • 6 Filter Pre-Entry (jam, posisi, spread, dll.)       │
│                                                          │
│  Lapisan 2: Neural Network Filter (ML-based)            │
│  ─────────────────────────────────────────────────────  │
│  • PrepareFeatures()    — Hitung 26 fitur real-time     │
│  • PredictConfidence()  — Inferensi ONNX XGBoost        │
│  • Threshold 0.51       — Terima jika confidence ≥ 0.51 │
└─────────────────────────────────────────────────────────┘
```

Sinyal **harus melewati KEDUA lapisan** sebelum order dikirim ke broker.

---

## 2. Input Parameters

### Grup: Entry Parameters

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `InpRSI_Period` | 14 | Periode RSI |
| `InpRSI_Oversold` | 40 | Threshold RSI oversold (syarat BUY) |
| `InpRSI_Overbought` | 70 | Threshold RSI overbought (syarat SELL) |
| `InpATR_Period` | 14 | Periode ATR |
| `InpATR_MinPoints` | 200 | ATR minimum dalam points (filter) |
| `InpATR_MaxPoints` | 2500 | ATR maksimum dalam points (filter) |
| `InpSNR_Tolerance` | 50 | Toleransi S/R dalam points |
| `InpDailyRangeThresh` | 1000 | Threshold daily range dalam points |
| `InpFractalLookback` | 2000 | Lookback fractal dalam bar M30 |
| `InpTradingHourStart` | 2 | Jam mulai trading (UTC) |
| `InpTradingHourEnd` | 18 | Jam akhir trading (UTC) |
| `InpMaxSpreadPoints` | 400 | Spread maksimum yang diperbolehkan (points) |
| `InpMinBarsElapsed` | 10 | Minimum bar M30 antara dua trade (= 5 jam) |

### Grup: Neural Network Filter

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `InpUseNNFilter` | true | Aktifkan/nonaktifkan filter NN |
| `InpConfidenceThresh` | 0.51 | Threshold confidence NN (0.0–1.0) |
| `InpOnnxModelFile` | `sr_mapping_nn.onnx` | Nama file model ONNX |

### Grup: Risk Management

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `InpLotSize` | 0.01 | Volume lot |
| `InpUseStopLoss` | true | Wajib aktifkan SL |
| `InpUseTakeProfit` | true | Aktifkan TP |
| `InpUseATR_SLTP` | true | Gunakan SL/TP berbasis ATR |
| `InpSL_ATR_Mult` | 1.6 | SL = ATR × 1.6 |
| `InpTP_ATR_Mult` | 2.0 | TP = ATR × 2.0 |
| `InpFixedSL_Points` | 5000 | SL fixed (jika tidak pakai ATR) |
| `InpFixedTP_Points` | 8000 | TP fixed (jika tidak pakai ATR) |
| `InpMaxOpenTrades` | 1 | Maksimum posisi terbuka sekaligus |

### Grup: Smart Trailing Stop

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `InpUseSmartTrailing` | true | Aktifkan smart trailing |
| `InpTrail_ATR_Mult` | 5.0 | Trigger trailing = ATR × 5.0 |
| `InpTrailStepPoints` | 300 | Langkah pergeseran SL trailing (points) |
| `InpMinSLMovement` | 50 | Minimum pergeseran SL (anti-spam, points) |

### Grup: EA Settings

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `InpMagicNumber` | 20250306 | Magic number order |
| `InpComment` | `SR_NN_v1` | Komentar order |

---

## 3. BuildSNR — Pemetaan Support/Resistance

**Fungsi:** `BuildSNR()`  
**Tujuan:** Menentukan level support (`g_currentSupport`) dan resistance (`g_currentResistance`) terdekat berdasarkan fraktal.

### Cara Kerja

```
1. Buat indikator iFractals pada timeframe M30
2. Lookback = min(InpFractalLookback, total_bar_M30) = min(2000, N)
3. Scan dari bar terbaru ke belakang:
   - Cari fractal ATAS pertama yang valid → g_currentResistance
   - Cari fractal BAWAH pertama yang valid → g_currentSupport
4. "Carry forward": nilai S/R tetap digunakan hingga fractal baru ditemukan
```

### Detail Implementasi

```mql5
// Indikator: iFractals(_Symbol, PERIOD_M30)
// Buffer 0: fractal atas  (UPPER)
// Buffer 1: fractal bawah (LOWER)
// Nilai valid: != EMPTY_VALUE && > 0
```

### Parameter Kunci

| Parameter | Nilai |
|-----------|-------|
| Timeframe fraktal | M30 |
| Lookback maksimum | 2.000 bar M30 |
| Logika | "Carry forward" — gunakan fraktal terakhir yang valid |

---

## 4. GetRSI_M1_Raw — Indikator RSI

**Fungsi:** `GetRSI_M1()` (disebut "GetRSI_M1_Raw" dalam konteks EA Foundation v7.1)  
**Tujuan:** Membaca nilai RSI current dari timeframe M1.

### Spesifikasi

| Parameter | Nilai |
|-----------|-------|
| Indikator | iRSI |
| Timeframe | M1 (PERIOD_M1) |
| Periode | 14 |
| Harga | PRICE_CLOSE |
| Oversold threshold | 40 (syarat BUY) |
| Overbought threshold | 70 (syarat SELL) |
| Nilai default (error) | 50.0 (netral) |

### Implementasi

```mql5
g_handleRSI_M1 = iRSI(_Symbol, PERIOD_M1, InpRSI_Period, PRICE_CLOSE);
// Dibuat saat OnInit(), digunakan sepanjang EA berjalan
// Dibersihkan di OnDeinit() via IndicatorRelease()
```

### Catatan

- RSI dihitung pada M1, bukan pada M30 atau H1, sehingga lebih responsif terhadap pergerakan harga jangka pendek
- Threshold yang digunakan (40/70) lebih longgar dari standar RSI (30/70) untuk menghasilkan lebih banyak sinyal

---

## 5. GetATR_M1_Points_Raw — Indikator ATR

**Fungsi:** `GetATR_M1_Points()` (disebut "GetATR_M1_Points_Raw" dalam konteks Foundation v7.1)  
**Tujuan:** Membaca nilai ATR dari M1 dan mengkonversinya ke satuan **points**.

### Spesifikasi

| Parameter | Nilai |
|-----------|-------|
| Indikator | iATR |
| Timeframe | M1 (PERIOD_M1) |
| Periode | 14 |
| Nilai minimum | 200 points (filter) |
| Nilai maksimum | 2.500 points (filter) |
| Konversi | ATR_nilai / g_point = ATR_points |

### Implementasi

```mql5
g_handleATR_M1 = iATR(_Symbol, PERIOD_M1, InpATR_Period);
// ATR dalam satuan harga → dibagi g_point → ATR dalam points
double atrPoints = atrBuffer[0] / g_point;
```

### Fungsi ATR dalam EA

ATR digunakan untuk:
1. **Filter volatilitas:** Hanya trade jika 200 ≤ ATR ≤ 2500 points
2. **Penentuan SL:** SL = ATR × 1.6 points dari entry
3. **Penentuan TP:** TP = ATR × 2.0 points dari entry
4. **Trailing trigger:** Trailing aktif jika profit > ATR × 5.0
5. **Feature NN:** Digunakan sebagai fitur `atr` untuk inferensi model

---

## 6. DetectDailyRange — Deteksi Arah Harian

**Fungsi:** `DetectDailyRange()`  
**Tujuan:** Menentukan arah bias harian sebelum membuka posisi.

### Logika

```
dailyOpen  = iOpen(_Symbol, PERIOD_D1, 0)   // Harga open hari ini
currentBid = SymbolInfoDouble(_Symbol, SYMBOL_BID)
rangePoints = (currentBid - dailyOpen) / g_point

Jika rangePoints >=  1000 → return  1  (BULLISH — bias beli)
Jika rangePoints <= -1000 → return -1  (BEARISH — bias jual)
Selainnya               → return  0  (TIDAK TRADE)
```

### Parameter

| Parameter | Nilai |
|-----------|-------|
| Threshold | 1.000 points (InpDailyRangeThresh) |
| Timeframe open | D1 |
| Harga referensi | Bid saat ini |

### Implikasi Trading

- `direction = +1` → Hanya buka posisi **BUY**
- `direction = -1` → Hanya buka posisi **SELL**
- `direction = 0`  → EA **tidak membuka posisi** (return dari OnTick)

---

## 7. Kondisi Entry — 6 Filter Utama

Sebelum sinyal BUY atau SELL dievaluasi, EA memeriksa **6 kondisi pre-entry** secara berurutan:

```
OnTick():
  ┌─ [1] IsWithinTradingHours()     → Jam 02:00–17:59 UTC
  ├─ [2] CountOpenPositions() < 1   → Maksimum 1 posisi terbuka
  ├─ [3] IsMinBarsElapsed()         → ≥ 10 bar M30 (300 menit) sejak trade terakhir
  ├─ [4] IsSpreadAcceptable()       → Spread ≤ 400 points
  ├─ [5] ATR in range               → 200 ≤ ATR_M1_points ≤ 2500
  └─ [6] DetectDailyRange() != 0   → Ada bias harian yang jelas
```

Jika salah satu kondisi tidak terpenuhi, `OnTick()` langsung `return` tanpa memproses lebih lanjut.

### Detail Setiap Filter

| # | Filter | Implementasi | Tujuan |
|---|--------|-------------|--------|
| 1 | Jam Trading | `dt.hour >= 2 && dt.hour < 18` | Hindari sesi pasar tipis |
| 2 | Posisi Terbuka | Loop semua posisi, filter symbol + magic | Cegah double-entry |
| 3 | Jeda Antar Trade | `elapsed >= InpMinBarsElapsed * 30 menit` | Hindari overtrading |
| 4 | Spread | `spread <= 400 points` | Hindari kondisi spread tinggi |
| 5 | ATR Range | `200 <= atrPoints <= 2500` | Hindari volatilitas terlalu rendah/tinggi |
| 6 | Daily Direction | `DetectDailyRange() != 0` | Hanya trade dengan tren harian |

---

## 8. Kondisi BUY

Setelah 6 filter pre-entry terpenuhi dan `direction == +1` (BULLISH):

```
Syarat BUY:
  1. dailyDirection == +1             (bias harian bullish)
  2. g_currentSupport > 0             (level support valid ditemukan)
  3. ask >= (support - 50 points)     (harga tidak terlalu jauh di bawah support)
  4. ask <= (support + 50 points)     (harga tidak terlalu jauh di atas support)
  5. RSI_M1 <= 40                     (kondisi oversold)
  6. NN confidence >= 0.51            (filter ML disetujui — jika InpUseNNFilter=true)
```

### Toleransi Harga

```
Zona BUY: [support - 50 pts] ─────── support ─────── [support + 50 pts]
```

Harga ASK harus berada dalam zona ±50 points dari level support terdekat.

### Parameter Entry BUY

```mql5
sl = ask - atrPoints * 1.6 * g_point   // Stop Loss
tp = ask + atrPoints * 2.0 * g_point   // Take Profit
```

---

## 9. Kondisi SELL

Setelah 6 filter pre-entry terpenuhi dan `direction == -1` (BEARISH):

```
Syarat SELL:
  1. dailyDirection == -1              (bias harian bearish)
  2. g_currentResistance > 0          (level resistance valid ditemukan)
  3. bid <= (resistance + 50 points)  (harga tidak terlalu jauh di atas resistance)
  4. bid >= (resistance - 50 points)  (harga tidak terlalu jauh di bawah resistance)
  5. RSI_M1 >= 70                     (kondisi overbought)
  6. NN confidence >= 0.51            (filter ML disetujui — jika InpUseNNFilter=true)
```

### Toleransi Harga

```
Zona SELL: [resistance - 50 pts] ─── resistance ─── [resistance + 50 pts]
```

Harga BID harus berada dalam zona ±50 points dari level resistance terdekat.

### Parameter Entry SELL

```mql5
sl = bid + atrPoints * 1.6 * g_point   // Stop Loss
tp = bid - atrPoints * 2.0 * g_point   // Take Profit
```

---

## 10. Smart Trailing Stop

**Fungsi:** `ManageSmartTrailing()`  
**Dipanggil:** Setiap tick, bahkan saat tidak ada sinyal entry baru.

### Logika Trailing

```
triggerDistance = entryATR * 5.0 * g_point   // Jarak trigger aktivasi
trailStep       = 300 * g_point               // Langkah pergeseran SL
minMove         = 50 * g_point                // Minimum pergeseran (anti-spam)
```

### Trailing BUY

```
Kondisi: bid - openPrice > triggerDistance
newSL = bid - 300 * g_point

Update SL jika:
  - newSL > currentSL + 50 points   (SL bergerak maju minimal 50 points)
  - newSL > openPrice               (SL tidak di bawah harga buka)
```

### Trailing SELL

```
Kondisi: openPrice - ask > triggerDistance
newSL = ask + 300 * g_point

Update SL jika:
  - newSL < currentSL - 50 points   (SL bergerak maju minimal 50 points)
  - newSL < openPrice               (SL tidak di atas harga buka)
```

### Anti-Spam Mechanism

Parameter `InpMinSLMovement = 50 points` memastikan SL hanya diperbarui jika ada pergeseran minimal 50 points. Ini mencegah request `OrderSend()` yang berlebihan saat harga bergerak kecil-kecil.

---

## 11. NN Filter — PrepareFeatures & PredictConfidence

### 11.1. PrepareFeatures()

**Tujuan:** Menghitung 26 fitur real-time untuk inferensi model ONNX.

Fitur dihitung dan dimasukkan ke array `float features[26]` dengan urutan yang **harus persis sama** dengan urutan training:

| Index | Nama Fitur | Sumber Data | Cara Hitung |
|-------|-----------|------------|-------------|
| f0 | `rsi` | RSI M1 bar 0 | `GetRSI_M1()` |
| f1 | `atr` | ATR M1 × g_point | `atrBuffer[0] / g_point * g_point` |
| f2 | `daily_range` | Bid − dailyOpen | `bid - iOpen(D1, 0)` |
| f3 | `daily_direction` | `DetectDailyRange()` | -1, 0, atau +1 |
| f4 | `dist_res_norm` | (resistance − bid) / ATR | Jarak ke resistance, ternormalisasi |
| f5 | `dist_sup_norm` | (bid − support) / ATR | Jarak ke support, ternormalisasi |
| f6 | `sr_position` | (bid − sup) / (res − sup) | Posisi relatif dalam rentang S/R |
| f7 | `hour` | `dt.hour` | Jam UTC saat ini |
| f8 | `dow` | `dt.day_of_week - 1` | 0=Senin, 4=Jumat |
| f9 | `ret_3` | (close[0]/close[3] − 1) × 100 | Return 3 bar M30 |
| f10 | `ret_5` | (close[0]/close[5] − 1) × 100 | Return 5 bar M30 |
| f11 | `ret_10` | (close[0]/close[10] − 1) × 100 | Return 10 bar M30 |
| f12 | `ret_20` | (close[0]/close[20] − 1) × 100 | Return 20 bar M30 |
| f13 | `atr_pctile` | Rank ATR di antara 168 bar terakhir | Persentil ATR |
| f14 | `atr_change` | (atr_now/atr_5bars_ago − 1) × 100 | Perubahan ATR % |
| f15 | `rsi_sma` | Rata-rata RSI 10 bar terakhir | SMA(RSI, 10) |
| f16 | `rsi_slope` | rsiArr[0] − rsiArr[3] | Kemiringan RSI 3 bar |
| f17 | `near_support` | `abs(distSup) <= ATR * 0.20` | 0 atau 1 |
| f18 | `near_resistance` | `abs(distRes) <= ATR * 0.20` | 0 atau 1 |
| f19 | `body_size` | `abs(close − open) / ATR` | Ukuran badan candle M30 |
| f20 | `upper_wick` | `(high − max(close,open)) / ATR` | Panjang ekor atas M30 |
| f21 | `lower_wick` | `(min(close,open) − low) / ATR` | Panjang ekor bawah M30 |
| f22 | `is_bullish` | `close > open ? 1 : 0` | 0 atau 1 |
| f23 | `bars_since_frac_up` | Scan iFractals M30 | Bar sejak fraktal naik terakhir |
| f24 | `bars_since_frac_down` | Scan iFractals M30 | Bar sejak fraktal turun terakhir |
| f25 | `vol_ratio` | `vol_now / avg(vol[-1:-20])` | Rasio volume vs rata-rata 20 bar |

### 11.2. PredictConfidence()

**Tujuan:** Menjalankan inferensi ONNX dan mengembalikan probabilitas kelas 1 (sinyal baik).

```
Input:  float inputData[26]      — fitur dari PrepareFeatures()
Output: double confidence        — probabilitas kelas 1 (0.0–1.0)

Proses ONNX:
  OnnxRun(handle, ONNX_NO_CONVERSION, inputData, predictedClass, probabilities)
  → probabilities[0] = P(kelas 0, sinyal buruk)
  → probabilities[1] = P(kelas 1, sinyal baik) ← ini yang digunakan
```

### 11.3. Logika Filter

```
if confidence >= InpConfidenceThresh (0.51):
  → ACCEPT: lanjut ke ExecuteBuy()/ExecuteSell()
  → Log: "NN FILTER: BUY signal ACCEPTED | Confidence: 0.XXXX"

if confidence < InpConfidenceThresh:
  → REJECT: return dari OnTick()
  → Log: "NN FILTER: BUY signal REJECTED | Confidence: 0.XXXX"
```

**Fallback:** Jika model ONNX gagal dimuat (`!g_onnxLoaded`), `PredictConfidence()` mengembalikan `1.0` — semua sinyal diterima (identik dengan EA tanpa filter).

---

## 12. Eksekusi Order

### ExecuteBuy()

```mql5
ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK)
sl  = ask - atrPoints * 1.6 * g_point
tp  = ask + atrPoints * 2.0 * g_point

ORDER_TYPE_BUY, volume=InpLotSize, deviation=30
magic=InpMagicNumber, comment="SR_NN_v1_BUY"
```

### ExecuteSell()

```mql5
bid = SymbolInfoDouble(_Symbol, SYMBOL_BID)
sl  = bid + atrPoints * 1.6 * g_point
tp  = bid - atrPoints * 2.0 * g_point

ORDER_TYPE_SELL, volume=InpLotSize, deviation=30
magic=InpMagicNumber, comment="SR_NN_v1_SELL"
```

Setelah order berhasil: `g_lastTradeTime = TimeCurrent()` dan `g_entryATR[0] = atrPoints` disimpan untuk keperluan trailing.

---

## 13. Perubahan dari Versi Sebelumnya

### Fitur yang DIHAPUS (dibanding versi sebelum v7.1/v_nn)

| Fitur | Parameter yang Dihapus | Alasan Penghapusan |
|-------|----------------------|-------------------|
| **Grid Trading** | `InpUseGrid`, `InpGridStepPoints`, `InpMartingaleFactor`, fungsi `ManageGrid()` | Memperbesar drawdown secara eksponensial, tidak kompatibel dengan strategi berbasis NN yang konservatif |
| **Martingale** | `InpMartingaleFactor` | Risiko blow-up akun saat rangkaian kerugian beruntun |
| **Basket Close** | `InpUseBasket`, fungsi `CheckBasketClose()`, fungsi `GetBasketProfit()` | Kompleksitas tidak diperlukan ketika max posisi = 1 |
| **Lot Cap** | `InpMaxLotCap` | Digantikan oleh single fixed lot size |

### Fitur yang DITAMBAHKAN

| Fitur | Parameter | Deskripsi |
|-------|-----------|-----------|
| **ONNX Inference** | `InpUseNNFilter`, `InpOnnxModelFile` | Muat dan jalankan model sr_mapping_nn.onnx |
| **Feature Preparation** | — | Fungsi `PrepareFeatures()` menghitung 26 fitur real-time |
| **Confidence Scoring** | `InpConfidenceThresh` | Fungsi `PredictConfidence()` menggunakan output probabilitas ONNX |
| **NN Enable/Disable** | `InpUseNNFilter = true/false` | Mudah dinonaktifkan untuk A/B testing |

---

## 14. Alur Lengkap OnTick()

```
OnTick() dipanggil setiap perubahan harga (tick)
│
├─► ManageSmartTrailing()          ← Selalu dijalankan (update trailing)
│
├─► IsWithinTradingHours()?        ← Filter 1: Jam 02–17 UTC
│   └── NO → return
│
├─► CountOpenPositions() >= 1?     ← Filter 2: Max 1 posisi
│   └── YES → return
│
├─► IsMinBarsElapsed()?            ← Filter 3: Jeda ≥ 300 menit
│   └── NO → return
│
├─► IsSpreadAcceptable()?          ← Filter 4: Spread ≤ 400 pts
│   └── NO → return
│
├─► ATR in [200, 2500] pts?        ← Filter 5: Range volatilitas
│   └── NO → return
│
├─► DetectDailyRange() != 0?       ← Filter 6: Ada bias harian
│   └── direction = 0 → return
│
├─► BuildSNR()                     ← Hitung level S/R dari fraktal M30
│
├─► Evaluasi BUY signal:
│   └── direction==+1 && ask near support ±50pts && RSI ≤ 40?
│       └── buySignal = true
│
├─► Evaluasi SELL signal:
│   └── direction==-1 && bid near resistance ±50pts && RSI ≥ 70?
│       └── sellSignal = true
│
├─► buySignal || sellSignal?
│   └── NO → return
│
├─► InpUseNNFilter == true?
│   ├─► PrepareFeatures(nnFeatures)  ← Hitung 26 fitur
│   ├─► PredictConfidence()          ← Inferensi ONNX
│   └─► confidence < 0.51?
│       └── YES → log "REJECTED" → return
│
└─► Execute Order:
    ├─► buySignal  → ExecuteBuy(atrPoints)
    └─► sellSignal → ExecuteSell(atrPoints)
```

---

*Dokumentasi ini dibuat berdasarkan kode sumber `SR_Mapping_NN_v1.mq5` di direktori `/home/user/workspace/sr_mapping_nn/`.*
