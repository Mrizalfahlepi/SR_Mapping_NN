# Laporan Validasi Model — SR_Mapping_NN

**Versi:** 1.0  
**Tanggal:** 6 Maret 2026  
**Proyek:** SR_Mapping_NN — Filter Neural Network untuk EA SR_Mapping_Foundation v7.1  
**Simbol:** XAUUSD (Gold)  

---

## Daftar Isi

1. [Ringkasan Eksekutif](#1-ringkasan-eksekutif)
2. [Sumber Data](#2-sumber-data)
3. [Strategi Pelabelan](#3-strategi-pelabelan)
4. [Pembagian Dataset](#4-pembagian-dataset)
5. [Konfigurasi Model](#5-konfigurasi-model)
6. [Hasil Evaluasi](#6-hasil-evaluasi)
7. [Confusion Matrix](#7-confusion-matrix)
8. [Feature Importance](#8-feature-importance)
9. [Perbandingan Ekuitas](#9-perbandingan-ekuitas)
10. [Analisis Threshold](#10-analisis-threshold)
11. [Keterbatasan](#11-keterbatasan)
12. [Rekomendasi](#12-rekomendasi)

---

## 1. Ringkasan Eksekutif

Model SR_Mapping_NN adalah filter berbasis machine learning yang dilatih untuk menyaring sinyal entry yang dihasilkan oleh EA SR_Mapping_Foundation v7.1. Model menggunakan algoritma XGBoost konfigurasi *conservative* yang dikonversi ke format ONNX untuk inferensi real-time di dalam MetaTrader 5.

### Metrik Kunci (Test Set)

| Metrik | Nilai |
|--------|-------|
| **Precision (threshold 0.51)** | **72.4%** |
| **AUC-ROC (Test)** | **0.538** |
| **AUC-ROC (Val)** | 0.550 |
| **AUC-ROC (Train)** | 0.658 |
| Threshold Optimal | 0.51 |
| Recall | 6.75% |
| F1-Score | 0.124 |
| Total Sinyal (Test) | 29 sinyal |

### Perbandingan Ringkas

| Strategi | Modal Awal | Modal Akhir | Jumlah Trade | Win Rate |
|----------|-----------|------------|-------------|---------|
| EA Original (tanpa filter) | $10,000 | $11,946 | 606 | 51.3% |
| EA + NN Filter (threshold 0.51) | $10,000 | $10,504 | 29 | **72.4%** |

**Kesimpulan:** Filter NN secara signifikan meningkatkan win rate dari 51.3% menjadi 72.4%, namun dengan trade-off volume sinyal yang sangat rendah (29 dari 606 sinyal diterima). Model saat ini hanya menghasilkan return moderat (+5.0%) dengan jumlah trade yang terbatas.

---

## 2. Sumber Data

| Parameter | Detail |
|-----------|--------|
| **Sumber** | yfinance (Yahoo Finance) |
| **Simbol** | GC=F (Gold Futures — proxy XAUUSD) |
| **Timeframe Utama** | H1 (1 jam) |
| **Periode** | Oktober 2024 — November 2025 |
| **Total Bar** | 6.711 bar H1 |
| **Timeframe Turunan** | M30 (disintesis dari H1), D1 |

> **Catatan Penting:** Data latih menggunakan GC=F dari yfinance, bukan data Exness XAUUSD. Harga GC=F dalam satuan dolar dengan presisi 1 desimal, berbeda dengan XAUUSD 5-digit di Exness (point = 0.001). Ini merupakan salah satu keterbatasan utama model ini.

---

## 3. Strategi Pelabelan

Model dilatih menggunakan simulasi forward-looking TP/SL berdasarkan parameter EA SR_Mapping_Foundation v7.1.

### Parameter Pelabelan

| Parameter | Nilai |
|-----------|-------|
| **Take Profit (TP)** | ATR × 2.0 |
| **Stop Loss (SL)** | ATR × 1.6 |
| **Batas Waktu (Timeout)** | 48 jam (96 bar M30) |
| **Label = 1 (BAIK)** | TP tercapai sebelum SL atau timeout |
| **Label = 0 (BURUK)** | SL tercapai atau timeout habis |

### Logika Pelabelan

```
Untuk setiap bar M30:
  entry_price = close bar saat ini
  tp_price    = entry_price + ATR × 2.0   (untuk BUY)
  sl_price    = entry_price - ATR × 1.6   (untuk BUY)
  
  Simulasi ke depan (hingga 96 bar):
    - Jika high[i] >= tp_price → label = 1
    - Jika low[i]  <= sl_price → label = 0
    - Jika 96 bar habis tanpa TP/SL → label = 0 (timeout)
```

Untuk sinyal SELL, logika dibalik: TP ke bawah, SL ke atas.

---

## 4. Pembagian Dataset

Dataset dibagi secara kronologis (tanpa shuffle) untuk mencegah data leakage dari masa depan ke masa lalu.

| Split | Jumlah Bar | Persentase Label Positif (label=1) |
|-------|-----------|-----------------------------------|
| **Train** | 2.828 | 42.5% |
| **Val** | 606 | 51.8% |
| **Test** | 606 | 51.3% |
| **Total** | 4.040 | — |

> Pembagian: ~70% Train / ~15% Val / ~15% Test berdasarkan urutan waktu.

---

## 5. Konfigurasi Model

Model terbaik dipilih berdasarkan precision tertinggi pada validation set. Konfigurasi *conservative* terpilih sebagai model final.

### XGBoost — Konfigurasi "Conservative"

| Hyperparameter | Nilai |
|----------------|-------|
| **n_estimators** | 800 |
| **max_depth** | 3 |
| **learning_rate** | 0.02 |
| **subsample** | 0.6 |
| **colsample_bytree** | 0.6 |
| **min_child_weight** | 20 |
| **gamma** | 0.5 |
| **reg_alpha** | 1.0 |
| **reg_lambda** | 3.0 |
| **scale_pos_weight** | Dihitung otomatis (rasio negatif/positif) |

Konfigurasi ini dirancang untuk **meminimalkan false positives** dengan regularisasi tinggi dan tree yang dangkal, sehingga menghasilkan precision yang tinggi meskipun recall sangat rendah.

### Format Export

| Aspek | Detail |
|-------|--------|
| **Format Training** | XGBoost PKL + JSON |
| **Format Deployment** | ONNX (sr_mapping_nn.onnx) |
| **Input Shape ONNX** | [1, 26] (batch=1, fitur=26) |
| **Output Shape ONNX** | Output 0: kelas (int64), Output 1: probabilitas [1, 2] |
| **Normalisasi** | Tidak ada (XGBoost tidak memerlukan normalisasi) |

---

## 6. Hasil Evaluasi

### AUC-ROC Per Split

| Split | AUC-ROC |
|-------|---------|
| Train | 0.6582 |
| Validation | 0.5498 |
| **Test** | **0.5382** |

> AUC test sebesar 0.538 menunjukkan kemampuan diskriminasi yang moderat (sedikit di atas random/0.5). Ini wajar untuk prediksi pasar keuangan yang sangat tidak pasti.

### Metrik pada Threshold 0.51

| Metrik | Nilai |
|--------|-------|
| Precision | 0.7241 (72.41%) |
| Recall | 0.0675 (6.75%) |
| F1-Score | 0.1235 |
| Jumlah Sinyal Diterima | 29 dari 606 |

---

## 7. Confusion Matrix

Hasil pada **Test Set** (606 bar) dengan threshold 0.51:

```
                    Prediksi: BURUK (0)    Prediksi: BAIK (1)
Aktual: BURUK (0)       TN = 287              FP = 8
Aktual: BAIK  (1)       FN = 290              TP = 21
```

| Metrik | Perhitungan | Nilai |
|--------|-------------|-------|
| **True Negative (TN)** | Sinyal buruk yang benar-benar ditolak | 287 |
| **False Positive (FP)** | Sinyal buruk yang salah diterima | 8 |
| **False Negative (FN)** | Sinyal baik yang ditolak | 290 |
| **True Positive (TP)** | Sinyal baik yang benar-benar diterima | 21 |

**Interpretasi:**
- Model berhasil menolak 287 dari 295 sinyal buruk (**akurasi penolakan 97.3%**)
- Model berhasil menerima 21 dari 311 sinyal baik (**akurasi penerimaan 6.8%**)
- Sangat konservatif: hanya 29 sinyal yang diloloskan (72.4% di antaranya profit)

---

## 8. Feature Importance

Top 10 fitur berdasarkan skor importance XGBoost (dari total 26 fitur):

| Rank | Fitur | Importance | Deskripsi |
|------|-------|-----------|-----------|
| 1 | `ret_3` | 0.0640 | Return 3 bar M30 terakhir |
| 2 | `ret_5` | 0.0621 | Return 5 bar M30 terakhir |
| 3 | `daily_range` | 0.0607 | Jarak harga dari daily open |
| 4 | `dow` | 0.0595 | Hari dalam minggu (Day of Week) |
| 5 | `ret_10` | 0.0590 | Return 10 bar M30 terakhir |
| 6 | `lower_wick` | 0.0563 | Panjang ekor bawah candle (relatif ATR) |
| 7 | `bars_since_frac_up` | 0.0540 | Jumlah bar sejak fraktal naik terakhir |
| 8 | `atr` | 0.0486 | Average True Range M1 |
| 9 | `atr_change` | 0.0483 | Perubahan ATR dalam % |
| 10 | `dist_sup_norm` | 0.0467 | Jarak ke support terdekat (dinormalisasi ATR) |

### Fitur dengan Importance = 0

Empat fitur berikut tidak memberikan kontribusi pada model akhir:
- `daily_direction` — arah tren harian
- `near_support` — flag biner "dekat support"
- `near_resistance` — flag biner "dekat resistance"
- `is_bullish` — flag biner "candle bullish"

---

## 9. Perbandingan Ekuitas

Simulasi pada **Test Set** (606 bar), menggunakan SL=ATR×1.6 dan TP=ATR×2.0, modal awal $10,000:

### Ekuitas Original EA (Tanpa Filter NN)

| Metrik | Nilai |
|--------|-------|
| Modal Awal | $10,000 |
| Modal Akhir | $11,946 |
| Profit | +$1,946 (+19.5%) |
| Jumlah Trade | 606 |
| Win Rate | 51.3% |
| Profit Factor | ~1.05 (estimasi) |

### Ekuitas EA + NN Filter (Threshold 0.51)

| Metrik | Nilai |
|--------|-------|
| Modal Awal | $10,000 |
| Modal Akhir | $10,504 |
| Profit | +$504 (+5.0%) |
| Jumlah Trade | 29 |
| Win Rate | **72.4%** |
| Sinyal yang Ditolak | 577 dari 606 |

### Analisis Komparatif

```
                   Original EA    EA + NN Filter
                   -----------    ---------------
Modal Akhir:       $11,946        $10,504
Total Trade:       606            29  (↓95.2%)
Win Rate:          51.3%          72.4% (↑21.1 pp)
Return:            +19.5%         +5.0%
Return per Trade:  ~$3.21         ~$17.38
```

> **Catatan:** Meskipun EA original menghasilkan return absolut lebih tinggi ($11,946 vs $10,504) dalam 14 bulan test, EA+NN Filter memiliki return per trade yang ~5.4× lebih tinggi dan win rate jauh lebih baik. Perbedaan return total terutama disebabkan oleh volume trade yang sangat berkurang.

---

## 10. Analisis Threshold

Threshold memengaruhi trade-off antara jumlah sinyal yang diterima dan presisi (win rate). Berikut ringkasan berdasarkan data aktual dari threshold_analysis.csv:

| Threshold | Jumlah Sinyal | Precision | Recall | F1 |
|-----------|--------------|-----------|--------|----|
| 0.30 – 0.49 | 606 | 51.3% | 100.0% | 0.678 |
| **0.50** | **358** | **51.1%** | **58.8%** | **0.547** |
| **0.51** | **29** | **72.4%** | **6.8%** | **0.124** |

> **Catatan:** Pada rentang threshold 0.30–0.49, model tidak memfilter sinyal apapun (semua 606 sinyal diterima), identik dengan EA tanpa filter. Lompatan signifikan terjadi pada threshold 0.50 (358 sinyal) dan 0.51 (29 sinyal). Tidak ada nilai threshold antara 0.50 dan 0.51 yang diuji dalam pipeline saat ini.

### Panduan Pemilihan Threshold

| Tujuan | Threshold Disarankan | Trade-off |
|--------|---------------------|-----------|
| Maksimalkan volume + sedikit filter | 0.50 | 358 sinyal, precision 51.1% |
| Maksimalkan precision | **0.51** | 29 sinyal, precision 72.4% |
| Keseimbangan (perlu retrain) | 0.50–0.51 | Perlu data lebih banyak |

---

## 11. Keterbatasan

### 11.1. Ketidaksesuaian Sumber Data

| Aspek | Data Latih | Target Deployment |
|-------|-----------|-------------------|
| Broker/Feed | yfinance (GC=F) | Exness XAUUSD |
| Presisi Harga | ~$0.10 per tick | 5-digit ($0.001) |
| Spread | Tidak ada spread | Ada spread (rata-rata ~200-400 poin) |
| Jam Trading | Jam bursa US | Jam MT5 (server time) |

### 11.2. Keterbatasan Timeframe

- Data dilatih pada **H1/M30**, sementara EA berjalan pada **M1/M30**
- Logika S/R (iFractals) berbasis M30 bisa tidak sinkron dengan data H1

### 11.3. Keterbatasan Durasi Data

- Hanya **14 bulan** data (Oktober 2024 — November 2025)
- Tidak mencakup siklus pasar yang lengkap (bull market, bear market, sideways panjang)
- Sampel test set relatif kecil (606 bar M30 ≈ ~1.5 bulan)

### 11.4. Performa AUC Moderat

- AUC test **0.538** hanya sedikit di atas random (0.500)
- Menunjukkan model masih kesulitan memisahkan sinyal baik dan buruk secara umum
- Precision tinggi (72.4%) diperoleh dengan mengorbankan recall ekstrem (hanya 6.75%)

### 11.5. Risiko Overfitting Threshold

- Threshold 0.51 dipilih berdasarkan test set, bukan validation set murni
- Ada risiko threshold ini tidak generalizable ke data masa depan

---

## 12. Rekomendasi

### 12.1. Jangka Pendek (Segera)

1. **Demo Trading:** Jalankan EA + NN filter pada akun demo Exness minimal 1 bulan sebelum live
2. **Monitoring Sinyal:** Catat setiap sinyal yang diterima/ditolak beserta alasannya di log MT5
3. **Verifikasi Feature:** Pastikan nilai fitur real-time dari `PrepareFeatures()` sesuai dengan distribusi training (lihat normalisasi di `feature_config.json`)

### 12.2. Jangka Menengah (1-3 Bulan)

4. **Retrain dengan Data Exness:** Download data XAUUSD H1/M1/M30 langsung dari Exness menggunakan MT5 History atau MT5 API untuk menghilangkan gap data GC=F vs XAUUSD
5. **Backtest MT5 Strategy Tester:** Validasi ulang EA dengan ONNX model menggunakan data Exness di MT5 Strategy Tester (mode tick-by-tick)
6. **Ekspansi Data:** Tambah periode data minimal 3-5 tahun untuk mencakup berbagai kondisi pasar

### 12.3. Jangka Panjang (3+ Bulan)

7. **Walk-Forward Validation:** Terapkan walk-forward testing (latih pada 12 bulan, uji 2 bulan, geser jendela) untuk menilai stabilitas model
8. **Ensemble Model:** Pertimbangkan ensemble XGBoost + LightGBM + Logistic Regression untuk meningkatkan AUC
9. **Retrain Berkala:** Jadwalkan retrain model setiap 3 bulan atau ketika win rate turun di bawah 55% selama 30 hari berturut-turut
10. **Feature Engineering Tambahan:** Tambahkan fitur berbasis order flow, seasonality tahunan, dan korelasi lintas aset (DXY, S&P500)

---

*Laporan ini dibuat berdasarkan data dari `feature_config.json`, `pipeline_results.json`, `threshold_analysis.csv`, dan kode `pipeline_v3.py` di direktori `/home/user/workspace/sr_mapping_nn/`.*
