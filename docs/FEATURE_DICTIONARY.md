# Kamus Data Fitur — SR_Mapping_NN

**Versi:** 1.0  
**Total Fitur:** 26  
**Format ONNX:** f0 – f25  
**Sumber:** `feature_config.json`, `pipeline_v3.py`, `SR_Mapping_NN_v1.mq5`  

---

## Daftar Isi

1. [Ringkasan 26 Fitur](#1-ringkasan-26-fitur)
2. [Grup A: Fitur Support/Resistance (S/R)](#2-grup-a-fitur-supportresistance-sr)
3. [Grup B: Fitur RSI](#3-grup-b-fitur-rsi)
4. [Grup C: Fitur ATR](#4-grup-c-fitur-atr)
5. [Grup D: Fitur Daily (Harian)](#5-grup-d-fitur-daily-harian)
6. [Grup E: Fitur Momentum (Return)](#6-grup-e-fitur-momentum-return)
7. [Grup F: Fitur Pola Candle](#7-grup-f-fitur-pola-candle)
8. [Grup G: Fitur Waktu](#8-grup-g-fitur-waktu)
9. [Grup H: Fitur Volume](#9-grup-h-fitur-volume)
10. [Catatan Komputasi Real-Time di MT5](#10-catatan-komputasi-real-time-di-mt5)
11. [Normalisasi Data](#11-normalisasi-data)

---

## 1. Ringkasan 26 Fitur

| Index | Nama Fitur | Grup | Importance | ONNX Alias |
|-------|-----------|------|-----------|------------|
| 0 | `rsi` | RSI | 0.0406 | f0 |
| 1 | `atr` | ATR | 0.0486 | f1 |
| 2 | `daily_range` | Daily | 0.0607 | f2 |
| 3 | `daily_direction` | Daily | 0.0000 | f3 |
| 4 | `dist_res_norm` | S/R | 0.0264 | f4 |
| 5 | `dist_sup_norm` | S/R | 0.0467 | f5 |
| 6 | `sr_position` | S/R | 0.0199 | f6 |
| 7 | `hour` | Waktu | 0.0381 | f7 |
| 8 | `dow` | Waktu | 0.0595 | f8 |
| 9 | `ret_3` | Momentum | **0.0640** | f9 |
| 10 | `ret_5` | Momentum | 0.0621 | f10 |
| 11 | `ret_10` | Momentum | 0.0590 | f11 |
| 12 | `ret_20` | Momentum | 0.0445 | f12 |
| 13 | `atr_pctile` | ATR | 0.0439 | f13 |
| 14 | `atr_change` | ATR | 0.0483 | f14 |
| 15 | `rsi_sma` | RSI | 0.0391 | f15 |
| 16 | `rsi_slope` | RSI | 0.0418 | f16 |
| 17 | `near_support` | S/R | 0.0000 | f17 |
| 18 | `near_resistance` | S/R | 0.0000 | f18 |
| 19 | `body_size` | Candle | 0.0384 | f19 |
| 20 | `upper_wick` | Candle | 0.0411 | f20 |
| 21 | `lower_wick` | Candle | 0.0563 | f21 |
| 22 | `is_bullish` | Candle | 0.0000 | f22 |
| 23 | `bars_since_frac_up` | S/R | 0.0540 | f23 |
| 24 | `bars_since_frac_down` | S/R | 0.0295 | f24 |
| 25 | `vol_ratio` | Volume | 0.0376 | f25 |

---

## 2. Grup A: Fitur Support/Resistance (S/R)

### `dist_res_norm` (f4)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Float, kontinu |
| **Deskripsi** | Jarak harga saat ini ke level resistance terdekat, dinormalisasi dengan ATR |
| **Cara Hitung** | `(resistance − currentBid) / ATR` |
| **Range Aktual** | −6.82 hingga +7.62 (dari data training) |
| **Nilai Typical** | ~0.83 (rata-rata) |
| **Std Dev** | 1.56 |
| **Importance** | 0.0264 (rendah) |
| **MT5 Real-time** | `(g_currentResistance - bid) / atr` |
| **Interpretasi** | Positif = harga di bawah resistance. Negatif = harga menembus resistance ke atas |

---

### `dist_sup_norm` (f5)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Float, kontinu |
| **Deskripsi** | Jarak harga saat ini ke level support terdekat, dinormalisasi dengan ATR |
| **Cara Hitung** | `(currentBid − support) / ATR` |
| **Range Aktual** | −6.22 hingga +9.81 |
| **Nilai Typical** | ~1.60 (rata-rata) |
| **Std Dev** | 1.48 |
| **Importance** | **0.0467 (top 10)** |
| **MT5 Real-time** | `(bid - g_currentSupport) / atr` |
| **Interpretasi** | Positif = harga di atas support. Semakin kecil → harga semakin dekat ke support |

---

### `sr_position` (f6)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Float, kontinu |
| **Deskripsi** | Posisi harga relatif di antara support dan resistance (0 = di support, 1 = di resistance) |
| **Cara Hitung** | `(bid − support) / (resistance − support)` |
| **Range Aktual** | −15.05 hingga +48.99 |
| **Nilai Typical** | ~0.78 (rata-rata) |
| **Std Dev** | 1.49 |
| **Importance** | 0.0199 (rendah) |
| **MT5 Real-time** | `(bid - g_currentSupport) / (g_currentResistance - g_currentSupport)` |
| **Catatan** | Nilai di luar [0,1] terjadi saat harga menembus S/R |

---

### `near_support` (f17)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Integer biner (0 atau 1) |
| **Deskripsi** | Flag: apakah harga berada sangat dekat dengan level support |
| **Cara Hitung** | `abs(bid − support) <= ATR * 0.20 ? 1 : 0` |
| **Range** | 0 atau 1 |
| **Nilai Typical** | ~0.032 (rata-rata, jarang terpenuhi) |
| **Importance** | **0.0000 (tidak berkontribusi pada model)** |
| **MT5 Real-time** | `(MathAbs(distSup) <= tolerance) ? 1 : 0` dimana `tolerance = atr * 0.20` |
| **Catatan** | Meskipun importance = 0, fitur tetap harus disertakan agar urutan array tidak berubah |

---

### `near_resistance` (f18)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Integer biner (0 atau 1) |
| **Deskripsi** | Flag: apakah harga berada sangat dekat dengan level resistance |
| **Cara Hitung** | `abs(resistance − bid) <= ATR * 0.20 ? 1 : 0` |
| **Range** | 0 atau 1 |
| **Nilai Typical** | ~0.091 (rata-rata) |
| **Importance** | **0.0000 (tidak berkontribusi pada model)** |
| **MT5 Real-time** | `(MathAbs(distRes) <= tolerance) ? 1 : 0` |

---

### `bars_since_frac_up` (f23)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Integer, kontinu |
| **Deskripsi** | Jumlah bar M30 sejak fraktal naik (upper fractal) terakhir terdeteksi |
| **Cara Hitung** | Scan buffer iFractals M30, cari nilai valid pertama dari bar terbaru |
| **Range Aktual** | 0 hingga 26 (dari data training) |
| **Nilai Typical** | ~4.73 (rata-rata) |
| **Std Dev** | 4.45 |
| **Importance** | **0.0540 (top 10)** |
| **MT5 Real-time** | Loop `CopyBuffer(handleFrac, 0, 0, 200, fUp)`, hitung index pertama yang valid |
| **Interpretasi** | Nilai kecil = fraktal resistance baru-baru ini terbentuk (fresh level) |

---

### `bars_since_frac_down` (f24)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Integer, kontinu |
| **Deskripsi** | Jumlah bar M30 sejak fraktal turun (lower fractal) terakhir terdeteksi |
| **Cara Hitung** | Scan buffer iFractals M30, cari nilai valid pertama dari bar terbaru |
| **Range Aktual** | 0 hingga 26 (dari data training) |
| **Nilai Typical** | ~4.20 (rata-rata) |
| **Std Dev** | 4.01 |
| **Importance** | 0.0295 |
| **MT5 Real-time** | Loop `CopyBuffer(handleFrac, 1, 0, 200, fDown)`, hitung index pertama yang valid |
| **Interpretasi** | Nilai kecil = fraktal support baru-baru ini terbentuk |

---

## 3. Grup B: Fitur RSI

### `rsi` (f0)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Float, kontinu |
| **Deskripsi** | Nilai RSI (Relative Strength Index) pada timeframe M1, periode 14, harga close |
| **Cara Hitung** | `iRSI(_Symbol, PERIOD_M1, 14, PRICE_CLOSE)` — bar ke-0 |
| **Range Teoritis** | 0 hingga 100 |
| **Range Aktual** | 12.76 hingga 89.66 |
| **Nilai Typical** | ~53.74 (rata-rata) |
| **Std Dev** | 13.07 |
| **Importance** | 0.0406 |
| **MT5 Real-time** | `GetRSI_M1()` — `g_handleRSI_M1` dibuat saat `OnInit()` |
| **Interpretasi** | < 40 = oversold (syarat BUY), > 70 = overbought (syarat SELL) |

---

### `rsi_sma` (f15)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Float, kontinu |
| **Deskripsi** | Simple Moving Average dari RSI selama 10 bar M1 terakhir |
| **Cara Hitung** | `mean(RSI[0], RSI[1], ..., RSI[9])` — 10 bar M1 |
| **Range Aktual** | 22.75 hingga 83.36 |
| **Nilai Typical** | ~53.33 (rata-rata) |
| **Std Dev** | 11.59 |
| **Importance** | 0.0391 |
| **MT5 Real-time** | `CopyBuffer(g_handleRSI_M1, 0, 0, 10, rsiArr)` → rata-rata |
| **Interpretasi** | Tren RSI jangka pendek. Membandingkan `rsi` vs `rsi_sma` menunjukkan momentum |

---

### `rsi_slope` (f16)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Float, kontinu |
| **Deskripsi** | Kemiringan (slope) RSI: selisih RSI bar 0 dengan RSI bar 3 |
| **Cara Hitung** | `rsiArr[0] - rsiArr[3]` (RSI saat ini minus RSI 3 bar lalu) |
| **Range Aktual** | −32.19 hingga +38.22 |
| **Nilai Typical** | ~0.35 (rata-rata, sedikit naik) |
| **Std Dev** | 8.21 |
| **Importance** | 0.0418 |
| **MT5 Real-time** | Dari array `rsiArr` yang sama dengan `rsi_sma` |
| **Interpretasi** | Positif = RSI sedang naik. Negatif = RSI sedang turun. Digunakan untuk deteksi divergensi |

---

## 4. Grup C: Fitur ATR

### `atr` (f1)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Float, kontinu |
| **Deskripsi** | Average True Range pada timeframe M1, periode 14, dalam satuan harga (bukan points) |
| **Cara Hitung** | `iATR(_Symbol, PERIOD_M1, 14)` — bar ke-0 |
| **Range Aktual** | 4.39 hingga 59.77 (satuan dolar/poin harga GC=F) |
| **Nilai Typical** | ~14.24 (rata-rata) |
| **Std Dev** | 8.95 |
| **Importance** | **0.0486 (top 10)** |
| **MT5 Real-time** | `atrBuffer[0]` (dalam satuan harga, bukan points) |
| **Catatan** | Di EA, dikonversi ke points dengan `atrBuffer[0] / g_point`. Dalam NN, digunakan nilai harga mentah |

---

### `atr_pctile` (f13)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Float, kontinu [0, 1] |
| **Deskripsi** | Persentil ATR saat ini dibandingkan dengan 168 bar ATR M1 terakhir |
| **Cara Hitung** | `count(atr[i] <= atr_now for i in [0, 168]) / 168` |
| **Range Teoritis** | 0.0 hingga 1.0 |
| **Range Aktual** | 0.006 hingga 1.0 |
| **Nilai Typical** | ~0.460 (rata-rata) |
| **Std Dev** | 0.330 |
| **Importance** | 0.0439 |
| **MT5 Real-time** | `CopyBuffer(g_handleATR_M1, 0, 0, 168, atrArr)` → hitung persentil |
| **Interpretasi** | 0.0 = volatilitas sangat rendah. 1.0 = volatilitas sangat tinggi. 168 bar M1 ≈ ~2.8 jam |

---

### `atr_change` (f14)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Float, kontinu |
| **Deskripsi** | Perubahan ATR dalam % dibandingkan 5 bar M1 yang lalu |
| **Cara Hitung** | `(atr_now / atr_5_bars_ago - 1) * 100` |
| **Range Aktual** | −26.07 hingga +267.31 |
| **Nilai Typical** | ~3.69 (rata-rata) |
| **Std Dev** | 15.92 |
| **Importance** | **0.0483 (top 10)** |
| **MT5 Real-time** | `atr_5 = atrArr[5] * g_point; atrChange = (atr/atr_5 - 1) * 100` |
| **Interpretasi** | Positif = volatilitas sedang meningkat. Negatif = volatilitas sedang menurun |

---

## 5. Grup D: Fitur Daily (Harian)

### `daily_range` (f2)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Float, kontinu |
| **Deskripsi** | Selisih harga bid saat ini dengan harga open hari ini (dalam satuan harga, bukan points) |
| **Cara Hitung** | `currentBid - iOpen(_Symbol, PERIOD_D1, 0)` |
| **Range Aktual** | −208.5 hingga +143.5 |
| **Nilai Typical** | ~23.0 (rata-rata, sedikit bullish) |
| **Std Dev** | 29.95 |
| **Importance** | **0.0607 (top 10)** |
| **MT5 Real-time** | `bid - dailyOpen` dimana `dailyOpen = iOpen(_Symbol, PERIOD_D1, 0)` |
| **Interpretasi** | Positif = harga di atas open hari ini (bullish intraday). Negatif = bearish intraday |

---

### `daily_direction` (f3)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Integer kategoris |
| **Deskripsi** | Arah bias harian berdasarkan DetectDailyRange() |
| **Cara Hitung** | daily_range >= 1000 pts → +1; <= −1000 pts → −1; lainnya → 0 |
| **Range** | −1, 0, +1 |
| **Nilai Typical** | ~0.71 (rata-rata) |
| **Std Dev** | 0.71 |
| **Importance** | **0.0000 (tidak berkontribusi)** |
| **MT5 Real-time** | `DetectDailyRange()` — sama dengan logika entry filter |
| **Catatan** | Importance 0 karena fitur ini selalu sama untuk semua sinyal yang masuk (sudah difilter oleh EA) |

---

## 6. Grup E: Fitur Momentum (Return)

Semua fitur return dihitung dari harga close M30.

### `ret_3` (f9)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Float, kontinu |
| **Deskripsi** | Return harga selama 3 bar M30 terakhir dalam % |
| **Cara Hitung** | `(close[0] / close[3] - 1) * 100` |
| **Range Aktual** | −2.74% hingga +1.95% |
| **Nilai Typical** | ~0.039% (rata-rata ≈ 0) |
| **Std Dev** | 0.443 |
| **Importance** | **0.0640 (TERTINGGI dari semua fitur)** |
| **MT5 Real-time** | `iClose(_Symbol, PERIOD_M30, 0)` dan `iClose(_Symbol, PERIOD_M30, 3)` |

---

### `ret_5` (f10)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Float, kontinu |
| **Deskripsi** | Return harga selama 5 bar M30 terakhir (2.5 jam) dalam % |
| **Cara Hitung** | `(close[0] / close[5] - 1) * 100` |
| **Range Aktual** | −3.33% hingga +2.25% |
| **Nilai Typical** | ~0.059% |
| **Std Dev** | 0.574 |
| **Importance** | **0.0621 (ke-2 tertinggi)** |
| **MT5 Real-time** | `iClose(_Symbol, PERIOD_M30, 5)` |

---

### `ret_10` (f11)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Float, kontinu |
| **Deskripsi** | Return harga selama 10 bar M30 terakhir (5 jam) dalam % |
| **Cara Hitung** | `(close[0] / close[10] - 1) * 100` |
| **Range Aktual** | −5.36% hingga +2.90% |
| **Nilai Typical** | ~0.109% |
| **Std Dev** | 0.765 |
| **Importance** | **0.0590 (top 10)** |
| **MT5 Real-time** | `iClose(_Symbol, PERIOD_M30, 10)` |

---

### `ret_20` (f12)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Float, kontinu |
| **Deskripsi** | Return harga selama 20 bar M30 terakhir (10 jam) dalam % |
| **Cara Hitung** | `(close[0] / close[20] - 1) * 100` |
| **Range Aktual** | −6.13% hingga +3.67% |
| **Nilai Typical** | ~0.181% |
| **Std Dev** | 1.115 |
| **Importance** | 0.0445 |
| **MT5 Real-time** | `iClose(_Symbol, PERIOD_M30, 20)` |

---

## 7. Grup F: Fitur Pola Candle

Semua fitur candle diambil dari bar M30 terbaru dan dinormalisasi dengan ATR.

### `body_size` (f19)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Float, kontinu ≥ 0 |
| **Deskripsi** | Ukuran badan candle (selisih close−open) dinormalisasi dengan ATR |
| **Cara Hitung** | `abs(close[0] - open[0]) / ATR` |
| **Range Aktual** | 0.0 hingga 5.57 |
| **Nilai Typical** | ~0.46 (rata-rata) |
| **Std Dev** | 0.449 |
| **Importance** | 0.0384 |
| **MT5 Real-time** | `MathAbs(close_now - open_0) / atr` dimana semua dalam satuan harga |
| **Interpretasi** | Nilai besar = candle dengan badan panjang (momentum kuat). Nilai 0 = doji |

---

### `upper_wick` (f20)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Float, kontinu ≥ 0 |
| **Deskripsi** | Panjang ekor atas candle dinormalisasi dengan ATR |
| **Cara Hitung** | `(high[0] - max(close[0], open[0])) / ATR` |
| **Range Aktual** | 0.0 hingga 3.12 |
| **Nilai Typical** | ~0.221 (rata-rata) |
| **Std Dev** | 0.208 |
| **Importance** | 0.0411 |
| **MT5 Real-time** | `(high_0 - MathMax(close_now, open_0)) / atr` |
| **Interpretasi** | Ekor atas panjang menunjukkan penolakan harga tinggi (tekanan jual) |

---

### `lower_wick` (f21)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Float, kontinu ≥ 0 |
| **Deskripsi** | Panjang ekor bawah candle dinormalisasi dengan ATR |
| **Cara Hitung** | `(min(close[0], open[0]) - low[0]) / ATR` |
| **Range Aktual** | 0.0 hingga 4.01 |
| **Nilai Typical** | ~0.379 (rata-rata) |
| **Std Dev** | 0.370 |
| **Importance** | **0.0563 (top 10)** |
| **MT5 Real-time** | `(MathMin(close_now, open_0) - low_0) / atr` |
| **Interpretasi** | Ekor bawah panjang menunjukkan penolakan harga rendah (tekanan beli) — sinyal reversal bullish |

---

### `is_bullish` (f22)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Integer biner (0 atau 1) |
| **Deskripsi** | Flag: apakah candle M30 terbaru adalah candle bullish |
| **Cara Hitung** | `close[0] > open[0] ? 1 : 0` |
| **Range** | 0 atau 1 |
| **Nilai Typical** | ~0.538 (rata-rata, sedikit lebih banyak candle bullish) |
| **Importance** | **0.0000 (tidak berkontribusi)** |
| **MT5 Real-time** | `(close_now > open_0) ? 1 : 0` |
| **Catatan** | Meskipun importance = 0, tetap harus disertakan di posisi f22 |

---

## 8. Grup G: Fitur Waktu

### `hour` (f7)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Integer kategoris |
| **Deskripsi** | Jam UTC saat sinyal terjadi (0–23) |
| **Cara Hitung** | `dt.hour` dari `TimeCurrent()` |
| **Range Aktual** | 2 hingga 17 (sesuai jam trading) |
| **Nilai Typical** | ~9.65 (rata-rata) |
| **Std Dev** | 4.63 |
| **Importance** | 0.0381 |
| **MT5 Real-time** | `MqlDateTime dt; TimeCurrent(dt); dt.hour` |
| **Interpretasi** | Jam trading server MT5 (UTC). Sesi London: 8–17, New York: 13–22 (overlap 13–17) |

---

### `dow` (f8)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Integer kategoris |
| **Deskripsi** | Hari dalam minggu (Day of Week), 0=Senin hingga 4=Jumat |
| **Cara Hitung** | `dt.day_of_week - 1` (MT5 mulai dari 0=Minggu, dikonversi agar 0=Senin) |
| **Range** | 0 (Senin) hingga 4 (Jumat) |
| **Nilai Typical** | ~2.01 (rata-rata ≈ Rabu) |
| **Std Dev** | 1.41 |
| **Importance** | **0.0595 (top 10)** |
| **MT5 Real-time** | `dt.day_of_week - 1; if(dow < 0) dow = 6` |
| **Interpretasi** | Pengaruh hari perdagangan. Hari Jumat (dow=4) sering memiliki volatilitas berbeda menjelang weekend |

---

## 9. Grup H: Fitur Volume

### `vol_ratio` (f25)

| Atribut | Detail |
|---------|--------|
| **Tipe** | Float, kontinu ≥ 0 |
| **Deskripsi** | Rasio volume bar M30 saat ini dibandingkan rata-rata volume 20 bar sebelumnya |
| **Cara Hitung** | `vol_now / mean(vol[-1], vol[-2], ..., vol[-20])` |
| **Range Aktual** | 0.0 hingga 11.38 |
| **Nilai Typical** | ~1.29 (rata-rata) |
| **Std Dev** | 1.06 |
| **Importance** | 0.0376 |
| **MT5 Real-time** | `iVolume(_Symbol, PERIOD_M30, 0)` dibandingkan rata-rata `iVolume(_Symbol, PERIOD_M30, i)` untuk i=1..20 |
| **Catatan** | Di XAUUSD MetaTrader 5 pada broker ECN, volume ini adalah "tick volume" (jumlah perubahan harga per bar), bukan volume kontrak sebenarnya |
| **Interpretasi** | > 1.0 = aktivitas trading di atas rata-rata. < 1.0 = aktivitas rendah |

---

## 10. Catatan Komputasi Real-Time di MT5

### Inisialisasi (OnInit)

Dua indikator dibuat saat EA diinisialisasi dan di-reuse setiap tick:

```mql5
// Dibuat sekali, digunakan berulang kali (efisien)
g_handleRSI_M1 = iRSI(_Symbol, PERIOD_M1, 14, PRICE_CLOSE);
g_handleATR_M1 = iATR(_Symbol, PERIOD_M1, 14);
```

Indikator `iFractals` untuk `bars_since_frac_up/down` dibuat dan dirilis setiap panggilan karena hanya diperlukan saat sinyal entry terdeteksi.

### Urutan Komputasi Kritis

Urutan fitur dalam array **harus persis sama** dengan urutan training:

```
features[0..25] = [rsi, atr, daily_range, daily_direction,
                   dist_res_norm, dist_sup_norm, sr_position,
                   hour, dow,
                   ret_3, ret_5, ret_10, ret_20,
                   atr_pctile, atr_change,
                   rsi_sma, rsi_slope,
                   near_support, near_resistance,
                   body_size, upper_wick, lower_wick, is_bullish,
                   bars_since_frac_up, bars_since_frac_down,
                   vol_ratio]
```

**Jangan pernah mengubah urutan ini** tanpa melatih ulang model.

### Perbedaan Sumber Data Training vs Real-Time

| Aspek | Training (yfinance GC=F) | Real-Time MT5 (Exness) |
|-------|------------------------|------------------------|
| ATR | ~$4–60 (satuan dolar) | ~4–60 (XAUUSD, 5-digit) |
| Volume | Volume futures GC=F | Tick volume XAUUSD |
| Daily open | Sesi CME (New York) | Sesi MT5 (server time) |
| M30 candle | Diturunkan dari H1 | Native M30 |

---

## 11. Normalisasi Data

Model XGBoost yang digunakan **tidak memerlukan normalisasi fitur** sebelum dimasukkan ke ONNX. Namun, nilai normalisasi (mean, std, min, max) tersimpan di `feature_config.json` untuk keperluan monitoring dan validasi fitur.

### Referensi Cepat Statistik Fitur

| Fitur | Mean | Std | Min | Max |
|-------|------|-----|-----|-----|
| rsi | 53.74 | 13.07 | 12.76 | 89.66 |
| atr | 14.24 | 8.95 | 4.39 | 59.77 |
| daily_range | 22.99 | 29.95 | −208.5 | 143.5 |
| daily_direction | 0.71 | 0.71 | −1.0 | 1.0 |
| dist_res_norm | 0.83 | 1.56 | −6.82 | 7.62 |
| dist_sup_norm | 1.60 | 1.48 | −6.22 | 9.81 |
| sr_position | 0.78 | 1.49 | −15.05 | 48.99 |
| hour | 9.65 | 4.63 | 2.0 | 17.0 |
| dow | 2.01 | 1.41 | 0.0 | 4.0 |
| ret_3 | 0.039 | 0.443 | −2.74 | 1.95 |
| ret_5 | 0.059 | 0.574 | −3.33 | 2.25 |
| ret_10 | 0.109 | 0.765 | −5.36 | 2.90 |
| ret_20 | 0.181 | 1.115 | −6.13 | 3.67 |
| atr_pctile | 0.460 | 0.330 | 0.006 | 1.0 |
| atr_change | 3.69 | 15.92 | −26.07 | 267.31 |
| rsi_sma | 53.33 | 11.59 | 22.75 | 83.36 |
| rsi_slope | 0.346 | 8.21 | −32.19 | 38.22 |
| near_support | 0.032 | 0.175 | 0.0 | 1.0 |
| near_resistance | 0.091 | 0.288 | 0.0 | 1.0 |
| body_size | 0.460 | 0.449 | 0.0 | 5.57 |
| upper_wick | 0.221 | 0.208 | 0.0 | 3.12 |
| lower_wick | 0.379 | 0.370 | 0.0 | 4.01 |
| is_bullish | 0.538 | 0.499 | 0.0 | 1.0 |
| bars_since_frac_up | 4.73 | 4.45 | 0.0 | 26.0 |
| bars_since_frac_down | 4.20 | 4.01 | 0.0 | 26.0 |
| vol_ratio | 1.29 | 1.06 | 0.0 | 11.38 |

---

*Statistik di atas bersumber dari `feature_config.json` yang dihasilkan oleh `pipeline_v3.py` berdasarkan data training GC=F yfinance Oktober 2024 – November 2025.*
