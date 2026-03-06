# Panduan Deployment SR_Mapping_NN ke MetaTrader 5

## Prasyarat

- MetaTrader 5 Build 3000+ (mendukung ONNX)
- Account Exness atau broker lain dengan XAUUSD
- File: `SR_Mapping_NN_v1.mq5` dan `sr_mapping_nn.onnx`

---

## Langkah 1: Copy File

### EA File
```
Dari:  mql5/SR_Mapping_NN_v1.mq5
Ke:    C:\Users\<NAMA>\AppData\Roaming\MetaQuotes\Terminal\<ID>\MQL5\Experts\SR_Mapping_NN_v1.mq5
```

### ONNX Model
```
Dari:  models/sr_mapping_nn.onnx
Ke:    C:\Users\<NAMA>\AppData\Roaming\MetaQuotes\Terminal\<ID>\MQL5\Files\sr_mapping_nn.onnx
```

**Cara cepat:**
1. Buka MetaEditor (F4 di MT5)
2. Klik kanan pada folder `Experts` → "Open Folder"
3. Copy-paste file `.mq5`
4. Klik kanan pada folder `Files` → "Open Folder"
5. Copy-paste file `.onnx`

---

## Langkah 2: Compile EA

1. Buka MetaEditor (tekan F4 di MT5)
2. Buka file `SR_Mapping_NN_v1.mq5`
3. Tekan **F7** untuk compile
4. Pastikan output: `0 error(s), 0 warning(s)`

Jika ada error terkait ONNX:
- Pastikan MT5 Build >= 3000
- Include `<ONNX\OnnxRuntime.mqh>` mungkin perlu diupdate

---

## Langkah 3: Attach ke Chart

1. Buka chart **XAUUSD** timeframe **M30** (atau H1)
2. Buka Navigator (Ctrl+N)
3. Drag `SR_Mapping_NN_v1` ke chart
4. Di dialog yang muncul:
   - Tab **Common**: centang "Allow algo trading"
   - Tab **Dependencies**: centang "Allow DLL imports" (jika diminta)
   - Tab **Inputs**: set parameter sesuai tabel di bawah

---

## Langkah 4: Setting Parameter

### Parameter Wajib (Jangan Diubah Kecuali Perlu)

| Parameter | Nilai | Fungsi |
|-----------|-------|--------|
| InpUseNNFilter | true | **Nyalakan NN filter** |
| InpConfidenceThresh | 0.51 | Threshold confidence (turunkan untuk lebih banyak trade) |
| InpOnnxModelFile | sr_mapping_nn.onnx | Nama file ONNX |
| InpUseStopLoss | true | **WAJIB true** (grid sudah dihapus) |
| InpUseATR_SLTP | true | SL/TP dinamis berbasis ATR |
| InpSL_ATR_Mult | 1.6 | SL distance = ATR x 1.6 |
| InpTP_ATR_Mult | 2.0 | TP distance = ATR x 2.0 |
| InpLotSize | 0.01 | Sesuaikan dengan modal |
| InpMaxOpenTrades | 1 | Maks 1 posisi terbuka |

### Parameter Entry (Default = Backtest Optimal)

| Parameter | Default | Range Aman |
|-----------|---------|------------|
| InpRSI_Oversold | 40 | 30-45 |
| InpRSI_Overbought | 70 | 65-75 |
| InpATR_MinPoints | 200 | 100-400 |
| InpATR_MaxPoints | 2500 | 2000-3000 |
| InpSNR_Tolerance | 50 | 30-100 |
| InpDailyRangeThresh | 1000 | 500-2000 |
| InpTradingHourStart | 2 | 0-5 |
| InpTradingHourEnd | 18 | 16-23 |
| InpMaxSpreadPoints | 400 | 200-600 |

---

## Langkah 5: Backtest (WAJIB Sebelum Live)

1. Buka **Strategy Tester** (Ctrl+R di MT5)
2. Setting:
   - Expert: SR_Mapping_NN_v1
   - Symbol: XAUUSD
   - Period: M30
   - Date: minimal 3 bulan terakhir
   - Model: Every tick
   - Deposit: sesuai akun real
3. Klik **Start**
4. Analisis hasil:
   - Win rate harus > 60%
   - Max DD harus < 20%
   - Profit factor > 1.5

---

## Langkah 6: Demo Trading (WAJIB)

1. Attach EA ke akun **DEMO**
2. Jalankan minimal **1 bulan**
3. Monitor:
   - Apakah NN filter berfungsi (cek log: "NN FILTER: ... ACCEPTED/REJECTED")
   - Apakah SL/TP terpasang di setiap trade
   - Apakah frekuensi trade sesuai (0.5-5 per hari)
4. Bandingkan dengan backtest

---

## Langkah 7: Live Trading

Setelah demo 1 bulan dengan hasil konsisten:

1. Attach ke akun **REAL** dengan lot kecil (0.01)
2. Monitor ketat selama 2 minggu pertama
3. Scale up lot secara bertahap jika konsisten

---

## Troubleshooting

### "ONNX model not loaded"
- Pastikan file `sr_mapping_nn.onnx` ada di folder `MQL5/Files/`
- Pastikan nama file persis sama (case-sensitive)
- Cek log di tab "Experts" untuk error detail

### "0 trades dalam backtest"
- Cek apakah InpUseNNFilter = true dan threshold terlalu tinggi
- Coba turunkan InpConfidenceThresh ke 0.40
- Pastikan data XAUUSD tersedia untuk periode backtest

### "Compile error"
- Update MetaTrader 5 ke build terbaru
- Pastikan semua include files tersedia
- Cek MetaEditor log untuk detail error

### "SL/TP not set"
- Pastikan InpUseStopLoss = true
- Cek apakah broker mengizinkan SL/TP pada saat entry
- Beberapa broker perlu jeda 1 detik setelah entry untuk modify SL/TP
