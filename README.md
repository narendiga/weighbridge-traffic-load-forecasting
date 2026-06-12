# Streamlit App Forecasting Volume Truk PT XYZ

Aplikasi ini membaca data historis dan model dari folder utama project menggunakan path relatif.

## Path yang Digunakan

- Data historis: `../weighbridge_fix.xlsx`
- Model Hybrid TCN-BiLSTM: `../models/hybrid.keras`
- Model TCN: `../models/tcn.keras`
- Model BiLSTM: `../models/bilstm.keras`

## Pipeline Inference

- Baca data historis dari `weighbridge_fix.xlsx`
- Gunakan kolom waktu `waktu_closing`
- Agregasi transaksi menjadi data harian `jumlah_truk`
- Reindex tanggal harian penuh
- Interpolasi polynomial order 2 seperti notebook
- Differencing harian
- Buat ulang `SimpleMinMaxScaler(feature_range=(0, 1))` dari data historis
- Gunakan `WINDOW_SIZE = 30`
- Jalankan recursive forecast
- Inverse scaling dan inverse differencing ke skala aktual `jumlah_truk`

## Navigasi

- Dashboard
- Forecast
- Detail Data

Halaman Forecast menyediakan dropdown model: Hybrid TCN-BiLSTM, TCN, dan BiLSTM.

## Menjalankan

```bash
streamlit run streamlit_app/app.py
```

Jika file model atau data tidak ditemukan, aplikasi menampilkan pesan error spesifik dan tidak crash.
