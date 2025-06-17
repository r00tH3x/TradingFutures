
# 💹 SuperSignal Futures Bot

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat&logo=python)
![Binance](https://img.shields.io/badge/Binance-Futures-yellow?style=flat&logo=binance)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Tentang Project

**SuperSignal Futures Bot** adalah bot trading otomatis berbasis Python yang powerful untuk platform **Binance Futures**. Tool ini menggabungkan:
- analisis teknikal,
- analisis sentimen berita dari berbagai sumber,
- serta prediksi harga menggunakan **LSTM + Random Forest (ensemble)**.

Dibuat untuk memindai, memilih, dan mengeksekusi posisi secara otomatis berdasarkan strategi yang teruji, dengan pengaturan risiko, trailing stop, dan performa yang dapat dikustomisasi.

## 🧠 Fitur Utama

✅ Prediksi harga dengan **LSTM Neural Network**  
✅ Analisis sentimen dari **RSS berita crypto**  
✅ Simulasi Monte Carlo & stress testing sebelum eksekusi  
✅ Auto-scan 30+ koin USDT untuk pilih koin terbaik  
✅ Support **trailing stop**, leverage dinamis, dan risk control  
✅ Eksekusi order Binance Futures secara otomatis  
✅ Logging transaksi + Database SQLite  
✅ Bisa jalan di Termux, Linux, Windows (CLI mode)  
✅ Tidak pakai AI eksternal, semua lokal!

## 📂 Struktur Project

```
.
├── Trading futures.txt     # Script utama (rename ke futures.py)
├── trade_history.db        # Database SQLite (dibuat otomatis)
├── trade_log.txt           # File log untuk semua transaksi
├── lstm_model_*.h5         # Model LSTM per coin (dibuat otomatis)
├── scaler_*.pkl            # Scaler MinMax untuk model
├── rf_model_*.pkl          # Model Random Forest
└── README.md               # Dokumentasi keren ini
```

## ⚙️ Cara Instalasi dan Menjalankan

### 1. Clone repo

```bash
git clone https://github.com/username/supersignal-futures.git
cd supersignal-futures
```

### 2. Rename file utama (wajib)

```bash
mv Trading\ futures.txt futures.py
```

### 3. Install dependensi

```bash
pip install -r requirements.txt
```

Isi `requirements.txt`:
```
binance
websocket-client
pandas
numpy
pandas-ta
feedparser
tensorflow
scikit-learn
colorama
requests
```

### 4. Konfigurasi API

Edit file `futures.py`:

```bash
nano futures.py
```

Isi API key dan secret Binance serta Etherscan pada bagian ini:
```python
API_KEY = 'api-key-kamu'
API_SECRET = 'api-secret-kamu'
ETHERSCAN_API_KEY = 'apikey-etherscanmu'
```

Simpan file dengan `CTRL + X`, lalu tekan `Y` dan `ENTER`.

## ✅ Cara Menjalankan

```bash
python futures.py
```

Akan muncul menu seperti ini:

```
1. Mulai Trading Otomatis (Single Coin)
2. Mulai Trading Otomatis (Auto-Scan Coins)
3. Analisis melalui Berita
4. Keluar
```

## 📈 Mode Trading

### 🔹 Single Coin Mode
Masukkan simbol koin dan timeframe (atau kosongkan untuk otomatis), bot akan memprediksi harga dan mengeksekusi posisi berdasarkan strategi.

### 🔸 Auto-Scan Mode
Bot akan memindai hingga 30+ koin dan memilih koin terbaik berdasarkan skor volatilitas, volume, dan prediksi harga.

## 📊 Komponen Analisis

| Komponen           | Teknologi                          |
|--------------------|-------------------------------------|
| Prediksi Harga     | LSTM + Random Forest (ensemble)     |
| Indikator Teknikal | SMA, RSI, MACD, VWAP, ADX, ATR      |
| Sentimen Berita    | RSS parser + keyword bullish/bearish|
| Timeframe Otomatis | MTF + backtesting sederhana         |
| Volatilitas Ekstrem| Deteksi persentase perubahan harga |
| Risk Management    | Stop loss, take profit, leverage    |

## 🔐 Keamanan

- Bot akan menghentikan trading jika saldo < $2
- Menahan eksekusi jika volatilitas ekstrem terdeteksi
- Maksimal posisi aktif: 2 posisi

## 🧑‍💻 Developer

Tool ini dibuat oleh **Imam Barmawi (a.k.a. Grok)** untuk eksplorasi dan edukasi trading futures berbasis AI.

## ⚠️ Disclaimer

> Tool ini hanya untuk tujuan edukasi dan eksperimen. Trading futures mengandung risiko tinggi. Segala bentuk kerugian bukan tanggung jawab developer.

## ☕ Support

Kalau kamu suka project ini, kasih ⭐ dan share ke teman-teman komunitas crypto-mu!
