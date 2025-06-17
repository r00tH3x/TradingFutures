import feedparser
from collections import Counter
import pandas as pd
from binance.client import Client
from binance.websockets import BinanceSocketManager
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
from colorama import init, Fore, Style
import os
import pickle
import websocket
import json
import threading
import time
from binance.enums import *
import logging
import sqlite3
import requests
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestRegressor

# Inisialisasi colorama
init()

# Binance API credentials (dari Anda)
API_KEY = 'api-key-kamu'
API_SECRET = 'api-secret-kamu'
client = Client(API_KEY, API_SECRET)

# Etherscan API key (dari Anda)
ETHERSCAN_API_KEY = 'VZFDUWB3YGQ1YCDKTCU1D6DDSS'

# Setup logging
logging.basicConfig(filename='trade_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# Setup SQLite database
conn = sqlite3.connect('trade_history.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS trades 
                  (timestamp TEXT, coin TEXT, direction TEXT, entry_price REAL, stop_loss REAL, 
                   take_profit REAL, leverage INTEGER, profit_loss REAL, balance REAL, order_id TEXT)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS predictions 
                  (timestamp TEXT, coin TEXT, current_price REAL, predicted_price REAL, mae REAL)''')
conn.commit()

# Daftar RSS feed publik
RSS_FEEDS = [
    "https://coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://www.theblock.co/feed",
    "https://crypto.news/feed/",
    "https://www.reuters.com/arc/outboundfeeds/rss/",
]

# Kata kunci untuk analisis sentimen
CRYPTO_KEYWORDS = ["bitcoin", "btc", "ethereum", "eth", "altcoin", "crypto", "blockchain", "defi", "nft", "xrp", "solana"]
BULLISH_KEYWORDS = ["rise", "surge", "rally", "increase", "boom", "up", "bullish", "adoption"]
BEARISH_KEYWORDS = ["drop", "fall", "crash", "decline", "down", "low", "bearish", "hack"]
GLOBAL_SENTIMENT_KEYWORDS = ["tariff", "trade", "inflation", "dollar", "stock", "economy", "recession", "policy", "fed"]

# Banner ASCII
BANNER = f"""
{Fore.CYAN + Style.BRIGHT}==========================================
     CRYPTO AUTO-TRADING BOT by Grok
=========================================={Style.RESET_ALL}
"""

# Global variables
real_time_data = []
error_count = 0
trade_count = 0
last_day = None
initial_capital = 10
ws_active = True
active_trades = {}  # Dictionary untuk menyimpan trade aktif: {order_id: {'coin': coin, ...}}
MAX_ACTIVE_POSITIONS = 2  # Batas maksimal posisi aktif
VOLATILITY_THRESHOLD = 0.05  # Ambang batas volatilitas ekstrem (5% per candle)

# Timeout posisi berdasarkan timeframe (dalam detik)
POSITION_TIMEOUT = {
    "1m": 4 * 3600,   # 4 jam untuk timeframe 1 menit
    "5m": 4 * 3600,   # 4 jam untuk timeframe 5 menit
    "15m": 6 * 3600,  # 6 jam untuk timeframe 15 menit
    "1h": 12 * 3600,  # 12 jam untuk timeframe 1 jam
    "4h": 24 * 3600,  # 24 jam untuk timeframe 4 jam
    "1d": 48 * 3600   # 48 jam untuk timeframe 1 hari
}

# Timeframe yang akan dianalisis untuk pemilihan otomatis
AVAILABLE_TIMEFRAMES = {
    "1m": Client.KLINE_INTERVAL_1MINUTE,
    "5m": Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "1d": Client.KLINE_INTERVAL_1DAY
}

# Cache untuk data historis
data_cache = {}

def on_message(ws, message):
    global real_time_data
    data = json.loads(message)
    if 'k' in data:
        candle = data['k']
        price = float(candle['c'])
        volume = float(candle['v'])
        timestamp = pd.to_datetime(candle['t'], unit='ms')
        real_time_data.append({'timestamp': timestamp, 'price': price, 'volume': volume})

def on_error(ws, error):
    global ws_active
    logging.error(f"WebSocket error: {error}")
    ws_active = False

def on_close(ws):
    global ws_active
    logging.warning("WebSocket ditutup.")
    ws_active = False

def start_websocket(coin, interval):
    symbol = f"{coin.lower()}usdt"
    ws_url = f"wss://fstream.binance.com/ws/{symbol}@kline_{interval}"
    ws = websocket.WebSocketApp(ws_url, on_message=on_message, on_error=on_error, on_close=on_close)
    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.daemon = True
    ws_thread.start()
    return ws

# --- Fungsi untuk Mendapatkan Daftar Koin dari Binance ---
def get_available_coins():
    try:
        exchange_info = client.get_exchange_info()
        symbols = [symbol['symbol'] for symbol in exchange_info['symbols'] if symbol['symbol'].endswith('USDT')]
        coins = [symbol.replace('USDT', '') for symbol in symbols]
        return [coin for coin in coins if len(coin) <= 5][:30]  # Batasi hingga 30 koin untuk performa di Termux
    except Exception as e:
        logging.error(f"Gagal mendapatkan daftar koin: {str(e)}")
        return []

# --- Fungsi untuk Mengambil Data Harga dengan Cache ---
def fetch_price_data(coin="BTC", interval=Client.KLINE_INTERVAL_1HOUR, limit=500):
    cache_key = f"{coin}_{interval}_{limit}"
    if cache_key in data_cache:
        return data_cache[cache_key]
    
    symbol = f"{coin}USDT"
    try:
        klines = client.get_historical_klines(symbol, interval, f"{limit} hours ago UTC")
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 
            'taker_buy_quote', 'ignored'
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["price"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df.set_index("timestamp", inplace=True)
        data_cache[cache_key] = df[["price", "volume"]]
        return data_cache[cache_key]
    except Exception as e:
        raise Exception(f"Koin {symbol} tidak ditemukan atau data tidak tersedia: {str(e)}")

# --- Fungsi untuk Mengambil Data On-Chain Gratis ---
def fetch_onchain_data(coin):
    try:
        if coin.lower() == "btc":
            # Ambil volume transaksi harian dari Blockchain.com
            response = requests.get("https://blockchain.info/charts/transactions-per-day?timespan=1days&format=json")
            data = response.json()
            latest_volume = data["values"][-1]["y"]  # Transaksi per hari terakhir
            return latest_volume
        elif coin.lower() == "eth":
            # Ambil gas fees rata-rata dari Etherscan
            api_key = ETHERSCAN_API_KEY
            response = requests.get(f"https://api.etherscan.io/api?module=gastracker&action=gasoracle&apikey={api_key}")
            data = response.json()
            if "result" in data and "SafeGasPrice" in data["result"]:
                avg_gas = float(data["result"]["SafeGasPrice"])
                return avg_gas
            else:
                return 0
        return 0
    except Exception as e:
        logging.error(f"Gagal mengambil data on-chain untuk {coin}: {str(e)}")
        return 0

# --- Fungsi untuk Menghitung Indikator Teknikal ---
def calculate_technicals(df):
    df["SMA_20"] = ta.sma(df["price"], length=20)
    df["RSI"] = ta.rsi(df["price"], length=14)
    bb = ta.bbands(df["price"], length=20)
    df["BB_upper"] = bb["BBU_20_2.0"]
    df["BB_lower"] = bb["BBL_20_2.0"]
    macd = ta.macd(df["price"], fast=12, slow=26, signal=9)
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_signal"] = macd["MACDS_12_26_9"]
    df["ATR"] = ta.atr(df["price"].shift(1), df["price"].shift(1), df["price"], length=14)
    df["OBV"] = ta.obv(df["price"], df["volume"])
    df["VWAP"] = ta.vwap(df["price"].shift(1), df["price"], df["volume"])
    df["ADX"] = ta.adx(df["price"], df["price"], df["price"], length=14)["ADX_14"]
    high = df["price"].max()
    low = df["price"].min()
    diff = high - low
    df["Fib_38.2"] = high - diff * 0.382
    df["Fib_61.8"] = high - diff * 0.618
    return df.dropna()

# --- Fungsi untuk Mendeteksi Volatilitas Ekstrem ---
def detect_extreme_volatility(df):
    if len(df) < 2:
        return False
    recent_changes = df["price"].pct_change().dropna()
    last_change = abs(recent_changes.iloc[-1])
    return last_change > VOLATILITY_THRESHOLD

# --- Fungsi untuk Simulasi Monte Carlo ---
def monte_carlo_simulation(df, num_simulations=500, prediction_horizon=10):
    returns = df["price"].pct_change().dropna()
    mean_return = returns.mean()
    std_return = returns.std()
    
    simulations = []
    for _ in range(num_simulations):
        simulation = [df["price"].iloc[-1]]
        for _ in range(prediction_horizon):
            random_shock = np.random.normal(mean_return, std_return)
            next_price = simulation[-1] * (1 + random_shock)
            simulation.append(next_price)
        simulations.append(simulation)
    
    simulated_final_prices = [sim[-1] for sim in simulations]
    expected_price = np.mean(simulated_final_prices)
    risk_of_loss = np.sum(np.array(simulated_final_prices) < df["price"].iloc[-1]) / num_simulations * 100
    return expected_price, risk_of_loss

# --- Fungsi untuk Stress Testing ---
def stress_test(df, crash_factor=-0.5, pump_factor=1.0):
    simulated_df = df.copy()
    
    # Simulasi Crash
    crash_prices = simulated_df["price"] * (1 + crash_factor)
    crash_volatility = crash_prices.pct_change().dropna().std()
    
    # Simulasi Pump
    pump_prices = simulated_df["price"] * (1 + pump_factor)
    pump_volatility = pump_prices.pct_change().dropna().std()
    
    return crash_volatility, pump_volatility

# --- Fungsi untuk Backtesting Sederhana pada Timeframe ---
def backtest_timeframe(coin, interval, look_back=30, num_candles=500):
    try:
        df = fetch_price_data(coin, interval, limit=num_candles)
        df = calculate_technicals(df)

        total_profit = 0
        trades = 0
        wins = 0

        for i in range(look_back, len(df) - 1):
            window = df.iloc[i-look_back:i]
            window["sentiment"] = 0  # Dummy sentiment untuk backtesting
            predicted_price, current_price, _, _ = train_and_predict_lstm(window, coin, interval, look_back=look_back)
            
            if predicted_price > current_price:  # Long
                entry_price = current_price
                future_price = df["price"].iloc[i+1]
                profit = (future_price - entry_price) / entry_price
                total_profit += profit
                trades += 1
                if profit > 0:
                    wins += 1
            elif predicted_price < current_price:  # Short
                entry_price = current_price
                future_price = df["price"].iloc[i+1]
                profit = (entry_price - future_price) / entry_price
                total_profit += profit
                trades += 1
                if profit > 0:
                    wins += 1

        win_rate = (wins / trades * 100) if trades > 0 else 0
        avg_profit_per_trade = (total_profit / trades) if trades > 0 else 0
        return win_rate, avg_profit_per_trade
    except Exception as e:
        logging.error(f"Backtesting gagal untuk {coin} pada timeframe {interval}: {str(e)}")
        return 0, 0

# --- Fungsi untuk Memilih Timeframe Otomatis dengan MTF dan Backtesting ---
def select_optimal_timeframe(coin):
    timeframe_scores = {}
    
    # Langkah 1: Analisis MTF untuk menentukan timeframe utama (tren utama)
    higher_timeframes = ["1h", "4h", "1d"]
    best_higher_tf = None
    best_higher_score = 0
    for tf_str in higher_timeframes:
        interval = AVAILABLE_TIMEFRAMES[tf_str]
        try:
            df = fetch_price_data(coin, interval, limit=500)
            df = calculate_technicals(df)

            adx = df["ADX"].iloc[-1]
            trend_strength_score = min(adx / 100, 1.0)  # Normalisasi ADX ke 0-1

            if trend_strength_score > best_higher_score:
                best_higher_score = trend_strength_score
                best_higher_tf = tf_str
        except Exception as e:
            logging.error(f"Gagal menganalisis {coin} pada timeframe {tf_str}: {str(e)}")
            continue

    if best_higher_tf is None:
        best_higher_tf = "1h"  # Default jika gagal

    # Langkah 2: Backtesting untuk timeframe sekunder (sinyal entry/exit)
    lower_timeframes = ["1m", "5m", "15m", "1h"]
    for tf_str in lower_timeframes:
        interval = AVAILABLE_TIMEFRAMES[tf_str]
        try:
            df = fetch_price_data(coin, interval, limit=500)
            df = calculate_technicals(df)

            # Hitung volatilitas (ATR sebagai persentase dari harga)
            atr = df["ATR"].iloc[-1] / df["price"].iloc[-1]
            volatility_score = min(atr * 100, 1.0) if atr > 0 else 0
            if volatility_score > 0.5:  # Penalti jika volatilitas terlalu tinggi
                volatility_score = 0.5 / volatility_score

            # Backtesting sederhana
            win_rate, avg_profit = backtest_timeframe(coin, interval)
            performance_score = (win_rate / 100) * 0.5 + (avg_profit * 100) * 0.5  # Kombinasi win rate dan profit

            # Simulasi Monte Carlo
            expected_price, risk_of_loss = monte_carlo_simulation(df)
            risk_score = 1.0 - (risk_of_loss / 100)  # Skor lebih tinggi jika risiko rendah

            # Skor total: kombinasi volatilitas, performa historis, dan risiko
            total_score = (0.3 * volatility_score) + (0.3 * performance_score) + (0.2 * best_higher_score) + (0.2 * risk_score)
            timeframe_scores[tf_str] = total_score
            print(f"{Fore.CYAN}Skor timeframe {tf_str} untuk {coin}: {total_score:.2f} (Volatilitas: {volatility_score:.2f}, Performa: {performance_score:.2f}, Tren Utama: {best_higher_score:.2f}, Risiko: {risk_score:.2f}){Style.RESET_ALL}")
        except Exception as e:
            logging.error(f"Gagal menghitung skor untuk timeframe {tf_str} pada {coin}: {str(e)}")
            timeframe_scores[tf_str] = 0

    # Pilih timeframe dengan skor tertinggi
    optimal_timeframe = max(timeframe_scores.items(), key=lambda x: x[1])[0]
    print(f"{Fore.GREEN}Timeframe optimal untuk {coin}: {optimal_timeframe} (skor: {timeframe_scores[optimal_timeframe]:.2f}), Tren Utama: {best_higher_tf}{Style.RESET_ALL}")
    return optimal_timeframe, AVAILABLE_TIMEFRAMES[optimal_timeframe], best_higher_tf, AVAILABLE_TIMEFRAMES[best_higher_tf]

# --- Fungsi untuk Fetch dan Analisis Sentimen Berita ---
def fetch_news(rss_feeds):
    news_titles = []
    for feed_url in rss_feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                news_titles.append(entry.title.lower())
        except Exception as e:
            logging.error(f"Gagal mengambil RSS feed dari {feed_url}: {str(e)}")
    return news_titles

def analyze_sentiment(news_titles):
    crypto_count = Counter()
    bullish_count = Counter()
    bearish_count = Counter()
    global_sentiment_count = Counter()

    for title in news_titles:
        title_lower = title.lower()
        for keyword in CRYPTO_KEYWORDS:
            if keyword in title_lower:
                crypto_count[keyword] += 1
        for keyword in BULLISH_KEYWORDS:
            if keyword in title_lower:
                bullish_count[keyword] += 1
        for keyword in BEARISH_KEYWORDS:
            if keyword in title_lower:
                bearish_count[keyword] += 1
        for keyword in GLOBAL_SENTIMENT_KEYWORDS:
            if keyword in title_lower:
                global_sentiment_count[keyword] += 1

    total_crypto = sum(crypto_count.values())
    total_bullish = sum(bullish_count.values())
    total_bearish = sum(bearish_count.values())
    total_global = sum(global_sentiment_count.values())
    sentiment_score = (total_bullish - total_bearish) / (total_bullish + total_bearish + 1)

    return total_crypto, total_bullish, total_bearish, total_global, sentiment_score, crypto_count

# --- Fungsi untuk Melatih dan Memprediksi dengan LSTM dan Random Forest (Ensemble) ---
def train_and_predict_lstm(df, coin, timeframe, units=75, epochs=30, look_back=30):
    news_titles = fetch_news(RSS_FEEDS)
    _, bullish_score, bearish_score, _, _, _ = analyze_sentiment(news_titles)
    sentiment_score = (bullish_score - bearish_score) / (bullish_score + bearish_score + 1)
    df["sentiment"] = sentiment_score
    
    # Tambahkan data on-chain
    onchain_volume = fetch_onchain_data(coin)
    df["onchain_volume"] = onchain_volume  # Volume transaksi harian sebagai fitur
    
    model_file = f"lstm_model_{coin}_{timeframe}.h5"
    scaler_file = f"scaler_{coin}_{timeframe}.pkl"
    rf_model_file = f"rf_model_{coin}_{timeframe}.pkl"
    
    if os.path.exists(model_file) and os.path.exists(scaler_file):
        model = load_model(model_file)
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
        logging.info(f"Memuat model LSTM untuk {coin} ({timeframe})")
        print(f"{Fore.GREEN}Memuat model LSTM untuk {coin} ({timeframe})...{Style.RESET_ALL}")
    else:
        X, y, scaler = prepare_lstm_data(df, look_back)
        model = Sequential()
        model.add(LSTM(units=units, return_sequences=True, input_shape=(look_back, len(df.columns)-2)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=units))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
        model.save(model_file)
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        logging.info(f"Model LSTM baru dilatih untuk {coin} ({timeframe})")
        print(f"{Fore.GREEN}Model LSTM baru dilatih dan disimpan untuk {coin} ({timeframe}).{Style.RESET_ALL}")
    
    last_data = scaler.transform(df.tail(look_back)[["price", "SMA_20", "RSI", "BB_upper", "BB_lower", "MACD", "MACD_signal", "ATR", "OBV", "VWAP", "sentiment", "ADX", "onchain_volume"]])
    last_data = np.reshape(last_data, (1, look_back, len(df.columns)-2))
    predicted_scaled_lstm = model.predict(last_data)
    predicted_price_lstm = scaler.inverse_transform(
        np.concatenate([predicted_scaled_lstm, np.zeros((1, len(df.columns)-3))), axis=1)
    )[0][0]
    
    X, y, _ = prepare_lstm_data(df, look_back)
    y_pred_scaled_lstm = model.predict(X)
    y_pred_lstm = scaler.inverse_transform(np.concatenate([y_pred_scaled_lstm, np.zeros((len(y_pred_scaled_lstm), len(df.columns)-3))), axis=1))[:, 0]
    y_true = scaler.inverse_transform(np.concatenate([y.reshape(-1, 1), np.zeros((len(y), len(df.columns)-3))), axis=1))[:, 0]
    mae_lstm = np.mean(np.abs(y_true - y_pred_lstm))
    
    # Random Forest untuk ensemble
    if os.path.exists(rf_model_file):
        with open(rf_model_file, 'rb') as f:
            rf_model = pickle.load(f)
    else:
        features = ["price", "SMA_20", "RSI", "BB_upper", "BB_lower", "MACD", "MACD_signal", "ATR", "OBV", "VWAP", "sentiment", "ADX", "onchain_volume"]
        X_rf = df[features].fillna(0).values
        y_rf = df["price"].shift(-1).fillna(method='ffill').values
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_rf[:-1], y_rf[:-1])
        with open(rf_model_file, 'wb') as f:
            pickle.dump(rf_model, f)
    
    features = ["price", "SMA_20", "RSI", "BB_upper", "BB_lower", "MACD", "MACD_signal", "ATR", "OBV", "VWAP", "sentiment", "ADX", "onchain_volume"]
    last_rf_data = df[features].tail(1).fillna(0).values
    predicted_price_rf = rf_model.predict(last_rf_data)[0]
    
    # Ensemble: rata-rata prediksi LSTM dan RF
    current_price = df["price"].iloc[-1]
    predicted_price = (0.6 * predicted_price_lstm + 0.4 * predicted_price_rf)
    
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("INSERT INTO predictions (timestamp, coin, current_price, predicted_price, mae) VALUES (?, ?, ?, ?, ?)",
                   (timestamp, coin, current_price, predicted_price, mae_lstm))
    conn.commit()
    
    return predicted_price, current_price, df, mae_lstm

# --- Fungsi untuk Menghitung Parameter Trading ---
def calculate_trading_params(df, current_price, predicted_price):
    atr = df["ATR"].iloc[-1]
    fib_38_2 = df["Fib_38.2"].iloc[-1]
    fib_61_8 = df["Fib_61.8"].iloc[-1]
    trend = "bullish" if predicted_price > current_price else "bearish"
    
    entry_price = current_price
    if trend == "bullish":
        stop_loss = max(entry_price - atr * 1.5, fib_61_8)
        take_profit = min(entry_price + atr * 3, fib_38_2)
        potential_direction = "Long (Beli)"
    else:
        stop_loss = min(entry_price + atr * 1.5, fib_38_2)
        take_profit = max(entry_price - atr * 3, fib_61_8)
        potential_direction = "Short (Jual)"
    
    risk_per_trade = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    risk_reward_ratio = reward / risk_per_trade if risk_per_trade > 0 else 0
    volatility_factor = atr / entry_price
    leverage = min(10, max(1, int(10 / (volatility_factor * 100))))
    
    return entry_price, stop_loss, take_profit, leverage, risk_reward_ratio, potential_direction

# --- Fungsi untuk Mengatur Mode Margin dan Alokasi Margin ---
def set_isolated_margin(symbol, leverage, margin_amount):
    try:
        # Set mode margin ke Isolated
        client.futures_change_margin_type(symbol=symbol, marginType="ISOLATED")
        print(f"{Fore.GREEN}Mode margin diatur ke Isolated untuk {symbol}{Style.RESET_ALL}")
    except Exception as e:
        if "No need to change margin type" in str(e):
            pass  # Margin sudah Isolated
        else:
            logging.error(f"Gagal mengatur mode margin untuk {symbol}: {str(e)}")
            print(f"{Fore.RED}Gagal mengatur mode margin untuk {symbol}: {str(e)}{Style.RESET_ALL}")

    try:
        # Alokasi margin untuk posisi
        client.futures_change_position_margin(symbol=symbol, amount=margin_amount, type=1)  # Type 1 = tambah margin
        print(f"{Fore.GREEN}Margin dialokasikan: {margin_amount} untuk {symbol}{Style.RESET_ALL}")
    except Exception as e:
        logging.error(f"Gagal mengatur margin untuk {symbol}: {str(e)}")
        print(f"{Fore.RED}Gagal mengatur margin untuk {symbol}: {str(e)}{Style.RESET_ALL}")

# --- Fungsi untuk Mengatur Trailing Stop-Loss ---
def set_trailing_stop(symbol, direction, quantity, activation_price, callback_rate=0.5):
    try:
        client.futures_create_order(
            symbol=symbol,
            side=SIDE_SELL if direction == "Long (Beli)" else SIDE_BUY,
            type="TRAILING_STOP_MARKET",
            quantity=quantity,
            activationPrice=activation_price,
            callbackRate=callback_rate
        )
        print(f"{Fore.GREEN}Trailing stop-loss diatur untuk {symbol} (Activation: {activation_price}, Callback: {callback_rate}%)}{Style.RESET_ALL}")
    except Exception as e:
        logging.error(f"Gagal mengatur trailing stop-loss untuk {symbol}: {str(e)}")
        print(f"{Fore.RED}Gagal mengatur trailing stop-loss untuk {symbol}: {str(e)}{Style.RESET_ALL}")

# --- Fungsi untuk Menghitung Skor Koin (Menggunakan Multithreading) ---
def calculate_coin_score_wrapper(coin, interval, look_back):
    return coin, calculate_coin_score(coin, interval, look_back)

# --- Fungsi untuk Menghitung Skor Koin ---
def calculate_coin_score(coin, interval=Client.KLINE_INTERVAL_1HOUR, look_back=30):
    try:
        df = fetch_price_data(coin, interval=interval, limit=500)
        df = calculate_technicals(df)

        # Volatilitas (diukur dengan ATR)
        atr = df["ATR"].iloc[-1] / df["price"].iloc[-1]  # Rasio ATR terhadap harga
        volatility_score = min(atr * 100, 1.0)  # Skala 0-1

        # Volume trading
        volume = df["volume"].mean()
        volume_score = min(volume / 1_000_000, 1.0)  # Skala 0-1, normalisasi arbitrer

        # Prediksi harga dengan LSTM
        predicted_price, current_price, df, mae = train_and_predict_lstm(df, coin, interval, look_back=look_back)
        price_movement = abs(predicted_price - current_price) / current_price
        price_score = min(price_movement * 10, 1.0)  # Skala 0-1

        # Skor total (weighted average)
        total_score = (0.4 * volatility_score) + (0.3 * volume_score) + (0.3 * price_score)
        return total_score, predicted_price, current_price, df, mae
    except Exception as e:
        logging.error(f"Gagal menghitung skor untuk {coin}: {str(e)}")
        return 0, 0, 0, None, float('inf')

# --- Fungsi untuk Scanning dan Pemilihan Koin dengan Multithreading ---
def scan_and_select_coin(coins, interval=Client.KLINE_INTERVAL_1HOUR, look_back=30, min_score=0.5):
    scores = []
    with ThreadPoolExecutor(max_workers=5) as executor:  # Batasi 5 worker untuk perangkat Anda
        futures = [executor.submit(calculate_coin_score_wrapper, coin, interval, look_back) for coin in coins]
        for future in futures:
            coin, result = future.result()
            score, predicted_price, current_price, df, mae = result
            if df is not None and score >= min_score:
                scores.append({
                    'coin': coin,
                    'score': score,
                    'predicted_price': predicted_price,
                    'current_price': current_price,
                    'df': df,
                    'mae': mae
                })
                logging.info(f"Skor {coin}: {score:.2f}")
                print(f"{Fore.YELLOW}Skor {coin}: {score:.2f}{Style.RESET_ALL}")

    if not scores:
        return None, None, None, None, None

    best_coin = max(scores, key=lambda x: x['score'])
    return best_coin['coin'], best_coin['predicted_price'], best_coin['current_price'], best_coin['df'], best_coin['mae']

# --- Fungsi untuk Eksekusi Trade ---
def execute_trade(coin, entry_price, stop_loss, take_profit, leverage, direction, capital, timeframe_str):
    global error_count, active_trades
    symbol = f"{coin}USDT"
    balance = check_balance()
    max_risk = balance * 0.5
    quantity = (max_risk * leverage) / entry_price
    margin_amount = max_risk / leverage  # Alokasi margin per posisi
    retries = 3
    
    for attempt in range(retries):
        try:
            client.futures_change_leverage(symbol=symbol, leverage=leverage)
            set_isolated_margin(symbol, leverage, margin_amount)

            if direction == "Long (Beli)":
                order = client.futures_create_order(
                    symbol=symbol,
                    side=SIDE_BUY,
                    type=ORDER_TYPE_MARKET,
                    quantity=round(quantity, 3)
                )
                msg = f"Membuka posisi LONG untuk {coin}USDT: {quantity:.3f} unit pada ${entry_price:.2f}"
                activation_price = entry_price * 1.02  # Aktivasi trailing stop pada profit 2%
            else:
                order = client.futures_create_order(
                    symbol=symbol,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_MARKET,
                    quantity=round(quantity, 3)
                )
                msg = f"Membuka posisi SHORT untuk {coin}USDT: {quantity:.3f} unit pada ${entry_price:.2f}"
                activation_price = entry_price * 0.98  # Aktivasi trailing stop pada profit 2%
            
            order_id = str(order['orderId'])
            
            # Set trailing stop-loss
            set_trailing_stop(symbol, direction, round(quantity, 3), activation_price)

            stop_msg = f"Stop Loss: ${stop_loss:.2f}, Take Profit: ${take_profit:.2f}"
            logging.info(msg)
            logging.info(stop_msg)
            print(f"{Fore.GREEN}{msg}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{stop_msg}{Style.RESET_ALL}")
            
            timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute("INSERT INTO trades (timestamp, coin, direction, entry_price, stop_loss, take_profit, leverage, profit_loss, balance, order_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (timestamp, coin, direction, entry_price, stop_loss, take_profit, leverage, 0, balance, order_id))
            conn.commit()
            
            active_trades[order_id] = {
                'coin': coin,
                'direction': direction,
                'entry_price': entry_price,
                'quantity': quantity,
                'leverage': leverage,
                'open_time': time.time(),
                'timeframe': timeframe_str  # Untuk menentukan timeout berdasarkan timeframe
            }
            return True
        except Exception as e:
            logging.error(f"Percobaan {attempt+1}/{retries} gagal: {str(e)}")
            if attempt < retries - 1:
                time.sleep(5)
            else:
                return False
    return False

# --- Fungsi untuk Memeriksa Saldo ---
def check_balance():
    try:
        balance = float(client.futures_account_balance()[0]['balance'])
        return balance
    except Exception as e:
        logging.error(f"Gagal memeriksa saldo: {str(e)}")
        return initial_capital

# --- Fungsi untuk Memperbarui Profit/Loss dan Timeout Posisi ---
def update_trade_profit_loss(higher_df, higher_trend):
    global active_trades
    try:
        positions = client.futures_position_information()
        current_time = time.time()
        for order_id, trade in list(active_trades.items()):
            symbol = f"{trade['coin']}USDT"
            timeframe = trade['timeframe']
            timeout = POSITION_TIMEOUT.get(timeframe, 12 * 3600)  # Default 12 jam jika timeframe tidak ditemukan

            # Timeout posisi jika tidak menguntungkan
            if (current_time - trade['open_time']) > timeout:
                current_price = float(client.get_symbol_ticker(symbol=symbol)["price"])
                unrealized_pnl = ((current_price - trade['entry_price']) / trade['entry_price'] * trade['leverage']) if trade['direction'] == "Long (Beli)" else ((trade['entry_price'] - current_price) / trade['entry_price'] * trade['leverage'])
                if unrealized_pnl < 0:
                    client.futures_create_order(
                        symbol=symbol,
                        side=SIDE_SELL if trade['direction'] == "Long (Beli)" else SIDE_BUY,
                        type=ORDER_TYPE_MARKET,
                        quantity=round(trade['quantity'], 3)
                    )
                    logging.info(f"Posisi {trade['direction']} untuk {trade['coin']}USDT ditutup karena timeout. Unrealized P/L: {unrealized_pnl:.2f}")
                    print(f"{Fore.YELLOW}Posisi {trade['direction']} untuk {trade['coin']}USDT ditutup karena timeout. Unrealized P/L: {unrealized_pnl:.2f}{Style.RESET_ALL}")
                    del active_trades[order_id]
                    continue

            # Tutup posisi jika tren utama berbalik
            if higher_trend == "bearish" and trade['direction'] == "Long (Beli)":
                current_price = float(client.get_symbol_ticker(symbol=symbol)["price"])
                unrealized_pnl = ((current_price - trade['entry_price']) / trade['entry_price'] * trade['leverage'])
                client.futures_create_order(
                    symbol=symbol,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_MARKET,
                    quantity=round(trade['quantity'], 3)
                )
                logging.info(f"Posisi {trade['direction']} untuk {trade['coin']}USDT ditutup karena tren utama berbalik. Unrealized P/L: {unrealized_pnl:.2f}")
                print(f"{Fore.YELLOW}Posisi {trade['direction']} untuk {trade['coin']}USDT ditutup karena tren utama berbalik menjadi bearish. Unrealized P/L: {unrealized_pnl:.2f}{Style.RESET_ALL}")
                del active_trades[order_id]
                continue
            elif higher_trend == "bullish" and trade['direction'] == "Short (Jual)":
                current_price = float(client.get_symbol_ticker(symbol=symbol)["price"])
                unrealized_pnl = ((trade['entry_price'] - current_price) / trade['entry_price'] * trade['leverage'])
                client.futures_create_order(
                    symbol=symbol,
                    side=SIDE_BUY,
                    type=ORDER_TYPE_MARKET,
                    quantity=round(trade['quantity'], 3)
                )
                logging.info(f"Posisi {trade['direction']} untuk {trade['coin']}USDT ditutup karena tren utama berbalik. Unrealized P/L: {unrealized_pnl:.2f}")
                print(f"{Fore.YELLOW}Posisi {trade['direction']} untuk {trade['coin']}USDT ditutup karena tren utama berbalik menjadi bullish. Unrealized P/L: {unrealized_pnl:.2f}{Style.RESET_ALL}")
                del active_trades[order_id]
                continue

            # Periksa jika posisi sudah ditutup oleh trailing stop atau lainnya
            for pos in positions:
                if pos['symbol'] == symbol and float(pos['positionAmt']) == 0:
                    realized_pnl = float(pos['realizedPnl'])
                    cursor.execute("UPDATE trades SET profit_loss = ? WHERE order_id = ? AND profit_loss = 0",
                                   (realized_pnl, order_id))
                    conn.commit()
                    logging.info(f"Posisi {trade['direction']} untuk {trade['coin']}USDT tertutup. Profit/Loss: ${realized_pnl:.2f}")
                    print(f"{Fore.GREEN if realized_pnl >= 0 else Fore.RED}Posisi {trade['direction']} untuk {trade['coin']}USDT tertutup. Profit/Loss: ${realized_pnl:.2f}{Style.RESET_ALL}")
                    del active_trades[order_id]
                    break
    except Exception as e:
        logging.error(f"Gagal memeriksa profit/loss: {str(e)}")

# --- Fungsi untuk Analisis Berita (opsional) ---
def analyze_news():
    news_titles = fetch_news(RSS_FEEDS)
    total_crypto, total_bullish, total_bearish, total_global, sentiment_score, crypto_count = analyze_sentiment(news_titles)
    output = f"\n{Fore.CYAN + Style.BRIGHT}=== Analisis Berita ==={Style.RESET_ALL}\n"
    output += f"Total Berita Crypto: {Fore.GREEN}{total_crypto}{Style.RESET_ALL}\n"
    output += f"Sentimen Bullish: {Fore.GREEN}{total_bullish}{Style.RESET_ALL}\n"
    output += f"Sentimen Bearish: {Fore.RED}{total_bearish}{Style.RESET_ALL}\n"
    output += f"Sentimen Global: {Fore.YELLOW}{total_global}{Style.RESET_ALL}\n"
    output += f"Skor Sentimen: {Fore.MAGENTA}{sentiment_score:.2f}{Style.RESET_ALL}\n"
    output += f"Koin yang Disebut: {Fore.CYAN}{dict(crypto_count)}{Style.RESET_ALL}\n"
    return output

# --- Fungsi Trading Otomatis (Single Coin Mode) ---
def auto_trade(coin, timeframe_str=None, interval_seconds=60, capital=initial_capital):
    global real_time_data, error_count, trade_count, last_day, ws_active

    # Informasi tentang kemampuan skrip
    print(f"{Fore.YELLOW}[INFO] Skrip ini dapat menganalisis dan mentradingkan semua koin yang tersedia di Binance Futures.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}[INFO] Data on-chain saat ini hanya tersedia untuk Bitcoin (volume transaksi) dan Ethereum (gas fees).{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}[INFO] Untuk koin lain, analisis menggunakan data harga, indikator teknikal, dan sentimen berita.{Style.RESET_ALL}")

    # Tentukan timeframe secara otomatis jika tidak diberikan
    if timeframe_str is None:
        timeframe_str, timeframe, higher_tf, higher_interval = select_optimal_timeframe(coin)
    else:
        timeframe_options = {
            "1m": Client.KLINE_INTERVAL_1MINUTE,
            "5m": Client.KLINE_INTERVAL_5MINUTE,
            "15m": Client.KLINE_INTERVAL_15MINUTE,
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "4h": Client.KLINE_INTERVAL_4HOUR,
            "1d": Client.KLINE_INTERVAL_1DAY
        }
        timeframe = timeframe_options.get(timeframe_str, Client.KLINE_INTERVAL_1HOUR)
        higher_tf = "1h"  # Default higher timeframe
        higher_interval = Client.KLINE_INTERVAL_1HOUR

    timeframe_display = timeframe_str if timeframe_str in AVAILABLE_TIMEFRAMES else "1h"

    units = 75
    epochs = 30
    look_back = 30

    print(f"\n{Fore.GREEN}Memulai trading otomatis untuk {coin}USDT pada timeframe {timeframe_display}...{Style.RESET_ALL}")
    price_df = fetch_price_data(coin, interval=timeframe, limit=500)
    price_df = calculate_technicals(price_df)

    # Simulasi Monte Carlo sebelum trading
    expected_price, risk_of_loss = monte_carlo_simulation(price_df)
    print(f"{Fore.CYAN}Simulasi Monte Carlo: Harga Ekspektasi: ${expected_price:.2f}, Risiko Kerugian: {risk_of_loss:.2f}%{Style.RESET_ALL}")

    # Stress testing sebelum trading
    crash_volatility, pump_volatility = stress_test(price_df)
    print(f"{Fore.CYAN}Stress Test: Volatilitas Crash (-50%): {crash_volatility:.4f}, Volatilitas Pump (+100%): {pump_volatility:.4f}{Style.RESET_ALL}")

    ws = start_websocket(coin, timeframe_str)
    print(f"{Fore.GREEN}WebSocket aktif untuk data real-time {coin}USDT...{Style.RESET_ALL}")

    while True:
        try:
            current_day = time.strftime("%Y-%m-%d")
            if last_day != current_day:
                trade_count = 0
                last_day = current_day
            
            if trade_count >= 5:
                logging.info("Batas perdagangan harian tercapai, jeda sampai besok.")
                print(f"{Fore.YELLOW}Batas perdagangan harian tercapai, jeda sampai besok.{Style.RESET_ALL}")
                time.sleep(86400 - (time.time() % 86400))
                continue
            
            if not ws_active:
                logging.warning("WebSocket mati, mencoba reconnect...")
                print(f"{Fore.YELLOW}WebSocket mati, mencoba reconnect...{Style.RESET_ALL}")
                ws = start_websocket(coin, timeframe_str)
                time.sleep(5)
                ws_active = True
            
            real_time_data = []
            time.sleep(10)
            
            if real_time_data:
                real_time_df = pd.DataFrame(real_time_data)
                real_time_df.set_index("timestamp", inplace=True)
                price_df = pd.concat([price_df, real_time_df]).drop_duplicates().sort_index().tail(500)
                price_df = calculate_technicals(price_df)

            # Deteksi volatilitas ekstrem
            if detect_extreme_volatility(price_df):
                print(f"{Fore.RED}Volatilitas ekstrem terdeteksi untuk {coin}. Menghentikan trading sementara...{Style.RESET_ALL}")
                time.sleep(300)  # Jeda 5 menit
                continue

            # Periksa tren utama pada higher timeframe
            higher_df = fetch_price_data(coin, higher_interval, limit=500)
            higher_df = calculate_technicals(higher_df)
            higher_adx = higher_df["ADX"].iloc[-1]
            higher_trend = "bullish" if higher_df["MACD"].iloc[-1] > higher_df["MACD_signal"].iloc[-1] else "bearish"

            predicted_price, current_price, df, mae = train_and_predict_lstm(price_df, coin, timeframe_display, units, epochs, look_back)
            entry_price, stop_loss, take_profit, leverage, risk_reward_ratio, potential_direction = calculate_trading_params(df, current_price, predicted_price)

            # Sesuaikan keputusan trading berdasarkan tren utama
            if higher_trend == "bullish" and potential_direction != "Long (Beli)":
                print(f"{Fore.YELLOW}Tren utama ({higher_tf}) adalah bullish, membatalkan sinyal short.{Style.RESET_ALL}")
                time.sleep(interval_seconds)
                continue
            if higher_trend == "bearish" and potential_direction != "Short (Jual)":
                print(f"{Fore.YELLOW}Tren utama ({higher_tf}) adalah bearish, membatalkan sinyal long.{Style.RESET_ALL}")
                time.sleep(interval_seconds)
                continue
            
            balance = check_balance()
            logging.info(f"Saldo saat ini: ${balance:.2f}")
            print(f"{Fore.WHITE}Saldo saat ini: ${balance:.2f}{Style.RESET_ALL}")
            
            update_trade_profit_loss(higher_df, higher_trend)
            
            if balance < 5:
                logging.warning(f"Saldo ${balance:.2f} terlalu rendah untuk trade, menunggu...")
                print(f"{Fore.RED}Saldo ${balance:.2f} terlalu rendah untuk trade, menunggu...{Style.RESET_ALL}")
                time.sleep(interval_seconds)
                continue
            
            if balance <= 2:
                logging.warning("Saldo di bawah $2, menghentikan bot.")
                print(f"{Fore.RED}Saldo di bawah $2, menghentikan bot.{Style.RESET_ALL}")
                break
            if balance >= 15:
                logging.info("Profit target $5 tercapai, menghentikan bot.")
                print(f"{Fore.GREEN}Profit target $5 tercapai, menghentikan bot.{Style.RESET_ALL}")
                break
            if error_count >= 3:
                logging.warning("Terlalu banyak error, jeda 5 menit.")
                print(f"{Fore.RED}Terlalu banyak error, jeda 5 menit.{Style.RESET_ALL}")
                time.sleep(300)
                error_count = 0
                continue
            
            output = f"\n{Fore.CYAN + Style.BRIGHT}=== Prediksi Harga {coin}USDT ({timeframe_display}) ==={Style.RESET_ALL}\n"
            output += f"Harga Terkini: {Fore.GREEN}${current_price:,.2f}{Style.RESET_ALL}\n"
            output += f"Harga Prediksi (LSTM + RF): {Fore.GREEN}${predicted_price:,.2f}{Style.RESET_ALL}\n"
            output += f"Tren Utama ({higher_tf}): {Fore.MAGENTA}{higher_trend}{Style.RESET_ALL}\n"
            output += f"\n{Fore.YELLOW + Style.BRIGHT}Rekomendasi Trading:{Style.RESET_ALL}\n"
            output += f"- Arah: {Fore.MAGENTA}{potential_direction}{Style.RESET_ALL}\n"
            output += f"- Entry Price: {Fore.GREEN}${entry_price:,.2f}{Style.RESET_ALL}\n"
            output += f"- Stop Loss: {Fore.RED}${stop_loss:,.2f}{Style.RESET_ALL}\n"
            output += f"- Take Profit: {Fore.GREEN}${take_profit:,.2f}{Style.RESET_ALL}\n"
            output += f"- Leverage: {Fore.CYAN}{leverage}x{Style.RESET_ALL}\n"
            output += f"- MAE: {Fore.YELLOW}${mae:.2f}{Style.RESET_ALL}\n"
            print(output)
            logging.info(output.replace(Fore.GREEN, "").replace(Fore.RED, "").replace(Fore.CYAN, "").replace(Fore.YELLOW, "").replace(Fore.MAGENTA, "").replace(Style.RESET_ALL, ""))
            
            if risk_reward_ratio >= 1.5:
                if execute_trade(coin, entry_price, stop_loss, take_profit, leverage, potential_direction, capital, timeframe_str):
                    trade_count += 1
                    error_count = 0
                else:
                    error_count += 1
            
            time.sleep(interval_seconds)
        except Exception as e:
            error_count += 1
            logging.error(f"Error: {str(e)}")
            print(f"{Fore.RED}Error: {str(e)}. Mencoba lagi dalam {interval_seconds} detik...{Style.RESET_ALL}")
            time.sleep(interval_seconds)

# --- Fungsi Trading Otomatis dengan Auto-Scanning (Opsi 3: Batasi Posisi Aktif) ---
def auto_trade_with_scanning(interval_seconds=60, capital=initial_capital, timeframe_str=None):
    global real_time_data, error_count, trade_count, last_day, ws_active

    # Informasi tentang kemampuan skrip
    print(f"{Fore.YELLOW}[INFO] Skrip ini akan memindai semua koin di Binance Futures dan memilih koin terbaik untuk ditradingkan.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}[INFO] Data on-chain saat ini hanya tersedia untuk Bitcoin (volume transaksi) dan Ethereum (gas fees).{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}[INFO] Untuk koin lain, analisis menggunakan data harga, indikator teknikal, dan sentimen berita.{Style.RESET_ALL}")

    # Tentukan timeframe secara otomatis jika tidak diberikan
    if timeframe_str is None:
        timeframe_str = "1h"  # Default sementara, akan diperbarui per koin
        timeframe = AVAILABLE_TIMEFRAMES[timeframe_str]
    else:
        timeframe_options = {
            "1m": Client.KLINE_INTERVAL_1MINUTE,
            "5m": Client.KLINE_INTERVAL_5MINUTE,
            "15m": Client.KLINE_INTERVAL_15MINUTE,
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "4h": Client.KLINE_INTERVAL_4HOUR,
            "1d": Client.KLINE_INTERVAL_1DAY
        }
        timeframe = timeframe_options.get(timeframe_str, Client.KLINE_INTERVAL_1HOUR)

    timeframe_display = timeframe_str if timeframe_str in AVAILABLE_TIMEFRAMES else "1h"

    units = 75
    epochs = 30
    look_back = 30

    # Dapatkan daftar koin yang tersedia
    coins = get_available_coins()
    if not coins:
        print(f"{Fore.RED}Tidak ada koin yang tersedia untuk scanning.{Style.RESET_ALL}")
        return

    current_coin = None
    ws = None
    higher_tf = "1h"  # Default higher timeframe
    higher_interval = Client.KLINE_INTERVAL_1HOUR

    while True:
        try:
            current_day = time.strftime("%Y-%m-%d")
            if last_day != current_day:
                trade_count = 0
                last_day = current_day

            if trade_count >= 5:
                logging.info("Batas perdagangan harian tercapai, jeda sampai besok.")
                print(f"{Fore.YELLOW}Batas perdagangan harian tercapai, jeda sampai besok.{Style.RESET_ALL}")
                time.sleep(86400 - (time.time() % 86400))
                continue

            # Periksa jumlah posisi aktif
            # Gunakan koin default (BTC) jika current_coin belum ada
            if current_coin is None:
                temp_coin = "BTC"
            else:
                temp_coin = current_coin
            higher_df = fetch_price_data(temp_coin, higher_interval, limit=500)
            higher_df = calculate_technicals(higher_df)
            higher_trend = "bullish" if higher_df["MACD"].iloc[-1] > higher_df["MACD_signal"].iloc[-1] else "bearish"
            update_trade_profit_loss(higher_df, higher_trend)  # Pastikan posisi yang tertutup diperbarui
            if len(active_trades) >= MAX_ACTIVE_POSITIONS:
                print(f"{Fore.YELLOW}Maksimum posisi aktif ({MAX_ACTIVE_POSITIONS}) tercapai. Menunggu posisi ditutup sebelum membuka posisi baru...{Style.RESET_ALL}")
                time.sleep(interval_seconds)
                continue

            print(f"{Fore.GREEN}Memulai scanning koin...{Style.RESET_ALL}")
            best_coin, predicted_price, current_price, df, mae = scan_and_select_coin(coins, timeframe, look_back)

            if best_coin is None:
                print(f"{Fore.RED}Tidak ada koin yang memenuhi kriteria (skor minimum 0.5). Tunggu {interval_seconds} detik sebelum scanning ulang...{Style.RESET_ALL}")
                time.sleep(interval_seconds)
                continue

            # Hindari membuka posisi baru jika koin sudah ada di posisi aktif
            if best_coin in [trade['coin'] for trade in active_trades.values()]:
                print(f"{Fore.YELLOW}{best_coin} sudah ada di posisi aktif. Memilih koin lain atau menunggu posisi selesai.{Style.RESET_ALL}")
                time.sleep(interval_seconds)
                continue

            # Tentukan timeframe optimal untuk koin ini
            if best_coin != current_coin or timeframe_str is None:
                timeframe_str, timeframe, higher_tf, higher_interval = select_optimal_timeframe(best_coin)
                timeframe_display = timeframe_str

            if best_coin != current_coin:
                if ws is not None:
                    ws.close()
                print(f"{Fore.GREEN}Berpindah ke koin terbaik: {best_coin}{Style.RESET_ALL}")
                current_coin = best_coin
                ws = start_websocket(current_coin, timeframe_str)
                print(f"{Fore.GREEN}WebSocket aktif untuk data real-time {current_coin}USDT...{Style.RESET_ALL}")

            if not ws_active:
                logging.warning("WebSocket mati, mencoba reconnect...")
                print(f"{Fore.YELLOW}WebSocket mati, mencoba reconnect...{Style.RESET_ALL}")
                ws = start_websocket(current_coin, timeframe_str)
                time.sleep(5)
                ws_active = True

            real_time_data = []
            time.sleep(10)

            if real_time_data:
                real_time_df = pd.DataFrame(real_time_data)
                real_time_df.set_index("timestamp", inplace=True)
                df = pd.concat([df, real_time_df]).drop_duplicates().sort_index().tail(500)
                df = calculate_technicals(df)

            # Deteksi volatilitas ekstrem
            if detect_extreme_volatility(df):
                print(f"{Fore.RED}Volatilitas ekstrem terdeteksi untuk {current_coin}. Menghentikan trading sementara...{Style.RESET_ALL}")
                time.sleep(300)  # Jeda 5 menit
                continue

            # Periksa tren utama pada higher timeframe
            higher_df = fetch_price_data(current_coin, higher_interval, limit=500)
            higher_df = calculate_technicals(higher_df)
            higher_adx = higher_df["ADX"].iloc[-1]
            higher_trend = "bullish" if higher_df["MACD"].iloc[-1] > higher_df["MACD_signal"].iloc[-1] else "bearish"

            predicted_price, current_price, df, mae = train_and_predict_lstm(df, current_coin, timeframe_display, units, epochs, look_back)
            entry_price, stop_loss, take_profit, leverage, risk_reward_ratio, potential_direction = calculate_trading_params(df, current_price, predicted_price)

            # Sesuaikan keputusan trading berdasarkan tren utama
            if higher_trend == "bullish" and potential_direction != "Long (Beli)":
                print(f"{Fore.YELLOW}Tren utama ({higher_tf}) adalah bullish, membatalkan sinyal short.{Style.RESET_ALL}")
                time.sleep(interval_seconds)
                continue
            if higher_trend == "bearish" and potential_direction != "Short (Jual)":
                print(f"{Fore.YELLOW}Tren utama ({higher_tf}) adalah bearish, membatalkan sinyal long.{Style.RESET_ALL}")
                time.sleep(interval_seconds)
                continue

            balance = check_balance()
            logging.info(f"Saldo saat ini: ${balance:.2f}")
            print(f"{Fore.WHITE}Saldo saat ini: ${balance:.2f}{Style.RESET_ALL}")

            update_trade_profit_loss(higher_df, higher_trend)

            if balance < 5:
                logging.warning(f"Saldo ${balance:.2f} terlalu rendah untuk trade, menunggu...")
                print(f"{Fore.RED}Saldo ${balance:.2f} terlalu rendah untuk trade, menunggu...{Style.RESET_ALL}")
                time.sleep(interval_seconds)
                continue

            if balance <= 2:
                logging.warning("Saldo di bawah $2, menghentikan bot.")
                print(f"{Fore.RED}Saldo di bawah $2, menghentikan bot.{Style.RESET_ALL}")
                break
            if balance >= 15:
                logging.info("Profit target $5 tercapai, menghentikan bot.")
                print(f"{Fore.GREEN}Profit target $5 tercapai, menghentikan bot.{Style.RESET_ALL}")
                break
            if error_count >= 3:
                logging.warning("Terlalu banyak error, jeda 5 menit.")
                print(f"{Fore.RED}Terlalu banyak error, jeda 5 menit.{Style.RESET_ALL}")
                time.sleep(300)
                error_count = 0
                continue

            output = f"\n{Fore.CYAN + Style.BRIGHT}=== Prediksi Harga {current_coin}USDT ({timeframe_display}) ==={Style.RESET_ALL}\n"
            output += f"Harga Terkini: {Fore.GREEN}${current_price:,.2f}{Style.RESET_ALL}\n"
            output += f"Harga Prediksi (LSTM + RF): {Fore.GREEN}${predicted_price:,.2f}{Style.RESET_ALL}\n"
            output += f"Tren Utama ({higher_tf}): {Fore.MAGENTA}{higher_trend}{Style.RESET_ALL}\n"
            output += f"\n{Fore.YELLOW + Style.BRIGHT}Rekomendasi Trading:{Style.RESET_ALL}\n"
            output += f"- Arah: {Fore.MAGENTA}{potential_direction}{Style.RESET_ALL}\n"
            output += f"- Entry Price: {Fore.GREEN}${entry_price:,.2f}{Style.RESET_ALL}\n"
            output += f"- Stop Loss: {Fore.RED}${stop_loss:,.2f}{Style.RESET_ALL}\n"
            output += f"- Take Profit: {Fore.GREEN}${take_profit:,.2f}{Style.RESET_ALL}\n"
            output += f"- Leverage: {Fore.CYAN}{leverage}x{Style.RESET_ALL}\n"
            output += f"- MAE: {Fore.YELLOW}${mae:.2f}{Style.RESET_ALL}\n"
            print(output)
            logging.info(output.replace(Fore.GREEN, "").replace(Fore.RED, "").replace(Fore.CYAN, "").replace(Fore.YELLOW, "").replace(Fore.MAGENTA, "").replace(Style.RESET_ALL, ""))

            if risk_reward_ratio >= 1.5:
                if execute_trade(current_coin, entry_price, stop_loss, take_profit, leverage, potential_direction, capital, timeframe_str):
                    trade_count += 1
                    error_count = 0
                else:
                    error_count += 1

            time.sleep(interval_seconds)
        except Exception as e:
            error_count += 1
            logging.error(f"Error: {str(e)}")
            print(f"{Fore.RED}Error: {str(e)}. Mencoba lagi dalam {interval_seconds} detik...{Style.RESET_ALL}")
            time.sleep(interval_seconds)

# --- Main Menu ---
def main_menu():
    print(f"{Fore.YELLOW}[INFO] Skrip ini dapat menganalisis dan mentradingkan semua koin di Binance Futures.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}[INFO] Data on-chain saat ini hanya tersedia untuk Bitcoin (volume transaksi) dan Ethereum (gas fees).{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}[INFO] Untuk koin lain (misalnya SOL, XRP), analisis menggunakan data harga, indikator teknikal, dan sentimen berita.{Style.RESET_ALL}")
    
    while True:
        print(BANNER)
        print(f"{Fore.YELLOW + Style.BRIGHT}1. Mulai Trading Otomatis (Single Coin){Style.RESET_ALL}")
        print(f"{Fore.YELLOW + Style.BRIGHT}2. Mulai Trading Otomatis (Auto-Scan Coins){Style.RESET_ALL}")
        print(f"{Fore.YELLOW + Style.BRIGHT}3. Analisis melalui Berita{Style.RESET_ALL}")
        print(f"{Fore.YELLOW + Style.BRIGHT}4. Keluar{Style.RESET_ALL}")
        choice = input(f"{Fore.WHITE}Pilih menu (1-4): {Style.RESET_ALL}")

        if choice == "1":
            coin = input(f"{Fore.YELLOW}Masukkan simbol crypto (contoh: BTC, ETH, XRP): {Style.RESET_ALL}").upper()
            timeframe_choice = input(f"{Fore.YELLOW}Masukkan timeframe (1m, 5m, 15m, 1h, 4h, 1d) atau tekan Enter untuk otomatis: {Style.RESET_ALL}").lower()
            timeframe = timeframe_choice if timeframe_choice else None
            auto_trade(coin, timeframe, interval_seconds=60, capital=10)
        elif choice == "2":
            # Melanjutkan kode yang terpotong di sini
            timeframe_choice = input(f"{Fore.YELLOW}Masukkan timeframe default untuk scanning (1m, 5m, 15m, 1h, 4h, 1d) atau tekan Enter untuk otomatis (1h): {Style.RESET_ALL}").lower()
            timeframe = timeframe_choice if timeframe_choice else None
            auto_trade_with_scanning(interval_seconds=60, capital=10, timeframe_str=timeframe) # Memanggil fungsi auto_trade_with_scanning
        elif choice == "3":
            news_output = analyze_news()
            print(news_output)
            input(f"{Fore.CYAN}Tekan Enter untuk kembali ke menu utama...{Style.RESET_ALL}") # Menambahkan jeda agar user bisa membaca output berita
        elif choice == "4":
            print(f"{Fore.GREEN}Menghentikan bot... Sampai jumpa lagi!{Style.RESET_ALL}")
            if ws_active: # Memastikan WebSocket ditutup sebelum keluar
                try:
                    client.futures_cancel_all_open_orders() # Membatalkan semua order terbuka
                    conn.close() # Menutup koneksi database
                    print(f"{Fore.GREEN}Order terbuka dibatalkan dan koneksi database ditutup.{Style.RESET_ALL}")
                except Exception as e:
                    logging.error(f"Gagal menutup sumber daya saat keluar: {str(e)}")
                    print(f"{Fore.RED}Gagal menutup sumber daya saat keluar: {str(e)}{Style.RESET_ALL}")
            break
        else:
            print(f"{Fore.RED}Pilihan tidak valid, silakan coba lagi.{Style.RESET_ALL}")

# Ini adalah bagian yang kemungkinan hilang untuk memulai menu saat skrip dijalankan
if __name__ == "__main__":
    main_menu()
