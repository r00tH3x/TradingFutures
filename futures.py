import feedparser
from collections import Counter
import pandas as pd
from binance.client import Client
import numpy as np
import pandas_ta as ta
from colorama import init, Fore, Style
import smtplib
from email.mime.text import MIMEText
import os,sys
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from binance.exceptions import BinanceAPIException
import math
import warnings
warnings.filterwarnings('ignore')

# Inisialisasi colorama
init(autoreset=True)

# IMPORTANT: Replace with your actual API credentials
API_KEY = 'your_api_key'.strip()
API_SECRET = 'your_api_key'.strip()

try:
    test_client = Client(API_KEY, API_SECRET, testnet=False)
    
    print("Testing Spot Account...")
    account = test_client.get_account(recvWindow=10000)
    print("✅ Spot account access OK")
    
    print("Testing Futures Account...")
    # Coba akses futures secara terpisah
    try:
        futures_account = test_client.futures_account(recvWindow=10000)
        print("✅ Futures account access OK")
        usdt_balance = [a['walletBalance'] for a in futures_account['assets'] if a['asset'] == 'USDT'][0]
        print(f"USDT Balance: ${usdt_balance}")
    except Exception as futures_error:
        print(f"❌ Futures access failed: {futures_error}")
        print("This means Futures trading is not enabled for your API key")
        
        # Coba cek apakah futures account sudah dibuka
        try:
            print("Checking if futures account exists...")
            # Test dengan endpoint yang lebih basic
            exchange_info = test_client.futures_exchange_info()
            print("✅ Futures endpoint accessible, but account access denied")
            print("Solution: Enable 'Futures' permission in API settings")
        except:
            print("❌ No futures access at all")
            print("Solution: 1) Open futures account first 2) Enable API futures permission")

except Exception as e:
    print(f"❌ General API error: {e}")

# Validasi API credentials
if not API_KEY or not API_SECRET or API_KEY == 'your_api_key_here' or API_SECRET == 'your_api_secret_here':
    print(f"{Fore.RED}PERINGATAN: Ganti API_KEY dan API_SECRET dengan credentials Binance Anda yang sebenarnya!")
    exit()

try:
    client = Client(API_KEY, API_SECRET, testnet=True)
    client.API_URL = 'https://fapi.binance.com'  # Futures API
    client.timestamp_offset = 0
except Exception as e:
    print(f"{Fore.RED}Error menginisialisasi Binance client: {e}")
    exit()

try:
    account_info = client.futures_account()
    print(f"{Fore.GREEN}API connection successful!")
    print(f"Account balance: ${float([a['walletBalance'] for a in account_info['assets'] if a['asset'] == 'USDT'][0]):.2f}")
except Exception as e:
    print(f"{Fore.RED}API connection failed: {e}")
    print(f"{Fore.YELLOW}Please check your API key and secret, and ensure futures trading is enabled.")
    exit()
    
# Setup logging dengan format yang lebih baik
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trade_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Setup SQLite database dengan error handling
try:
    conn = sqlite3.connect('trade_history.db', check_same_thread=False)
    cursor = conn.cursor()
    
    # Improved table structure
    cursor.execute('''CREATE TABLE IF NOT EXISTS trades 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       timestamp TEXT, 
                       coin TEXT, 
                       direction TEXT, 
                       entry_price REAL, 
                       stop_loss REAL, 
                       take_profit REAL, 
                       leverage INTEGER, 
                       profit_loss REAL, 
                       balance REAL, 
                       order_id TEXT,
                       timeframe TEXT,
                       confidence REAL,
                       status TEXT DEFAULT 'OPEN')''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS predictions 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       timestamp TEXT, 
                       coin TEXT, 
                       current_price REAL, 
                       predicted_price REAL, 
                       mae REAL,
                       model_type TEXT,
                       accuracy REAL)''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS performance_metrics 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       timestamp TEXT,
                       total_trades INTEGER,
                       winning_trades INTEGER,
                       win_rate REAL,
                       total_profit REAL,
                       max_drawdown REAL,
                       sharpe_ratio REAL)''')
    conn.commit()
except Exception as e:
    logger.error(f"Database initialization error: {e}")
    exit()

# Enhanced RSS feeds with more sources
RSS_FEEDS = [
    "https://coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://www.theblock.co/feed",
    "https://crypto.news/feed/",
    "https://decrypt.co/feed",
    "https://www.coinbase.com/blog/rss.xml",
    "https://blog.binance.com/en/rss.xml"
]

# Enhanced keyword sets for better sentiment analysis
CRYPTO_KEYWORDS = {
    "bitcoin": ["bitcoin", "btc", "bitcoin price", "btc/usd"],
    "ethereum": ["ethereum", "eth", "ethereum price", "eth/usd"],
    "general": ["crypto", "cryptocurrency", "blockchain", "defi", "nft", "altcoin", "digital asset"]
}

BULLISH_KEYWORDS = [
    "rally", "surge", "boom", "bullish", "rise", "up", "increase", "pump",
    "adoption", "breakthrough", "positive", "growth", "gains", "momentum",
    "institutional", "investment", "milestone", "upgrade"
]

BEARISH_KEYWORDS = [
    "crash", "drop", "fall", "bearish", "decline", "down", "dump", "sell-off",
    "correction", "bearish", "negative", "losses", "regulatory", "ban",
    "hack", "exploit", "vulnerability", "scam"
]

def send_email_notification(subject, body, to_email='your_email@gmail.com'):
    try:
        from_email = 'your_email@gmail.com'
        password = 'your_password_app'  # Generate app password di Gmail settings > Security
        
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = from_email
        msg['To'] = to_email
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(from_email, password)
            server.sendmail(from_email, to_email, msg.as_string())
        
        logger.info(f"Email notification sent: {subject}")
    except Exception as e:
        logger.error(f"Email notification error: {e}")
        
# Enhanced global variables with better structure
class TradingState:
    def __init__(self):
        self.real_time_data = []
        self.error_count = 0
        self.trade_count = 0
        self.last_day = None
        self.initial_capital = 100  # Increased default capital
        self.ws_active = False
        self.active_trades = {}
        self.MAX_ACTIVE_POSITIONS = 2  # Increased limit
        self.VOLATILITY_THRESHOLD = 0.08  # Adjusted threshold
        self.data_cache = {}
        self.model_cache = {}
        
        # Risk management parameters
        self.MAX_RISK_PER_TRADE = 0.02  # 2% risk per trade
        self.MAX_DAILY_LOSS = 0.1  # 10% max daily loss
        self.TARGET_WIN_RATE = 0.6  # 60% target win rate
        
        # Performance tracking
        self.daily_pnl = 0
        self.total_trades_today = 0
        self.winning_trades_today = 0

# Initialize trading state
trading_state = TradingState()

# Enhanced timeframe configuration
AVAILABLE_TIMEFRAMES = {
    "1m": Client.KLINE_INTERVAL_1MINUTE,
    "5m": Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "30m": Client.KLINE_INTERVAL_30MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "1d": Client.KLINE_INTERVAL_1DAY
}

POSITION_TIMEOUT = {
    "1m": 2 * 3600,   # 2 hours for 1m
    "5m": 4 * 3600,   # 4 hours for 5m  
    "15m": 8 * 3600,  # 8 hours for 15m
    "30m": 12 * 3600, # 12 hours for 30m
    "1h": 24 * 3600,  # 24 hours for 1h
    "4h": 48 * 3600,  # 48 hours for 4h
    "1d": 96 * 3600   # 96 hours for 1d
}

# Enhanced banner
BANNER = f"""
{Fore.CYAN + Style.BRIGHT}
==================================================
    🚀 ENHANCED CRYPTO AUTO-TRADING BOT V2.0 🚀
         Advanced AI Trading System
==================================================
{Style.RESET_ALL}
"""

class WebSocketManager:
    """Enhanced WebSocket manager with reconnection logic"""
    
    def __init__(self):
        self.ws = None
        self.symbol = None
        self.interval = None
        self.reconnect_count = 0
        self.max_reconnects = 5
        
    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            if 'k' in data:
                candle = data['k']
                price = float(candle['c'])
                volume = float(candle['v'])
                timestamp = pd.to_datetime(candle['t'], unit='ms')
                
                trading_state.real_time_data.append({
                    'timestamp': timestamp, 
                    'price': price, 
                    'volume': volume,
                    'high': float(candle['h']),
                    'low': float(candle['l']),
                    'open': float(candle['o'])
                })
                
                # Keep only recent data
                if len(trading_state.real_time_data) > 100:
                    trading_state.real_time_data = trading_state.real_time_data[-50:]
                    
        except Exception as e:
            logger.error(f"WebSocket message processing error: {e}")
    
    def on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")
        trading_state.ws_active = False
        self.attempt_reconnect()
    
    def on_close(self, ws, close_status_code, close_msg):
        logger.warning("WebSocket connection closed")
        trading_state.ws_active = False
        self.attempt_reconnect()
    
    def attempt_reconnect(self):
        if self.reconnect_count < self.max_reconnects:
            self.reconnect_count += 1
            logger.info(f"Attempting reconnection {self.reconnect_count}/{self.max_reconnects}")
            time.sleep(5 * self.reconnect_count)  # Exponential backoff
            self.start_websocket(self.symbol, self.interval)
        else:
            logger.error("Maximum reconnection attempts reached")
    
    def start_websocket(self, coin, interval):
        try:
            if self.ws:
                self.ws.close()
                
            self.symbol = coin
            self.interval = interval
            symbol = f"{coin.lower()}usdt"
            ws_url = f"wss://fstream.binance.com/ws/{symbol}@kline_{interval}"
            
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            trading_state.ws_active = True
            self.reconnect_count = 0
            logger.info(f"WebSocket started for {coin} on {interval}")
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket: {e}")
            trading_state.ws_active = False

# Initialize WebSocket manager
ws_manager = WebSocketManager()

class EnhancedDataFetcher:
    """Enhanced data fetching with caching and error handling"""
    
    @staticmethod
    def get_available_coins():
        try:
            cache_key = "available_coins"
            if cache_key in trading_state.data_cache:
                return trading_state.data_cache[cache_key]
                
            exchange_info = client.get_exchange_info()
            futures_symbols = []
            
            for symbol in exchange_info['symbols']:
                if (symbol['symbol'].endswith('USDT') and 
                    symbol['status'] == 'TRADING' and
                    len(symbol['symbol'].replace('USDT', '')) <= 6):
                    futures_symbols.append(symbol['symbol'].replace('USDT', ''))
            
            # Filter top coins by volume
            top_coins = futures_symbols[:50]  # Limit to top 50 coins
            trading_state.data_cache[cache_key] = top_coins
            return top_coins
            
        except Exception as e:
            logger.error(f"Failed to get available coins: {e}")
            return ['BTC', 'ETH', 'BNB', 'XRP', 'ADA']  # Fallback coins
    
    @staticmethod
    def fetch_price_data(coin="BTC", interval=Client.KLINE_INTERVAL_1HOUR, limit=500):
        cache_key = f"{coin}_{interval}_{limit}"
        
        # Check cache first
        if cache_key in trading_state.data_cache:
            cache_time, df = trading_state.data_cache[cache_key]
            if time.time() - cache_time < 300:  # 5-minute cache
                return df
        
        symbol = f"{coin}USDT"
        try:
            klines = client.get_historical_klines(symbol, interval, f"{limit} hours ago UTC")
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignored'
            ])
            
            # Convert data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
            
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            
            # Cache the result
            trading_state.data_cache[cache_key] = (time.time(), df)
            return df
            
        except Exception as e:
            raise Exception(f"Failed to fetch data for {symbol}: {str(e)}")

class EnhancedTechnicalAnalysis:
    """Enhanced technical analysis with more indicators"""
    
    @staticmethod
    def calculate_advanced_technicals(df):
        try:
            # Basic indicators
            df["SMA_20"] = ta.sma(df["close"], length=20)
            df["SMA_50"] = ta.sma(df["close"], length=50)
            df["EMA_12"] = ta.ema(df["close"], length=12)
            df["EMA_26"] = ta.ema(df["close"], length=26)
            
            # Momentum indicators
            df["RSI"] = ta.rsi(df["close"], length=14)
            df["RSI_SMA"] = ta.sma(df["RSI"], length=5)
            
            # Bollinger Bands
            bb = ta.bbands(df["close"], length=20, std=2)
            if bb is not None and len(bb.columns) >= 3:
                df["BB_upper"] = bb.iloc[:, 0]
                df["BB_middle"] = bb.iloc[:, 1]
                df["BB_lower"] = bb.iloc[:, 2]
                df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]
            
            # MACD
            macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
            if macd is not None and len(macd.columns) >= 3:
                df["MACD"] = macd.iloc[:, 0]
                df["MACD_signal"] = macd.iloc[:, 2]
                df["MACD_histogram"] = macd.iloc[:, 1]
            
            # Volatility indicators
            df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
            df["ATR_percent"] = df["ATR"] / df["close"] * 100
            
            # Volume indicators
            df["OBV"] = ta.obv(df["close"], df["volume"])
            df["VWAP"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
            
            # Trend indicators
            adx = ta.adx(df["high"], df["low"], df["close"], length=14)
            if adx is not None and len(adx.columns) >= 3:
                df["ADX"] = adx.iloc[:, 0]
                df["DI_plus"] = adx.iloc[:, 1]
                df["DI_minus"] = adx.iloc[:, 2]
            
            # Stochastic
            stoch = ta.stoch(df["high"], df["low"], df["close"])
            if stoch is not None and len(stoch.columns) >= 2:
                df["STOCH_k"] = stoch.iloc[:, 1]
                df["STOCH_d"] = stoch.iloc[:, 0]
            
            # Williams %R
            df["WILLIAMS_R"] = ta.willr(df["high"], df["low"], df["close"])
            
            # Price channels
            df["DONCHIAN_upper"] = df["high"].rolling(20).max()
            df["DONCHIAN_lower"] = df["low"].rolling(20).min()
            
            # Fibonacci retracement levels
            period = min(50, len(df))
            high_period = df["high"].rolling(period).max()
            low_period = df["low"].rolling(period).min()
            diff = high_period - low_period

            # Standard Fibonacci levels
            df["Fib_0"] = high_period  # 0% (High)
            df["Fib_23.6"] = high_period - diff * 0.236
            df["Fib_38.2"] = high_period - diff * 0.382
            df["Fib_50.0"] = high_period - diff * 0.500
            df["Fib_61.8"] = high_period - diff * 0.618
            df["Fib_78.6"] = high_period - diff * 0.786
            df["Fib_100"] = low_period   # 100% (Low)

            # Fibonacci extensions
            df["Fib_ext_127.2"] = high_period + diff * 0.272
            df["Fib_ext_161.8"] = high_period + diff * 0.618
            df["Fib_ext_261.8"] = high_period + diff * 1.618

            # Fibonacci support/resistance strength
            current_price = df["close"].iloc[-1] if len(df) > 0 else 0
            fib_levels = [df["Fib_23.6"].iloc[-1], df["Fib_38.2"].iloc[-1], 
                          df["Fib_50.0"].iloc[-1], df["Fib_61.8"].iloc[-1], df["Fib_78.6"].iloc[-1]]
            df["Fib_support_resistance"] = min(fib_levels, key=lambda x: abs(x - current_price)) if current_price > 0 else 0
            
            return df.dropna()
            
        except Exception as e:
            logger.error(f"Technical analysis calculation error: {e}")
            return df

class EnhancedMLModels:
    """Enhanced machine learning models with ensemble approach"""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'lr': LinearRegression()
        }
        self.scaler = StandardScaler()
        self.feature_importance = {}
    
    def prepare_features(self, df, look_back=20):
        """Prepare features for ML models"""
        try:
            if len(df) < 50:
                raise ValueError(f"Data too short: {len(df)} rows < 50")
            
            feature_columns = [
                'close', 'volume', 'high', 'low', 'open',
                'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                'RSI', 'RSI_SMA', 'BB_upper', 'BB_lower', 'BB_width',
                'MACD', 'MACD_signal', 'MACD_histogram',
                'ATR', 'ATR_percent', 'OBV', 'VWAP',
                'ADX', 'DI_plus', 'DI_minus',
                'STOCH_k', 'STOCH_d', 'WILLIAMS_R'
            ]
            
            available_features = [col for col in feature_columns if col in df.columns]
            df_features = df[available_features].copy()
            
            # CLEAN DATA LEBIH KUAT
            df_features = df_features.replace([np.inf, -np.inf], np.nan)
            for col in df_features.columns:
                if df_features[col].isnull().all():
                    raise ValueError(f"Column {col} all NaN, skipping coin")
                mean_val = df_features[col].mean(skipna=True)
                if np.isnan(mean_val):  # Kalau mean NaN, skip
                    raise ValueError(f"Mean NaN in column {col}, skipping coin")
                df_features[col] = df_features[col].fillna(mean_val)
                std_val = df_features[col].std(skipna=True)
                if np.isnan(std_val) or std_val == 0:
                    df_features[col] = mean_val  # Constant value
                else:
                    df_features[col] = np.clip(df_features[col], mean_val - 3*std_val, mean_val + 3*std_val)
            
            df_features['price_change_1'] = df_features['close'].pct_change(1).fillna(0).replace([np.inf, -np.inf], 0)
            df_features['price_change_5'] = df_features['close'].pct_change(5).fillna(0).replace([np.inf, -np.inf], 0)
            df_features['volume_change'] = df_features['volume'].pct_change(1).fillna(0).replace([np.inf, -np.inf], 0)
            
            if 'RSI' in df_features.columns:
                df_features['RSI_MA'] = df_features['RSI'].rolling(3).mean().fillna(df_features['RSI'].mean())
            if 'MACD' in df_features.columns:
                df_features['MACD_MA'] = df_features['MACD'].rolling(3).mean().fillna(df_features['MACD'].mean())
            
            df_features['target'] = df_features['close'].shift(-1).fillna(df_features['close'].mean())
            
            df_features = df_features.dropna()
            
            if len(df_features) < look_back + 10:
                raise ValueError(f"Insufficient data after clean: {len(df_features)} rows")
            
            X, y = [], []
            features = df_features.drop(['target'], axis=1).values
            targets = df_features['target'].values
            
            for i in range(look_back, len(df_features)):
                X.append(features[i-look_back:i])
                y.append(targets[i])
            
            X_array = np.array(X)
            y_array = np.array(y)
            
            if np.any(np.isnan(X_array)) or np.any(np.isinf(X_array)):
                raise ValueError("Data still has NaN/inf after cleaning")
            
            return X_array, y_array, df_features.drop(['target'], axis=1).columns.tolist()
            
        except Exception as e:
            logger.error(f"Feature preparation error: {e}")
            raise
    
    def train_ensemble(self, df, coin, timeframe):
        """Train ensemble of models"""
        try:
            X, y, feature_names = self.prepare_features(df)
            
            if len(X) == 0:
                raise ValueError("No training data available")
            
            X_flat = X.reshape(X.shape[0], -1)
            
            # Split data
            split_idx = int(0.8 * len(X_flat))
            X_train, X_test = X_flat[:split_idx], X_flat[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features (fit ulang)
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train models
            predictions = {}
            scores = {}
            
            for name, model in self.models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    mae = mean_absolute_error(y_test, y_pred)
                    scores[name] = mae
                    predictions[name] = y_pred
                    
                    logger.info(f"Model {name} trained with MAE: {mae:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {name}: {e}")
                    continue
            
            if predictions:
                ensemble_pred = np.mean(list(predictions.values()), axis=0)
                ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
                
                os.makedirs("models", exist_ok=True)
                model_path = f"models/{coin}_{timeframe}_ensemble.pkl"
                
                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'models': self.models,
                        'scaler': self.scaler,
                        'feature_names': feature_names,
                        'scores': scores,
                        'ensemble_mae': ensemble_mae
                    }, f)
                
                return ensemble_mae, scores
            else:
                raise ValueError("No models trained successfully")
                
        except Exception as e:
            logger.error(f"Ensemble training error for {coin}: {e}")
            raise
    
    def predict(self, df, coin, timeframe, look_back=20):
        try:
            os.makedirs("models", exist_ok=True)
            model_path = f"models/{coin}_{timeframe}_ensemble.pkl"
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.models = model_data['models']
                    self.scaler = model_data['scaler']
                    feature_names = model_data['feature_names']
                    scores = model_data.get('scores', {})
                    
                    if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
                        logger.warning(f"Scaler not fitted for {coin}, retraining...")
                        mae, scores = self.train_ensemble(df, coin, timeframe)
            else:
                logger.info(f"Training new model for {coin}_{timeframe}")
                mae, scores = self.train_ensemble(df, coin, timeframe)
            
            X, y, _ = self.prepare_features(df, look_back)
            if len(X) == 0:
                raise ValueError("No data for prediction")
            
            last_sequence = X[-1].reshape(1, -1)
            if last_sequence.shape[1] != self.scaler.n_features_in_:
                logger.warning(f"Feature mismatch for {coin}, retraining...")
                mae, scores = self.train_ensemble(df, coin, timeframe)
                X, y, _ = self.prepare_features(df, look_back)
                last_sequence = X[-1].reshape(1, -1)
            
            last_sequence_scaled = self.scaler.transform(last_sequence)
            
            predictions = {}
            for name, model in self.models.items():
                try:
                    if not hasattr(model, 'predict') or model is None:
                        logger.warning(f"Model {name} not fitted for {coin}, skipping")
                        continue
                    # FIX: Check if GB fitted (tree_ exists)
                    if name == 'gb' and not hasattr(model, 'estimators_') or model.estimators_ is None:
                        logger.warning(f"GB model not fitted, skipping for {coin}")
                        continue
                    pred = model.predict(last_sequence_scaled)[0]
                    predictions[name] = pred
                except Exception as e:
                    logger.error(f"Prediction error for {name}: {e}")
                    continue
            
            if not predictions:
                raise ValueError("No successful predictions")
            
            weights = {name: 1.0 / (scores.get(name, 1.0) + 1e-6) for name in predictions.keys()}
            total_weight = sum(weights.values())
            weights = {name: w / total_weight for name, w in weights.items()}
            
            ensemble_prediction = sum(pred * weights[name] for name, pred in predictions.items())
            
            current_price = df['close'].iloc[-1]
            confidence = self.calculate_prediction_confidence(predictions, ensemble_prediction)
            
            return ensemble_prediction, current_price, confidence
            
        except ValueError as e:
            logger.warning(f"Skipping prediction for {coin}: {e}")
            return current_price * 0.99, current_price, 0.5
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return df['close'].iloc[-1] * 1.001, df['close'].iloc[-1], 0.1
    
    def calculate_prediction_confidence(self, predictions, ensemble_pred):
        """Calculate prediction confidence based on model agreement"""
        if len(predictions) <= 1:
            return 0.5
        
        pred_values = list(predictions.values())
        std_dev = np.std(pred_values)
        mean_pred = np.mean(pred_values)
        
        confidence = 1.0 / (1.0 + std_dev / (abs(mean_pred) + 1e-6))
        return min(max(confidence, 0.1), 0.9)

class RiskManager:
    """Enhanced risk management system"""
    
    def __init__(self):
        self.max_risk_per_trade = 0.02  # 2% per trade
        self.max_daily_risk = 0.1      # 10% daily
        self.max_drawdown = 0.15       # 15% max drawdown
        
    def calculate_position_size(self, balance, entry_price, stop_loss, confidence):
        """Calculate optimal position size based on Kelly criterion and risk management"""
        try:
            # Basic risk calculation
            if balance < 100:
                risk_percent = 0.20  # 20% untuk modal kecil
            elif balance < 1000:
                risk_percent = 0.10  # 10%
            else:
                risk_percent = 0.02  # 2% default
            risk_amount = balance * risk_percent
            
            price_risk = abs(entry_price - stop_loss)
            base_quantity = risk_amount / price_risk if price_risk > 0 else 0
            
            # Adjust based on confidence
            confidence_multiplier = min(confidence * 2, 1.0)  # Scale 0.5-1.0 to 0-1.0
            adjusted_quantity = base_quantity * confidence_multiplier
            
            # Ensure minimum and maximum limits
            min_notional = 5  # $5 minimum
            max_notional = balance * 0.3  # 50% of balance maximum
            
            notional_value = adjusted_quantity * entry_price
            if notional_value < min_notional:
                adjusted_quantity = min_notional / entry_price
            elif notional_value > max_notional:
                adjusted_quantity = max_notional / entry_price
            
            print(f"Debug position_size: Risk ${risk_amount:.2f}, Price risk ${price_risk:.4f}, Base qty {base_quantity:.4f}, Adjusted qty {adjusted_quantity:.4f}")
            
            return adjusted_quantity
            
        except Exception as e:
            logger.error(f"Position size calculation error: {e}")
            return 0
    
    def calculate_optimal_leverage(self, volatility, confidence, timeframe, balance):
        """Calculate optimal leverage based on volatility and confidence"""
        try:
            # Dynamic leverage: Modal kecil = tinggi (10-20x), besar = rendah (5-10x)
            if balance < 100:
                max_lev = 20
                min_lev = 10
            elif balance < 1000:
                max_lev = 15
                min_lev = 5
            else:
                max_lev = 10
                min_lev = 5
                
            # Base leverage calculation
            base_leverage = 1.0
            
            # Adjust based on volatility (higher volatility = lower leverage)
            volatility_factor = max(0.1, min(1.0, 1.0 - volatility))
            
            # Adjust based on confidence
            confidence_factor = max(0.5, confidence)
            
            # Adjust based on timeframe (shorter timeframe = lower leverage)
            timeframe_factors = {
                "1m": 0.5, "5m": 0.7, "15m": 0.8, "30m": 0.9,
                "1h": 1.0, "4h": 1.2, "1d": 1.5
            }
            timeframe_factor = timeframe_factors.get(timeframe, 1.0)
            
            optimal_leverage = base_leverage * volatility_factor * confidence_factor * timeframe_factor
            optimal_leverage = max(min_lev, min(max_lev, int(optimal_leverage)))            
            
            # Clamp between 1 and 10
            return optimal_leverage
            
        except Exception as e:
            logger.error(f"Leverage calculation error: {e}")
            return 1

class EnhancedTradingBot:
    """Main enhanced trading bot class"""
    
    def __init__(self):
        self.data_fetcher = EnhancedDataFetcher()
        self.technical_analysis = EnhancedTechnicalAnalysis()
        self.ml_models = EnhancedMLModels()
        self.risk_manager = RiskManager()
        self.performance_tracker = self.init_performance_tracker()
    
    def init_performance_tracker(self):
        """Initialize performance tracking"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'daily_pnl': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0
        }
    
    def get_market_sentiment(self):
        """Enhanced market sentiment analysis"""
        try:
            news_titles = []
            for feed_url in RSS_FEEDS:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:5]:  # Limit to recent news
                        news_titles.append(entry.title.lower())
                except Exception as e:
                    logger.error(f"Failed to fetch RSS feed {feed_url}: {e}")
                    continue
            
            if not news_titles:
                return 0.0  # Neutral sentiment if no news
            
            # Count sentiment keywords
            bullish_count = sum(1 for title in news_titles 
                              for keyword in BULLISH_KEYWORDS 
                              if keyword in title)
            
            bearish_count = sum(1 for title in news_titles 
                              for keyword in BEARISH_KEYWORDS 
                              if keyword in title)
            
            total_sentiment = bullish_count + bearish_count
            if total_sentiment == 0:
                return 0.0
            
            sentiment_score = (bullish_count - bearish_count) / total_sentiment
            return max(-1.0, min(1.0, sentiment_score))  # Clamp between -1 and 1
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return 0.0
    
    def analyze_coin_score(self, coin, timeframe):
        """Enhanced coin scoring system"""
        try:
            # Fetch data
            df = self.data_fetcher.fetch_price_data(coin, AVAILABLE_TIMEFRAMES[timeframe])
            df = self.technical_analysis.calculate_advanced_technicals(df)
            
            # Technical indicators score
            rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
            rsi_score = 1.0 - abs(50 - rsi) / 50  # Higher score for RSI near extremes
            
            # Volatility score (ATR)
            atr_percent = df['ATR_percent'].iloc[-1] if 'ATR_percent' in df.columns else 2.0
            volatility_score = min(atr_percent / 10, 1.0)  # Normalize to 0-1
            
            # Volume score
            volume_ma = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_score = min(current_volume / volume_ma, 2.0) / 2.0 if volume_ma > 0 else 0.5
            
            # Trend strength score (ADX)
            adx = df['ADX'].iloc[-1] if 'ADX' in df.columns else 20
            trend_score = min(adx / 50, 1.0)
            
            # Price momentum score
            price_change = df['close'].pct_change(5).iloc[-1]
            momentum_score = min(abs(price_change) * 100, 1.0)
            
            # ML prediction score
            try:
                predicted_price, current_price, confidence = self.ml_models.predict(df, coin, timeframe)
                price_change_pred = abs(predicted_price - current_price) / current_price
                ml_score = min(price_change_pred * 10, 1.0) * confidence
            except:
                ml_score = 0.3
                
            try:
               current_price = df['close'].iloc[-1]
               fib_support = df["Fib_support_resistance"].iloc[-1] if 'Fib_support_resistance' in df.columns else current_price
               fib_distance = abs(current_price - fib_support) / current_price if current_price > 0 else 0
               fib_score = max(0, 1.0 - fib_distance * 10)  # Closer to Fib level = higher score
            except:
                fib_score = 0.5
            
            # Market sentiment
            sentiment = self.get_market_sentiment()
            sentiment_score = (sentiment + 1) / 2  # Normalize -1,1 to 0,1
            
            # Weighted total score
            total_score = (
                0.2 * rsi_score +
                0.15 * volatility_score +
                0.15 * volume_score +
                0.2 * trend_score +
                0.15 * momentum_score +
                0.1 * ml_score +
                0.07 * fib_score +
                0.05 * sentiment_score
            )
            
            if df is not None and len(df) > 20:  # Pastikan df valid
                return {
                    'coin': coin,
                    'total_score': total_score,
                    'rsi_score': rsi_score,
                    'volatility_score': volatility_score,
                    'volume_score': volume_score,
                    'trend_score': trend_score,
                    'momentum_score': momentum_score,
                    'ml_score': ml_score,
                    'sentiment_score': sentiment_score,
                    'fib_score': fib_score if 'fib_score' in locals() else 0.3,
                    'df': df
                }
            else:
                logger.warning(f"Invalid dataframe for {coin}")
                return None
                
        except Exception as e:  # TAMBAHKAN INI
            logger.error(f"Coin analysis error for {coin}: {e}")
            return None
    
    def scan_best_coins(self, timeframe="1h", top_n=5):
        """Scan and return best coins for trading"""
        coins = self.data_fetcher.get_available_coins()
        coin_scores = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_coin = {
                executor.submit(self.analyze_coin_score, coin, timeframe): coin 
                for coin in coins[:20]  # Limit to top 20 for performance
            }
            
            for future in future_to_coin:
                coin = future_to_coin[future]
                try:
                   result = future.result(timeout=30)
                   if result and result.get('total_score', 0) > 0.4 and 'df' in result and result['df'] is not None and len(result['df']) > 0:
                       coin_scores.append(result)
                   else:
                       logger.warning(f"Skipping {coin}: insufficient data or low score")
                except ValueError as e:
                    logger.warning(f"Skipping {coin}: {e}")
                except Exception as e:
                    logger.error(f"Error analyzing {coin}: {e}")
                    continue
        
        # Sort by total score
        coin_scores.sort(key=lambda x: x['total_score'], reverse=True)
        return coin_scores[:top_n]
    
    def calculate_trading_signals(self, df, predicted_price, current_price, confidence):
        """Calculate comprehensive trading signals"""
        try:
            signals = {
                'direction': None,
                'entry_price': current_price,
                'stop_loss': 0,
                'take_profit': 0,
                'leverage': 1,
                'confidence': confidence,
                'risk_reward_ratio': 0
            }
            
            # Determine direction
            price_change = (predicted_price - current_price) / current_price
            if abs(price_change) < 0.005:  # Less than 0.5% change
                return signals  # No signal
            
            signals['direction'] = 'LONG' if price_change > 0 else 'SHORT'
            
            # Calculate ATR for stop loss and take profit
            atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else current_price * 0.02
            atr_multiplier = 2.0  # Base ATR multiplier
            
            # Adjust based on volatility
            volatility = df['ATR_percent'].iloc[-1] if 'ATR_percent' in df.columns else 2.0
            if volatility > 5:  # High volatility
                atr_multiplier *= 1.5
            elif volatility < 1:  # Low volatility
                atr_multiplier *= 0.7
            
            if signals['direction'] == 'LONG':
                signals['stop_loss'] = current_price - (atr * atr_multiplier)
                signals['take_profit'] = current_price + (atr * atr_multiplier * 2)
            else:
                signals['stop_loss'] = current_price + (atr * atr_multiplier)
                signals['take_profit'] = current_price - (atr * atr_multiplier * 2)
                
            # ========== FIX: CLAMP TP SUPAYA POSITIF DAN REALISTIS ==========
            min_tp_distance = current_price * 0.01  # Min 1% dari entry
            if signals['take_profit'] < 0.01:  # Kalau negatif atau terlalu kecil
               signals['take_profit'] = current_price - min_tp_distance if signals['direction'] == 'SHORT' else current_price + min_tp_distance
               logger.warning(f"TP clamped to {signals['take_profit']:.4f} (was negative/unrealistic)")
            # ===============================================================                
            
            # Calculate risk-reward ratio
            risk = abs(current_price - signals['stop_loss'])
            reward = abs(signals['take_profit'] - current_price)
            signals['risk_reward_ratio'] = reward / risk if risk > 0 else 0
            
            # Filter: Skip if risk-reward ratio < 2 (best practice)
            if signals['risk_reward_ratio'] < 2:
               logger.info(f"Signal skipped: Risk-reward ratio {signals['risk_reward_ratio']:.2f} < 2")
               return signals  # No trade
            
            # Calculate optimal leverage
            timeframe = "1h"  # Default, should be passed as parameter
            signals['leverage'] = self.risk_manager.calculate_optimal_leverage(
                volatility / 100, confidence, timeframe, balance=self.get_account_balance()
            )
            
            return signals
            
        except Exception as e:
            logger.error(f"Signal calculation error: {e}")
            return signals
            
    def confirm_signal_lower_tf(self, coin, main_timeframe, signals, lower_tf="15m"):
        """Konfirmasi signal dari timeframe lebih kecil"""
        try:
            lower_interval = AVAILABLE_TIMEFRAMES[lower_tf]
            df_lower = self.data_fetcher.fetch_price_data(coin, lower_interval, limit=100)  # Data pendek buat konfirmasi
            df_lower = self.technical_analysis.calculate_advanced_technicals(df_lower)
        
            # Cek alignment: RSI & MACD searah dengan main signal
            rsi_lower = df_lower['RSI'].iloc[-1] if 'RSI' in df_lower.columns else 50
            macd_lower = df_lower['MACD'].iloc[-1] if 'MACD' in df_lower.columns else 0
            macd_signal_lower = df_lower['MACD_signal'].iloc[-1] if 'MACD_signal' in df_lower.columns else 0
        
            confirm_score = 0
            if signals['direction'] == 'SHORT':
                if rsi_lower > 70:  # Oversold? Wait, SHORT butuh overbought >70 buat konfirm bearish
                    confirm_score += 0.5
                if macd_lower < macd_signal_lower:  # MACD cross down
                    confirm_score += 0.5
            else:  # LONG
                if rsi_lower < 30:  # Oversold <30 buat bullish
                    confirm_score += 0.5
                if macd_lower > macd_signal_lower:  # MACD cross up
                    confirm_score += 0.5
        
            confirmed = confirm_score >= 0.5  # Minimal 1 konfirmasi
            logger.info(f"Lower TF ({lower_tf}) confirmation for {coin} {signals['direction']}: Score {confirm_score:.1f}, Confirmed: {confirmed}")
            return confirmed
        
        except Exception as e:
            logger.error(f"Lower TF confirmation error: {e}")
            return True  # Fallback: Lanjut kalau error            
    
    def execute_trade(self, coin, signals, balance):
        """Execute trade with enhanced error handling"""
        try:
            symbol = f"{coin}USDT"
        
            # ========== FETCH PRESISI DARI EXCHANGE ==========
            try:
                exchange_info = client.futures_exchange_info()
                symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
                if symbol_info:
                    tick_size = float(next((f['tickSize'] for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), '0.01'))
                    price_precision = abs(int(math.log10(tick_size))) if tick_size > 0 else 2
                
                    step_size = float(next((f['stepSize'] for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), '0.001'))
                    quantity_precision = abs(int(math.log10(step_size))) if step_size > 0 else 3
                
                    print(f"Debug: Presisi harga {price_precision} desimal, quantity {quantity_precision} desimal untuk {symbol}")
                else:
                    price_precision = 2
                    quantity_precision = 3
                    print(f"Debug: Pakai fallback presisi untuk {symbol}")
            except Exception as e:
                logger.warning(f"Gagal ambil presisi: {e}. Pakai default.")
                price_precision = 2
                quantity_precision = 3
            # =================================================
        
            # ========== BULATKAN HARGA DAN QUANTITY ==========
            signals['entry_price'] = round(signals['entry_price'], price_precision)
            signals['stop_loss'] = round(signals['stop_loss'], price_precision)
            signals['take_profit'] = round(signals['take_profit'], price_precision)
        
            # Hitung position_size (kode lama)
            position_size = self.risk_manager.calculate_position_size(
                balance, signals['entry_price'], signals['stop_loss'], signals['confidence']
            )
        
            # ========== FIX: BULATKAN SEBELUM CHECK <=0 ==========
            position_size = round(position_size, quantity_precision)
            # ====================================================
        
            if position_size <= 0:
                logger.warning(f"Invalid position size for {coin}: {position_size} (too small after rounding)")
                return False
        
            print(f"Debug: Position size after rounding: {position_size} {coin}")
            # =================================================
        
            # Set leverage
            try:
                client.futures_change_leverage(symbol=symbol, leverage=signals['leverage'])
            except Exception as e:
                logger.warning(f"Leverage setting failed: {e}")
                signals['leverage'] = 1
        
            # PLACE MARKET ORDER DENGAN RETRY
            side = SIDE_BUY if signals['direction'] == 'LONG' else SIDE_SELL
        
            def place_order_with_retry():
                nonlocal position_size  # FIX: Pakai nonlocal biar bisa akses dari outer scope
                try:
                    return client.futures_create_order(
                        symbol=symbol,
                        side=side,
                        type=ORDER_TYPE_MARKET,
                        quantity=position_size,
                        reduceOnly=False
                    )
                except BinanceAPIException as e:
                    if e.code == -1111:  # Precision error
                        logger.warning(f"Precision error: {e}. Mencoba bulatkan lebih kasar.")
                        quantity_precision -= 1
                        position_size = round(position_size, max(quantity_precision, 0))  # Jangan negatif
                    
                        if quantity_precision < 0:
                            raise e
                    
                        return client.futures_create_order(
                            symbol=symbol,
                            side=side,
                            type=ORDER_TYPE_MARKET,
                            quantity=position_size,
                            reduceOnly=False
                        )
                    else:
                        raise e
        
            order = place_order_with_retry()
            order_id = str(order['orderId'])
            # ================================================
        
            # Place stop loss order
            try:
                sl_side = SIDE_SELL if signals['direction'] == 'LONG' else SIDE_BUY
                sl_stop_price = round(signals['stop_loss'], price_precision)
                client.futures_create_order(
                    symbol=symbol,
                    side=sl_side,
                    type='STOP_MARKET',
                    quantity=position_size,
                    stopPrice=sl_stop_price,
                    reduceOnly=True
                )
                logger.info(f"Stop loss set at {sl_stop_price}")
            except BinanceAPIException as e:
                if e.code == -1111:
                    logger.warning(f"SL precision error, skip set manual.")
                else:
                    logger.error(f"Failed to set stop loss: {e}")
            except Exception as e:
                logger.error(f"Failed to set stop loss: {e}")
        
            # Place take profit order
            try:
                if signals['take_profit'] <= 0.01:  # FIX: Skip kalau TP invalid
                     logger.warning(f"Invalid TP {signals['take_profit']:.4f}, skipping TP order")
                else:
                    tp_side = SIDE_SELL if signals['direction'] == 'LONG' else SIDE_BUY
                    tp_price = round(signals['take_profit'], price_precision)
                    client.futures_create_order(
                        symbol=symbol,
                        side=tp_side,
                        type='LIMIT',  # <-- Pastikan string, bukan enum kalau bermasalah
                        quantity=position_size,
                        price=tp_price,
                        timeInForce=TIME_IN_FORCE_GTC,
                        reduceOnly=True
                    )
                    logger.info(f"Take profit set at {tp_price}")
            except BinanceAPIException as e:
                if e.code == -1102 or e.code == -1111:
                    logger.warning(f"TP error (code {e.code}), skip set manual.")
                else:
                    logger.error(f"Failed to set take profit: {e}")
            except Exception as e:
                logger.error(f"Failed to set take profit: {e}")
        
            # Record trade
            timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute("""
                INSERT INTO trades 
                (timestamp, coin, direction, entry_price, stop_loss, take_profit, 
                 leverage, profit_loss, balance, order_id, confidence, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, coin, signals['direction'], signals['entry_price'],
                signals['stop_loss'], signals['take_profit'], signals['leverage'],
                0, balance, order_id, signals['confidence'], 'OPEN'
            ))
            conn.commit()
        
            # Add to active trades
            trading_state.active_trades[order_id] = {
                'coin': coin,
                'direction': signals['direction'],
                'entry_price': signals['entry_price'],
                'quantity': position_size,
                'leverage': signals['leverage'],
                'open_time': time.time(),
                'stop_loss': signals['stop_loss'],
                'take_profit': signals['take_profit']
            }
        
            logger.info(f"Trade executed: {signals['direction']} {coin} at ${signals['entry_price']:.{price_precision}f}")
            print(f"{Fore.GREEN}Trade executed: {signals['direction']} {coin} at ${signals['entry_price']:.{price_precision}f}")
            body = f"Trade OPEN: {signals['direction']} {coin} at ${signals['entry_price']:.4f}\nSL: ${signals['stop_loss']:.4f}\nTP: ${signals['take_profit']:.4f}\nQuantity: {position_size}\nLeverage: {signals['leverage']}x\nConfidence: {signals['confidence']:.2f}"
            send_email_notification("New Trade Executed", body)
            return True
        
        except Exception as e:
            logger.error(f"Trade execution failed for {coin}: {e}")
            print(f"{Fore.RED}Trade execution failed for {coin}: {e}")
            return False
    
    def monitor_positions(self):
        """Monitor and manage active positions"""
        try:
            positions = client.futures_position_information()
            current_time = time.time()
            
            for order_id, trade in list(trading_state.active_trades.items()):
                symbol = f"{trade['coin']}USDT"
                
                # Check if position still exists
                position_exists = False
                current_pnl = 0
                
                for pos in positions:
                    if pos['symbol'] == symbol and float(pos['positionAmt']) != 0:
                        position_exists = True
                        current_pnl = float(pos['unRealizedProfit'])
                        break
                        
                if position_exists:
                    current_price = float(pos['markPrice'])  # Ambil dari positions
                    if trade['direction'] == 'SHORT' and current_price < trade['entry_price']:
                        profit_pct = (trade['entry_price'] - current_price) / trade['entry_price']
                        if profit_pct > 0.005:  # >0.5% profit
                           new_sl = current_price + (trade['entry_price'] - trade['take_profit']) * 0.5                        
                
                if not position_exists:
                    # Position closed, update database
                    try:
                        # Get realized PnL from account
                        account_info = client.futures_account()
                        for asset in account_info['assets']:
                            if asset['asset'] == 'USDT':
                                # This is a simplified approach - in production, you'd track PnL more precisely
                                break
                        
                        cursor.execute("""
                            UPDATE trades SET status = 'CLOSED', profit_loss = ?
                            WHERE order_id = ?
                        """, (current_pnl, order_id))
                        conn.commit()
                        
                        logger.info(f"Position closed: {trade['coin']} PnL: ${current_pnl:.2f}")
                        print(f"{Fore.GREEN if current_pnl >= 0 else Fore.RED}Position closed: {trade['coin']} PnL: ${current_pnl:.2f}")
                        
                        # Update performance tracking
                        self.performance_tracker['win_rate'] = (self.performance_tracker['winning_trades'] / self.performance_tracker['total_trades']) * 100 if self.performance_tracker['total_trades'] > 0 else 0
                        body = f"Position CLOSED: {trade['coin']} {trade['direction']}\nPnL: ${current_pnl:.2f}\nStatus: {'Profit' if current_pnl > 0 else 'Loss'}"
                        send_email_notification("Position Closed", body)
                        
                    except Exception as e:
                        logger.error(f"Error updating closed position: {e}")
                    
                    del trading_state.active_trades[order_id]
                
                # Check for timeout
                elif current_time - trade['open_time'] > POSITION_TIMEOUT.get("1h", 3600):
                    logger.info(f"Position timeout: {trade['coin']}")
                    try:
                        # Close position due to timeout
                        side = SIDE_SELL if trade['direction'] == 'LONG' else SIDE_BUY
                        client.futures_create_order(
                            symbol=symbol,
                            side=side,
                            type=ORDER_TYPE_MARKET,
                            quantity=trade['quantity'],
                            reduceOnly=True
                        )
                        logger.info(f"Position closed due to timeout: {trade['coin']}")
                    except Exception as e:
                        logger.error(f"Failed to close position on timeout: {e}")
        
        except Exception as e:
            logger.error(f"Position monitoring error: {e}")
    
    def get_account_balance(self):
        """Get current account balance"""
        try:
            account = client.futures_account(recvWindow=10000)
            for asset in account['assets']:
                if asset['asset'] == 'USDT':
                    return float(asset['walletBalance'])
            return 0.0
        except Exception as e:
            logger.error(f"Balance fetch error: {e}")
            return trading_state.initial_capital
    
    def run_single_coin_bot(self, coin, timeframe="1h", interval_seconds=60):
        """Run bot for single coin"""
        print(f"{Fore.GREEN}Starting single coin bot for {coin} on {timeframe} timeframe")
        
        ws_manager.start_websocket(coin, timeframe)
        last_trade_time = 0
        
        while True:
            try:
                # Rate limiting
                if time.time() - last_trade_time < interval_seconds:
                    time.sleep(1)
                    continue
                
                # Get current balance
                balance = self.get_account_balance()
                print(f"{Fore.WHITE}Current balance: ${balance:.2f}")
                
                # Check balance limits
                if balance < 10:
                    print(f"{Fore.RED}Balance too low: ${balance:.2f}")
                    break
                
                # Monitor existing positions
                self.monitor_positions()
                
                if any(trade['coin'] == coin for trade in trading_state.active_trades.values()):
                     print(f"{Fore.YELLOW}Already have active position for {coin}, waiting for close...")
                     time.sleep(10)
                     continue                
                
                # Check if we can open new position
                if len(trading_state.active_trades) >= trading_state.MAX_ACTIVE_POSITIONS:
                    print(f"{Fore.YELLOW}Maximum positions reached: {len(trading_state.active_trades)}")
                    time.sleep(10)
                    continue
                
                # Analyze coin
                coin_data = self.analyze_coin_score(coin, timeframe)
                if not coin_data or coin_data['total_score'] < 0.5:
                    print(f"{Fore.YELLOW}Low score for {coin}: {coin_data['total_score'] if coin_data else 'N/A'}")
                    time.sleep(10)
                    continue
                
                # Get ML prediction
                df = coin_data['df']
                predicted_price, current_price, confidence = self.ml_models.predict(df, coin, timeframe)
                
                # Calculate trading signals
                signals = self.calculate_trading_signals(df, predicted_price, current_price, confidence)
                
                if signals['direction'] and signals['risk_reward_ratio'] >= 2 and confidence > 0.4:
                    print(f"{Fore.CYAN}Trading signal: {signals['direction']} {coin}")
                    print(f"Entry: ${signals['entry_price']:.4f}")
                    print(f"Stop Loss: ${signals['stop_loss']:.4f}")
                    print(f"Take Profit: ${signals['take_profit']:.4f}")
                    print(f"Confidence: {confidence:.2f}")
                    
                    # Multi-Timeframe Confirmation (MTC) from 15m
                    if not self.confirm_signal_lower_tf(coin, timeframe, signals, lower_tf="15m"):
                         print(f"{Fore.YELLOW}Signal not confirmed on 15m, skipping entry")
                         continue
                    
                    if self.execute_trade(coin, signals, balance):
                        last_trade_time = time.time()
                        trading_state.trade_count += 1
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Bot error: {e}")
                print(f"{Fore.RED}Bot error: {e}")
                time.sleep(10)
    
    def run_multi_coin_bot(self, timeframe="1h", interval_seconds=120):
        """Run bot with automatic coin scanning"""
        print(f"{Fore.GREEN}Starting multi-coin scanning bot on {timeframe} timeframe")
        
        while True:
            try:
                coin_data = None  # Pindahkan ke dalam loop
                best_coins = []
                # Get current balance
                balance = self.get_account_balance()
                print(f"{Fore.WHITE}Current balance: ${balance:.2f}")
                
                # Check balance limits
                if balance < 10:
                    print(f"{Fore.RED}Balance too low: ${balance:.2f}")
                    break
                
                # Monitor existing positions
                self.monitor_positions()
                
                # Check if we can open new position
                if len(trading_state.active_trades) >= trading_state.MAX_ACTIVE_POSITIONS:
                    print(f"{Fore.YELLOW}Maximum positions reached: {len(trading_state.active_trades)}")
                    time.sleep(30)
                    continue
                
                # Scan for best coins
                print(f"{Fore.CYAN}Scanning coins...")
                best_coins = self.scan_best_coins(timeframe, top_n=3)
                
                if not best_coins:
                    print(f"{Fore.YELLOW}No suitable coins found")
                    time.sleep(interval_seconds)
                    continue
                
                # Try to trade the best coin not already in active trades
                for coin_data in best_coins:
                    coin = None  # Initialize
                    try:
                        if not coin_data or not isinstance(coin_data, dict):
                           continue
            
                        coin = coin_data.get('coin', 'Unknown')
        
                        # Skip if already trading this coin
                        if any(trade['coin'] == coin for trade in trading_state.active_trades.values()):
                           continue
        
                        # Pastikan coin_data punya 'df'
                        if 'df' not in coin_data or coin_data['df'] is None:
                            print(f"{Fore.YELLOW}No data available for {coin}")
                            continue
            
                        # Get ML prediction
                        df = coin_data['df']
                        predicted_price, current_price, confidence = self.ml_models.predict(df, coin, timeframe)
        
                        # Calculate trading signals
                        signals = self.calculate_trading_signals(df, predicted_price, current_price, confidence)
        
                        if signals['direction'] and signals['risk_reward_ratio'] >= 2 and confidence > 0.4:
                            print(f"{Fore.CYAN}Best coin: {coin} (Score: {coin_data['total_score']:.3f})")
                            print(f"Trading signal: {signals['direction']}")
                            print(f"Entry: ${signals['entry_price']:.4f}")
                            print(f"Confidence: {confidence:.2f}")
                            
                            # Multi-Timeframe Confirmation (MTC) from 15m
                            if not self.confirm_signal_lower_tf(coin, timeframe, signals, lower_tf="15m"):
                               print(f"{Fore.YELLOW}Signal not confirmed on 15m, skipping entry")
                               continue                            
                            
                            for active_id, active_trade in trading_state.active_trades.items():
                                if active_trade['coin'] == coin and abs(signals['entry_price'] - active_trade['entry_price']) / active_trade['entry_price'] < 0.005:  # <0.5% beda
                                    print(f"{Fore.YELLOW}Duplicate signal for {coin}, skipping...")
                                    break  # Skip execute
                            else:  # Kalau no duplicate
                                if self.execute_trade(coin, signals, balance):
                                   trading_state.trade_count += 1
                                   break  # Only trade one coin per cycle
                
                    except Exception as e:
                         logger.error(f"Error processing coin {coin if 'coin' in locals() else 'unknown'}: {e}")
                         continue
                         
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Multi-coin bot error: {e}")
                print(f"{Fore.RED}Multi-coin bot error: {e}")
                time.sleep(30)                                                

def display_performance_summary():
    """Display trading performance summary"""
    try:
        cursor.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10")
        recent_trades = cursor.fetchall()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(profit_loss) as total_profit,
                AVG(profit_loss) as avg_profit
            FROM trades WHERE status = 'CLOSED'
        """)
        stats = cursor.fetchone()
        
        print(f"\n{Fore.CYAN + Style.BRIGHT}=== PERFORMANCE SUMMARY ==={Style.RESET_ALL}")
        if stats and stats[0] > 0:
            total_trades, winning_trades, total_profit, avg_profit = stats
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            print(f"Total Trades: {Fore.WHITE}{total_trades}{Style.RESET_ALL}")
            print(f"Winning Trades: {Fore.GREEN}{winning_trades}{Style.RESET_ALL}")
            print(f"Win Rate: {Fore.YELLOW}{win_rate:.1f}%{Style.RESET_ALL}")  # Dari DB
            print(f"Total Profit: {Fore.GREEN if total_profit >= 0 else Fore.RED}${total_profit:.2f}{Style.RESET_ALL}")
            print(f"Average Profit per Trade: {Fore.GREEN if avg_profit >= 0 else Fore.RED}${avg_profit:.2f}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}No completed trades yet{Style.RESET_ALL}")
            print(f"Win Rate: {Fore.YELLOW}0.0%{Style.RESET_ALL}")  # Fallback
        
        if recent_trades:
            print(f"\n{Fore.CYAN}Recent Trades:{Style.RESET_ALL}")
            for trade in recent_trades[:5]:
                color = Fore.GREEN if trade[8] >= 0 else Fore.RED  # profit_loss column
                print(f"  {trade[2]} {trade[3]} - PnL: {color}${trade[8]:.2f}{Style.RESET_ALL}")
    
    except Exception as e:
        logger.error(f"Performance summary error: {e}")
        
def sync_timestamp():
    try:
        server_time = client.get_server_time()
        local_time = int(time.time() * 1000)
        client.timestamp_offset = server_time['serverTime'] - local_time
        print(f"Timestamp synced. Offset: {client.timestamp_offset}ms")
    except Exception as e:
        print(f"Timestamp sync failed: {e}")

# Panggil sync timestamp
sync_timestamp()

def safe_get(dictionary, key, default=None):
    """Safe dictionary get with default"""
    if isinstance(dictionary, dict):
        return dictionary.get(key, default)
    return default

def main_menu():
    """Enhanced main menu"""
    bot = EnhancedTradingBot()
    
    while True:
        print(BANNER)
        print(f"[ENHANCED CRYPTO TRADING BOT V2.0]")
        print(f"Current Balance: ${bot.get_account_balance():.2f}")
        print(f"Active Positions: {len(trading_state.active_trades)}")
        print()
        
        print("1. Single Coin Trading")
        print("2. Multi-Coin Auto-Scanning") 
        print("3. View Performance Summary")
        print("4. Market Sentiment Analysis")
        print("5. Coin Analysis")
        print("6. Exit")
        
        choice = input("Select option (1-6): ").strip()
        
        try:
            if choice == "1":
                coin = input(f"{Fore.YELLOW}Enter coin symbol (e.g., BTC, ETH): {Style.RESET_ALL}").upper().strip()
                timeframe = input(f"{Fore.YELLOW}Enter timeframe (1m,5m,15m,30m,1h,4h,1d) [default: 1h]: {Style.RESET_ALL}").lower().strip()
                timeframe = timeframe if timeframe in AVAILABLE_TIMEFRAMES else "1h"
                
                bot.run_single_coin_bot(coin, timeframe)
                
            elif choice == "2":
                try:
                    timeframe = input(f"{Fore.YELLOW}Enter timeframe (1m,5m,15m,30m,1h,4h,1d) [default: 1h]: {Style.RESET_ALL}").lower().strip()
                    timeframe = timeframe if timeframe in AVAILABLE_TIMEFRAMES else "1h"
        
                    print(f"{Fore.GREEN}Starting multi-coin bot...{Style.RESET_ALL}")
                    bot.run_multi_coin_bot(timeframe)
        
                except Exception as e:
                    logger.error(f"Multi-coin bot error: {e}")
                    print(f"{Fore.RED}Multi-coin bot error: {e}{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}Try again or use single coin trading instead{Style.RESET_ALL}")
                    input(f"{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
                
            elif choice == "3":
                display_performance_summary()
                input(f"{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
                
            elif choice == "4":
                sentiment = bot.get_market_sentiment()
                sentiment_text = "Bullish" if sentiment > 0.1 else "Bearish" if sentiment < -0.1 else "Neutral"
                color = Fore.GREEN if sentiment > 0 else Fore.RED if sentiment < 0 else Fore.YELLOW
                
                print(f"\n{Fore.CYAN}Market Sentiment Analysis:{Style.RESET_ALL}")
                print(f"Sentiment Score: {color}{sentiment:.3f}{Style.RESET_ALL}")
                print(f"Market Mood: {color}{sentiment_text}{Style.RESET_ALL}")
                input(f"{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
                
            elif choice == "5":
                try:
                    coin = input("Enter coin symbol for analysis: ").upper().strip()
                    timeframe = input("Enter timeframe [default: 1h]: ").lower().strip()
                    timeframe = timeframe if timeframe in AVAILABLE_TIMEFRAMES else "1h"
        
                    print(f"Analyzing {coin}...")
                    coin_data = bot.analyze_coin_score(coin, timeframe)
        
                    if coin_data and isinstance(coin_data, dict):
                        print(f"\n=== {coin} Analysis ===")
                        print(f"Total Score: {coin_data.get('total_score', 0):.3f}")
                        print(f"RSI Score: {coin_data.get('rsi_score', 0):.3f}")
                        print(f"Volatility Score: {coin_data.get('volatility_score', 0):.3f}")
                        print(f"Volume Score: {coin_data.get('volume_score', 0):.3f}")
                        print(f"Trend Score: {coin_data.get('trend_score', 0):.3f}")
                        print(f"Momentum Score: {coin_data.get('momentum_score', 0):.3f}")
                        print(f"ML Score: {coin_data.get('ml_score', 0):.3f}")
                        print(f"Fibonacci Score: {coin_data.get('fib_score', 0):.3f}")
                        print(f"Sentiment Score: {coin_data.get('sentiment_score', 0):.3f}")
                    else:
                        print(f"Failed to analyze {coin}")
        
                except Exception as e:
                    print(f"Analysis error: {e}")
    
                input("Press Enter to continue...")
                
            elif choice == "6":
                print(f"{Fore.GREEN}Closing all positions and exiting...{Style.RESET_ALL}")
                
                # Close all active positions
                try:
                    for order_id, trade in trading_state.active_trades.items():
                        symbol = f"{trade['coin']}USDT"
                        side = SIDE_SELL if trade['direction'] == 'LONG' else SIDE_BUY
                        
                        client.futures_create_order(
                            symbol=symbol,
                            side=side,
                            type=ORDER_TYPE_MARKET,
                            quantity=trade['quantity'],
                            reduceOnly=True
                        )
                        print(f"Closed position: {trade['coin']}")
                        
                except Exception as e:
                    logger.error(f"Error closing positions: {e}")
                
                # Close database connection
                conn.close()
                print(f"{Fore.GREEN}Bot stopped successfully!{Style.RESET_ALL}")
                break
                
            else:
                print(f"{Fore.RED}Invalid option. Please try again.{Style.RESET_ALL}")
                
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Operation cancelled by user{Style.RESET_ALL}")
            continue
        except Exception as e:
            logger.error(f"Menu option error: {e}")
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
            continue

if __name__ == "__main__":
    try:
        print("Starting bot initialization...")
        
        print("Display startup information...")
        print("[INFO] Enhanced Crypto Trading Bot V2.0")
        print("[INFO] Make sure to set your Binance API credentials")
        print("[INFO] This bot uses advanced ML models and risk management")
        print("[WARNING] Use at your own risk. Start with small amounts.")
        print()
        
        print("Clearing screen...")
        os.system('clear')
        
        print("Starting main menu...")
        main_menu()
        
    except Exception as e:
        print(f"MAIN ERROR LOCATION: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'conn' in locals():
            conn.close()
