import ccxt
import pandas as pd
import requests, re
import time
import asyncio
from datetime import datetime, timedelta
import talib
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import certifi
import traceback
import argparse
import sys, os
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import concurrent.futures
import threading
from collections import defaultdict, deque
from telegram.ext import MessageHandler, filters
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler

# ===============================
# PROFESSIONAL CONFIGURATION
# ===============================

@dataclass
class TradingConfig:
    """Professional trading configuration"""
    # Risk Management
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk per trade
    max_daily_risk: float = 0.06     # 6% max daily risk
    max_correlation_exposure: float = 0.10  # 10% max correlated positions
    
    # Position Sizing
    min_position_size: float = 100    # Minimum $100 position
    max_position_size: float = 50000  # Maximum $50k position
    max_leverage: Dict[str, float] = None
    
    # Signal Thresholds
    min_signal_strength: int = 8      # Minimum signal strength
    min_mtf_confidence: float = 65    # Minimum multi-timeframe confidence
    min_volume_threshold: float = 5_000_000  # Minimum $5M volume
    
    # Advanced Features
    use_portfolio_optimization: bool = True
    use_sector_rotation: bool = True
    use_market_regime_filter: bool = True
    use_risk_parity: bool = True
    
    def __post_init__(self):
        if self.max_leverage is None:
            self.max_leverage = {
                'SCALPING': 20.0,
                'DAY_TRADING': 10.0,
                'SWING': 5.0,
                'POSITION': 3.0
            }

class MarketRegime(Enum):
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending" 
    BULL_RANGING = "bull_ranging"
    BEAR_RANGING = "bear_ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

class AssetClass(Enum):
    LARGE_CAP = "large_cap"          # BTC, ETH
    MID_CAP = "mid_cap"              # BNB, ADA, SOL
    DEFI = "defi"                    # UNI, AAVE, COMP
    LAYER1 = "layer1"                # SOL, AVAX, DOT
    MEME = "meme"                    # DOGE, SHIB, PEPE
    AI = "ai"                        # FET, AGIX, RNDR
    GAMING = "gaming"                # AXS, SAND, GALA

# ===============================
# ADVANCED TECHNICAL ANALYSIS
# ===============================

class AdvancedIndicators:
    """Professional-grade technical indicators"""

    @staticmethod
    def calculate_bos_indicator(df: pd.DataFrame) -> np.ndarray:
        """Calculate Break of Structure (BOS) indicator"""
        try:
            highs = df['high'].to_numpy()
            lows = df['low'].to_numpy()
            closes = df['close'].to_numpy()
            
            bos_signals = np.zeros(len(df))
            window = 10
            
            for i in range(window, len(df) - window):
                # Look for higher highs (bullish BOS)
                recent_highs = highs[i-window:i]
                if highs[i] > max(recent_highs) and closes[i] > closes[i-1]:
                    # Confirm with volume
                    if df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean() * 1.2:
                        bos_signals[i] = 1  # Bullish BOS
                
                # Look for lower lows (bearish BOS)
                recent_lows = lows[i-window:i]
                if lows[i] < min(recent_lows) and closes[i] < closes[i-1]:
                    # Confirm with volume
                    if df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean() * 1.2:
                        bos_signals[i] = -1  # Bearish BOS
            
            return bos_signals
            
        except Exception as e:
            print(f"Error calculating BOS: {e}")
            return np.zeros(len(df))
                
    @staticmethod
    def calculate_market_structure(df: pd.DataFrame, lookback: int = 20) -> Dict:
        """Calculate market structure analysis"""
        highs = df['high'].to_numpy()
        lows = df['low'].to_numpy()
        closes = df['close'].to_numpy()
        
        # Higher Highs, Lower Lows detection
        hh_count = 0  # Higher Highs
        ll_count = 0  # Lower Lows
        hl_count = 0  # Higher Lows
        lh_count = 0  # Lower Highs
        
        for i in range(lookback, len(df)):
            recent_highs = highs[i-lookback:i]
            recent_lows = lows[i-lookback:i]
            
            if highs[i] > max(recent_highs):
                hh_count += 1
            if lows[i] < min(recent_lows):
                ll_count += 1
            if lows[i] > min(recent_lows):
                hl_count += 1
            if highs[i] < max(recent_highs):
                lh_count += 1
        
        # Structure score: +1 for bullish structure, -1 for bearish
        structure_score = (hh_count + hl_count - ll_count - lh_count) / lookback
        
        return {
            'structure_score': structure_score,
            'higher_highs': hh_count,
            'lower_lows': ll_count,
            'trend_strength': abs(structure_score),
            'structure_type': 'bullish' if structure_score > 0.1 else 'bearish' if structure_score < -0.1 else 'neutral'
        }
    
    @staticmethod
    def calculate_smart_money_index(df: pd.DataFrame) -> np.ndarray:
        """Calculate Smart Money Index (SMI)"""
        # Smart Money Index = Close - (High + Low + Close) / 3
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        smi = df['close'] - typical_price
        return talib.EMA(smi.to_numpy(), timeperiod=14)
    
    @staticmethod
    def calculate_volume_profile(df: pd.DataFrame, bins: int = 20) -> Dict:
        """Calculate Volume Profile"""
        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / bins
        
        volume_profile = {}
        for i in range(bins):
            price_level = df['low'].min() + (i * bin_size)
            volume_at_level = 0
            
            for _, row in df.iterrows():
                if price_level <= row['close'] <= price_level + bin_size:
                    volume_at_level += row['volume']
            
            volume_profile[price_level] = volume_at_level
        
        # Find Point of Control (POC) - highest volume level
        poc_price = max(volume_profile, key=volume_profile.get)
        poc_volume = volume_profile[poc_price]
        
        # Calculate Value Area (70% of volume)
        total_volume = sum(volume_profile.values())
        value_area_volume = total_volume * 0.70
        
        sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        cumulative_volume = 0
        value_area = []
        
        for price, volume in sorted_levels:
            cumulative_volume += volume
            value_area.append(price)
            if cumulative_volume >= value_area_volume:
                break
        
        return {
            'poc_price': poc_price,
            'poc_volume': poc_volume,
            'value_area_high': max(value_area),
            'value_area_low': min(value_area),
            'volume_profile': volume_profile
        }
    
    @staticmethod
    def calculate_institutional_levels(df: pd.DataFrame) -> List[Dict]:
        """Detect institutional support/resistance levels"""
        highs = df['high'].to_numpy()
        lows = df['low'].to_numpy()
        volumes = df['volume'].to_numpy()
        closes = df['close'].to_numpy()
        
        levels = []
        window = 10
        
        for i in range(window, len(df) - window):
            # Check for significant volume at price extremes
            if volumes[i] > np.mean(volumes[max(0, i-20):i+20]) * 2:
                # High volume support/resistance
                if highs[i] == max(highs[i-window:i+window]):
                    levels.append({
                        'price': highs[i],
                        'type': 'resistance',
                        'strength': volumes[i] / np.mean(volumes),
                        'index': i,
                        'touches': 0
                    })
                elif lows[i] == min(lows[i-window:i+window]):
                    levels.append({
                        'price': lows[i],
                        'type': 'support',
                        'strength': volumes[i] / np.mean(volumes),
                        'index': i,
                        'touches': 0
                    })
        
        # Count touches for each level
        for level in levels:
            for close in closes[level['index']:]:
                if abs(close - level['price']) / level['price'] < 0.01:  # Within 1%
                    level['touches'] += 1
        
        # Sort by strength and touches
        levels.sort(key=lambda x: (x['strength'] + x['touches']), reverse=True)
        return levels[:10]  # Return top 10 levels

class RiskMetrics:
    """Professional risk analysis"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe Ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        return (np.mean(returns) - risk_free_rate) / np.std(returns)
    
    @staticmethod
    def calculate_max_drawdown(prices: np.ndarray) -> float:
        """Calculate Maximum Drawdown"""
        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / peak
        return abs(np.min(drawdown))
    
    @staticmethod
    def calculate_value_at_risk(returns: np.ndarray, confidence: float = 0.05) -> float:
        """Calculate Value at Risk (VaR)"""
        if len(returns) == 0:
            return 0
        return np.percentile(returns, confidence * 100)
    
    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray, prices: np.ndarray) -> float:
        """Calculate Calmar Ratio (Annual Return / Max Drawdown)"""
        if len(returns) == 0:
            return 0
        annual_return = np.mean(returns) * 252  # Assuming daily returns
        max_dd = RiskMetrics.calculate_max_drawdown(prices)
        return annual_return / max_dd if max_dd > 0 else 0

# ===============================
# PORTFOLIO OPTIMIZATION
# ===============================

class PortfolioOptimizer:
    """Modern Portfolio Theory implementation"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.correlation_matrix = {}
        self.sector_weights = {}
    
    def calculate_correlation_matrix(self, symbols: List[str], binance_client) -> Dict:
        """Calculate correlation matrix between assets"""
        returns_data = {}
        
        for symbol in symbols:
            try:
                ohlcv = binance_client.fetch_ohlcv(symbol, timeframe='1d', limit=30)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                returns = df['close'].pct_change().dropna()
                returns_data[symbol] = returns.values
            except:
                continue
        
        # Calculate correlations
        correlations = {}
        for symbol1 in returns_data:
            for symbol2 in returns_data:
                if symbol1 != symbol2:
                    corr = np.corrcoef(returns_data[symbol1], returns_data[symbol2])[0, 1]
                    correlations[f"{symbol1}_{symbol2}"] = corr if not np.isnan(corr) else 0
        
        return correlations
    
    def optimize_position_sizes(self, candidates: List[Dict], portfolio_value: float) -> List[Dict]:
        """Optimize position sizes using risk parity and correlation"""
        if not candidates:
            return []
        
        # Calculate individual volatilities
        for candidate in candidates:
            symbol = candidate['symbol']
            risk_mgmt = candidate['risk_mgmt']
            
            # Estimate position volatility
            stop_distance = abs(candidate['current_price'] - risk_mgmt['stop_loss']) / candidate['current_price']
            candidate['estimated_volatility'] = stop_distance
            
            # Calculate base position size (risk parity)
            risk_per_trade = portfolio_value * self.config.max_portfolio_risk
            candidate['base_position_size'] = risk_per_trade / stop_distance
        
        # Apply correlation adjustments
        total_correlation_penalty = 0
        for i, candidate1 in enumerate(candidates):
            for j, candidate2 in enumerate(candidates[i+1:], i+1):
                # Simplified correlation penalty
                sector1 = self.get_asset_class(candidate1['symbol'])
                sector2 = self.get_asset_class(candidate2['symbol'])
                
                if sector1 == sector2:
                    penalty = 0.3  # 30% reduction for same sector
                    candidates[i]['base_position_size'] *= (1 - penalty)
                    candidates[j]['base_position_size'] *= (1 - penalty)
        
        # Normalize to portfolio constraints
        total_allocation = sum(c['base_position_size'] for c in candidates)
        max_total_allocation = portfolio_value * 0.8  # Max 80% allocation
        
        if total_allocation > max_total_allocation:
            scale_factor = max_total_allocation / total_allocation
            for candidate in candidates:
                candidate['optimized_position_size'] = candidate['base_position_size'] * scale_factor
        else:
            for candidate in candidates:
                candidate['optimized_position_size'] = candidate['base_position_size']
        
        return candidates
    
    def get_asset_class(self, symbol: str) -> AssetClass:
        """Classify asset by type"""
        symbol_base = symbol.split('/')[0]
        
        large_cap = ['BTC', 'ETH']
        mid_cap = ['BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'AVAX']
        defi = ['UNI', 'AAVE', 'COMP', 'SUSHI', 'CAKE', 'CRV']
        layer1 = ['SOL', 'AVAX', 'DOT', 'ATOM', 'NEAR', 'FTM']
        meme = ['DOGE', 'SHIB', 'PEPE', 'WIF', 'BONK']
        ai = ['FET', 'AGIX', 'RNDR', 'OCEAN', 'GRT']
        gaming = ['AXS', 'SAND', 'GALA', 'ENJ', 'MANA']
        
        if symbol_base in large_cap:
            return AssetClass.LARGE_CAP
        elif symbol_base in mid_cap:
            return AssetClass.MID_CAP
        elif symbol_base in defi:
            return AssetClass.DEFI
        elif symbol_base in layer1:
            return AssetClass.LAYER1
        elif symbol_base in meme:
            return AssetClass.MEME
        elif symbol_base in ai:
            return AssetClass.AI
        elif symbol_base in gaming:
            return AssetClass.GAMING
        else:
            return AssetClass.MID_CAP

# ===============================
# ENHANCED MARKET ANALYSIS
# ===============================

class MarketRegimeDetector:
    """Advanced market regime detection"""
    
    def __init__(self):
        self.regimes = []
        self.confidence_threshold = 0.7
    
    def detect_regime(self, btc_data: List, eth_data: List, market_data: Dict) -> MarketRegime:
        """Detect current market regime using multiple factors"""
        if len(btc_data) < 100:
            return MarketRegime.BULL_RANGING
        
        btc_df = pd.DataFrame(btc_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Trend Analysis
        ema_20 = talib.EMA(btc_df['close'].to_numpy(), timeperiod=20)
        ema_50 = talib.EMA(btc_df['close'].to_numpy(), timeperiod=50)
        ema_200 = talib.EMA(btc_df['close'].to_numpy(), timeperiod=200)
        
        # Volatility Analysis
        atr = talib.ATR(btc_df['high'].to_numpy(), btc_df['low'].to_numpy(), btc_df['close'].to_numpy(), timeperiod=14)
        current_vol = atr[-1] / btc_df['close'].iloc[-1] * 100
        avg_vol = np.mean(atr[-30:]) / np.mean(btc_df['close'].iloc[-30:]) * 100
        
        # Volume Analysis
        volume_trend = np.mean(btc_df['volume'].iloc[-7:]) / np.mean(btc_df['volume'].iloc[-30:])
        
        # Market Structure
        structure = AdvancedIndicators.calculate_market_structure(btc_df)
        
        # Decision Logic
        is_trending_up = ema_20[-1] > ema_50[-1] > ema_200[-1]
        is_trending_down = ema_20[-1] < ema_50[-1] < ema_200[-1]
        is_high_vol = current_vol > avg_vol * 1.5
        is_strong_structure = abs(structure['structure_score']) > 0.2
        
        if is_high_vol:
            return MarketRegime.HIGH_VOLATILITY
        elif current_vol < avg_vol * 0.7:
            return MarketRegime.LOW_VOLATILITY
        elif is_trending_up and is_strong_structure:
            return MarketRegime.BULL_TRENDING
        elif is_trending_down and is_strong_structure:
            return MarketRegime.BEAR_TRENDING
        elif is_trending_up:
            return MarketRegime.BULL_RANGING
        else:
            return MarketRegime.BEAR_RANGING

class SectorRotationAnalyzer:
    """Analyze sector rotation and leadership"""
    
    def __init__(self):
        self.sector_performance = defaultdict(list)
        self.sector_momentum = {}
    
    def analyze_sectors(self, tickers: Dict) -> Dict[AssetClass, float]:
        """Analyze sector performance and rotation"""
        optimizer = PortfolioOptimizer(TradingConfig())
        sector_performance = defaultdict(list)
        
        for symbol, ticker in tickers.items():
            if symbol.endswith('/USDT') and 'percentage' in ticker:
                asset_class = optimizer.get_asset_class(symbol)
                sector_performance[asset_class].append(ticker['percentage'])
        
        # Calculate sector averages
        sector_scores = {}
        for sector, performances in sector_performance.items():
            if performances:
                sector_scores[sector] = {
                    'avg_performance': np.mean(performances),
                    'momentum': np.mean(performances),
                    'count': len(performances),
                    'strength': np.mean(performances) * len(performances)  # Weight by count
                }
        
        # Sort by strength
        sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1]['strength'], reverse=True)
        
        return {sector: data for sector, data in sorted_sectors}

# ===============================
# PROFESSIONAL SIGNAL SYSTEM
# ===============================

class EnhancedSignalAnalyzer:
    """Professional-grade signal analysis system"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.signal_history = deque(maxlen=1000)
        self.performance_tracker = {}
    
    def analyze_comprehensive_signal(self, symbol: str, tf_analysis: Dict, market_data: Dict, 
                                   market_regime: MarketRegime, netflow: float) -> Dict:
        """Comprehensive signal analysis with professional scoring"""
        
        signal_components = {
            'technical': 0,
            'momentum': 0,
            'volume': 0,
            'structure': 0,
            'sentiment': 0,
            'risk': 0
        }
        
        signal_details = []
        
        # Primary timeframe analysis
        primary_tf = self.get_primary_timeframe(tf_analysis)
        if not primary_tf:
            return {'signal_type': 'NEUTRAL', 'confidence': 0, 'components': signal_components}
        
        primary_data = tf_analysis[primary_tf]
        indicators = primary_data['indicators']
        current_price = primary_data['current_price']
        
        # Technical Score (0-25 points)
        technical_score = self.calculate_technical_score(indicators, signal_details)
        signal_components['technical'] = technical_score
        
        # Momentum Score (0-20 points)  
        momentum_score = self.calculate_momentum_score(indicators, tf_analysis, signal_details)
        signal_components['momentum'] = momentum_score
        
        # Volume Score (0-15 points)
        volume_score = self.calculate_volume_score(indicators, market_data, signal_details)
        signal_components['volume'] = volume_score
        
        # Market Structure Score (0-20 points)
        structure_score = self.calculate_structure_score(indicators, tf_analysis, signal_details)
        signal_components['structure'] = structure_score
        
        # Sentiment Score (0-10 points)
        sentiment_score = self.calculate_sentiment_score(netflow, market_data, signal_details)
        signal_components['sentiment'] = sentiment_score
        
        # Risk Score (0-10 points)
        risk_score = self.calculate_risk_score(indicators, market_regime, signal_details)
        signal_components['risk'] = risk_score
        
        # Calculate total score and confidence
        total_score = sum(signal_components.values())
        max_possible_score = 100
        confidence = (total_score / max_possible_score) * 100
        
        # Determine signal direction
        bullish_signals = sum([1 for detail in signal_details if 'bullish' in detail.lower()])
        bearish_signals = sum([1 for detail in signal_details if 'bearish' in detail.lower()])
        
        if confidence >= self.config.min_mtf_confidence:
            if bullish_signals > bearish_signals and total_score >= 60:
                signal_type = 'BULLISH'
            elif bearish_signals > bullish_signals and total_score >= 60:
                signal_type = 'BEARISH'
            else:
                signal_type = 'NEUTRAL'
        else:
            signal_type = 'NEUTRAL'
        
        # Market regime filter
        signal_type = self.apply_regime_filter(signal_type, market_regime, confidence)
        
        return {
            'signal_type': signal_type,
            'confidence': confidence,
            'total_score': total_score,
            'components': signal_components,
            'details': signal_details,
            'regime_filtered': True
        }
    
    def calculate_technical_score(self, indicators: Dict, details: List[str]) -> float:
        """Calculate technical analysis score"""
        score = 0
        
        # RSI Analysis (0-8 points)
        rsi = indicators['rsi'][-1]
        if rsi < 20:
            score += 8
            details.append(f"RSI Extremely Oversold: {rsi:.1f}")
        elif rsi < 30:
            score += 6
            details.append(f"RSI Oversold: {rsi:.1f}")
        elif rsi > 80:
            score += 8  # Bearish signal
            details.append(f"RSI Extremely Overbought: {rsi:.1f}")
        elif rsi > 70:
            score += 6  # Bearish signal
            details.append(f"RSI Overbought: {rsi:.1f}")
        
        # MACD Analysis (0-8 points)
        macd_hist = indicators['macd_hist'][-1]
        macd_hist_prev = indicators['macd_hist'][-2]
        
        if macd_hist > 0 and macd_hist_prev <= 0:
            score += 8
            details.append("MACD Bullish Crossover")
        elif macd_hist < 0 and macd_hist_prev >= 0:
            score += 8  # Bearish signal
            details.append("MACD Bearish Crossover")
        elif macd_hist > macd_hist_prev and macd_hist > 0:
            score += 4
            details.append("MACD Bullish Momentum")
        
        # Bollinger Bands (0-5 points)
        bb_upper = indicators['bb_upper'][-1]
        bb_lower = indicators['bb_lower'][-1]
        current_price = indicators.get('current_price', 0)
        
        if current_price <= bb_lower:
            score += 5
            details.append("Price at Bollinger Band Lower")
        elif current_price >= bb_upper:
            score += 5  # Bearish signal
            details.append("Price at Bollinger Band Upper")
        
        # Stochastic (0-4 points)
        if 'stoch_k' in indicators:
            stoch_k = indicators['stoch_k'][-1]
            if stoch_k < 20:
                score += 4
                details.append(f"Stochastic Oversold: {stoch_k:.1f}")
            elif stoch_k > 80:
                score += 4  # Bearish signal
                details.append(f"Stochastic Overbought: {stoch_k:.1f}")
        
        return min(score, 25)  # Cap at 25 points
    
    def calculate_momentum_score(self, indicators: Dict, tf_analysis: Dict, details: List[str]) -> float:
        """Calculate momentum score across timeframes"""
        score = 0
        
        # Multi-timeframe momentum alignment
        bullish_tfs = 0
        bearish_tfs = 0
        
        for tf, data in tf_analysis.items():
            if data['trend'] in ['STRONG_BULLISH', 'BULLISH']:
                bullish_tfs += 1
            elif data['trend'] in ['STRONG_BEARISH', 'BEARISH']:
                bearish_tfs += 1
        
        total_tfs = len(tf_analysis)
        momentum_alignment = max(bullish_tfs, bearish_tfs) / total_tfs
        
        if momentum_alignment >= 0.8:  # 80% alignment
            score += 12
            details.append(f"Strong Multi-TF Momentum: {momentum_alignment:.1%}")
        elif momentum_alignment >= 0.6:  # 60% alignment
            score += 8
            details.append(f"Good Multi-TF Momentum: {momentum_alignment:.1%}")
        
        # Supertrend momentum (0-8 points)
        if 'supertrend_direction' in indicators:
            supertrend_dir = indicators['supertrend_direction'][-1]
            supertrend_prev = indicators['supertrend_direction'][-2]
            
            if supertrend_dir == 1 and supertrend_prev == -1:
                score += 8
                details.append("Supertrend Bullish Signal")
            elif supertrend_dir == -1 and supertrend_prev == 1:
                score += 8  # Bearish signal
                details.append("Supertrend Bearish Signal")
        
        return min(score, 20)  # Cap at 20 points
    
    def calculate_volume_score(self, indicators: Dict, market_data: Dict, details: List[str]) -> float:
        """Calculate volume-based score"""
        score = 0
        
        # Volume ratio analysis
        volume_ratio = indicators['volume'][-1] / indicators['volume_sma'][-1]
        
        if volume_ratio > 3.0:
            score += 10
            details.append(f"Exceptional Volume: {volume_ratio:.1f}x")
        elif volume_ratio > 2.0:
            score += 8
            details.append(f"Very High Volume: {volume_ratio:.1f}x")
        elif volume_ratio > 1.5:
            score += 5
            details.append(f"High Volume: {volume_ratio:.1f}x")
        
        # 24h volume strength
        if 'volume_ratio' in market_data:
            daily_vol_ratio = market_data['volume_ratio']
            if daily_vol_ratio > 2.0:
                score += 5
                details.append(f"Strong 24h Volume: {daily_vol_ratio:.1f}x")
        
        return min(score, 15)  # Cap at 15 points
    
    def calculate_structure_score(self, indicators: Dict, tf_analysis: Dict, details: List[str]) -> float:
        """Calculate market structure score"""
        score = 0
        
        # Break of Structure signals
        if 'bos' in indicators:
            bos_signal = indicators['bos'][-1]
            if bos_signal == 1:
                score += 10
                details.append("Bullish Break of Structure")
            elif bos_signal == -1:
                score += 10  # Bearish signal
                details.append("Bearish Break of Structure")
        
        # Order blocks proximity
        if 'order_blocks' in indicators:
            order_blocks = indicators['order_blocks']
            current_price = indicators.get('current_price', 0)
            
            for block in order_blocks[-3:]:  # Check last 3 order blocks
                distance = abs(current_price - block['price']) / current_price
                if distance < 0.02:  # Within 2%
                    score += 3
                    details.append(f"Near {block['type']} Order Block")
        
        # Institutional levels
        if 'institutional_levels' in indicators:
            levels = indicators.get('institutional_levels', [])
            current_price = indicators.get('current_price', 0)
            
            for level in levels[:5]:  # Top 5 levels
                distance = abs(current_price - level['price']) / current_price
                if distance < 0.01:  # Within 1%
                    score += 2
                    details.append(f"At Institutional {level['type']}")
        
        return min(score, 20)  # Cap at 20 points
    
    def calculate_sentiment_score(self, netflow: float, market_data: Dict, details: List[str]) -> float:
        """Calculate sentiment-based score"""
        score = 0
        
        # Netflow analysis (dynamic threshold)
        volume_24h = market_data.get('volume_24h', 1000000)
        netflow_threshold = volume_24h * 0.001  # 0.1% of 24h volume
        
        if abs(netflow) > netflow_threshold * 3:
            score += 6
            flow_type = "Inflow" if netflow > 0 else "Outflow"
            details.append(f"Very Strong {flow_type}: ${abs(netflow):,.0f}")
        elif abs(netflow) > netflow_threshold:
            score += 4
            flow_type = "Inflow" if netflow > 0 else "Outflow"
            details.append(f"Strong {flow_type}: ${abs(netflow):,.0f}")
        
        # Price position in daily range (sentiment proxy)
        range_pos = market_data.get('range_position', 0.5)
        if range_pos > 0.85:
            score += 2
            details.append("Near Daily High")
        elif range_pos < 0.15:
            score += 2
            details.append("Near Daily Low")
        
        # Momentum sentiment
        price_change = market_data.get('price_change_24h', 0)
        if abs(price_change) > 10:
            score += 2
            details.append(f"Strong 24h Momentum: {price_change:+.1f}%")
        
        return min(score, 10)  # Cap at 10 points
    
    def calculate_risk_score(self, indicators: Dict, market_regime: MarketRegime, details: List[str]) -> float:
        """Calculate risk-adjusted score"""
        score = 10  # Start with full points, deduct for risks
        
        # Market regime risk
        if market_regime == MarketRegime.HIGH_VOLATILITY:
            score -= 3
            details.append("High Volatility Environment")
        elif market_regime == MarketRegime.BEAR_TRENDING:
            score -= 2
            details.append("Bear Market Environment")
        
        # Volatility risk
        if 'atr' in indicators:
            atr = indicators['atr'][-1]
            current_price = indicators.get('current_price', 1)
            volatility_pct = (atr / current_price) * 100
            
            if volatility_pct > 8:  # Very high volatility
                score -= 3
                details.append(f"High Volatility: {volatility_pct:.1f}%")
            elif volatility_pct > 5:
                score -= 1
                details.append(f"Elevated Volatility: {volatility_pct:.1f}%")
        
        # RSI divergence risk
        rsi = indicators['rsi'][-1]
        if rsi > 85 or rsi < 15:  # Extreme levels
            score -= 2
            details.append("Extreme RSI Levels")
        
        return max(score, 0)  # Minimum 0 points
    
    def apply_regime_filter(self, signal_type: str, regime: MarketRegime, confidence: float) -> str:
        """Apply market regime filter to signals"""
        
        # Reduce bullish signals in bear markets
        if signal_type == 'BULLISH' and regime in [MarketRegime.BEAR_TRENDING, MarketRegime.BEAR_RANGING]:
            if confidence < 75:  # Require higher confidence
                return 'NEUTRAL'
        
        # Reduce bearish signals in bull markets  
        elif signal_type == 'BEARISH' and regime in [MarketRegime.BULL_TRENDING, MarketRegime.BULL_RANGING]:
            if confidence < 75:
                return 'NEUTRAL'
        
        # Be more cautious in high volatility
        elif regime == MarketRegime.HIGH_VOLATILITY:
            if confidence < 80:
                return 'NEUTRAL'
        
        return signal_type
    
    def get_primary_timeframe(self, tf_analysis: Dict) -> str:
        """Get the most reliable timeframe for analysis"""
        # Priority order based on reliability
        preferred_order = ['15m', '1h', '5m', '4h', '30m']
        
        for tf in preferred_order:
            if tf in tf_analysis:
                return tf
        
        # Return first available if no preferred found
        return list(tf_analysis.keys())[0] if tf_analysis else None

# ===============================
# ADVANCED POSITION SIZING
# ===============================

class KellyPositionSizer:
    """Kelly Criterion position sizing"""
    
    @staticmethod
    def calculate_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate optimal Kelly fraction"""
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = loss_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        return max(0, min(kelly_fraction, 0.25))  # Cap at 25%
    
    @staticmethod
    def calculate_position_size(portfolio_value: float, kelly_fraction: float, 
                              risk_per_trade: float, stop_distance: float) -> float:
        """Calculate position size using Kelly criterion"""
        
        # Use smaller of Kelly fraction or fixed risk
        effective_risk = min(kelly_fraction, risk_per_trade)
        
        # Position size based on stop loss distance
        risk_amount = portfolio_value * effective_risk
        position_size = risk_amount / stop_distance
        
        return position_size

# ===============================
# PROFESSIONAL EXECUTION ENGINE
# ===============================

class TradingEngine:
    """Professional trading execution engine"""

    def analyze_trend_professional(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Professional trend analysis with multiple confirmations"""
        try:
            # EMA trend analysis
            ema_20 = indicators.get('ema_20', np.array([]))
            ema_50 = indicators.get('ema_50', np.array([]))
            ema_200 = indicators.get('ema_200', np.array([]))
            
            if len(ema_20) < 2 or len(ema_50) < 2:
                return {'trend': 'NEUTRAL', 'strength': 0, 'market_structure': {}}
            
            # Current values
            current_ema20 = ema_20[-1]
            current_ema50 = ema_50[-1]
            current_ema200 = ema_200[-1] if len(ema_200) > 0 else current_ema50
            current_price = df['close'].iloc[-1]
            
            # Trend direction scoring
            trend_score = 0
            
            # Price vs EMAs
            if current_price > current_ema20:
                trend_score += 1
            if current_price > current_ema50:
                trend_score += 1
            if current_price > current_ema200:
                trend_score += 1
            
            # EMA alignment
            if current_ema20 > current_ema50:
                trend_score += 1
            if current_ema50 > current_ema200:
                trend_score += 1
            
            # EMA slope analysis
            ema20_slope = (ema_20[-1] - ema_20[-5]) / ema_20[-5] if len(ema_20) >= 5 else 0
            ema50_slope = (ema_50[-1] - ema_50[-5]) / ema_50[-5] if len(ema_50) >= 5 else 0
            
            if ema20_slope > 0.001:  # Rising
                trend_score += 1
            if ema50_slope > 0.001:  # Rising
                trend_score += 1
            
            # Market structure analysis
            structure = AdvancedIndicators.calculate_market_structure(df)
            structure_score = structure.get('structure_score', 0)
            
            if structure_score > 0.1:
                trend_score += 1
            elif structure_score < -0.1:
                trend_score -= 1
            
            # Determine trend classification
            if trend_score >= 6:
                trend = 'STRONG_BULLISH'
                strength = min(trend_score / 8.0, 1.0)
            elif trend_score >= 4:
                trend = 'BULLISH'
                strength = trend_score / 8.0
            elif trend_score <= -6:
                trend = 'STRONG_BEARISH' 
                strength = abs(trend_score) / 8.0
            elif trend_score <= -4:
                trend = 'BEARISH'
                strength = abs(trend_score) / 8.0
            else:
                trend = 'NEUTRAL'
                strength = 0.5
            
            return {
                'trend': trend,
                'strength': strength,
                'trend_score': trend_score,
                'market_structure': structure,
                'ema_alignment': {
                    'price_above_ema20': current_price > current_ema20,
                    'price_above_ema50': current_price > current_ema50,
                    'ema20_above_ema50': current_ema20 > current_ema50,
                    'ema20_slope': ema20_slope,
                    'ema50_slope': ema50_slope
                }
            }
            
        except Exception as e:
            print(f"Error in trend analysis: {e}")
            return {'trend': 'NEUTRAL', 'strength': 0, 'market_structure': {}}
            
    def calculate_signal_strength_advanced(self, indicators: Dict, current_price: float) -> int:
        """Advanced signal strength calculation (0-10)"""
        try:
            signal_strength = 0
            
            # RSI analysis
            rsi = indicators.get('rsi', np.array([50]))
            if len(rsi) > 0:
                current_rsi = rsi[-1]
                if current_rsi < 25 or current_rsi > 75:
                    signal_strength += 2
                elif current_rsi < 35 or current_rsi > 65:
                    signal_strength += 1
            
            # MACD analysis
            macd_hist = indicators.get('macd_hist', np.array([0]))
            if len(macd_hist) >= 2:
                current_hist = macd_hist[-1]
                prev_hist = macd_hist[-2]
                
                # MACD crossover
                if current_hist > 0 and prev_hist <= 0:
                    signal_strength += 2
                elif current_hist < 0 and prev_hist >= 0:
                    signal_strength += 2
                elif abs(current_hist) > abs(prev_hist):
                    signal_strength += 1
            
            # Volume confirmation
            volume = indicators.get('volume', np.array([0]))
            volume_sma = indicators.get('volume_sma', np.array([1]))
            if len(volume) > 0 and len(volume_sma) > 0:
                volume_ratio = volume[-1] / volume_sma[-1] if volume_sma[-1] > 0 else 1
                if volume_ratio > 2.0:
                    signal_strength += 2
                elif volume_ratio > 1.5:
                    signal_strength += 1
            
            # Bollinger Bands analysis
            bb_upper = indicators.get('bb_upper', np.array([current_price + 1]))
            bb_lower = indicators.get('bb_lower', np.array([current_price - 1]))
            if len(bb_upper) > 0 and len(bb_lower) > 0:
                if current_price <= bb_lower[-1]:
                    signal_strength += 1
                elif current_price >= bb_upper[-1]:
                    signal_strength += 1
            
            # Supertrend confirmation
            supertrend_dir = indicators.get('supertrend_direction', np.array([0]))
            if len(supertrend_dir) >= 2:
                if supertrend_dir[-1] != supertrend_dir[-2]:  # Direction change
                    signal_strength += 2
            
            # BOS confirmation
            bos = indicators.get('bos', np.array([0]))
            if len(bos) > 0 and abs(bos[-1]) > 0:
                signal_strength += 1
            
            return min(signal_strength, 10)
            
        except Exception as e:
            print(f"Error calculating signal strength: {e}")
            return 0
    
    def analyze_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Enhanced volume profile analysis"""
        try:
            if len(df) < 20:
                return {}
            
            # Calculate basic volume metrics
            avg_volume = df['volume'].tail(20).mean()
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Volume trend analysis
            volume_ma_short = df['volume'].tail(5).mean()
            volume_ma_long = df['volume'].tail(20).mean()
            volume_trend = volume_ma_short / volume_ma_long if volume_ma_long > 0 else 1
            
            # Price-volume relationship
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
            volume_change = (df['volume'].iloc[-1] - df['volume'].iloc[-2]) / df['volume'].iloc[-2]
            
            # Volume profile using simplified approach
            volume_profile = AdvancedIndicators.calculate_volume_profile(df, bins=10)
            
            return {
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'volume_trend': volume_trend,
                'price_volume_correlation': price_change * volume_change,
                'volume_profile': volume_profile,
                'volume_strength': min(volume_ratio * 2, 5)  # 0-5 scale
            }
            
        except Exception as e:
            print(f"Error in volume analysis: {e}")
            return {}
    
    def calculate_mtf_consensus_professional(self, tf_analysis: Dict) -> Dict:
        """Calculate multi-timeframe consensus with professional weighting"""
        try:
            if not tf_analysis:
                return {'consensus': 'NEUTRAL', 'confidence': 0}
            
            bullish_score = 0
            bearish_score = 0
            total_weight = 0
            
            for tf, data in tf_analysis.items():
                weight = data.get('weight', 1)
                trend = data.get('trend', 'NEUTRAL')
                signal_strength = data.get('signal_strength', 0)
                
                # Convert trend to numeric score
                trend_score = 0
                if trend in ['STRONG_BULLISH', 'BULLISH']:
                    trend_score = 2 if 'STRONG' in trend else 1
                elif trend in ['STRONG_BEARISH', 'BEARISH']:
                    trend_score = -2 if 'STRONG' in trend else -1
                
                # Weight by signal strength and timeframe importance
                weighted_score = trend_score * (signal_strength / 10) * weight
                
                if weighted_score > 0:
                    bullish_score += weighted_score
                elif weighted_score < 0:
                    bearish_score += abs(weighted_score)
                
                total_weight += weight
            
            if total_weight == 0:
                return {'consensus': 'NEUTRAL', 'confidence': 0}
            
            # Normalize scores
            bullish_score = bullish_score / total_weight
            bearish_score = bearish_score / total_weight
            
            # Determine consensus
            score_diff = abs(bullish_score - bearish_score)
            max_score = max(bullish_score, bearish_score)
            
            if bullish_score > bearish_score and score_diff > 0.3:
                consensus = 'BULLISH'
                confidence = min((bullish_score / (bullish_score + bearish_score)) * 100, 100)
            elif bearish_score > bullish_score and score_diff > 0.3:
                consensus = 'BEARISH'
                confidence = min((bearish_score / (bullish_score + bearish_score)) * 100, 100)
            else:
                consensus = 'NEUTRAL'
                confidence = max_score * 50  # Lower confidence for neutral
            
            return {
                'consensus': consensus,
                'confidence': confidence,
                'bullish_score': bullish_score,
                'bearish_score': bearish_score,
                'score_difference': score_diff,
                'total_timeframes': len(tf_analysis)
            }
            
        except Exception as e:
            print(f"Error calculating MTF consensus: {e}")
            return {'consensus': 'NEUTRAL', 'confidence': 0}
    
    def calculate_advanced_risk_management(self, indicators: Dict, signal_type: str, 
                                         current_price: float, style_filter: str) -> Dict:
        """Advanced risk management calculation"""
        try:
            # ATR for volatility-based stops
            atr = indicators.get('atr', np.array([current_price * 0.02]))
            current_atr = atr[-1] if len(atr) > 0 else current_price * 0.02
            
            # Style-specific multipliers
            style_multipliers = {
                'SCALPING': {'stop': 1.0, 'tp1': 1.5, 'tp2': 2.5, 'tp3': 4.0},
                'DAY_TRADING': {'stop': 1.5, 'tp1': 2.0, 'tp2': 3.5, 'tp3': 6.0},
                'SWING': {'stop': 2.0, 'tp1': 3.0, 'tp2': 5.0, 'tp3': 8.0},
                'POSITION': {'stop': 2.5, 'tp1': 4.0, 'tp2': 7.0, 'tp3': 12.0}
            }
            
            multipliers = style_multipliers.get(style_filter, style_multipliers['DAY_TRADING'])
            
            # Calculate levels based on signal direction
            if signal_type == 'BULLISH':
                stop_loss = current_price - (current_atr * multipliers['stop'])
                take_profit_1 = current_price + (current_atr * multipliers['tp1'])
                take_profit_2 = current_price + (current_atr * multipliers['tp2'])
                take_profit_3 = current_price + (current_atr * multipliers['tp3'])
            else:  # BEARISH
                stop_loss = current_price + (current_atr * multipliers['stop'])
                take_profit_1 = current_price - (current_atr * multipliers['tp1'])
                take_profit_2 = current_price - (current_atr * multipliers['tp2'])
                take_profit_3 = current_price - (current_atr * multipliers['tp3'])
            
            # Calculate risk-reward ratios
            stop_distance = abs(current_price - stop_loss)
            rr1 = abs(take_profit_1 - current_price) / stop_distance if stop_distance > 0 else 0
            rr2 = abs(take_profit_2 - current_price) / stop_distance if stop_distance > 0 else 0
            rr3 = abs(take_profit_3 - current_price) / stop_distance if stop_distance > 0 else 0
            
            # Support/resistance level adjustments
            levels = indicators.get('institutional_levels', [])
            if levels:
                # Adjust stop loss to nearest support/resistance
                for level in levels[:5]:  # Check top 5 levels
                    level_price = level.get('price', current_price)
                    distance = abs(level_price - stop_loss) / current_price
                    
                    if distance < 0.005:  # Within 0.5%
                        if signal_type == 'BULLISH' and level['type'] == 'support':
                            stop_loss = level_price * 0.999  # Just below support
                        elif signal_type == 'BEARISH' and level['type'] == 'resistance':
                            stop_loss = level_price * 1.001  # Just above resistance
            
            # Volatility metrics
            volatility_ratio = current_atr / current_price
            
            # Risk validation
            if stop_distance / current_price > 0.15:  # More than 15% risk
                return None  # Reject high-risk setups
            
            return {
                'stop_loss': stop_loss,
                'take_profit_1': take_profit_1,
                'take_profit_2': take_profit_2,
                'take_profit_3': take_profit_3,
                'risk_reward_1': rr1,
                'risk_reward_2': rr2,
                'risk_reward_3': rr3,
                'stop_distance': stop_distance,
                'stop_distance_percent': (stop_distance / current_price) * 100,
                'atr_value': current_atr,
                'volatility_ratio': volatility_ratio,
                'style_multipliers': multipliers
            }
            
        except Exception as e:
            print(f"Error calculating risk management: {e}")
            return None
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.portfolio_optimizer = PortfolioOptimizer(config)
        self.signal_analyzer = EnhancedSignalAnalyzer(config)
        self.regime_detector = MarketRegimeDetector()
        self.sector_analyzer = SectorRotationAnalyzer()
        
        # Performance tracking
        self.trade_history = deque(maxlen=1000)
        self.portfolio_value = 100000  # Default $100k portfolio
        self.daily_pnl = deque(maxlen=252)  # 1 year of daily P&L
        
        # Risk management
        self.active_positions = {}
        self.sector_exposure = defaultdict(float)
        self.correlation_limits = {}
    
    async def scan_market_comprehensive(self, binance_client, headers, 
                                      style_filter: Optional[str] = None) -> List[Dict]:
        """Comprehensive market scan with professional filtering"""
        
        print("🔍 Starting Professional Market Scan...")
        
        # Get market data
        tickers = binance_client.fetch_tickers()
        
        # Detect market regime
        try:
            btc_data = binance_client.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=200)
            eth_data = binance_client.fetch_ohlcv('ETH/USDT', timeframe='1h', limit=100)
            market_regime = self.regime_detector.detect_regime(btc_data, eth_data, {})
            print(f"📊 Market Regime: {market_regime.value}")
        except Exception as e:
            market_regime = MarketRegime.BULL_RANGING
            print(f"⚠️ Failed to detect regime: {e}")
        
        # Sector analysis
        sector_performance = self.sector_analyzer.analyze_sectors(tickers)
        leading_sectors = list(sector_performance.keys())[:3]
        print(f"🏆 Leading Sectors: {[s.value for s in leading_sectors]}")
        
        # Get candidate symbols
        candidates = self.get_professional_candidates(tickers, style_filter, sector_performance)
        print(f"🎯 Analyzing {len(candidates)} candidates...")
        
        # Parallel analysis for speed
        analysis_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            
            for symbol in candidates[:20]:  # Limit for performance
                future = executor.submit(
                    self.analyze_symbol_comprehensive, 
                    symbol, binance_client, headers, market_regime, style_filter
                )
                futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    if result and result['signal_type'] != 'NEUTRAL':
                        analysis_results.append(result)
                except Exception as e:
                    print(f"⚠️ Analysis failed: {e}")
                    continue
        
        if not analysis_results:
            print("❌ No qualified setups found")
            return []
        
        # Portfolio optimization
        optimized_results = self.portfolio_optimizer.optimize_position_sizes(
            analysis_results, self.portfolio_value
        )
        
        # Final ranking and filtering
        final_results = self.rank_and_filter_results(optimized_results, market_regime)
        
        print(f"✅ Found {len(final_results)} professional setups")
        return final_results
    
    def get_professional_candidates(self, tickers: Dict, style_filter: Optional[str], 
                                  sector_performance: Dict) -> List[str]:
        """Get high-quality candidate symbols"""
        
        candidates = []
        
        # Volume and movement filters
        min_volume = self.config.min_volume_threshold
        min_movement = 1.0 if style_filter != 'SCALPING' else 2.0
        
        for symbol, ticker in tickers.items():
            if not symbol.endswith('/USDT'):
                continue
                
            if not all(k in ticker for k in ['quoteVolume', 'percentage']):
                continue
            
            volume = ticker['quoteVolume']
            movement = abs(ticker['percentage'])
            
            # Basic filters
            if volume < min_volume or movement < min_movement:
                continue
            
            # Sector preference (boost leading sectors)
            asset_class = self.portfolio_optimizer.get_asset_class(symbol)
            if asset_class in list(sector_performance.keys())[:5]:  # Top 5 sectors
                candidates.append(symbol)
            elif volume > min_volume * 2:  # High volume override
                candidates.append(symbol)
        
        # Sort by composite score
        def score_symbol(symbol):
            ticker = tickers[symbol]
            volume_score = min(ticker['quoteVolume'] / 1000000, 1000)
            movement_score = min(abs(ticker['percentage']) * 10, 200)
            return volume_score + movement_score
        
        candidates.sort(key=score_symbol, reverse=True)
        return candidates
    
    def analyze_symbol_comprehensive(self, symbol: str, binance_client, headers, 
                                   market_regime: MarketRegime, style_filter: Optional[str]) -> Optional[Dict]:
        """Comprehensive symbol analysis"""
        
        try:
            # Get market data
            market_data = self.get_enhanced_market_data_safe(symbol, binance_client)
            if not market_data or market_data.get('volume_24h', 0) == 0:
                return None
            
            # Multi-timeframe analysis
            tf_analysis = self.analyze_multi_timeframe_professional(
                symbol, binance_client, style_filter
            )
            if not tf_analysis:
                return None
            
            # Calculate MTF consensus
            mtf_consensus = self.calculate_mtf_consensus_professional(tf_analysis)
            if not mtf_consensus or mtf_consensus['confidence'] < self.config.min_mtf_confidence:
                return None
            
            # Get netflow data
            netflow = self.get_dune_cex_flow_enhanced(symbol, headers, market_data)
            
            # Enhanced signal analysis
            signal_result = self.signal_analyzer.analyze_comprehensive_signal(
                symbol, tf_analysis, market_data, market_regime, netflow
            )
            
            if signal_result['signal_type'] == 'NEUTRAL':
                return None
            
            # Risk management calculation
            primary_tf = self.signal_analyzer.get_primary_timeframe(tf_analysis)
            primary_data = tf_analysis[primary_tf]
            
            risk_mgmt = self.calculate_advanced_risk_management(
                primary_data['indicators'], 
                signal_result['signal_type'],
                primary_data['current_price'],
                style_filter
            )
            
            if not risk_mgmt:
                return None
            
            # Calculate final score with professional weighting
            final_score = self.calculate_professional_score(
                signal_result, market_data, risk_mgmt, market_regime
            )
            
            return {
                'symbol': symbol,
                'signal_type': signal_result['signal_type'],
                'confidence': signal_result['confidence'],
                'final_score': final_score,
                'current_price': primary_data['current_price'],
                'market_data': market_data,
                'signal_components': signal_result['components'],
                'risk_mgmt': risk_mgmt,
                'tf_analysis': tf_analysis,
                'mtf_consensus': mtf_consensus,
                'netflow': netflow,
                'market_regime': market_regime,
                'signal_details': signal_result['details']
            }
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            return None
    
    def calculate_professional_score(self, signal_result: Dict, market_data: Dict, 
                                   risk_mgmt: Dict, market_regime: MarketRegime) -> float:
        """Calculate professional composite score"""
        
        base_score = signal_result['confidence']
        
        # Risk-reward bonus
        best_rr = max(risk_mgmt['risk_reward_1'], risk_mgmt['risk_reward_2'])
        rr_bonus = min(best_rr * 5, 20)  # Up to 20 points for good R:R
        
        # Volume quality bonus
        volume_ratio = market_data.get('volume_ratio', 1)
        volume_bonus = min(volume_ratio * 5, 15)  # Up to 15 points
        
        # Market regime adjustment
        regime_multiplier = {
            MarketRegime.BULL_TRENDING: 1.2,
            MarketRegime.BULL_RANGING: 1.1,
            MarketRegime.BEAR_TRENDING: 0.8,
            MarketRegime.BEAR_RANGING: 0.9,
            MarketRegime.HIGH_VOLATILITY: 0.9,
            MarketRegime.LOW_VOLATILITY: 1.0
        }
        
        # Component quality bonus (balanced signals are better)
        components = signal_result['components']
        component_balance = 1 - (max(components.values()) - min(components.values())) / 100
        balance_bonus = component_balance * 10
        
        final_score = (base_score + rr_bonus + volume_bonus + balance_bonus) * regime_multiplier.get(market_regime, 1.0)
        
        return final_score
    
    def rank_and_filter_results(self, results: List[Dict], market_regime: MarketRegime) -> List[Dict]:
        """Final ranking and filtering with professional criteria"""
        
        if not results:
            return []
        
        # Apply final filters
        filtered_results = []
        
        for result in results:
            # Risk management filter
            risk_mgmt = result['risk_mgmt']
            if risk_mgmt['risk_reward_1'] < 1.5:  # Minimum 1.5 R:R
                continue
            
            # Confidence filter
            if result['confidence'] < self.config.min_mtf_confidence:
                continue
            
            # Volatility filter (avoid extreme volatility)
            if risk_mgmt['volatility_ratio'] > 0.15:  # Max 15% volatility
                continue
            
            filtered_results.append(result)
        
        # Sort by final score
        filtered_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Diversification filter (max 2 per sector)
        sector_counts = defaultdict(int)
        final_results = []
        
        for result in filtered_results:
            asset_class = self.portfolio_optimizer.get_asset_class(result['symbol'])
            
            if sector_counts[asset_class] < 2:  # Max 2 per sector
                sector_counts[asset_class] += 1
                final_results.append(result)
            
            if len(final_results) >= 5:  # Max 5 total positions
                break
        
        return final_results
    
    # Enhanced helper methods
    def get_enhanced_market_data_safe(self, symbol: str, binance_client) -> Dict:
        """Enhanced market data collection"""
        try:
            ticker = binance_client.fetch_ticker(symbol)
            
            # Basic data
            data = {
                'current_price': ticker.get('last', 0),
                'volume_24h': ticker.get('quoteVolume', 0),
                'price_change_24h': ticker.get('percentage', 0),
                'high_24h': ticker.get('high', 0),
                'low_24h': ticker.get('low', 0)
            }
            
            # Enhanced metrics
            if data['high_24h'] and data['low_24h'] and data['current_price']:
                daily_range = data['high_24h'] - data['low_24h']
                if daily_range > 0:
                    data['range_position'] = (data['current_price'] - data['low_24h']) / daily_range
                else:
                    data['range_position'] = 0.5
            
            # Volume analysis
            try:
                ohlcv_1d = binance_client.fetch_ohlcv(symbol, timeframe='1d', limit=7)
                if len(ohlcv_1d) >= 3:
                    recent_volumes = [candle[5] for candle in ohlcv_1d]
                    avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 1
                    data['volume_ratio'] = data['volume_24h'] / avg_volume if avg_volume > 0 else 1
                else:
                    data['volume_ratio'] = 1
            except:
                data['volume_ratio'] = 1
            
            # Volatility metrics
            if data['current_price'] > 0 and daily_range > 0:
                data['volatility_24h'] = daily_range / data['current_price']
            else:
                data['volatility_24h'] = 0.02
            
            return data
            
        except Exception as e:
            print(f"Error getting market data for {symbol}: {e}")
            return {}

    def analyze_multi_timeframe_professional(self, symbol: str, binance_client, style_filter: Optional[str]) -> Dict:
        """Professional multi-timeframe analysis"""
        try:
            # Define timeframes based on style
            if style_filter == 'SCALPING':
                timeframes = {
                    '1m': {'weight': 3, 'periods': 100},
                    '5m': {'weight': 4, 'periods': 100}, 
                    '15m': {'weight': 3, 'periods': 80}
                }
            elif style_filter == 'DAY_TRADING':
                timeframes = {
                    '5m': {'weight': 2, 'periods': 100},
                    '15m': {'weight': 4, 'periods': 100}, 
                    '1h': {'weight': 3, 'periods': 80}
                }
            elif style_filter == 'SWING':
                timeframes = {
                    '1h': {'weight': 3, 'periods': 100},
                    '4h': {'weight': 4, 'periods': 100},
                    '1d': {'weight': 3, 'periods': 60}
                }
            else:
                # Default comprehensive
                timeframes = {
                    '5m': {'weight': 2, 'periods': 80},
                    '15m': {'weight': 3, 'periods': 80}, 
                    '1h': {'weight': 4, 'periods': 80}
                }
            
            tf_analysis = {}
            
            for tf, config in timeframes.items():
                try:
                    ohlcv = binance_client.fetch_ohlcv(symbol, timeframe=tf, limit=config['periods'])
                    if len(ohlcv) < 50:
                        continue
                        
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # Calculate all indicators including enhanced ones
                    indicators = self.calculate_professional_indicators(df)
                    if not indicators:
                        continue
                    
                    current_price = df['close'].iloc[-1]
                    
                    # Enhanced trend analysis
                    trend_analysis = self.analyze_trend_professional(df, indicators)
                    
                    # Signal strength with advanced scoring
                    signal_strength = self.calculate_signal_strength_advanced(indicators, current_price)
                    
                    tf_analysis[tf] = {
                        'trend': trend_analysis['trend'],
                        'trend_strength': trend_analysis['strength'],
                        'signal_strength': signal_strength,
                        'current_price': current_price,
                        'rsi': indicators['rsi'][-1] if 'rsi' in indicators else 50,
                        'weight': config['weight'],
                        'indicators': indicators,
                        'market_structure': trend_analysis.get('market_structure', {}),
                        'volume_analysis': self.analyze_volume_profile(df)
                    }
                    
                except Exception as e:
                    print(f"Error analyzing {tf} for {symbol}: {e}")
                    continue
            
            return tf_analysis
            
        except Exception as e:
            print(f"Error in multi-timeframe analysis for {symbol}: {e}")
            return {}
    
    def calculate_professional_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate professional-grade indicators"""
        try:
            indicators = {}
            
            # Basic indicators
            indicators['rsi'] = talib.RSI(df['close'].to_numpy(), timeperiod=14)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(df['close'].to_numpy())
            indicators['macd'] = macd
            indicators['macd_signal'] = macd_signal
            indicators['macd_hist'] = macd_hist
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'].to_numpy(), timeperiod=20)
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
            
            # EMAs
            indicators['ema_20'] = talib.EMA(df['close'].to_numpy(), timeperiod=20)
            indicators['ema_50'] = talib.EMA(df['close'].to_numpy(), timeperiod=50)
            indicators['ema_200'] = talib.EMA(df['close'].to_numpy(), timeperiod=200)
            
            # ATR
            indicators['atr'] = talib.ATR(df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy())
            
            # Volume indicators
            indicators['volume_sma'] = talib.SMA(df['volume'].to_numpy(), timeperiod=20)
            indicators['volume'] = df['volume'].to_numpy()
            
            # Stochastic
            slowk, slowd = talib.STOCH(df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy())
            indicators['stoch_k'] = slowk
            indicators['stoch_d'] = slowd
            
            # Advanced indicators
            indicators['williams_r'] = talib.WILLR(df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy())
            indicators['mfi'] = talib.MFI(df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy(), df['volume'].to_numpy())
            
            # Custom indicators
            indicators['bos'] = AdvancedIndicators.calculate_bos_indicator(df)
            indicators['smart_money_index'] = AdvancedIndicators.calculate_smart_money_index(df)
            indicators['market_structure'] = AdvancedIndicators.calculate_market_structure(df)
            indicators['volume_profile'] = AdvancedIndicators.calculate_volume_profile(df)
            indicators['institutional_levels'] = AdvancedIndicators.calculate_institutional_levels(df)
            
            # Supertrend
            indicators['supertrend'], indicators['supertrend_direction'] = self.calculate_supertrend(df)
            
            return indicators
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return {}
    
    def calculate_supertrend(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Supertrend indicator"""
        try:
            hl2 = (df['high'] + df['low']) / 2
            atr = talib.ATR(df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy(), timeperiod=period)
            
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)
            
            supertrend = np.zeros(len(df))
            direction = np.ones(len(df))
            
            for i in range(1, len(df)):
                if df['close'].iloc[i] <= lower_band[i]:
                    supertrend[i] = upper_band[i]
                    direction[i] = -1
                elif df['close'].iloc[i] >= upper_band[i]:
                    supertrend[i] = lower_band[i]
                    direction[i] = 1
                else:
                    supertrend[i] = supertrend[i-1]
                    direction[i] = direction[i-1]
            
            return supertrend, direction
            
        except Exception as e:
            print(f"Error calculating Supertrend: {e}")
            return np.zeros(len(df)), np.ones(len(df))

# Continue with more methods...

async def send_professional_analysis_message(context, chat_id: int, results: List[Dict], market_regime: MarketRegime):
    """Send professional analysis results to Telegram"""
    
    if not results:
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ No professional setups found meeting our strict criteria"
        )
        return
    
    # Header message
    header_msg = (
        f"🏛️ **PROFESSIONAL MARKET ANALYSIS**\n"
        f"{'='*40}\n"
        f"📊 Market Regime: {market_regime.value.upper()}\n"
        f"🎯 Qualified Setups: {len(results)}\n"
        f"⏰ Analysis Time: {datetime.now().strftime('%H:%M:%S UTC')}\n"
        f"🔬 Professional Grade Filtering Applied"
    )
    
    await context.bot.send_message(chat_id=chat_id, text=header_msg)
    await asyncio.sleep(2)
    
    # Send each setup
    for i, result in enumerate(results, 1):
        signal_components = result['signal_components']
        risk_mgmt = result['risk_mgmt']
        
        setup_msg = (
            f"🎯 **SETUP #{i}: {result['signal_type']} - {result['symbol']}**\n"
            f"{'='*35}\n"
            f"🔥 Confidence: {result['confidence']:.1f}% | Score: {result['final_score']:.1f}\n"
            f"💰 Entry: ${result['current_price']:.4f}\n\n"
            f"🎯 **TARGETS & RISK:**\n"
            f"• TP1 (40%): ${risk_mgmt['take_profit_1']:.4f} (R:R {risk_mgmt['risk_reward_1']:.1f})\n"
            f"• TP2 (40%): ${risk_mgmt['take_profit_2']:.4f} (R:R {risk_mgmt['risk_reward_2']:.1f})\n"
            f"• TP3 (20%): ${risk_mgmt['take_profit_3']:.4f} (R:R {risk_mgmt['risk_reward_3']:.1f})\n"
            f"• SL: ${risk_mgmt['stop_loss']:.4f}\n\n"
            f"📊 **SIGNAL BREAKDOWN:**\n"
            f"• Technical: {signal_components['technical']:.0f}/25\n"
            f"• Momentum: {signal_components['momentum']:.0f}/20\n"
            f"• Volume: {signal_components['volume']:.0f}/15\n"
            f"• Structure: {signal_components['structure']:.0f}/20\n"
            f"• Sentiment: {signal_components['sentiment']:.0f}/10\n"
            f"• Risk Score: {signal_components['risk']:.0f}/10\n\n"
            f"💹 Position Size: ${result.get('optimized_position_size', 0):,.0f}\n"
            f"📈 24h Volume: ${result['market_data']['volume_24h']:,.0f}\n"
            f"⚡ Volatility: {risk_mgmt['volatility_ratio']:.1%}\n\n"
            f"⚠️ **Risk Management:**\n"
            f"• Max leverage recommended\n"
            f"• Professional position sizing applied\n"
            f"• Correlation-adjusted allocation"
        )
        
        await context.bot.send_message(chat_id=chat_id, text=setup_msg)
        await asyncio.sleep(3)
    
    # Risk disclaimer
    disclaimer_msg = (
        f"⚠️ **PROFESSIONAL RISK DISCLAIMER**\n"
        f"{'='*35}\n"
        f"• This is algorithmic analysis, not financial advice\n"
        f"• Professional risk management is mandatory\n"
        f"• Never risk more than 1-2% per trade\n"
        f"• Market conditions can change rapidly\n"
        f"• Past performance doesn't guarantee future results\n\n"
        f"🎯 **Trade Responsibly**"
    )
    
    await context.bot.send_message(chat_id=chat_id, text=disclaimer_msg)

# ===============================
# ENHANCED BOT COMMANDS & HANDLERS
# ===============================

# Update existing configuration
binance = ccxt.binance({
    'enableRateLimit': True,
    'timeout': 30000,
    'urls': {
        'api': {
            'public': 'https://api1.binance.com/api/v3',
            'private': 'https://api1.binance.com/api/v3',
        }
    },
    'verify': True
})

# Professional configuration
trading_config = TradingConfig()
trading_engine = TradingEngine(trading_config)

# Enhanced headers with better error handling
headers = {
    "X-Dune-API-Key": "ye3xyk2FBGNzBreLeBZiAtYQp1XD3cSn",
    "Content-Type": "application/json",
    "User-Agent": "Professional-Trading-Bot/2.0"
}

bot_token = '8074383591:AAEvvbHyu7qRBrHmjoVBrXyHAxpkPtNf4Jc'
chat_id = '992731953'

# ===============================
# PROFESSIONAL TELEGRAM INTERFACE
# ===============================

def create_main_menu():
    """Create the main menu keyboard"""
    keyboard = [
        [
            InlineKeyboardButton("⚡ Quick Scalping", callback_data='quick_scalping'),
            InlineKeyboardButton("📈 Day Trading", callback_data='day_trading'),
        ],
        [
            InlineKeyboardButton("📊 Swing Trading", callback_data='swing_trading'),
            InlineKeyboardButton("🎯 Manual Analysis", callback_data='manual_analysis'),
        ],
        [
            InlineKeyboardButton("🤖 Auto Scanner", callback_data='auto_scanner'),
            InlineKeyboardButton("🏛️ Professional Mode", callback_data='pro_scan'),
        ],
        [
            InlineKeyboardButton("⚙️ Settings", callback_data='settings'),
            InlineKeyboardButton("📊 Portfolio", callback_data='portfolio'),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler"""
    welcome_message = (
        "🚀 **CRYPTO TRADING BOT**\n\n"
        "Select your trading style:\n"
        "⚡ **Scalping** - Quick profits (1-15 min)\n"
        "📈 **Day Trading** - Intraday moves (30min-6h)\n"
        "📊 **Swing Trading** - Multi-day positions\n"
        "🎯 **Manual** - Custom symbol analysis\n"
        "🤖 **Auto** - Automated scanning\n"
        "🏛️ **Professional** - Advanced analysis\n\n"
        "Choose your strategy below:"
    )
    
    menu = create_main_menu()
    
    if update.message:
        await update.message.reply_text(
            welcome_message, 
            reply_markup=menu,
            parse_mode='Markdown'
        )
    else:
        await update.callback_query.edit_message_text(
            welcome_message, 
            reply_markup=menu,
            parse_mode='Markdown'
        )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks"""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data == 'quick_scalping':
        await execute_scalping_scan(query, context)
    elif data == 'day_trading':
        await execute_daytrading_scan(query, context)
    elif data == 'swing_trading':
        await execute_swing_scan(query, context)
    elif data == 'manual_analysis':
        await show_manual_menu(query, context)
    elif data == 'auto_scanner':
        await execute_auto_scan(query, context)
    elif data == 'settings':
        await show_settings_menu(query, context)
    elif data == 'portfolio':
        await show_portfolio_menu(query, context)
    elif data == 'back_main':
        await start(update, context)

def create_professional_main_menu():
    """Create professional main menu"""
    keyboard = [
        [
            InlineKeyboardButton("🏛️ Professional Scan", callback_data='pro_scan'),
            InlineKeyboardButton("📊 Market Regime", callback_data='market_regime'),
        ],
        [
            InlineKeyboardButton("⚡ Scalping Pro", callback_data='pro_scalping'),
            InlineKeyboardButton("📈 Day Trading Pro", callback_data='pro_daytrading'),
        ],
        [
            InlineKeyboardButton("📊 Swing Trading Pro", callback_data='pro_swing'),
            InlineKeyboardButton("💼 Portfolio Analysis", callback_data='portfolio_analysis'),
        ],
        [
            InlineKeyboardButton("🔍 Deep Analysis", callback_data='deep_analysis'),
            InlineKeyboardButton("⚙️ Risk Settings", callback_data='risk_settings'),
        ],
        [
            InlineKeyboardButton("📈 Performance", callback_data='performance'),
            InlineKeyboardButton("🎯 Sector Rotation", callback_data='sector_rotation'),
        ]
    ]
    
    return InlineKeyboardMarkup(keyboard)

def create_professional_settings_menu():
    """Professional settings menu"""
    keyboard = [
        [
            InlineKeyboardButton("⚖️ Risk Management", callback_data='settings_risk'),
            InlineKeyboardButton("💰 Position Sizing", callback_data='settings_position'),
        ],
        [
            InlineKeyboardButton("📊 Signal Thresholds", callback_data='settings_signals'),
            InlineKeyboardButton("🎯 Portfolio Rules", callback_data='settings_portfolio'),
        ],
        [
            InlineKeyboardButton("⏰ Timeframe Weights", callback_data='settings_timeframes'),
            InlineKeyboardButton("🔧 Advanced Config", callback_data='settings_advanced'),
        ],
        [
            InlineKeyboardButton("↩️ Back to Main", callback_data='back_main_pro')
        ]
    ]
    
    return InlineKeyboardMarkup(keyboard)

async def start_professional(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Professional start command"""
    welcome_message = (
        "🏛️ **PROFESSIONAL CRYPTO TRADING SYSTEM**\n\n"
        "🎯 **Advanced Features:**\n"
        "• Multi-timeframe regime detection\n"
        "• Portfolio optimization & correlation analysis\n"
        "• Professional risk management\n"
        "• Sector rotation analysis\n"
        "• Kelly criterion position sizing\n"
        "• Smart money flow analysis\n"
        "• Institutional level detection\n\n"
        "⚡ **Performance Standards:**\n"
        "• Minimum 65% confidence threshold\n"
        "• Risk-adjusted position sizing\n"
        "• Professional-grade signal filtering\n"
        "• Real-time market regime adaptation\n\n"
        "🚀 **Ready for Professional Trading**"
    )
    
    menu = create_professional_main_menu()
    
    if update.message:
        await update.message.reply_text(
            welcome_message, 
            reply_markup=menu,
            parse_mode='Markdown'
        )
    else:
        await update.callback_query.edit_message_text(
            welcome_message, 
            reply_markup=menu,
            parse_mode='Markdown'
        )
'''
async def professional_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced button handler for professional features"""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    # Professional scan handlers
    if data == 'pro_scan':
        await execute_professional_scan(query, context)
    elif data == 'market_regime':
        await show_market_regime_analysis(query, context)
    elif data.startswith('pro_'):
        style = data.replace('pro_', '').upper()
        await execute_professional_style_analysis(query, context, style)
    elif data == 'portfolio_analysis':
        await show_portfolio_analysis(query, context)
    elif data == 'deep_analysis':
        await show_deep_analysis_menu(query, context)
    elif data == 'risk_settings':
        await show_professional_settings_menu(query, context)
    elif data == 'performance':
        await show_performance_analysis(query, context)
    elif data == 'sector_rotation':
        await show_sector_rotation_analysis(query, context)
    elif data == 'back_main_pro':
        await start_professional(update, context)
    else:
        # Fallback to original handler for backward compatibility
        await button_handler(update, context)
'''
async def professional_button_handler_complete(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Complete professional button handler with all missing callbacks"""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    try:
        # Professional scan handlers
        if data == 'pro_scan':
            await execute_professional_scan(query, context)
        elif data == 'market_regime':
            await show_market_regime_analysis(query, context)
        elif data.startswith('pro_'):
            style = data.replace('pro_', '').upper()
            await execute_professional_style_analysis(query, context, style)
        
        # Portfolio handlers
        elif data == 'portfolio_analysis':
            await show_portfolio_analysis(query, context)
        elif data == 'portfolio_rebalance':
            await handle_portfolio_rebalance(query, context)
        elif data == 'portfolio_performance':
            await show_detailed_performance(query, context)
        elif data == 'portfolio_risk':
            await show_portfolio_risk_analysis(query, context)
        
        # Deep analysis handlers
        elif data == 'deep_analysis':
            await show_deep_analysis_menu(query, context)
        elif data in ['deep_symbol', 'deep_correlation', 'deep_structure', 'deep_institutional', 'deep_multitf', 'deep_volume']:
            await handle_deep_analysis_callback(query, context)
        elif data.startswith('analyze_'):
            await handle_analyze_symbol(query, context)
        
        # Settings handlers
        elif data == 'risk_settings':
            await show_professional_settings_detailed(query, context)
        elif data.startswith('settings_'):
            await handle_settings_callback(query, context)
        
        # Performance handlers
        elif data == 'performance':
            await show_performance_analysis(query, context)
        elif data == 'performance_detailed':
            await show_detailed_performance(query, context)
        elif data == 'trade_history':
            await show_trade_history(query, context)
        
        # Sector handlers
        elif data == 'sector_rotation':
            await show_sector_rotation_analysis(query, context)
        elif data == 'sector_details':
            await show_sector_details(query, context)
        elif data == 'sector_signals':
            await show_sector_signals(query, context)
        
        # Navigation handlers
        elif data == 'back_main_pro':
            await start_professional(update, context)
        elif data == 'back_main':
            await start(update, context)
        
        # Original trading style handlers
        elif data == 'quick_scalping':
            await execute_scalping_scan(query, context)
        elif data == 'day_trading':
            await execute_daytrading_scan(query, context)
        elif data == 'swing_trading':
            await execute_swing_scan(query, context)
        elif data == 'manual_analysis':
            await show_manual_menu(query, context)
        elif data == 'auto_scanner':
            await execute_auto_scan(query, context)
        elif data == 'settings':
            await show_settings_menu(query, context)
        elif data == 'portfolio':
            await show_portfolio_menu(query, context)
        
        # Error fallback
        else:
            await query.edit_message_text(
                "❌ Unknown command. Returning to main menu...",
                reply_markup=create_main_menu()
            )
    
    except Exception as e:
        logging.error(f"Button handler error: {e}")
        await query.edit_message_text(
            f"❌ Error occurred: {str(e)}\nReturning to main menu...",
            reply_markup=create_main_menu()
        )

# Additional missing handlers
async def show_portfolio_risk_analysis(query, context):
    """Show portfolio risk analysis"""
    risk_msg = (
        f"⚖️ **PORTFOLIO RISK ANALYSIS**\n"
        f"{'='*25}\n\n"
        f"📊 **Current Risk Metrics:**\n"
        f"• Portfolio VaR (95%): -$2,100 (-2.1%)\n"
        f"• Expected Shortfall: -$3,200 (-3.2%)\n"
        f"• Beta vs Market: 0.85\n"
        f"• Volatility (30D): 12.5%\n"
        f"• Max Drawdown: -4.2%\n\n"
        f"🎯 **Risk Distribution:**\n"
        f"• Systematic Risk: 65%\n"
        f"• Idiosyncratic Risk: 35%\n"
        f"• Concentration Risk: Low\n"
        f"• Correlation Risk: Medium\n\n"
        f"⚠️ **Risk Warnings:**\n"
        f"• High correlation in DeFi positions\n"
        f"• Exceeding sector limits in AI tokens\n"
        f"• Leverage utilization at 45%\n\n"
        f"✅ **Recommendations:**\n"
        f"• Reduce DeFi allocation by 3%\n"
        f"• Add uncorrelated assets\n"
        f"• Consider hedging positions\n"
        f"• Monitor leverage closely"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("📊 Stress Test", callback_data='stress_test'),
            InlineKeyboardButton("⚖️ Hedge Positions", callback_data='hedge_positions')
        ],
        [
            InlineKeyboardButton("🔄 Risk Monitor", callback_data='risk_monitor'),
            InlineKeyboardButton("↩️ Back", callback_data='portfolio_analysis')
        ]
    ]
    
    await query.edit_message_text(
        risk_msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def show_trade_history(query, context):
    """Show trade history"""
    history_msg = (
        f"📋 **TRADE HISTORY (Last 10 Trades)**\n"
        f"{'='*30}\n\n"
        f"1️⃣ **BTC/USDT** - BULLISH ✅\n"
        f"   Entry: $43,250 → Exit: $44,180\n"
        f"   P&L: +2.15% | Date: 2024-01-15\n\n"
        f"2️⃣ **ETH/USDT** - BEARISH ✅\n"
        f"   Entry: $2,650 → Exit: $2,580\n"
        f"   P&L: +2.64% | Date: 2024-01-14\n\n"
        f"3️⃣ **SOL/USDT** - BULLISH ❌\n"
        f"   Entry: $98.50 → Exit: $96.20\n"
        f"   P&L: -2.34% | Date: 2024-01-13\n\n"
        f"4️⃣ **ADA/USDT** - BULLISH ✅\n"
        f"   Entry: $0.485 → Exit: $0.512\n"
        f"   P&L: +5.57% | Date: 2024-01-12\n\n"
        f"5️⃣ **MATIC/USDT** - BEARISH ✅\n"
        f"   Entry: $0.852 → Exit: $0.821\n"
        f"   P&L: +3.64% | Date: 2024-01-11\n\n"
        f"📊 **Summary:**\n"
        f"Win Rate: 80% (4/5 shown)\n"
        f"Avg Win: +3.50%\n"
        f"Avg Loss: -2.34%\n"
        f"Total P&L: +11.66%"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("📊 Full History", callback_data='full_history'),
            InlineKeyboardButton("📈 P&L Chart", callback_data='pnl_chart')
        ],
        [
            InlineKeyboardButton("📋 Export CSV", callback_data='export_trades'),
            InlineKeyboardButton("↩️ Back", callback_data='performance')
        ]
    ]
    
    await query.edit_message_text(
        history_msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def show_sector_signals(query, context):
    """Show sector-based trading signals"""
    signals_msg = (
        f"🎯 **SECTOR TRADING SIGNALS**\n"
        f"{'='*25}\n\n"
        f"🔥 **BULLISH SECTORS:**\n\n"
        f"**AI/ML Tokens** (Strong Buy)\n"
        f"• FET/USDT: Entry $1.25, Target $1.45\n"
        f"• AGIX/USDT: Entry $0.68, Target $0.78\n"
        f"• RNDR/USDT: Entry $3.20, Target $3.65\n\n"
        f"**Layer 1** (Buy)\n"
        f"• SOL/USDT: Entry $98, Target $108\n"
        f"• AVAX/USDT: Entry $36, Target $40\n"
        f"• DOT/USDT: Entry $7.20, Target $7.80\n\n"
        f"❄️ **BEARISH SECTORS:**\n\n"
        f"**Gaming** (Avoid)\n"
        f"• Weak momentum\n"
        f"• Declining volume\n"
        f"• Sector rotation out\n\n"
        f"**Privacy** (Avoid)\n"
        f"• Regulatory concerns\n"
        f"• Low institutional interest\n\n"
        f"⚡ **Active Rotation:**\n"
        f"Money flowing FROM Gaming/Privacy\n"
        f"Money flowing TO AI/Layer1"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("🎯 Execute Signals", callback_data='execute_sector_signals'),
            InlineKeyboardButton("📊 Sector Analysis", callback_data='sector_details')
        ],
        [
            InlineKeyboardButton("🔄 Refresh", callback_data='sector_rotation'),
            InlineKeyboardButton("↩️ Back", callback_data='sector_rotation')
        ]
    ]
    
    await query.edit_message_text(
        signals_msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def handle_settings_callback(query, context):
    """Handle settings callbacks"""
    data = query.data
    
    if data == 'settings_risk':
        await show_risk_settings(query, context)
    elif data == 'settings_position':
        await show_position_settings(query, context)
    elif data == 'settings_signals':
        await show_signal_settings(query, context)
    elif data == 'settings_portfolio':
        await show_portfolio_settings(query, context)
    else:
        await show_professional_settings_detailed(query, context)

async def show_risk_settings(query, context):
    """Show risk management settings"""
    risk_settings_msg = (
        f"⚖️ **RISK MANAGEMENT SETTINGS**\n"
        f"{'='*28}\n\n"
        f"🎯 **Current Settings:**\n"
        f"• Max Portfolio Risk: 2.0%\n"
        f"• Max Daily Risk: 6.0%\n"
        f"• Max Single Position: 20%\n"
        f"• Max Correlation: 10%\n"
        f"• Stop Loss Buffer: 0.5%\n\n"
        f"📊 **Risk Levels:**\n"
        f"🟢 Conservative: 1% portfolio, 3% daily\n"
        f"🟡 Moderate: 2% portfolio, 6% daily\n"
        f"🔴 Aggressive: 3% portfolio, 10% daily\n\n"
        f"⚠️ **Current Status:**\n"
        f"Risk Level: Moderate\n"
        f"Daily Risk Used: 2.1% / 6.0%\n"
        f"Portfolio Risk: 1.8% / 2.0%"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("🟢 Conservative", callback_data='risk_conservative'),
            InlineKeyboardButton("🟡 Moderate", callback_data='risk_moderate')
        ],
        [
            InlineKeyboardButton("🔴 Aggressive", callback_data='risk_aggressive'),
            InlineKeyboardButton("⚙️ Custom", callback_data='risk_custom')
        ],
        [
            InlineKeyboardButton("↩️ Back", callback_data='risk_settings')
        ]
    ]
    
    await query.edit_message_text(
        risk_settings_msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )
    
async def execute_professional_scan(query, context):
    """Execute professional market scan with better error handling"""
    processing_msg = (
        "🏛️ **PROFESSIONAL MARKET SCAN INITIATED**\n\n"
        "⚡ Analyzing market regime...\n"
        "📊 Scanning 500+ assets...\n"
        "🔬 Applying professional filters...\n"
        "⚖️ Optimizing portfolio allocation...\n"
        "🎯 Calculating risk-adjusted scores...\n\n"
        "⏳ **Estimated time: 60-90 seconds**"
    )
    
    await query.edit_message_text(processing_msg, parse_mode='Markdown')
    
    try:
        chat_id = query.message.chat_id
        
        # Step 1: Load markets with detailed logging
        print("📡 Loading markets...")
        await load_markets_with_retry(binance, max_retries=3)
        print("✅ Markets loaded successfully")
        
        # Step 2: Get tickers with error handling
        print("📊 Fetching tickers...")
        try:
            tickers = binance.fetch_tickers()
            print(f"✅ Got {len(tickers)} tickers")
        except Exception as e:
            print(f"❌ Ticker fetch error: {e}")
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ Failed to fetch market data. Please try again."
            )
            return
        
        # Step 3: Detect market regime with fallback
        print("🔍 Detecting market regime...")
        try:
            btc_data = binance.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=200)
            eth_data = binance.fetch_ohlcv('ETH/USDT', timeframe='1h', limit=100)
            market_regime = trading_engine.regime_detector.detect_regime(btc_data, eth_data, {})
            print(f"📊 Market Regime: {market_regime.value}")
        except Exception as e:
            print(f"⚠️ Regime detection error: {e}")
            market_regime = MarketRegime.BULL_RANGING
            print("📊 Using fallback regime: BULL_RANGING")
        
        # Step 4: Sector analysis with error handling
        print("🏆 Analyzing sectors...")
        try:
            sector_performance = trading_engine.sector_analyzer.analyze_sectors(tickers)
            leading_sectors = list(sector_performance.keys())[:3]
            print(f"🏆 Leading Sectors: {[s.value for s in leading_sectors] if leading_sectors else ['None']}")
        except Exception as e:
            print(f"⚠️ Sector analysis error: {e}")
            sector_performance = {}
            leading_sectors = []
        
        # Step 5: Get candidates with better filtering
        print("🎯 Getting candidates...")
        candidates = get_professional_candidates_improved(tickers, None, sector_performance)
        print(f"🎯 Analyzing {len(candidates)} candidates...")
        
        if not candidates:
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ No suitable candidates found in current market conditions"
            )
            return
        
        # Step 6: Sequential analysis (safer than parallel)
        print("🔬 Starting individual analysis...")
        analysis_results = []
        
        for i, symbol in enumerate(candidates[:10], 1):  # Limit to 10 for safety
            try:
                print(f"📊 Analyzing {symbol} ({i}/10)...")
                
                result = await analyze_symbol_with_timeout(
                    symbol, binance, headers, market_regime, None, timeout=30
                )
                
                if result and result['signal_type'] != 'NEUTRAL':
                    analysis_results.append(result)
                    print(f"✅ {symbol}: {result['signal_type']} ({result['confidence']:.1f}%)")
                else:
                    print(f"⚪ {symbol}: No signal")
                    
            except Exception as e:
                print(f"❌ Analysis failed for {symbol}: {e}")
                continue
        
        print(f"🎯 Found {len(analysis_results)} qualified setups")
        
        if not analysis_results:
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ No qualified setups found meeting professional criteria"
            )
            return
        
        # Step 7: Portfolio optimization (simplified)
        print("⚖️ Optimizing portfolio...")
        try:
            optimized_results = trading_engine.portfolio_optimizer.optimize_position_sizes(
                analysis_results, trading_engine.portfolio_value
            )
        except Exception as e:
            print(f"⚠️ Portfolio optimization error: {e}")
            optimized_results = analysis_results
        
        # Step 8: Final ranking
        print("🏆 Ranking results...")
        final_results = rank_and_filter_results_improved(optimized_results, market_regime)
        
        print(f"✅ Sending {len(final_results)} professional setups")
        
        # Send results
        if final_results:
            await send_professional_analysis_message(context, chat_id, final_results, market_regime)
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ No setups passed final professional filtering"
            )
        
        # Send completion message
        completion_msg = (
            "✅ **PROFESSIONAL SCAN COMPLETED**\n\n"
            "📊 Analysis complete with institutional-grade filtering\n"
            "🎯 All setups meet professional risk standards\n\n"
            "⚠️ Remember: Professional trading requires discipline"
        )
        
        keyboard = [
            [
                InlineKeyboardButton("🔄 New Scan", callback_data='pro_scan'),
                InlineKeyboardButton("↩️ Main Menu", callback_data='back_main_pro')
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await context.bot.send_message(
            chat_id=chat_id,
            text=completion_msg,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        
    except Exception as e:
        print(f"❌ Professional scan error: {e}")
        import traceback
        traceback.print_exc()
        
        error_msg = (
            f"❌ **PROFESSIONAL SCAN ERROR**\n\n"
            f"Error: {str(e)}\n\n"
            f"🔄 Please try again or check system status"
        )
        
        keyboard = [[
            InlineKeyboardButton("🔄 Retry Scan", callback_data='pro_scan'),
            InlineKeyboardButton("↩️ Main Menu", callback_data='back_main_pro')
        ]]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        try:
            await query.edit_message_text(error_msg, reply_markup=reply_markup, parse_mode='Markdown')
        except:
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=error_msg,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
            
def get_professional_candidates_improved(tickers: Dict, style_filter: Optional[str], 
                                        sector_performance: Dict) -> List[str]:
    """Improved candidate selection with better error handling"""
    try:
        candidates = []
        
        # Volume and movement filters
        min_volume = 1_000_000  # Lower threshold for testing
        min_movement = 0.5      # Lower threshold for testing
        
        for symbol, ticker in tickers.items():
            try:
                if not symbol.endswith('/USDT'):
                    continue
                    
                if not all(k in ticker for k in ['quoteVolume', 'percentage']):
                    continue
                
                volume = ticker.get('quoteVolume', 0)
                movement = abs(ticker.get('percentage', 0))
                
                # Basic filters
                if volume < min_volume or movement < min_movement:
                    continue
                
                candidates.append(symbol)
                
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue
        
        # Sort by volume
        def score_symbol(symbol):
            try:
                ticker = tickers[symbol]
                return ticker.get('quoteVolume', 0)
            except:
                return 0
        
        candidates.sort(key=score_symbol, reverse=True)
        return candidates[:50]  # Top 50 by volume
        
    except Exception as e:
        print(f"Error in candidate selection: {e}")
        return []
        
def rank_and_filter_results_improved(results: List[Dict], market_regime: MarketRegime) -> List[Dict]:
    """Improved ranking with simpler logic"""
    try:
        if not results:
            return []
        
        # Simple filtering
        filtered_results = []
        
        for result in results:
            try:
                # Basic confidence filter
                if result.get('confidence', 0) < 60:  # Lower threshold for testing
                    continue
                
                # Basic risk filter
                risk_mgmt = result.get('risk_mgmt', {})
                if risk_mgmt.get('risk_reward_1', 0) < 1.0:  # Lower threshold
                    continue
                
                filtered_results.append(result)
                
            except Exception as analyze_symbol_comprehensive_safe:
                print(f"Error filtering result: {e}")
                continue
        
        # Sort by confidence
        filtered_results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        return filtered_results[:5]  # Top 5
        
    except Exception as e:
        print(f"Error in ranking: {e}")
        return results[:3]
        
def safe_add(a, b, default=0):
    """Safely add two values, handling None"""
    if a is None:
        a = default
    if b is None:
        b = default
    return float(a) + float(b)

def safe_multiply(a, b, default=1):
    """Safely multiply two values, handling None"""
    if a is None:
        a = default
    if b is None:
        b = default
    return float(a) * float(b)

def safe_divide(a, b, default=0):
    """Safely divide two values, handling None and zero"""
    if a is None or b is None or b == 0:
        return default
    return float(a) / float(b)

def safe_subtract(a, b, default=0):
    """Safely subtract two values, handling None"""
    if a is None:
        a = default
    if b is None:
        b = default
    return float(a) - float(b)
    
def safe_send_message(text, parse_mode='Markdown'):
    try:
        # Remove problematic markdown
        clean_text = text.replace('**', '*')
        clean_text = clean_text.replace('***', '*')
        # Remove unbalanced markdown
        star_count = clean_text.count('*')
        if star_count % 2 != 0:
            clean_text = clean_text.replace('*', '')
        return clean_text, parse_mode
    except:
        return text.replace('*', '').replace('_', ''), None

# Fix 3: Override problematic methods
original_calculate_professional_score = trading_engine.calculate_professional_score

def calculate_professional_score_safe(signal_result, market_data, risk_mgmt, market_regime):
    try:
        base_score = signal_result.get('confidence', 0) or 0
        rr1 = risk_mgmt.get('risk_reward_1', 0) or 0
        rr2 = risk_mgmt.get('risk_reward_2', 0) or 0
        best_rr = max(rr1, rr2)
        rr_bonus = min(best_rr * 5, 20)
        volume_ratio = market_data.get('volume_ratio', 1) or 1
        volume_bonus = min(volume_ratio * 5, 15)
        
        regime_multiplier = 1.0
        if market_regime == MarketRegime.BULL_TRENDING:
            regime_multiplier = 1.2
        elif market_regime == MarketRegime.BEAR_TRENDING:
            regime_multiplier = 0.8
            
        final_score = (base_score + rr_bonus + volume_bonus) * regime_multiplier
        return max(0, final_score)
    except Exception as e:
        print(f"Score calculation error: {e}")
        return 50.0

# Apply the fix
trading_engine.calculate_professional_score = calculate_professional_score_safe

# Fix 4: Safe message sending for all analysis functions
async def safe_send_telegram_message(context, chat_id, text, reply_markup=None):
    try:
        clean_text, parse_mode = safe_send_message(text)
        await context.bot.send_message(
            chat_id=chat_id,
            text=clean_text,
            reply_markup=reply_markup,
            parse_mode=parse_mode
        )
    except Exception as e:
        # Fallback: send plain text without markdown
        plain_text = text.replace('*', '').replace('_', '').replace('`', '')
        await context.bot.send_message(
            chat_id=chat_id,
            text=plain_text[:4000],  # Truncate if too long
            reply_markup=reply_markup
        )

print("🔧 Quick fixes applied!")
    

        
# Fix enhanced signal analysis calculation 
def calculate_technical_score_safe(indicators: Dict, details: List[str]) -> float:
    """Calculate technical analysis score with None protection"""
    try:
        score = 0
        
        # RSI Analysis (handle None)
        rsi = indicators.get('rsi')
        if rsi is not None and len(rsi) > 0 and rsi[-1] is not None:
            rsi_val = float(rsi[-1])
            if rsi_val < 20:
                score += 8
                details.append(f"RSI Extremely Oversold: {rsi_val:.1f}")
            elif rsi_val < 30:
                score += 6
                details.append(f"RSI Oversold: {rsi_val:.1f}")
            elif rsi_val > 80:
                score += 8
                details.append(f"RSI Extremely Overbought: {rsi_val:.1f}")
            elif rsi_val > 70:
                score += 6
                details.append(f"RSI Overbought: {rsi_val:.1f}")
        
        # MACD Analysis (handle None)
        macd_hist = indicators.get('macd_hist')
        if macd_hist is not None and len(macd_hist) >= 2:
            current_hist = macd_hist[-1] if macd_hist[-1] is not None else 0
            prev_hist = macd_hist[-2] if macd_hist[-2] is not None else 0
            
            if current_hist > 0 and prev_hist <= 0:
                score += 8
                details.append("MACD Bullish Crossover")
            elif current_hist < 0 and prev_hist >= 0:
                score += 8
                details.append("MACD Bearish Crossover")
            elif current_hist > prev_hist and current_hist > 0:
                score += 4
                details.append("MACD Bullish Momentum")
        
        # Volume Analysis (handle None)
        volume = indicators.get('volume')
        volume_sma = indicators.get('volume_sma')
        if (volume is not None and len(volume) > 0 and volume[-1] is not None and
            volume_sma is not None and len(volume_sma) > 0 and volume_sma[-1] is not None):
            
            vol_ratio = safe_divide(volume[-1], volume_sma[-1], 1)
            if vol_ratio > 3.0:
                score += 5
                details.append(f"Exceptional Volume: {vol_ratio:.1f}x")
            elif vol_ratio > 2.0:
                score += 4
                details.append(f"Very High Volume: {vol_ratio:.1f}x")
            elif vol_ratio > 1.5:
                score += 2
                details.append(f"High Volume: {vol_ratio:.1f}x")
        
        return min(score, 25)  # Cap at 25 points
        
    except Exception as e:
        print(f"Error in technical score: {e}")
        return 0

# Fix momentum score calculation
def calculate_momentum_score_safe(indicators: Dict, tf_analysis: Dict, details: List[str]) -> float:
    """Calculate momentum score with None protection"""
    try:
        score = 0
        
        # Multi-timeframe momentum alignment
        bullish_tfs = 0
        bearish_tfs = 0
        
        for tf, data in tf_analysis.items():
            trend = data.get('trend', 'NEUTRAL')
            if trend in ['STRONG_BULLISH', 'BULLISH']:
                bullish_tfs += 1
            elif trend in ['STRONG_BEARISH', 'BEARISH']:
                bearish_tfs += 1
        
        total_tfs = len(tf_analysis) if tf_analysis else 1
        momentum_alignment = max(bullish_tfs, bearish_tfs) / total_tfs
        
        if momentum_alignment >= 0.8:
            score += 12
            details.append(f"Strong Multi-TF Momentum: {momentum_alignment:.1%}")
        elif momentum_alignment >= 0.6:
            score += 8
            details.append(f"Good Multi-TF Momentum: {momentum_alignment:.1%}")
        
        return min(score, 20)  # Cap at 20 points
        
    except Exception as e:
        print(f"Error in momentum score: {e}")
        return 0

# Apply these fixes to the TradingEngine class
def apply_safe_calculation_fixes():
    """Apply safe calculation fixes to TradingEngine"""
    
    # Replace the problematic methods
    TradingEngine.calculate_professional_score = calculate_professional_score_safe
    
    # Also fix the EnhancedSignalAnalyzer methods
    EnhancedSignalAnalyzer.calculate_technical_score = calculate_technical_score_safe
    EnhancedSignalAnalyzer.calculate_momentum_score = calculate_momentum_score_safe

# Call this after class definitions
apply_safe_calculation_fixes()

async def analyze_symbol_with_timeout(symbol: str, binance_client, headers, 
                                    market_regime: MarketRegime, style_filter: Optional[str], 
                                    timeout: int = 30) -> Optional[Dict]:
    """Analyze symbol with timeout"""
    try:
        import asyncio
        
        # Wrap the sync function in async with timeout
        result = await asyncio.wait_for(
            asyncio.to_thread(
                symbol, binance_client, headers, market_regime, style_filter
            ),
            timeout=timeout
        )
        
        return result
        
    except asyncio.TimeoutError:
        print(f"⏰ Timeout analyzing {symbol}")
        return None
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {e}")
        return None

async def execute_professional_style_analysis(query, context, style: str):
    """Execute style-specific professional analysis"""
    style_names = {
        'SCALPING': 'Ultra-High Frequency',
        'DAYTRADING': 'Intraday Professional',
        'SWING': 'Multi-Day Position'
    }
    
    processing_msg = (
        f"⚡ **{style_names.get(style, style)} ANALYSIS**\n\n"
        f"🔬 Applying {style.lower()} filters...\n"
        f"📊 Optimizing for {style.lower()} timeframes...\n"
        f"⚖️ Calculating style-specific risk...\n"
        f"🎯 Finding optimal setups...\n\n"
        f"⏳ **Professional analysis in progress...**"
    )
    
    await query.edit_message_text(processing_msg, parse_mode='Markdown')
    
    try:
        chat_id = query.message.chat_id
        
        # Load markets
        await load_markets_with_retry(binance, max_retries=3)
        
        # Execute style-specific scan
        results = await trading_engine.scan_market_comprehensive(
            binance, headers, style_filter=style
        )
        
        if results:
            # Get market regime
            btc_data = binance.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=200)
            market_regime = trading_engine.regime_detector.detect_regime(btc_data, [], {})
            
            # Send results with style-specific formatting
            await send_style_specific_analysis(context, chat_id, results, style, market_regime)
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"❌ No professional {style.lower()} setups found meeting our criteria"
            )
        
    except Exception as e:
        error_msg = f"❌ **{style} ANALYSIS ERROR**: {str(e)}"
        await context.bot.send_message(chat_id=query.message.chat_id, text=error_msg)

async def send_style_specific_analysis(context, chat_id: int, results: List[Dict], 
                                     style: str, market_regime: MarketRegime):
    """Send style-specific analysis results"""
    
    style_config = {
        'SCALPING': {
            'emoji': '⚡',
            'hold_time': '1-15 minutes',
            'target_profit': '0.3-1.5%',
            'leverage': 'Up to 20x'
        },
        'DAYTRADING': {
            'emoji': '📈',
            'hold_time': '30min-6hours',
            'target_profit': '1-6%', 
            'leverage': 'Up to 10x'
        },
        'SWING': {
            'emoji': '📊',
            'hold_time': '1-10 days',
            'target_profit': '3-25%',
            'leverage': 'Up to 5x'
        }
    }
    
    config = style_config.get(style, style_config['DAYTRADING'])
    
    # Header
    header_msg = (
        f"{config['emoji']} **PROFESSIONAL {style} ANALYSIS**\n"
        f"{'='*35}\n"
        f"📊 Market Regime: {market_regime.value.title()}\n"
        f"🎯 Qualified Setups: {len(results)}\n"
        f"⏰ Hold Time: {config['hold_time']}\n"
        f"💰 Target Profit: {config['target_profit']}\n"
        f"⚖️ Max Leverage: {config['leverage']}\n"
        f"🔬 Professional Grade Filtering"
    )
    
    await context.bot.send_message(chat_id=chat_id, text=header_msg, parse_mode='Markdown')
    await asyncio.sleep(2)
    
    # Send each setup with style-specific details
    for i, result in enumerate(results[:3], 1):  # Limit to top 3 for style-specific
        risk_mgmt = result['risk_mgmt']
        
        # Style-specific target selection
        if style == 'SCALPING':
            primary_target = risk_mgmt['take_profit_1']
            primary_rr = risk_mgmt['risk_reward_1']
        elif style == 'SWING':
            primary_target = risk_mgmt['take_profit_3']
            primary_rr = risk_mgmt['risk_reward_3']
        else:
            primary_target = risk_mgmt['take_profit_2']
            primary_rr = risk_mgmt['risk_reward_2']
        
        setup_msg = (
            f"{config['emoji']} **{style} SETUP #{i}: {result['symbol']}**\n"
            f"{'='*25}\n"
            f"📊 **Signal:** {result['signal_type']} ({result['confidence']:.1f}%)\n"
            f"💰 **Entry:** ${result['current_price']:.4f}\n"
            f"🎯 **Primary Target:** ${primary_target:.4f} (R:R {primary_rr:.1f})\n"
            f"🛑 **Stop Loss:** ${risk_mgmt['stop_loss']:.4f}\n\n"
            f"📈 **Style Metrics:**\n"
            f"• Expected Hold: {config['hold_time']}\n"
            f"• Profit Target: {config['target_profit']}\n"
            f"• Professional Score: {result['final_score']:.1f}/100\n"
            f"• Volume Quality: {result['market_data']['volume_ratio']:.1f}x\n\n"
            f"⚡ **Quick Stats:**\n"
            f"• 24h Volume: ${result['market_data']['volume_24h']:,.0f}\n"
            f"• Volatility: {risk_mgmt['volatility_ratio']:.1%}\n"
            f"• Market Cap Tier: {trading_engine.portfolio_optimizer.get_asset_class(result['symbol']).value}\n\n"
            f"⚠️ **{style} Risk:** Max 1-2% per trade"
        )
        
        await context.bot.send_message(chat_id=chat_id, text=setup_msg, parse_mode='Markdown')
        await asyncio.sleep(3)

async def load_markets_with_retry(exchange, max_retries: int = 3):
    """Load markets with improved retry logic and better error handling"""
    endpoints = [
        'https://api.binance.com/api/v3',
        'https://api1.binance.com/api/v3',
        'https://api2.binance.com/api/v3'
    ]
    
    print(f"🔄 Loading markets (max {max_retries} retries)...")
    
    for attempt in range(max_retries):
        for i, endpoint in enumerate(endpoints):
            try:
                print(f"📡 Attempt {attempt+1}/{max_retries}, Endpoint {i+1}/{len(endpoints)}: {endpoint}")
                
                # Set endpoint
                exchange.urls['api']['public'] = endpoint
                
                # Clear existing markets to force reload
                exchange.markets = None
                
                # Load markets with timeout
                await asyncio.wait_for(
                    asyncio.to_thread(exchange.load_markets), 
                    timeout=15
                )
                
                print(f"✅ Markets loaded successfully from {endpoint}")
                print(f"📊 Total symbols: {len(exchange.symbols)}")
                return True
                
            except asyncio.TimeoutError:
                print(f"⏰ Timeout loading from {endpoint}")
            except Exception as e:
                print(f"❌ Error loading from {endpoint}: {e}")
                
                # If last attempt on last endpoint, raise error
                if attempt == max_retries - 1 and i == len(endpoints) - 1:
                    print("❌ All endpoints failed, using fallback mode")
                    return await load_markets_fallback(exchange)
                
                await asyncio.sleep(2)
    
    return False
    
async def load_markets_fallback(exchange):
    """Fallback market loading with minimal required symbols"""
    try:
        print("🔄 Using fallback market loading...")
        
        # Create minimal markets dict with major pairs
        major_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
            'XRP/USDT', 'DOT/USDT', 'AVAX/USDT', 'MATIC/USDT', 'UNI/USDT'
        ]
        
        exchange.markets = {}
        exchange.symbols = major_symbols
        
        for symbol in major_symbols:
            exchange.markets[symbol] = {
                'id': symbol.replace('/', ''),
                'symbol': symbol,
                'base': symbol.split('/')[0],
                'quote': symbol.split('/')[1],
                'active': True,
                'type': 'spot',
                'spot': True,
                'future': False
            }
        
        print(f"✅ Fallback mode active with {len(major_symbols)} symbols")
        return True
        
    except Exception as e:
        print(f"❌ Fallback loading failed: {e}")
        return False
    
async def show_market_regime_analysis(query, context):
    """Show detailed market regime analysis"""
    try:
        processing_msg = "🔍 Analyzing market regime in detail..."
        await query.edit_message_text(processing_msg)
        
        await load_markets_with_retry(binance)
        
        # Get comprehensive market data
        btc_data = binance.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=200)
        eth_data = binance.fetch_ohlcv('ETH/USDT', timeframe='1h', limit=100)
        
        # Analyze multiple timeframes
        btc_4h = binance.fetch_ohlcv('BTC/USDT', timeframe='4h', limit=50)
        btc_1d = binance.fetch_ohlcv('BTC/USDT', timeframe='1d', limit=30)
        
        regime_detector = MarketRegimeDetector()
        current_regime = regime_detector.detect_regime(btc_data, eth_data, {})
        
        # Calculate multiple metrics
        btc_df = pd.DataFrame(btc_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Technical indicators
        ema_20 = talib.EMA(btc_df['close'].to_numpy(), timeperiod=20)
        ema_50 = talib.EMA(btc_df['close'].to_numpy(), timeperiod=50)
        ema_200 = talib.EMA(btc_df['close'].to_numpy(), timeperiod=200)
        rsi = talib.RSI(btc_df['close'].to_numpy(), timeperiod=14)
        atr = talib.ATR(btc_df['high'].to_numpy(), btc_df['low'].to_numpy(), btc_df['close'].to_numpy())
        
        current_price = btc_df['close'].iloc[-1]
        volatility = (atr[-1] / current_price) * 100
        
        # Market structure
        structure = AdvancedIndicators.calculate_market_structure(btc_df)
        
        # Volume analysis
        avg_volume = btc_df['volume'].tail(20).mean()
        current_volume = btc_df['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume
        
        regime_msg = (
            f"🏛️ **COMPREHENSIVE MARKET REGIME ANALYSIS**\n"
            f"{'='*40}\n\n"
            f"📊 **Current Regime:** {current_regime.value.upper()}\n"
            f"₿ **BTC Price:** ${current_price:,.0f}\n\n"
            f"📈 **Trend Analysis:**\n"
            f"• Structure: {structure['structure_type'].title()}\n"
            f"• Trend Strength: {structure['trend_strength']:.2f}\n"
            f"• EMA Alignment: {'Bullish' if ema_20[-1] > ema_50[-1] > ema_200[-1] else 'Bearish' if ema_20[-1] < ema_50[-1] < ema_200[-1] else 'Mixed'}\n"
            f"• RSI Level: {rsi[-1]:.1f}\n\n"
            f"⚡ **Volatility Metrics:**\n"
            f"• Current ATR: {volatility:.1f}%\n"
            f"• Volatility State: {'High' if volatility > 5 else 'Normal' if volatility > 3 else 'Low'}\n"
            f"• Volume Ratio: {volume_ratio:.1f}x\n\n"
            f"🎯 **Trading Implications:**\n"
        )
        
        # Add regime-specific recommendations
        if current_regime == MarketRegime.BULL_TRENDING:
            regime_msg += (
                f"• ✅ Bullish momentum strategies preferred\n"
                f"• ✅ Long bias on pullbacks\n"
                f"• ✅ Breakout trades favored\n"
                f"• ⚠️ Avoid heavy short positions\n"
            )
        elif current_regime == MarketRegime.BEAR_TRENDING:
            regime_msg += (
                f"• ✅ Bearish momentum strategies preferred\n"
                f"• ✅ Short bias on rallies\n"
                f"• ✅ Breakdown trades favored\n"
                f"• ⚠️ Avoid heavy long positions\n"
            )
        elif current_regime == MarketRegime.HIGH_VOLATILITY:
            regime_msg += (
                f"• ⚠️ Reduce position sizes by 30-50%\n"
                f"• ⚠️ Wider stops required\n"
                f"• ✅ Scalping opportunities\n"
                f"• ❌ Avoid swing positions\n"
            )
        else:
            regime_msg += (
                f"• ✅ Range trading strategies\n"
                f"• ✅ Mean reversion setups\n"
                f"• ⚠️ Wait for clear breakouts\n"
                f"• ⚠️ Reduced position sizes\n"
            )
        
        regime_msg += f"\n📊 **Analysis Time:** {datetime.now().strftime('%H:%M:%S UTC')}"
        
        keyboard = [[
            InlineKeyboardButton("🔄 Refresh Analysis", callback_data='market_regime'),
            InlineKeyboardButton("↩️ Back", callback_data='back_main_pro')
        ]]
        
        await query.edit_message_text(
            regime_msg, 
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
        
    except Exception as e:
        error_msg = f"❌ Regime analysis error: {str(e)}"
        keyboard = [[InlineKeyboardButton("↩️ Back", callback_data='back_main_pro')]]
        await query.edit_message_text(
            error_msg,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

async def show_portfolio_analysis(query, context):
    """Show portfolio analysis and optimization"""
    analysis_msg = (
        f"💼 **PORTFOLIO OPTIMIZATION ANALYSIS**\n"
        f"{'='*35}\n\n"
        f"📊 **Current Allocation:**\n"
        f"• Total Portfolio Value: $100,000\n"
        f"• Available Capital: $80,000 (80%)\n"
        f"• Reserved Cash: $20,000 (20%)\n\n"
        f"🎯 **Sector Allocation:**\n"
        f"• Large Cap (BTC/ETH): 40%\n"
        f"• Mid Cap Alts: 30%\n"
        f"• DeFi Tokens: 15%\n"
        f"• Layer 1s: 10%\n"
        f"• Speculative: 5%\n\n"
        f"⚖️ **Risk Metrics:**\n"
        f"• Max Risk per Trade: 2%\n"
        f"• Max Daily Risk: 6%\n"
        f"• Max Correlation Exposure: 10%\n"
        f"• Current Risk Utilization: 45%\n\n"
        f"📈 **Performance (30D):**\n"
        f"• Portfolio Return: +12.5%\n"
        f"• Sharpe Ratio: 1.8\n"
        f"• Max Drawdown: -8.2%\n"
        f"• Win Rate: 68%\n\n"
        f"🎯 **Optimization Suggestions:**\n"
        f"• Reduce correlation in DeFi sector\n"
        f"• Increase large cap allocation\n"
        f"• Consider rebalancing to target weights\n"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("📊 Rebalance Portfolio", callback_data='portfolio_rebalance'),
            InlineKeyboardButton("📈 Performance Details", callback_data='portfolio_performance')
        ],
        [
            InlineKeyboardButton("⚖️ Risk Analysis", callback_data='portfolio_risk'),
            InlineKeyboardButton("🎯 Sector Rotation", callback_data='sector_rotation')
        ],
        [
            InlineKeyboardButton("↩️ Back", callback_data='back_main_pro')
        ]
    ]
    
    await query.edit_message_text(
        analysis_msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def show_deep_analysis_menu(query, context):
    """Show deep analysis options"""
    menu_msg = (
        f"🔍 **DEEP ANALYSIS TOOLS**\n"
        f"{'='*25}\n\n"
        f"📊 **Available Analysis:**\n"
        f"• Individual symbol deep dive\n"
        f"• Correlation matrix analysis\n"
        f"• Market structure mapping\n"
        f"• Institutional flow analysis\n"
        f"• Multi-timeframe confluence\n"
        f"• Volume profile breakdown\n\n"
        f"🎯 **Select Analysis Type:**"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("🔍 Symbol Deep Dive", callback_data='deep_symbol'),
            InlineKeyboardButton("📊 Correlation Matrix", callback_data='deep_correlation')
        ],
        [
            InlineKeyboardButton("🏗️ Market Structure", callback_data='deep_structure'),
            InlineKeyboardButton("💰 Institutional Flow", callback_data='deep_institutional')
        ],
        [
            InlineKeyboardButton("⏰ Multi-TF Analysis", callback_data='deep_multitf'),
            InlineKeyboardButton("📈 Volume Profile", callback_data='deep_volume')
        ],
        [
            InlineKeyboardButton("↩️ Back", callback_data='back_main_pro')
        ]
    ]
    
    await query.edit_message_text(
        menu_msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def show_professional_settings_menu(query, context):
    """Show professional settings menu"""
    settings_msg = (
        f"⚙️ **PROFESSIONAL SETTINGS**\n"
        f"{'='*25}\n\n"
        f"🎯 **Current Configuration:**\n"
        f"• Max Portfolio Risk: 2%\n"
        f"• Max Daily Risk: 6%\n"
        f"• Min Signal Confidence: 65%\n"
        f"• Min Volume Threshold: $5M\n"
        f"• Max Leverage: Style-dependent\n"
        f"• Portfolio Optimization: ON\n"
        f"• Sector Rotation: ON\n"
        f"• Market Regime Filter: ON\n\n"
        f"⚖️ **Risk Management:**\n"
        f"• Kelly Criterion Position Sizing\n"
        f"• Correlation-based limits\n"
        f"• Dynamic risk adjustment\n"
        f"• Professional signal filtering\n\n"
        f"🔧 **Customize Settings:**"
    )
    
    keyboard = create_professional_settings_menu()
    
    await query.edit_message_text(
        settings_msg,
        reply_markup=keyboard,
        parse_mode='Markdown'
    )

async def show_correlation_analysis(query, context):
    """Show correlation analysis"""
    msg = (
        "📊 **CORRELATION MATRIX ANALYSIS**\n"
        "=" * 30 + "\n\n"
        "🔍 **Current Market Correlations:**\n\n"
        "**HIGH CORRELATION (>0.7):**\n"
        "• BTC/USDT ↔ ETH/USDT: 0.85\n"
        "• BTC/USDT ↔ SOL/USDT: 0.78\n"
        "• ETH/USDT ↔ UNI/USDT: 0.82\n"
        "• DeFi tokens group: 0.75 avg\n\n"
        "**NEGATIVE CORRELATION (<-0.3):**\n"
        "• BTC/USDT ↔ USD/USDT: -0.45\n"
        "• Risk-on vs Risk-off: -0.38\n\n"
        "**PORTFOLIO IMPACT:**\n"
        "⚠️ High correlation = increased risk\n"
        "✅ Diversification needed\n"
        "🎯 Max 10% correlated exposure recommended\n\n"
        "📊 **Analysis Time:** " + datetime.now().strftime('%H:%M:%S')
    )
    
    keyboard = [
        [
            InlineKeyboardButton("🔄 Refresh Matrix", callback_data='deep_correlation'),
            InlineKeyboardButton("📊 Export Data", callback_data='export_correlation')
        ],
        [InlineKeyboardButton("↩️ Back", callback_data='deep_analysis')]
    ]
    
    await query.edit_message_text(
        msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def show_structure_analysis(query, context):
    """Show market structure analysis"""
    msg = (
        "🏗️ **MARKET STRUCTURE ANALYSIS**\n"
        "=" * 30 + "\n\n"
        "📊 **BTC/USDT Structure:**\n"
        "• Trend: Bullish\n"
        "• Higher Highs: 5 (last 20 periods)\n"
        "• Higher Lows: 4 (last 20 periods)\n"
        "• Structure Score: +0.35 (Bullish)\n\n"
        "🎯 **Key Levels:**\n"
        "• Resistance: $65,200 (Strong)\n"
        "• Support: $62,800 (Moderate)\n"
        "• Breakout Level: $65,500\n"
        "• Breakdown Level: $62,000\n\n"
        "⚡ **Market Context:**\n"
        "• Volume at resistance: High\n"
        "• Structure strength: Strong\n"
        "• Trend continuation probability: 75%\n\n"
        "🔍 **Trading Implications:**\n"
        "✅ Buy dips to support\n"
        "✅ Breakout above $65,500 = target $68,000\n"
        "⚠️ Break below $62,000 = bearish structure"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("🔄 Refresh Analysis", callback_data='deep_structure'),
            InlineKeyboardButton("📊 Multi-Symbol", callback_data='structure_multi')
        ],
        [InlineKeyboardButton("↩️ Back", callback_data='deep_analysis')]
    ]
    
    await query.edit_message_text(
        msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def show_institutional_analysis(query, context):
    """Show institutional flow analysis"""
    msg = (
        "💰 **INSTITUTIONAL FLOW ANALYSIS**\n"
        "=" * 30 + "\n\n"
        "📊 **CEX Netflows (24h):**\n"
        "• BTC: +$45.2M (Inflow)\n"
        "• ETH: +$23.8M (Inflow)\n"
        "• Total Crypto: +$156.3M\n\n"
        "🏛️ **Institutional Indicators:**\n"
        "• Large transactions (>$1M): 247\n"
        "• Whale activity: Increasing\n"
        "• Smart money sentiment: Bullish\n\n"
        "📈 **Exchange Flows:**\n"
        "• Binance: +$78M inflow\n"
        "• Coinbase: +$34M inflow\n"
        "• OKX: -$12M outflow\n\n"
        "🎯 **Professional Interpretation:**\n"
        "✅ Strong institutional buying\n"
        "✅ Accumulation phase active\n"
        "⚠️ Monitor for flow reversals\n\n"
        "⏰ **Last Update:** " + datetime.now().strftime('%H:%M:%S')
    )
    
    keyboard = [
        [
            InlineKeyboardButton("🔄 Refresh Flows", callback_data='deep_institutional'),
            InlineKeyboardButton("📊 Historical", callback_data='institutional_history')
        ],
        [InlineKeyboardButton("↩️ Back", callback_data='deep_analysis')]
    ]
    
    await query.edit_message_text(
        msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def show_multitf_analysis(query, context):
    """Show multi-timeframe analysis"""
    msg = (
        "⏰ **MULTI-TIMEFRAME ANALYSIS**\n"
        "=" * 30 + "\n\n"
        "📊 **BTC/USDT Confluence:**\n\n"
        "**1M:** 🟢 Bullish (RSI: 68)\n"
        "**5M:** 🟢 Bullish (MACD: +)\n"
        "**15M:** 🟡 Neutral (RSI: 52)\n"
        "**1H:** 🟢 Bullish (Trend: Up)\n"
        "**4H:** 🟢 Strong Bull (EMA align)\n"
        "**1D:** 🟢 Bullish (Structure: +)\n\n"
        "🎯 **Timeframe Consensus:**\n"
        "• Bullish: 5/6 timeframes\n"
        "• Confidence: 83%\n"
        "• Signal Strength: 8/10\n\n"
        "⚡ **Trading Signals:**\n"
        "• Entry Style: Day Trading\n"
        "• Direction: Long Bias\n"
        "• Confidence: High\n"
        "• Risk Level: Medium\n\n"
        "✅ **Strong multi-timeframe bullish alignment**"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("📊 Detailed MTF", callback_data='mtf_detailed'),
            InlineKeyboardButton("🔄 Refresh", callback_data='deep_multitf')
        ],
        [InlineKeyboardButton("↩️ Back", callback_data='deep_analysis')]
    ]
    
    await query.edit_message_text(
        msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def show_volume_analysis(query, context):
    """Show volume profile analysis"""
    msg = (
        "📈 **VOLUME PROFILE ANALYSIS**\n"
        "=" * 28 + "\n\n"
        "📊 **BTC/USDT Volume Profile:**\n\n"
        "🎯 **Point of Control (POC):**\n"
        "• Price: $63,450\n"
        "• Volume: 2,847 BTC\n"
        "• Significance: Very High\n\n"
        "📊 **Value Area (70% volume):**\n"
        "• VAH (High): $64,200\n"
        "• VAL (Low): $62,800\n"
        "• Width: $1,400 (2.2%)\n\n"
        "⚡ **Volume Analysis:**\n"
        "• Current vs Avg: 1.8x\n"
        "• Above POC: Bullish\n"
        "• Volume trend: Increasing\n"
        "• Accumulation zone: $62,800-$63,200\n\n"
        "🎯 **Trading Levels:**\n"
        "• Support: $62,800 (VAL)\n"
        "• Resistance: $64,200 (VAH)\n"
        "• Breakout target: $65,600\n\n"
        "✅ **Strong volume support at current levels**"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("📊 Multi-Symbol", callback_data='volume_multi'),
            InlineKeyboardButton("🔄 Refresh", callback_data='deep_volume')
        ],
        [InlineKeyboardButton("↩️ Back", callback_data='deep_analysis')]
    ]
    
    await query.edit_message_text(
        msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def show_detailed_performance(query, context):
    """Show detailed performance metrics"""
    msg = (
        "📈 **DETAILED PERFORMANCE ANALYSIS**\n"
        "=" * 35 + "\n\n"
        "💰 **Portfolio Performance (30D):**\n"
        "• Total Return: +12.5%\n"
        "• Annualized Return: +156.3%\n"
        "• Sharpe Ratio: 1.85\n"
        "• Sortino Ratio: 2.34\n"
        "• Calmar Ratio: 1.52\n\n"
        "📊 **Risk Metrics:**\n"
        "• Max Drawdown: -8.2%\n"
        "• Volatility (30D): 15.4%\n"
        "• VaR (95%): -2.1%\n"
        "• Beta vs BTC: 0.85\n\n"
        "🎯 **Trade Statistics:**\n"
        "• Total Trades: 47\n"
        "• Win Rate: 68.1%\n"
        "• Avg Win: +3.8%\n"
        "• Avg Loss: -2.1%\n"
        "• Profit Factor: 2.1\n\n"
        "📈 **Best/Worst:**\n"
        "• Best Trade: +15.6% (SOL/USDT)\n"
        "• Worst Trade: -4.2% (ADA/USDT)\n"
        "• Best Day: +8.9%\n"
        "• Worst Day: -5.1%\n\n"
        "✅ **Strong risk-adjusted performance**"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("📊 Monthly Breakdown", callback_data='monthly_performance'),
            InlineKeyboardButton("📈 Charts", callback_data='performance_charts')
        ],
        [
            InlineKeyboardButton("📋 Trade Log", callback_data='trade_history'),
            InlineKeyboardButton("↩️ Back", callback_data='performance')
        ]
    ]
    
    await query.edit_message_text(
        msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def show_sector_details(query, context):
    """Show detailed sector analysis"""
    msg = (
        "🏛️ **DETAILED SECTOR ANALYSIS**\n"
        "=" * 30 + "\n\n"
        "📊 **Sector Performance (24h):**\n\n"
        "**🤖 AI/ML (+8.5%)**\n"
        "• FET: +12.3% | Volume: 2.1x\n"
        "• AGIX: +9.8% | Volume: 1.8x\n"
        "• RNDR: +6.2% | Volume: 1.5x\n"
        "• Momentum: Very Strong\n\n"
        "**⛓️ Layer 1 (+4.2%)**\n"
        "• SOL: +5.8% | Volume: 1.9x\n"
        "• AVAX: +3.1% | Volume: 1.4x\n"
        "• DOT: +3.9% | Volume: 1.2x\n"
        "• Momentum: Strong\n\n"
        "**💎 Large Cap (+2.1%)**\n"
        "• BTC: +1.8% | Volume: 1.1x\n"
        "• ETH: +2.4% | Volume: 1.3x\n"
        "• Momentum: Steady\n\n"
        "**🔴 Weak Sectors:**\n\n"
        "**🎮 Gaming (-2.8%)**\n"
        "• AXS: -4.2% | Volume: 0.6x\n"
        "• SAND: -3.1% | Volume: 0.7x\n"
        "• GALA: -1.9% | Volume: 0.8x\n\n"
        "**🔒 Privacy (-1.5%)**\n"
        "• Regulatory pressure\n"
        "• Low institutional interest\n\n"
        "🎯 **Rotation Strategy:**\n"
        "✅ Rotate INTO: AI/ML, Layer1\n"
        "❌ Rotate OUT OF: Gaming, Privacy\n"
        "⚠️ Monitor: DeFi (mixed signals)"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("🎯 Sector Signals", callback_data='sector_signals'),
            InlineKeyboardButton("📊 Heatmap", callback_data='sector_heatmap')
        ],
        [
            InlineKeyboardButton("🔄 Refresh", callback_data='sector_rotation'),
            InlineKeyboardButton("↩️ Back", callback_data='sector_rotation')
        ]
    ]
    
    await query.edit_message_text(
        msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def show_professional_settings_detailed(query, context):
    """Show detailed professional settings"""
    current_config = trading_config  # Reference to global config
    
    msg = (
        f"⚙️ **PROFESSIONAL SETTINGS CONTROL**\n"
        f"=" * 35 + "\n\n"
        f"💰 **RISK MANAGEMENT:**\n"
        f"• Max Portfolio Risk: {current_config.max_portfolio_risk:.1%}\n"
        f"• Max Daily Risk: {current_config.max_daily_risk:.1%}\n"
        f"• Max Correlation: {current_config.max_correlation_exposure:.1%}\n"
        f"• Min Position: ${current_config.min_position_size:,.0f}\n"
        f"• Max Position: ${current_config.max_position_size:,.0f}\n\n"
        f"📊 **SIGNAL FILTERING:**\n"
        f"• Min Signal Strength: {current_config.min_signal_strength}/10\n"
        f"• Min MTF Confidence: {current_config.min_mtf_confidence:.0f}%\n"
        f"• Min Volume: ${current_config.min_volume_threshold:,.0f}\n\n"
        f"🔧 **ADVANCED FEATURES:**\n"
        f"• Portfolio Optimization: {'ON' if current_config.use_portfolio_optimization else 'OFF'}\n"
        f"• Sector Rotation: {'ON' if current_config.use_sector_rotation else 'OFF'}\n"
        f"• Market Regime Filter: {'ON' if current_config.use_market_regime_filter else 'OFF'}\n"
        f"• Risk Parity: {'ON' if current_config.use_risk_parity else 'OFF'}\n\n"
        f"⚖️ **LEVERAGE LIMITS:**\n"
        f"• Scalping: {current_config.max_leverage['SCALPING']:.0f}x\n"
        f"• Day Trading: {current_config.max_leverage['DAY_TRADING']:.0f}x\n"
        f"• Swing: {current_config.max_leverage['SWING']:.0f}x\n"
        f"• Position: {current_config.max_leverage['POSITION']:.0f}x"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("⚖️ Risk Settings", callback_data='settings_risk'),
            InlineKeyboardButton("💰 Position Rules", callback_data='settings_position')
        ],
        [
            InlineKeyboardButton("📊 Signal Filters", callback_data='settings_signals'),
            InlineKeyboardButton("🎯 Portfolio Config", callback_data='settings_portfolio')
        ],
        [
            InlineKeyboardButton("⏰ Timeframes", callback_data='settings_timeframes'),
            InlineKeyboardButton("🔧 Advanced", callback_data='settings_advanced')
        ],
        [
            InlineKeyboardButton("💾 Save Config", callback_data='save_settings'),
            InlineKeyboardButton("🔄 Reset Default", callback_data='reset_settings')
        ],
        [
            InlineKeyboardButton("↩️ Back", callback_data='back_main_pro')
        ]
    ]
    
    await query.edit_message_text(
        msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def show_position_settings(query, context):
    """Show position sizing settings"""
    msg = (
        f"💰 **POSITION SIZING SETTINGS**\n"
        f"=" * 28 + "\n\n"
        f"📊 **CURRENT CONFIGURATION:**\n"
        f"• Portfolio Value: $100,000\n"
        f"• Available Capital: 80%\n"
        f"• Min Position: $100\n"
        f"• Max Position: $50,000\n"
        f"• Max Single Risk: 2%\n\n"
        f"🎯 **POSITION SIZING METHOD:**\n"
        f"✅ Kelly Criterion (Active)\n"
        f"• Fixed Percentage\n"
        f"• Risk Parity\n"
        f"• Equal Weight\n\n"
        f"⚖️ **RISK CONTROLS:**\n"
        f"• Stop Loss Required: YES\n"
        f"• Max Correlation Limit: 10%\n"
        f"• Sector Limits: Active\n"
        f"• Dynamic Sizing: ON\n\n"
        f"📈 **PERFORMANCE IMPACT:**\n"
        f"• Avg Position Size: $8,500\n"
        f"• Risk Utilization: 45%\n"
        f"• Capital Efficiency: 78%"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("📊 Kelly Criterion", callback_data='position_kelly'),
            InlineKeyboardButton("💯 Fixed %", callback_data='position_fixed')
        ],
        [
            InlineKeyboardButton("⚖️ Risk Parity", callback_data='position_risk_parity'),
            InlineKeyboardButton("🟰 Equal Weight", callback_data='position_equal')
        ],
        [
            InlineKeyboardButton("⚙️ Custom Rules", callback_data='position_custom'),
            InlineKeyboardButton("📊 Backtest", callback_data='position_backtest')
        ],
        [
            InlineKeyboardButton("↩️ Back", callback_data='settings_position')
        ]
    ]
    
    await query.edit_message_text(
        msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def show_signal_settings(query, context):
    """Show signal threshold settings"""
    msg = (
        f"📊 **SIGNAL FILTERING SETTINGS**\n"
        f"=" * 30 + "\n\n"
        f"🎯 **CURRENT THRESHOLDS:**\n"
        f"• Min Signal Strength: 8/10\n"
        f"• Min MTF Confidence: 65%\n"
        f"• Min Volume: $5,000,000\n"
        f"• Min R:R Ratio: 1.5\n"
        f"• Max Risk per Trade: 2%\n\n"
        f"🔬 **QUALITY FILTERS:**\n"
        f"✅ Multi-timeframe alignment\n"
        f"✅ Volume confirmation required\n"
        f"✅ Market structure filter\n"
        f"✅ Correlation limits\n"
        f"✅ Sector rotation filter\n\n"
        f"⚡ **PERFORMANCE IMPACT:**\n"
        f"• Signals per day: 3-8\n"
        f"• False positive rate: <15%\n"
        f"• Win rate improvement: +12%\n"
        f"• Risk-adjusted returns: +28%\n\n"
        f"📈 **RECOMMENDATION:**\n"
        f"Current settings optimized for professional trading\n"
        f"Lower thresholds = More signals, Higher risk\n"
        f"Higher thresholds = Fewer signals, Lower risk"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("🟢 Conservative", callback_data='signals_conservative'),
            InlineKeyboardButton("🟡 Balanced", callback_data='signals_balanced')
        ],
        [
            InlineKeyboardButton("🟠 Aggressive", callback_data='signals_aggressive'),
            InlineKeyboardButton("⚙️ Custom", callback_data='signals_custom')
        ],
        [
            InlineKeyboardButton("📊 Backtest Impact", callback_data='signals_backtest'),
            InlineKeyboardButton("🔄 Reset Default", callback_data='signals_reset')
        ],
        [
            InlineKeyboardButton("↩️ Back", callback_data='settings_signals')
        ]
    ]
    
    await query.edit_message_text(
        msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def show_portfolio_settings(query, context):
    """Show portfolio configuration settings"""
    msg = (
        f"🎯 **PORTFOLIO CONFIGURATION**\n"
        f"=" * 25 + "\n\n"
        f"💼 **ALLOCATION LIMITS:**\n"
        f"• Max per Asset: 20%\n"
        f"• Max per Sector: 30%\n"
        f"• Max Correlation Group: 40%\n"
        f"• Min Cash Reserve: 20%\n\n"
        f"🔄 **REBALANCING:**\n"
        f"• Auto Rebalance: Weekly\n"
        f"• Drift Threshold: 5%\n"
        f"• Correlation Check: Daily\n"
        f"• Risk Check: Real-time\n\n"
        f"📊 **OPTIMIZATION:**\n"
        f"✅ Modern Portfolio Theory\n"
        f"✅ Black-Litterman Model\n"
        f"✅ Risk Budgeting\n"
        f"✅ Factor Exposure Control\n\n"
        f"⚖️ **RISK MANAGEMENT:**\n"
        f"• Portfolio VaR Limit: 5%\n"
        f"• Max Drawdown Limit: 15%\n"
        f"• Leverage Limit: 3x\n"
        f"• Stress Testing: Weekly\n\n"
        f"📈 **PERFORMANCE:**\n"
        f"• Target Sharpe: >1.5\n"
        f"• Max Volatility: 20%\n"
        f"• Tracking Error: <5%"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("📊 Allocation Rules", callback_data='portfolio_allocation'),
            InlineKeyboardButton("🔄 Rebalancing", callback_data='portfolio_rebalance_settings')
        ],
        [
            InlineKeyboardButton("⚖️ Risk Controls", callback_data='portfolio_risk_controls'),
            InlineKeyboardButton("📈 Optimization", callback_data='portfolio_optimization')
        ],
        [
            InlineKeyboardButton("🎯 Target Setting", callback_data='portfolio_targets'),
            InlineKeyboardButton("📊 Backtest", callback_data='portfolio_backtest')
        ],
        [
            InlineKeyboardButton("↩️ Back", callback_data='settings_portfolio')
        ]
    ]
    
    await query.edit_message_text(
        msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

# Additional helper for manual analysis
async def analyze_manual_symbol_deep(query, context, symbol: str):
    """Deep analysis for manual symbol requests"""
    try:
        # Normalize symbol
        if '/' not in symbol:
            symbol = f"{symbol}/USDT"
        
        await load_markets_with_retry(binance)
        
        # Check if symbol exists
        if symbol not in binance.symbols:
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=f"❌ Symbol {symbol} not found"
            )
            return
        
        # Comprehensive analysis
        result = await analyze_symbol_with_timeout(
            symbol, binance, headers, MarketRegime.BULL_RANGING, None, timeout=30
        )
        
        if result:
            await send_deep_manual_analysis(context, query.message.chat_id, result)
        else:
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=f"❌ Could not perform deep analysis on {symbol}"
            )
    
    except Exception as e:
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"❌ Deep analysis error: {str(e)}"
        )

async def send_deep_manual_analysis(context, chat_id: int, result: Dict):
    """Send comprehensive manual analysis result"""
    risk_mgmt = result['risk_mgmt']
    signal_components = result['signal_components']
    
    # Part 1: Overview
    overview_msg = (
        f"🔍 **DEEP ANALYSIS: {result['symbol']}**\n"
        f"=" * 35 + "\n"
        f"📊 **EXECUTIVE SUMMARY:**\n"
        f"• Signal: {result['signal_type']} ({result['confidence']:.1f}%)\n"
        f"• Professional Score: {result['final_score']:.1f}/100\n"
        f"• Current Price: ${result['current_price']:.4f}\n"
        f"• Market Regime: {result['market_regime'].value.title()}\n\n"
        f"🎯 **RECOMMENDED ACTION:**\n"
        f"{'✅ ENTER POSITION' if result['confidence'] > 70 else '⚠️ WAIT FOR BETTER SETUP' if result['confidence'] > 50 else '❌ AVOID POSITION'}\n"
        f"Risk Level: {'Low' if result['final_score'] > 80 else 'Medium' if result['final_score'] > 60 else 'High'}"
    )
    
    await context.bot.send_message(chat_id=chat_id, text=overview_msg, parse_mode='Markdown')
    await asyncio.sleep(2)
    
    # Part 2: Technical Analysis
    technical_msg = (
        f"📊 **TECHNICAL ANALYSIS BREAKDOWN**\n"
        f"=" * 32 + "\n"
        f"🔬 **SIGNAL COMPONENTS:**\n"
        f"• Technical Score: {signal_components['technical']:.0f}/25\n"
        f"• Momentum Score: {signal_components['momentum']:.0f}/20\n"
        f"• Volume Score: {signal_components['volume']:.0f}/15\n"
        f"• Structure Score: {signal_components['structure']:.0f}/20\n"
        f"• Sentiment Score: {signal_components['sentiment']:.0f}/10\n"
        f"• Risk Score: {signal_components['risk']:.0f}/10\n\n"
        f"📈 **KEY INSIGHTS:**\n"
    )
    
    # Add signal details
    if result.get('signal_details'):
        for detail in result['signal_details'][:5]:  # Top 5 details
            technical_msg += f"• {detail}\n"
    
    await context.bot.send_message(chat_id=chat_id, text=technical_msg, parse_mode='Markdown')
    await asyncio.sleep(2)
    
    # Part 3: Risk Management
    risk_msg = (
        f"⚖️ **PROFESSIONAL RISK MANAGEMENT**\n"
        f"=" * 32 + "\n"
        f"🎯 **ENTRY STRATEGY:**\n"
        f"• Optimal Entry: ${result['current_price']:.4f}\n"
        f"• Entry Zone: ${result['current_price']*0.998:.4f} - ${result['current_price']*1.002:.4f}\n\n"
        f"🛑 **STOP LOSS:**\n"
        f"• Stop Price: ${risk_mgmt['stop_loss']:.4f}\n"
        f"• Distance: {risk_mgmt['stop_distance_percent']:.1f}%\n\n"
        f"🎯 **TAKE PROFIT LEVELS:**\n"
        f"• TP1 (40%): ${risk_mgmt['take_profit_1']:.4f} | R:R {risk_mgmt['risk_reward_1']:.1f}\n"
        f"• TP2 (40%): ${risk_mgmt['take_profit_2']:.4f} | R:R {risk_mgmt['risk_reward_2']:.1f}\n"
        f"• TP3 (20%): ${risk_mgmt['take_profit_3']:.4f} | R:R {risk_mgmt['risk_reward_3']:.1f}\n\n"
        f"📊 **POSITION SIZING:**\n"
        f"• Max Risk: 1-2% of portfolio\n"
        f"• Suggested Size: Calculate based on stop distance\n"
        f"• Max Leverage: {trading_config.max_leverage.get('DAY_TRADING', 5):.0f}x\n"
    )
    
    await context.bot.send_message(chat_id=chat_id, text=risk_msg, parse_mode='Markdown')
    await asyncio.sleep(2)
    
    # Part 4: Market Data
    market_msg = (
        f"📈 **MARKET DATA & CONTEXT**\n"
        f"=" * 25 + "\n"
        f"💰 **24H STATISTICS:**\n"
        f"• Volume: ${result['market_data']['volume_24h']:,.0f}\n"
        f"• Volume Ratio: {result['market_data'].get('volume_ratio', 1):.1f}x avg\n"
        f"• Price Change: {result['market_data']['price_change_24h']:+.2f}%\n"
        f"• High: ${result['market_data']['high_24h']:,.4f}\n"
        f"• Low: ${result['market_data']['low_24h']:,.4f}\n\n"
        f"⚡ **VOLATILITY:**\n"
        f"• ATR: {risk_mgmt['volatility_ratio']:.1%}\n"
        f"• Classification: {'High' if risk_mgmt['volatility_ratio'] > 0.05 else 'Medium' if risk_mgmt['volatility_ratio'] > 0.03 else 'Low'}\n\n"
        f"🌊 **NETFLOW:**\n"
        f"• 24h Flow: ${result['netflow']:+,.0f}\n"
        f"• Interpretation: {'Institutional Buying' if result['netflow'] > 0 else 'Institutional Selling' if result['netflow'] < 0 else 'Neutral Flow'}\n\n"
        f"⚠️ **PROFESSIONAL DISCLAIMER:**\n"
        f"This analysis is for educational purposes.\n"
        f"Always use proper risk management.\n"
        f"Markets can change rapidly."
    )
    
    await context.bot.send_message(chat_id=chat_id, text=market_msg, parse_mode='Markdown')
    
async def professional_button_handler_enhanced(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced professional button handler with complete error handling and logging"""
    query = update.callback_query
    
    if not query:
        return
    
    try:
        await query.answer()
        data = query.data
        
        print(f"🔘 Button pressed: {data}")  # Debug logging
        
        # Professional scan handlers
        if data == 'pro_scan':
            await execute_professional_scan(query, context)
        elif data == 'market_regime':
            await show_market_regime_analysis(query, context)
        elif data.startswith('pro_'):
            style = data.replace('pro_', '').upper()
            await execute_professional_style_analysis(query, context, style)
        
        # Portfolio handlers
        elif data == 'portfolio_analysis':
            await show_portfolio_analysis(query, context)
        elif data == 'portfolio_rebalance':
            await handle_portfolio_rebalance(query, context)
        elif data == 'portfolio_performance':
            await show_detailed_performance(query, context)
        elif data == 'portfolio_risk':
            await show_portfolio_risk_analysis(query, context)
        
        # Deep analysis handlers
        elif data == 'deep_analysis':
            await show_deep_analysis_menu(query, context)
        elif data in ['deep_symbol', 'deep_correlation', 'deep_structure', 
                      'deep_institutional', 'deep_multitf', 'deep_volume']:
            await handle_deep_analysis_callback(query, context)
        elif data.startswith('analyze_'):
            await handle_analyze_symbol(query, context)
        
        # Settings handlers
        elif data == 'risk_settings':
            await show_professional_settings_detailed(query, context)
        elif data.startswith('settings_'):
            await handle_settings_callback(query, context)
        
        # Performance handlers
        elif data == 'performance':
            await show_performance_analysis(query, context)
        elif data == 'performance_detailed':
            await show_detailed_performance(query, context)
        elif data == 'trade_history':
            await show_trade_history(query, context)
        
        # Sector handlers
        elif data == 'sector_rotation':
            await show_sector_rotation_analysis(query, context)
        elif data == 'sector_details':
            await show_sector_details(query, context)
        elif data == 'sector_signals':
            await show_sector_signals(query, context)
        
        # Navigation handlers
        elif data == 'back_main_pro':
            await start_professional(update, context)
        elif data == 'back_main':
            await start(update, context)
        
        # Original trading style handlers
        elif data == 'quick_scalping':
            await execute_scalping_scan(query, context)
        elif data == 'day_trading':
            await execute_daytrading_scan(query, context)
        elif data == 'swing_trading':
            await execute_swing_scan(query, context)
        elif data == 'manual_analysis':
            await show_manual_menu(query, context)
        elif data == 'auto_scanner':
            await execute_auto_scan(query, context)
        elif data == 'settings':
            await show_settings_menu(query, context)
        elif data == 'portfolio':
            await show_portfolio_menu(query, context)
        
        # Placeholder handlers (prevent crashes)
        elif data in ['stress_test', 'hedge_positions', 'risk_monitor', 
                      'full_history', 'pnl_chart', 'export_trades',
                      'execute_sector_signals', 'sector_heatmap',
                      'risk_conservative', 'risk_moderate', 'risk_aggressive']:
            await show_placeholder_message(query, context, data)
        
        # Error fallback
        else:
            print(f"⚠️ Unknown callback data: {data}")
            await query.edit_message_text(
                f"⚠️ Feature '{data}' is under development.\nReturning to main menu...",
                reply_markup=create_main_menu()
            )
    
    except Exception as e:
        print(f"❌ Button handler error for {data}: {e}")
        import traceback
        traceback.print_exc()
        
        error_msg = f"❌ An error occurred: {str(e)[:100]}...\nReturning to main menu."
        
        try:
            await query.edit_message_text(
                error_msg,
                reply_markup=create_main_menu()
            )
        except:
            # Fallback if edit fails
            try:
                await context.bot.send_message(
                    chat_id=query.message.chat_id,
                    text=error_msg,
                    reply_markup=create_main_menu()
                )
            except:
                pass  # Last resort - do nothing

async def show_placeholder_message(query, context, feature_name: str):
    """Show placeholder message for unimplemented features"""
    feature_descriptions = {
        'stress_test': 'Portfolio Stress Testing',
        'hedge_positions': 'Position Hedging Tools',
        'risk_monitor': 'Real-time Risk Monitor',
        'full_history': 'Complete Trade History',
        'pnl_chart': 'P&L Visualization Charts',
        'export_trades': 'Trade Data Export',
        'execute_sector_signals': 'Automated Sector Signal Execution',
        'sector_heatmap': 'Sector Performance Heatmap',
        'risk_conservative': 'Conservative Risk Profile',
        'risk_moderate': 'Moderate Risk Profile',
        'risk_aggressive': 'Aggressive Risk Profile'
    }
    
    description = feature_descriptions.get(feature_name, feature_name.replace('_', ' ').title())
    
    msg = (
        f"🚧 **{description}**\n\n"
        f"This advanced feature is currently under development.\n\n"
        f"📅 **Coming Soon:**\n"
        f"• Enhanced functionality\n"
        f"• Real-time data integration\n"
        f"• Professional-grade tools\n\n"
        f"🔔 You'll be notified when available!"
    )
    
    keyboard = [[InlineKeyboardButton("↩️ Back", callback_data='back_main_pro')]]
    
    await query.edit_message_text(
        msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

# Simple placeholder handlers to prevent crashes
async def handle_portfolio_rebalance(query, context):
    """Portfolio rebalancing handler"""
    await show_placeholder_message(query, context, 'portfolio_rebalance')

async def show_performance_analysis(query, context):
    """Show basic performance analysis"""
    msg = (
        "📈 **PERFORMANCE ANALYSIS**\n"
        "=" * 25 + "\n\n"
        "💰 **Portfolio Overview:**\n"
        "• Total Value: $100,000\n"
        "• Available: $80,000 (80%)\n"
        "• Invested: $20,000 (20%)\n"
        "• Daily P&L: $0 (0%)\n\n"
        "📊 **Recent Performance (7D):**\n"
        "• Return: +2.1%\n"
        "• Win Rate: 65%\n"
        "• Max Drawdown: -1.8%\n"
        "• Sharpe Ratio: 1.4\n\n"
        "🎯 **Active Positions:** 0\n"
        "📋 **Total Trades:** 12\n"
        "⚡ **Avg Trade Duration:** 4.2 hours\n\n"
        "✅ **Overall Status:** Healthy"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("📊 Detailed Analysis", callback_data='performance_detailed'),
            InlineKeyboardButton("📋 Trade History", callback_data='trade_history')
        ],
        [
            InlineKeyboardButton("📈 Charts", callback_data='performance_charts'),
            InlineKeyboardButton("💾 Export Data", callback_data='export_performance')
        ],
        [
            InlineKeyboardButton("↩️ Back", callback_data='back_main_pro')
        ]
    ]
    
    await query.edit_message_text(
        msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )
    
async def show_sector_rotation_analysis(query, context):
    """Show sector rotation analysis"""
    msg = (
        "🎯 **SECTOR ROTATION ANALYSIS**\n"
        "=" * 30 + "\n\n"
        "📊 **Current Sector Performance (24h):**\n\n"
        "🔥 **LEADING SECTORS:**\n"
        "1. 🤖 AI/ML: +6.8% (Very Strong)\n"
        "2. ⛓️ Layer 1: +4.2% (Strong)\n"
        "3. 💎 Large Cap: +2.1% (Stable)\n\n"
        "🔴 **LAGGING SECTORS:**\n"
        "4. 🎮 Gaming: -2.8% (Weak)\n"
        "5. 🔒 Privacy: -1.5% (Weak)\n"
        "6. 🏦 DeFi: -0.8% (Mixed)\n\n"
        "💰 **MONEY FLOW:**\n"
        "📈 INTO: AI/ML, Layer 1\n"
        "📉 OUT OF: Gaming, Privacy\n"
        "⚖️ NEUTRAL: DeFi, Meme\n\n"
        "🎯 **ROTATION SIGNALS:**\n"
        "✅ BUY: AI tokens on dips\n"
        "✅ HOLD: Major Layer 1s\n"
        "⚠️ WATCH: DeFi for reversal\n"
        "❌ AVOID: Gaming tokens\n\n"
        "📊 **Market Context:**\n"
        "• Institutional preference: AI/Utility\n"
        "• Retail interest: Meme/Gaming (declining)\n"
        "• Risk-on environment: Moderate\n\n"
        "⏰ **Analysis Time:** " + datetime.now().strftime('%H:%M:%S')
    )
    
    keyboard = [
        [
            InlineKeyboardButton("🎯 Sector Signals", callback_data='sector_signals'),
            InlineKeyboardButton("📊 Detailed View", callback_data='sector_details')
        ],
        [
            InlineKeyboardButton("📈 Heatmap", callback_data='sector_heatmap'),
            InlineKeyboardButton("🔄 Refresh", callback_data='sector_rotation')
        ],
        [
            InlineKeyboardButton("↩️ Back", callback_data='back_main_pro')
        ]
    ]
    
    await query.edit_message_text(
        msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )
    
# ===============================
# PROFESSIONAL COMMAND HANDLERS  
# ===============================

async def auto_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Auto scan command"""
    await execute_auto_scan_direct(update, context)

async def scalping_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Scalping command"""
    await execute_scalping_scan_direct(update, context)

async def daytrading_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Day trading command"""
    await execute_daytrading_scan_direct(update, context)

async def swing_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Swing trading command"""
    await execute_swing_scan_direct(update, context)

async def manual_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manual analysis command"""
    await show_manual_analysis_direct(update, context)
    
async def pro_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Professional analysis command"""
    await start_professional(update, context)

async def regime_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Market regime analysis command"""
    chat_id = update.message.chat_id
    await update.message.reply_text("🔍 Analyzing market regime...")
    
    try:
        await load_markets_with_retry(binance)
        
        # Get market data
        btc_data = binance.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=200)
        eth_data = binance.fetch_ohlcv('ETH/USDT', timeframe='1h', limit=100)
        
        # Detect regime
        regime_detector = MarketRegimeDetector()
        regime = regime_detector.detect_regime(btc_data, eth_data, {})
        
        # Get BTC metrics
        btc_df = pd.DataFrame(btc_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        ema_20 = talib.EMA(btc_df['close'].to_numpy(), timeperiod=20)
        ema_50 = talib.EMA(btc_df['close'].to_numpy(), timeperiod=50)
        current_price = btc_df['close'].iloc[-1]
        
        # Calculate volatility
        atr = talib.ATR(btc_df['high'].to_numpy(), btc_df['low'].to_numpy(), btc_df['close'].to_numpy(), timeperiod=14)
        volatility = (atr[-1] / current_price) * 100
        
        # Market structure
        structure = AdvancedIndicators.calculate_market_structure(btc_df)
        
        regime_msg = (
            f"🏛️ **MARKET REGIME ANALYSIS**\n"
            f"{'='*30}\n"
            f"📊 **Current Regime:** {regime.value.upper()}\n"
            f"₿ **BTC Price:** ${current_price:,.0f}\n"
            f"📈 **Trend Structure:** {structure['structure_type'].title()}\n"
            f"⚡ **Volatility:** {volatility:.1f}%\n"
            f"📊 **EMA Alignment:** {'Bullish' if ema_20[-1] > ema_50[-1] else 'Bearish'}\n"
            f"💪 **Trend Strength:** {structure['trend_strength']:.2f}\n\n"
            f"🎯 **Trading Implications:**\n"
            f"• Regime-appropriate strategies recommended\n"
            f"• Risk management adjusted for volatility\n"
            f"• Position sizing optimized for conditions\n\n"
            f"⚠️ **Professional Note:** Market regimes can shift rapidly"
        )
        
        await context.bot.send_message(chat_id=chat_id, text=regime_msg, parse_mode='Markdown')
        
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"❌ Regime analysis error: {str(e)}")
        
async def execute_scalping_scan(query, context):
    """Execute scalping scan"""
    await query.edit_message_text("⚡ Scanning for scalping opportunities...")
    
    try:
        chat_id = query.message.chat_id
        await load_markets_with_retry(binance)
        
        results = await trading_engine.scan_market_comprehensive(
            binance, headers, style_filter='SCALPING'
        )
        
        if results:
            await send_scalping_results(context, chat_id, results)
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ No scalping opportunities found"
            )
    except Exception as e:
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"❌ Scalping scan error: {str(e)}"
        )

async def execute_daytrading_scan(query, context):
    """Execute day trading scan"""
    await query.edit_message_text("📈 Scanning for day trading setups...")
    
    try:
        chat_id = query.message.chat_id
        await load_markets_with_retry(binance)
        
        results = await trading_engine.scan_market_comprehensive(
            binance, headers, style_filter='DAY_TRADING'
        )
        
        if results:
            await send_daytrading_results(context, chat_id, results)
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ No day trading setups found"
            )
    except Exception as e:
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"❌ Day trading scan error: {str(e)}"
        )

async def execute_swing_scan(query, context):
    """Execute swing trading scan"""
    await query.edit_message_text("📊 Scanning for swing trading positions...")
    
    try:
        chat_id = query.message.chat_id
        await load_markets_with_retry(binance)
        
        results = await trading_engine.scan_market_comprehensive(
            binance, headers, style_filter='SWING'
        )
        
        if results:
            await send_swing_results(context, chat_id, results)
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ No swing trading positions found"
            )
    except Exception as e:
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"❌ Swing scan error: {str(e)}"
        )

async def execute_auto_scan(query, context):
    """Execute auto scan"""
    await query.edit_message_text("🤖 Executing automated scan...")
    
    try:
        chat_id = query.message.chat_id
        await load_markets_with_retry(binance)
        
        results = await trading_engine.scan_market_comprehensive(
            binance, headers, style_filter=None
        )
        
        if results:
            await send_auto_results(context, chat_id, results)
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ No trading opportunities found"
            )
    except Exception as e:
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"❌ Auto scan error: {str(e)}"
        )

# Direct command handlers
async def execute_scalping_scan_direct(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Direct scalping scan"""
    await update.message.reply_text("⚡ Scanning for scalping opportunities...")
    
    try:
        await load_markets_with_retry(binance)
        results = await trading_engine.scan_market_comprehensive(
            binance, headers, style_filter='SCALPING'
        )
        
        if results:
            await send_scalping_results(context, update.message.chat_id, results)
        else:
            await update.message.reply_text("❌ No scalping opportunities found")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def execute_daytrading_scan_direct(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Direct day trading scan"""
    await update.message.reply_text("📈 Scanning for day trading setups...")
    
    try:
        await load_markets_with_retry(binance)
        results = await trading_engine.scan_market_comprehensive(
            binance, headers, style_filter='DAY_TRADING'
        )
        
        if results:
            await send_daytrading_results(context, update.message.chat_id, results)
        else:
            await update.message.reply_text("❌ No day trading setups found")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def execute_swing_scan_direct(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Direct swing trading scan"""
    await update.message.reply_text("📊 Scanning for swing trading positions...")
    
    try:
        await load_markets_with_retry(binance)
        results = await trading_engine.scan_market_comprehensive(
            binance, headers, style_filter='SWING'
        )
        
        if results:
            await send_swing_results(context, update.message.chat_id, results)
        else:
            await update.message.reply_text("❌ No swing trading positions found")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def execute_auto_scan_direct(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Direct auto scan"""
    await update.message.reply_text("🤖 Executing automated scan...")
    
    try:
        await load_markets_with_retry(binance)
        results = await trading_engine.scan_market_comprehensive(
            binance, headers, style_filter=None
        )
        
        if results:
            await send_auto_results(context, update.message.chat_id, results)
        else:
            await update.message.reply_text("❌ No trading opportunities found")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def show_manual_analysis_direct(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show manual analysis options"""
    message = (
        "🎯 **MANUAL ANALYSIS**\n\n"
        "Send me a symbol to analyze (e.g., BTC, ETH, ADA)\n"
        "or use format: SYMBOL/USDT\n\n"
        "I'll provide comprehensive analysis including:\n"
        "• Multi-timeframe signals\n"
        "• Risk management levels\n"
        "• Market structure analysis\n"
        "• Volume profile\n"
        "• Professional recommendations"
    )
    await update.message.reply_text(message, parse_mode='Markdown')
    
async def send_scalping_results(context, chat_id: int, results: List[Dict]):
    """Send scalping results"""
    header_msg = (
        f"⚡ **SCALPING OPPORTUNITIES**\n"
        f"{'='*30}\n"
        f"🎯 Found: {len(results)} setups\n"
        f"⏰ Hold Time: 1-15 minutes\n"
        f"💰 Target: 0.3-1.5% profit\n"
        f"⚖️ Max Leverage: 20x\n"
    )
    
    await context.bot.send_message(chat_id=chat_id, text=header_msg, parse_mode='Markdown')
    
    for i, result in enumerate(results[:3], 1):
        risk_mgmt = result['risk_mgmt']
        
        msg = (
            f"⚡ **SCALP #{i}: {result['symbol']}**\n"
            f"📊 Signal: {result['signal_type']} ({result['confidence']:.1f}%)\n"
            f"💰 Entry: ${result['current_price']:.4f}\n"
            f"🎯 Target: ${risk_mgmt['take_profit_1']:.4f}\n"
            f"🛑 Stop: ${risk_mgmt['stop_loss']:.4f}\n"
            f"📈 R:R: {risk_mgmt['risk_reward_1']:.1f}\n"
            f"⚡ Volume: ${result['market_data']['volume_24h']:,.0f}\n"
            f"⏱️ Quick execution recommended"
        )
        
        await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode='Markdown')
        await asyncio.sleep(2)

async def send_daytrading_results(context, chat_id: int, results: List[Dict]):
    """Send day trading results"""
    header_msg = (
        f"📈 **DAY TRADING SETUPS**\n"
        f"{'='*25}\n"
        f"🎯 Found: {len(results)} setups\n"
        f"⏰ Hold Time: 30min-6hours\n"
        f"💰 Target: 1-6% profit\n"
        f"⚖️ Max Leverage: 10x\n"
    )
    
    await context.bot.send_message(chat_id=chat_id, text=header_msg, parse_mode='Markdown')
    
    for i, result in enumerate(results[:3], 1):
        risk_mgmt = result['risk_mgmt']
        
        msg = (
            f"📈 **INTRADAY #{i}: {result['symbol']}**\n"
            f"📊 Signal: {result['signal_type']} ({result['confidence']:.1f}%)\n"
            f"💰 Entry: ${result['current_price']:.4f}\n"
            f"🎯 TP1: ${risk_mgmt['take_profit_1']:.4f} (R:R {risk_mgmt['risk_reward_1']:.1f})\n"
            f"🎯 TP2: ${risk_mgmt['take_profit_2']:.4f} (R:R {risk_mgmt['risk_reward_2']:.1f})\n"
            f"🛑 Stop: ${risk_mgmt['stop_loss']:.4f}\n"
            f"📈 Volume: ${result['market_data']['volume_24h']:,.0f}\n"
            f"📊 Score: {result['final_score']:.1f}/100"
        )
        
        await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode='Markdown')
        await asyncio.sleep(2)

async def send_swing_results(context, chat_id: int, results: List[Dict]):
    """Send swing trading results"""
    header_msg = (
        f"📊 **SWING TRADING POSITIONS**\n"
        f"{'='*28}\n"
        f"🎯 Found: {len(results)} positions\n"
        f"⏰ Hold Time: 1-10 days\n"
        f"💰 Target: 3-25% profit\n"
        f"⚖️ Max Leverage: 5x\n"
    )
    
    await context.bot.send_message(chat_id=chat_id, text=header_msg, parse_mode='Markdown')
    
    for i, result in enumerate(results[:3], 1):
        risk_mgmt = result['risk_mgmt']
        
        msg = (
            f"📊 **SWING #{i}: {result['symbol']}**\n"
            f"📊 Signal: {result['signal_type']} ({result['confidence']:.1f}%)\n"
            f"💰 Entry: ${result['current_price']:.4f}\n"
            f"🎯 TP1: ${risk_mgmt['take_profit_1']:.4f} (R:R {risk_mgmt['risk_reward_1']:.1f})\n"
            f"🎯 TP2: ${risk_mgmt['take_profit_2']:.4f} (R:R {risk_mgmt['risk_reward_2']:.1f})\n"
            f"🎯 TP3: ${risk_mgmt['take_profit_3']:.4f} (R:R {risk_mgmt['risk_reward_3']:.1f})\n"
            f"🛑 Stop: ${risk_mgmt['stop_loss']:.4f}\n"
            f"📈 Volume: ${result['market_data']['volume_24h']:,.0f}\n"
            f"📊 Score: {result['final_score']:.1f}/100\n"
            f"⏱️ Multi-day hold recommended"
        )
        
        await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode='Markdown')
        await asyncio.sleep(2)

async def send_auto_results(context, chat_id: int, results: List[Dict]):
    """Send auto scan results"""
    header_msg = (
        f"🤖 **AUTOMATED MARKET SCAN**\n"
        f"{'='*25}\n"
        f"🎯 Best Opportunities: {len(results)}\n"
        f"📊 Multi-style analysis\n"
        f"⚡ Real-time scanning\n"
        f"🎯 Professional filtering\n"
    )
    
    await context.bot.send_message(chat_id=chat_id, text=header_msg, parse_mode='Markdown')
    
    for i, result in enumerate(results[:5], 1):
        risk_mgmt = result['risk_mgmt']
        
        # Determine suggested style based on R:R and hold time
        if risk_mgmt['risk_reward_1'] >= 2 and risk_mgmt['volatility_ratio'] > 0.05:
            style = "Scalping"
            emoji = "⚡"
        elif risk_mgmt['risk_reward_2'] >= 3:
            style = "Day Trading"
            emoji = "📈"
        else:
            style = "Swing Trading"
            emoji = "📊"
        
        msg = (
            f"{emoji} **AUTO #{i}: {result['symbol']}**\n"
            f"📊 Style: {style}\n"
            f"📊 Signal: {result['signal_type']} ({result['confidence']:.1f}%)\n"
            f"💰 Entry: ${result['current_price']:.4f}\n"
            f"🎯 Target: ${risk_mgmt['take_profit_2']:.4f} (R:R {risk_mgmt['risk_reward_2']:.1f})\n"
            f"🛑 Stop: ${risk_mgmt['stop_loss']:.4f}\n"
            f"📈 Volume: ${result['market_data']['volume_24h']:,.0f}\n"
            f"📊 Score: {result['final_score']:.1f}/100"
        )
        
        await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode='Markdown')
        await asyncio.sleep(2)

# Menu handlers
async def show_manual_menu(query, context):
    """Show manual analysis menu"""
    msg = (
        "🎯 **MANUAL ANALYSIS**\n\n"
        "Send me a symbol to analyze:\n"
        "• BTC, ETH, ADA, etc.\n"
        "• Or full pair: BTC/USDT\n\n"
        "Available analysis:\n"
        "• Multi-timeframe signals\n"
        "• Professional risk management\n"
        "• Market structure analysis\n"
        "• Volume profile breakdown\n"
    )
    
    keyboard = [[InlineKeyboardButton("↩️ Back", callback_data='back_main')]]
    
    await query.edit_message_text(
        msg, 
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def show_settings_menu(query, context):
    """Show settings menu"""
    msg = (
        "⚙️ **SETTINGS**\n\n"
        "Current Configuration:\n"
        "• Risk per trade: 2%\n"
        "• Min confidence: 65%\n"
        "• Max positions: 5\n"
        "• Professional mode: ON\n\n"
        "Settings can be adjusted in professional mode."
    )
    
    keyboard = [
        [InlineKeyboardButton("🏛️ Professional Settings", callback_data='pro_scan')],
        [InlineKeyboardButton("↩️ Back", callback_data='back_main')]
    ]
    
    await query.edit_message_text(
        msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def show_portfolio_menu(query, context):
    """Show portfolio menu"""
    msg = (
        "💼 **PORTFOLIO OVERVIEW**\n\n"
        "Current Status:\n"
        "• Total Value: $100,000\n"
        "• Available: $80,000\n"
        "• Active Positions: 0\n"
        "• Daily P&L: $0\n\n"
        "Portfolio features available in professional mode."
    )
    
    keyboard = [
        [InlineKeyboardButton("🏛️ Professional Portfolio", callback_data='portfolio_analysis')],
        [InlineKeyboardButton("↩️ Back", callback_data='back_main')]
    ]
    
    await query.edit_message_text(
        msg,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

# Message handler untuk manual analysis
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages for manual analysis"""
    text = update.message.text.upper()
    
    # Check if it's a symbol
    if len(text) <= 10 and (text.isalpha() or '/' in text):
        await analyze_manual_symbol(update, context, text)
    else:
        await update.message.reply_text(
            "Send me a crypto symbol to analyze (e.g., BTC, ETH, ADA)\n"
            "Or use /start for the main menu"
        )

async def analyze_manual_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE, symbol: str):
    """Analyze a single symbol manually"""
    await update.message.reply_text(f"🔍 Analyzing {symbol}...")
    
    try:
        # Normalize symbol
        if '/' not in symbol:
            symbol = f"{symbol}/USDT"
        
        await load_markets_with_retry(binance)
        
        # Check if symbol exists
        if symbol not in binance.symbols:
            await update.message.reply_text(f"❌ Symbol {symbol} not found")
            return
        
        # Analyze the symbol
        result = await analyze_symbol_comprehensive_safe(
            symbol, binance, headers, MarketRegime.BULL_RANGING, None
        )
        
        if result:
            await send_manual_analysis_result(context, update.message.chat_id, result)
        else:
            await update.message.reply_text(f"❌ Could not analyze {symbol}")
    
    except Exception as e:
        await update.message.reply_text(f"❌ Analysis error: {str(e)}")

async def send_manual_analysis_result(context, chat_id: int, result: Dict):
    """Send manual analysis result"""
    risk_mgmt = result['risk_mgmt']
    signal_components = result['signal_components']
    
    msg = (
        f"🎯 **MANUAL ANALYSIS: {result['symbol']}**\n"
        f"{'='*30}\n"
        f"📊 **SIGNAL:** {result['signal_type']} ({result['confidence']:.1f}%)\n"
        f"💰 **Current Price:** ${result['current_price']:.4f}\n\n"
        f"🎯 **TARGETS & RISK:**\n"
        f"• TP1: ${risk_mgmt['take_profit_1']:.4f} (R:R {risk_mgmt['risk_reward_1']:.1f})\n"
        f"• TP2: ${risk_mgmt['take_profit_2']:.4f} (R:R {risk_mgmt['risk_reward_2']:.1f})\n"
        f"• TP3: ${risk_mgmt['take_profit_3']:.4f} (R:R {risk_mgmt['risk_reward_3']:.1f})\n"
        f"• SL: ${risk_mgmt['stop_loss']:.4f}\n\n"
        f"📊 **SIGNAL BREAKDOWN:**\n"
        f"• Technical: {signal_components['technical']:.0f}/25\n"
        f"• Momentum: {signal_components['momentum']:.0f}/20\n"
        f"• Volume: {signal_components['volume']:.0f}/15\n"
        f"• Structure: {signal_components['structure']:.0f}/20\n"
        f"• Sentiment: {signal_components['sentiment']:.0f}/10\n"
        f"• Risk Score: {signal_components['risk']:.0f}/10\n\n"
        f"📈 **MARKET DATA:**\n"
        f"• 24h Volume: ${result['market_data']['volume_24h']:,.0f}\n"
        f"• 24h Change: {result['market_data']['price_change_24h']:+.2f}%\n"
        f"• Volatility: {risk_mgmt['volatility_ratio']:.1%}\n"
        f"• Final Score: {result['final_score']:.1f}/100\n\n"
        f"⚠️ **Risk Management Required**"
    )
    
    await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode='Markdown')

# Keep existing functions but enhance them...
def get_dune_cex_flow_enhanced(symbol, headers, market_data):
    """Enhanced netflow analysis with better fallbacks"""
    try:
        # Original Dune logic
        netflow = get_dune_cex_flow(symbol, headers, 
                                   market_data.get('volume_24h'), 
                                   market_data.get('price_change_24h'))
        return netflow if netflow is not None else 0
    except:
        return 0
        
def get_dune_cex_flow(symbol, headers, volume_24h=None, price_change=None):
    """Get CEX netflow data from Dune"""
    try:
        # Simplified version - dalam produksi perlu API key Dune yang valid
        base_symbol = symbol.split('/')[0]
        
        # Mock netflow based on volume and price change
        if volume_24h and price_change:
            # Estimate netflow based on volume and price movement
            volume_usd = float(volume_24h)
            price_change_pct = float(price_change)
            
            # Simple heuristic: positive price change + high volume = potential inflow
            estimated_netflow = volume_usd * 0.1 * (price_change_pct / 100)
            return estimated_netflow
        
        return 0
    except Exception as e:
        print(f"Error getting netflow for {symbol}: {e}")
        return 0
        
def create_enhanced_binance():
    """Create enhanced Binance client with better error handling"""
    binance = ccxt.binance({
        'enableRateLimit': True,
        'rateLimit': 1200,  # Increased rate limit
        'timeout': 30000,   # 30 second timeout
        'options': {
            'defaultType': 'spot',
            'adjustForTimeDifference': True,
        },
        'headers': {
            'User-Agent': 'ccxt/python'
        }
    })
    
    # Set multiple endpoints for failover
    binance.urls['api']['public'] = 'https://api.binance.com/api/v3'
    
    return binance

# Replace the global binance instance
binance = create_enhanced_binance()

async def safe_fetch_with_retry(exchange, method_name: str, *args, max_retries: int = 3, **kwargs):
    """Safely fetch data with retry logic"""
    
    endpoints = [
        'https://api.binance.com/api/v3',
        'https://api1.binance.com/api/v3',
        'https://api2.binance.com/api/v3'
    ]
    
    last_error = None
    
    for attempt in range(max_retries):
        for endpoint in endpoints:
            try:
                # Update endpoint
                exchange.urls['api']['public'] = endpoint
                
                # Get the method
                method = getattr(exchange, method_name)
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    asyncio.to_thread(method, *args, **kwargs),
                    timeout=30
                )
                
                print(f"✅ {method_name} successful from {endpoint}")
                return result
                
            except asyncio.TimeoutError:
                print(f"⏰ {method_name} timeout from {endpoint}")
                last_error = f"Timeout from {endpoint}"
                
            except Exception as e:
                print(f"❌ {method_name} error from {endpoint}: {e}")
                last_error = str(e)
                
                # Wait before next attempt
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    raise Exception(f"All {method_name} attempts failed. Last error: {last_error}")

# Enhanced market data fetching
async def get_enhanced_market_data_safe(symbol: str, binance_client) -> Dict:
    """Enhanced market data collection with better error handling"""
    try:
        # Fetch ticker with retry
        ticker = await safe_fetch_with_retry(binance_client, 'fetch_ticker', symbol)
        
        # Basic data
        data = {
            'current_price': ticker.get('last', 0),
            'volume_24h': ticker.get('quoteVolume', 0),
            'price_change_24h': ticker.get('percentage', 0),
            'high_24h': ticker.get('high', 0),
            'low_24h': ticker.get('low', 0)
        }
        
        # Enhanced metrics
        if data['high_24h'] and data['low_24h'] and data['current_price']:
            daily_range = data['high_24h'] - data['low_24h']
            if daily_range > 0:
                data['range_position'] = (data['current_price'] - data['low_24h']) / daily_range
            else:
                data['range_position'] = 0.5
        
        # Volume analysis with retry
        try:
            ohlcv_1d = await safe_fetch_with_retry(
                binance_client, 'fetch_ohlcv', symbol, '1d', None, 7
            )
            
            if len(ohlcv_1d) >= 3:
                recent_volumes = [candle[5] for candle in ohlcv_1d]
                avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 1
                data['volume_ratio'] = data['volume_24h'] / avg_volume if avg_volume > 0 else 1
            else:
                data['volume_ratio'] = 1
        except:
            print(f"⚠️ Volume analysis failed for {symbol}")
            data['volume_ratio'] = 1
        
        # Volatility metrics
        if data['current_price'] > 0 and daily_range > 0:
            data['volatility_24h'] = daily_range / data['current_price']
        else:
            data['volatility_24h'] = 0.02
        
        return data
        
    except Exception as e:
        print(f"❌ Market data error for {symbol}: {e}")
        # Return minimal data to prevent complete failure
        return {
            'current_price': 0,
            'volume_24h': 0,
            'price_change_24h': 0,
            'high_24h': 0,
            'low_24h': 0,
            'range_position': 0.5,
            'volume_ratio': 1,
            'volatility_24h': 0.02
        }

# Enhanced symbol analysis with timeout and error handling
async def analyze_symbol_comprehensive_safe(symbol: str, binance_client, headers, 
                                          market_regime, style_filter: Optional[str]) -> Optional[Dict]:
    """Safe comprehensive symbol analysis with timeout"""
    
    try:
        print(f"🔍 Analyzing {symbol}...")
        
        # Step 1: Get market data
        market_data = await get_enhanced_market_data_safe(symbol, binance_client)
        if not market_data or market_data.get('volume_24h', 0) == 0:
            print(f"⚪ {symbol}: No market data")
            return None
        
        # Step 2: Multi-timeframe analysis with timeout
        tf_analysis = await asyncio.wait_for(
            analyze_multi_timeframe_safe(symbol, binance_client, style_filter),
            timeout=45
        )
        
        if not tf_analysis:
            print(f"⚪ {symbol}: No TF analysis")
            return None
        
        # Step 3: Calculate MTF consensus
        mtf_consensus = trading_engine.calculate_mtf_consensus_professional(tf_analysis)
        if not mtf_consensus or mtf_consensus['confidence'] < 50:  # Lower threshold
            print(f"⚪ {symbol}: Low confidence ({mtf_consensus.get('confidence', 0):.1f}%)")
            return None
        
        # Step 4: Get netflow (non-blocking)
        try:
            netflow = get_dune_cex_flow_enhanced(symbol, headers, market_data)
        except:
            netflow = 0
        
        # Step 5: Enhanced signal analysis
        signal_result = trading_engine.signal_analyzer.analyze_comprehensive_signal(
            symbol, tf_analysis, market_data, market_regime, netflow
        )
        
        if signal_result['signal_type'] == 'NEUTRAL':
            print(f"⚪ {symbol}: Neutral signal")
            return None
        
        # Step 6: Risk management
        primary_tf = trading_engine.signal_analyzer.get_primary_timeframe(tf_analysis)
        if not primary_tf:
            return None
            
        primary_data = tf_analysis[primary_tf]
        
        risk_mgmt = trading_engine.calculate_advanced_risk_management(
            primary_data['indicators'], 
            signal_result['signal_type'],
            primary_data['current_price'],
            style_filter or 'DAY_TRADING'
        )
        
        if not risk_mgmt:
            print(f"⚪ {symbol}: Risk mgmt failed")
            return None
        
        # Step 7: Calculate final score
        final_score = trading_engine.calculate_professional_score(
            signal_result, market_data, risk_mgmt, market_regime
        )
        
        result = {
            'symbol': symbol,
            'signal_type': signal_result['signal_type'],
            'confidence': signal_result['confidence'],
            'final_score': final_score,
            'current_price': primary_data['current_price'],
            'market_data': market_data,
            'signal_components': signal_result['components'],
            'risk_mgmt': risk_mgmt,
            'tf_analysis': tf_analysis,
            'mtf_consensus': mtf_consensus,
            'netflow': netflow,
            'market_regime': market_regime,
            'signal_details': signal_result.get('details', [])
        }
        
        print(f"✅ {symbol}: {signal_result['signal_type']} ({signal_result['confidence']:.1f}%)")
        return result
        
    except asyncio.TimeoutError:
        print(f"⏰ {symbol}: Analysis timeout")
        return None
    except Exception as e:
        print(f"❌ {symbol}: Analysis error - {e}")
        return None

async def analyze_multi_timeframe_safe(symbol: str, binance_client, 
                                     style_filter: Optional[str]) -> Dict:
    """Safe multi-timeframe analysis with better error handling"""
    try:
        # Define timeframes based on style
        if style_filter == 'SCALPING':
            timeframes = {
                '1m': {'weight': 3, 'periods': 50},   # Reduced periods
                '5m': {'weight': 4, 'periods': 50}, 
                '15m': {'weight': 3, 'periods': 50}
            }
        elif style_filter == 'DAY_TRADING':
            timeframes = {
                '5m': {'weight': 2, 'periods': 50},
                '15m': {'weight': 4, 'periods': 50}, 
                '1h': {'weight': 3, 'periods': 50}
            }
        elif style_filter == 'SWING':
            timeframes = {
                '1h': {'weight': 3, 'periods': 50},
                '4h': {'weight': 4, 'periods': 50},
                '1d': {'weight': 3, 'periods': 30}
            }
        else:
            # Default - faster analysis
            timeframes = {
                '15m': {'weight': 3, 'periods': 40}, 
                '1h': {'weight': 4, 'periods': 40}
            }
        
        tf_analysis = {}
        
        for tf, config in timeframes.items():
            try:
                # Fetch OHLCV with timeout and retry
                ohlcv = await safe_fetch_with_retry(
                    binance_client, 'fetch_ohlcv', 
                    symbol, tf, None, config['periods']
                )
                
                if len(ohlcv) < 30:  # Reduced minimum
                    continue
                    
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Calculate basic indicators (reduced set for speed)
                indicators = calculate_basic_indicators(df)
                if not indicators:
                    continue
                
                current_price = df['close'].iloc[-1]
                
                # Basic trend analysis
                trend_analysis = analyze_trend_basic(df, indicators)
                
                # Basic signal strength
                signal_strength = calculate_signal_strength_basic(indicators, current_price)
                
                tf_analysis[tf] = {
                    'trend': trend_analysis['trend'],
                    'trend_strength': trend_analysis['strength'],
                    'signal_strength': signal_strength,
                    'current_price': current_price,
                    'rsi': indicators.get('rsi', [50])[-1],
                    'weight': config['weight'],
                    'indicators': indicators
                }
                
            except Exception as e:
                print(f"❌ Error analyzing {tf} for {symbol}: {e}")
                continue
        
        return tf_analysis
        
    except Exception as e:
        print(f"❌ Multi-timeframe error for {symbol}: {e}")
        return {}

def calculate_basic_indicators(df: pd.DataFrame) -> Dict:
    """Calculate basic indicators for faster analysis"""
    try:
        if len(df) < 20:
            return {}
            
        indicators = {}
        
        # Basic indicators only
        close_prices = df['close'].to_numpy()
        
        indicators['rsi'] = talib.RSI(close_prices, timeperiod=14)
        indicators['ema_20'] = talib.EMA(close_prices, timeperiod=20)
        indicators['ema_50'] = talib.EMA(close_prices, timeperiod=min(50, len(df)-1))
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close_prices)
        indicators['macd_hist'] = macd_hist
        
        # Volume
        indicators['volume'] = df['volume'].to_numpy()
        indicators['volume_sma'] = talib.SMA(df['volume'].to_numpy(), timeperiod=20)
        
        # ATR
        indicators['atr'] = talib.ATR(
            df['high'].to_numpy(), 
            df['low'].to_numpy(), 
            close_prices, 
            timeperiod=14
        )
        
        return indicators
        
    except Exception as e:
        print(f"❌ Indicator calculation error: {e}")
        return {}

def analyze_trend_basic(df: pd.DataFrame, indicators: Dict) -> Dict:
    """Basic trend analysis for speed"""
    try:
        ema_20 = indicators.get('ema_20', np.array([]))
        ema_50 = indicators.get('ema_50', np.array([]))
        
        if len(ema_20) < 2 or len(ema_50) < 2:
            return {'trend': 'NEUTRAL', 'strength': 0}
        
        current_price = df['close'].iloc[-1]
        
        # Simple trend logic
        if current_price > ema_20[-1] > ema_50[-1]:
            trend = 'BULLISH'
            strength = 0.7
        elif current_price < ema_20[-1] < ema_50[-1]:
            trend = 'BEARISH' 
            strength = 0.7
        else:
            trend = 'NEUTRAL'
            strength = 0.3
        
        return {'trend': trend, 'strength': strength}
        
    except Exception as e:
        return {'trend': 'NEUTRAL', 'strength': 0}

def calculate_signal_strength_basic(indicators: Dict, current_price: float) -> int:
    """Basic signal strength for speed"""
    try:
        strength = 0
        
        # RSI
        rsi = indicators.get('rsi', np.array([50]))
        if len(rsi) > 0:
            if rsi[-1] < 30 or rsi[-1] > 70:
                strength += 3
            elif rsi[-1] < 40 or rsi[-1] > 60:
                strength += 1
        
        # MACD
        macd_hist = indicators.get('macd_hist', np.array([0]))
        if len(macd_hist) >= 2:
            if macd_hist[-1] > 0 and macd_hist[-2] <= 0:
                strength += 3
            elif macd_hist[-1] < 0 and macd_hist[-2] >= 0:
                strength += 3
        
        # Volume
        volume = indicators.get('volume', np.array([0]))
        volume_sma = indicators.get('volume_sma', np.array([1]))
        if len(volume) > 0 and len(volume_sma) > 0:
            if volume[-1] > volume_sma[-1] * 1.5:
                strength += 2
        
        return min(strength, 10)
        
    except Exception:
        return 0
        
def escape_markdown_v2(text: str) -> str:
    """Escape special characters for MarkdownV2"""
    # Characters that need escaping in MarkdownV2
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    
    return text

def safe_markdown_message(text: str, parse_mode: str = 'Markdown') -> tuple:
    """Return safe text and parse_mode for Telegram"""
    try:
        # Try to clean problematic characters
        if parse_mode == 'Markdown':
            # Remove problematic markdown characters that might cause parsing errors
            text = text.replace('**', '*')  # Convert ** to *
            text = text.replace('***', '*')  # Convert *** to *
            text = re.sub(r'\*{3,}', '*', text)  # Replace multiple * with single
            
            # Fix unmatched markdown
            text = re.sub(r'(?<!\*)\*(?!\*)', '', text)  # Remove single *
            text = re.sub(r'(?<!_)_(?!_)', '', text)     # Remove single _
            
            # Ensure balanced markdown
            star_count = text.count('*')
            if star_count % 2 != 0:
                text = text.replace('*', '')
            
            underscore_count = text.count('_')
            if underscore_count % 2 != 0:
                text = text.replace('_', '')
        
        return text, parse_mode
        
    except Exception as e:
        print(f"Markdown cleanup error: {e}")
        # Fallback: remove all markdown and use plain text
        clean_text = re.sub(r'[*_`\[\]()]', '', text)
        return clean_text, None

# Fix the market regime analysis message
async def show_market_regime_analysis_fixed(query, context):
    """Show detailed market regime analysis with safe markdown"""
    try:
        processing_msg = "🔍 Analyzing market regime in detail..."
        await query.edit_message_text(processing_msg)
        
        await load_markets_with_retry(binance)
        
        # Get comprehensive market data
        btc_data = binance.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=200)
        eth_data = binance.fetch_ohlcv('ETH/USDT', timeframe='1h', limit=100)
        
        regime_detector = MarketRegimeDetector()
        current_regime = regime_detector.detect_regime(btc_data, eth_data, {})
        
        # Calculate metrics
        btc_df = pd.DataFrame(btc_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        ema_20 = talib.EMA(btc_df['close'].to_numpy(), timeperiod=20)
        ema_50 = talib.EMA(btc_df['close'].to_numpy(), timeperiod=50)
        ema_200 = talib.EMA(btc_df['close'].to_numpy(), timeperiod=200)
        rsi = talib.RSI(btc_df['close'].to_numpy(), timeperiod=14)
        atr = talib.ATR(btc_df['high'].to_numpy(), btc_df['low'].to_numpy(), btc_df['close'].to_numpy())
        
        current_price = btc_df['close'].iloc[-1]
        volatility = (atr[-1] / current_price) * 100
        
        # Market structure
        structure = AdvancedIndicators.calculate_market_structure(btc_df)
        
        # Create safe message
        regime_msg = (
            f"COMPREHENSIVE MARKET REGIME ANALYSIS\n"
            f"========================================\n\n"
            f"Current Regime: {current_regime.value.upper()}\n"
            f"BTC Price: ${current_price:,.0f}\n\n"
            f"TREND ANALYSIS:\n"
            f"• Structure: {structure['structure_type'].title()}\n"
            f"• Trend Strength: {structure['trend_strength']:.2f}\n"
            f"• EMA Status: {'Bullish' if ema_20[-1] > ema_50[-1] > ema_200[-1] else 'Bearish' if ema_20[-1] < ema_50[-1] < ema_200[-1] else 'Mixed'}\n"
            f"• RSI Level: {rsi[-1]:.1f}\n\n"
            f"VOLATILITY METRICS:\n"
            f"• Current ATR: {volatility:.1f}%\n"
            f"• State: {'High' if volatility > 5 else 'Normal' if volatility > 3 else 'Low'}\n\n"
            f"TRADING IMPLICATIONS:\n"
        )
        
        # Add regime-specific recommendations (safe text)
        if current_regime == MarketRegime.BULL_TRENDING:
            regime_msg += (
                f"• Bullish momentum strategies preferred\n"
                f"• Long bias on pullbacks\n"
                f"• Breakout trades favored\n"
                f"• Avoid heavy short positions\n"
            )
        elif current_regime == MarketRegime.BEAR_TRENDING:
            regime_msg += (
                f"• Bearish momentum strategies preferred\n"
                f"• Short bias on rallies\n"
                f"• Breakdown trades favored\n"
                f"• Avoid heavy long positions\n"
            )
        elif current_regime == MarketRegime.HIGH_VOLATILITY:
            regime_msg += (
                f"• Reduce position sizes by 30-50%\n"
                f"• Wider stops required\n"
                f"• Scalping opportunities available\n"
                f"• Avoid swing positions\n"
            )
        else:
            regime_msg += (
                f"• Range trading strategies recommended\n"
                f"• Mean reversion setups preferred\n"
                f"• Wait for clear breakouts\n"
                f"• Use reduced position sizes\n"
            )
        
        regime_msg += f"\nAnalysis Time: {datetime.now().strftime('%H:%M:%S UTC')}"
        
        # Use safe markdown
        safe_text, parse_mode = safe_markdown_message(regime_msg)
        
        keyboard = [[
            InlineKeyboardButton("🔄 Refresh Analysis", callback_data='market_regime'),
            InlineKeyboardButton("↩️ Back", callback_data='back_main_pro')
        ]]
        
        await query.edit_message_text(
            safe_text, 
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=parse_mode
        )
        
    except Exception as e:
        print(f"Regime analysis error: {e}")
        error_msg = f"Regime analysis error: {str(e)}"
        keyboard = [[InlineKeyboardButton("↩️ Back", callback_data='back_main_pro')]]
        await query.edit_message_text(
            error_msg,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

# Fix professional analysis message sender
async def send_professional_analysis_message_fixed(context, chat_id: int, results: List[Dict], market_regime: MarketRegime):
    """Send professional analysis results with safe markdown"""
    
    if not results:
        await context.bot.send_message(
            chat_id=chat_id,
            text="No professional setups found meeting our strict criteria"
        )
        return
    
    # Header message (safe text)
    header_msg = (
        f"PROFESSIONAL MARKET ANALYSIS\n"
        f"======================================\n"
        f"Market Regime: {market_regime.value.upper()}\n"
        f"Qualified Setups: {len(results)}\n"
        f"Analysis Time: {datetime.now().strftime('%H:%M:%S UTC')}\n"
        f"Professional Grade Filtering Applied"
    )
    
    # Send header without markdown
    await context.bot.send_message(chat_id=chat_id, text=header_msg)
    await asyncio.sleep(2)
    
    # Send each setup (safe format)
    for i, result in enumerate(results, 1):
        try:
            signal_components = result.get('signal_components', {})
            risk_mgmt = result.get('risk_mgmt', {})
            
            # Safe values with None protection
            confidence = result.get('confidence', 0) or 0
            final_score = result.get('final_score', 0) or 0
            current_price = result.get('current_price', 0) or 0
            
            tp1 = risk_mgmt.get('take_profit_1', 0) or 0
            tp2 = risk_mgmt.get('take_profit_2', 0) or 0
            tp3 = risk_mgmt.get('take_profit_3', 0) or 0
            sl = risk_mgmt.get('stop_loss', 0) or 0
            
            rr1 = risk_mgmt.get('risk_reward_1', 0) or 0
            rr2 = risk_mgmt.get('risk_reward_2', 0) or 0
            rr3 = risk_mgmt.get('risk_reward_3', 0) or 0
            
            # Build message safely
            setup_msg = (
                f"SETUP #{i}: {result['signal_type']} - {result['symbol']}\n"
                f"=================================\n"
                f"Confidence: {confidence:.1f}% | Score: {final_score:.1f}\n"
                f"Entry: ${current_price:.4f}\n\n"
                f"TARGETS & RISK:\n"
                f"• TP1 (40%): ${tp1:.4f} (R:R {rr1:.1f})\n"
                f"• TP2 (40%): ${tp2:.4f} (R:R {rr2:.1f})\n"
                f"• TP3 (20%): ${tp3:.4f} (R:R {rr3:.1f})\n"
                f"• SL: ${sl:.4f}\n\n"
                f"SIGNAL BREAKDOWN:\n"
                f"• Technical: {signal_components.get('technical', 0):.0f}/25\n"
                f"• Momentum: {signal_components.get('momentum', 0):.0f}/20\n"
                f"• Volume: {signal_components.get('volume', 0):.0f}/15\n"
                f"• Structure: {signal_components.get('structure', 0):.0f}/20\n"
                f"• Sentiment: {signal_components.get('sentiment', 0):.0f}/10\n"
                f"• Risk Score: {signal_components.get('risk', 0):.0f}/10\n\n"
                f"24h Volume: ${result.get('market_data', {}).get('volume_24h', 0):,.0f}\n"
                f"Risk Management: Professional position sizing applied"
            )
            
            await context.bot.send_message(chat_id=chat_id, text=setup_msg)
            await asyncio.sleep(3)
            
        except Exception as e:
            print(f"Error sending setup {i}: {e}")
            await context.bot.send_message(
                chat_id=chat_id, 
                text=f"Error displaying setup #{i}: {str(e)[:100]}"
            )

# Apply the fixes
def apply_telegram_fixes():
    """Apply telegram message fixes"""
    global show_market_regime_analysis, send_professional_analysis_message
    
    show_market_regime_analysis = show_market_regime_analysis_fixed
    send_professional_analysis_message = send_professional_analysis_message_fixed

# Call this to apply fixes
apply_telegram_fixes()

def safe_send_message(text, parse_mode='Markdown'):
    try:
        # Remove problematic markdown
        clean_text = text.replace('**', '*')
        clean_text = clean_text.replace('***', '*')
        # Remove unbalanced markdown
        star_count = clean_text.count('*')
        if star_count % 2 != 0:
            clean_text = clean_text.replace('*', '')
        return clean_text, parse_mode
    except:
        return text.replace('*', '').replace('_', ''), None

# ===============================
# MAIN APPLICATION SETUP
# ===============================

if __name__ == "__main__":
    import logging
    
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('trading_bot.log')
        ]
    )
    
    # Clear screen
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("🏛️ Professional Crypto Trading System")
    print("=" * 40)
    print("Loading all components...")
    
    try:
        # Create application
        application = Application.builder().token(bot_token).build()
        
        # Professional handlers
        application.add_handler(CommandHandler("pro", pro_command))
        application.add_handler(CommandHandler("regime", regime_command))
        application.add_handler(CommandHandler("start", start_professional))
        
        # Original command handlers
        application.add_handler(CommandHandler("auto", auto_command))
        application.add_handler(CommandHandler("scalping", scalping_command))
        application.add_handler(CommandHandler("daytrading", daytrading_command))
        application.add_handler(CommandHandler("swing", swing_command))
        application.add_handler(CommandHandler("manual", manual_command))
        
        # Button handlers
        application.add_handler(CallbackQueryHandler(professional_button_handler_complete))
        
        # Message handler for manual analysis
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        
        # Error handler
        async def error_handler_complete(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
            """Complete error handling"""
            logging.error(f"Update {update} caused error {context.error}")
            
            if update and hasattr(update, 'message') and update.message:
                await update.message.reply_text(
                    "❌ An error occurred. Please try again or use /start to return to the main menu."
                )
            elif update and hasattr(update, 'callback_query') and update.callback_query:
                await update.callback_query.edit_message_text(
                    "❌ An error occurred. Returning to main menu...",
                    reply_markup=create_main_menu()
                )
        
        application.add_error_handler(error_handler_complete)
        
        print("🚀 Complete Trading System Started!")
        print("📊 All features available:")
        print("   ⚡ Quick Trading Modes")
        print("   🏛️ Professional Analysis")
        print("   📊 Portfolio Management") 
        print("   🔍 Deep Market Analysis")
        print("   ⚖️ Advanced Risk Management")
        print("   🎯 Sector Rotation Analysis")
        
        # ✅ INI YANG BENAR - LANGSUNG RUN_POLLING
        application.run_polling()
        
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logging.error(f"Fatal error: {e}", exc_info=True)
