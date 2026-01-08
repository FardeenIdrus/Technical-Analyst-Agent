"""
Signal Generator Module
Combines technical indicators with regime context to generate trading signals.

Features:
- Regime-adaptive strategy selection (trend-following vs mean-reversion)
- Multi-indicator confluence scoring
- Confidence-weighted signals (0-1)
- Adaptive thresholds based on market regime
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Signal(Enum):
    """Trading signal types"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class Strategy(Enum):
    """Strategy types based on regime"""
    TREND_FOLLOWING = "TREND_FOLLOWING"
    MEAN_REVERSION = "MEAN_REVERSION"
    DEFENSIVE = "DEFENSIVE"  # High volatility - reduce exposure


@dataclass
class TradeSignal:
    """Complete trade signal with metadata"""
    signal: Signal
    confidence: float  # 0-1
    strategy: Strategy
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_modifier: float  # 0-1, multiplier for position sizing
    indicators_bullish: int  # Count of bullish indicators
    indicators_bearish: int  # Count of bearish indicators
    reasoning: str  # Human-readable explanation

    def to_dict(self) -> dict:
        return {
            'signal': self.signal.value,
            'confidence': self.confidence,
            'strategy': self.strategy.value,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size_modifier': self.position_size_modifier,
            'indicators_bullish': self.indicators_bullish,
            'indicators_bearish': self.indicators_bearish,
            'reasoning': self.reasoning
        }


class SignalGenerator:
    """
    Generates trading signals by combining technical indicators with regime context.

    Strategy Selection Logic:
    - STRONG_BULL/MODERATE_BULL + TRENDING → Trend-following (long bias)
    - STRONG_BEAR/MODERATE_BEAR + TRENDING → Trend-following (short bias)
    - SIDEWAYS + MEAN_REVERTING → Mean-reversion (fade extremes)
    - CHOPPY/HIGH_VOLATILITY → Defensive (reduce exposure)

    Confluence Scoring:
    - Each indicator contributes a vote (+1 bullish, -1 bearish, 0 neutral)
    - More indicators agreeing = higher confidence
    - Regime alignment boosts confidence
    """

    # Thresholds for different regimes (adaptive)
    THRESHOLDS = {
        'trending': {
            'rsi_overbought': 75,  # More lenient in trends
            'rsi_oversold': 25,
            'bb_upper': 0.9,  # Allow riding the band
            'bb_lower': 0.1,
        },
        'mean_reverting': {
            'rsi_overbought': 70,  # Standard thresholds
            'rsi_oversold': 30,
            'bb_upper': 1.0,  # Fade at the bands
            'bb_lower': 0.0,
        },
        'defensive': {
            'rsi_overbought': 80,  # Very conservative
            'rsi_oversold': 20,
            'bb_upper': 1.1,
            'bb_lower': -0.1,
        }
    }

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with data containing indicators and regime classifications.

        Args:
            data: DataFrame with OHLCV, technical indicators, and regime columns
                  Required: Close, RSI, MACD, SMA_50, SMA_200, BB_Percent_B, ATR
                  Required regime: Market_Regime, Volatility_Regime, Trend_Persistence
        """
        self.data = data.copy()
        self._validate_data()

    def _validate_data(self):
        """Validate required columns exist"""
        required_indicators = ['Close', 'RSI', 'MACD', 'MACD_Signal']
        required_regime = ['Market_Regime', 'Volatility_Regime', 'Trend_Persistence']

        missing_ind = [col for col in required_indicators if col not in self.data.columns]
        missing_reg = [col for col in required_regime if col not in self.data.columns]

        if missing_ind:
            raise ValueError(f"Missing indicator columns: {missing_ind}")
        if missing_reg:
            raise ValueError(f"Missing regime columns: {missing_reg}. Run RegimeDetector first.")

    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals for all rows in the dataset.

        Returns:
            DataFrame with signal columns added
        """
        print("\n Generating Trading Signals...")

        # Select strategy based on regime
        self._select_strategies()

        # Generate indicator signals
        self._generate_indicator_signals()

        # Calculate confluence scores
        self._calculate_confluence()

        # Generate final signals
        self._generate_final_signals()

        # Calculate entry/exit levels
        self._calculate_trade_levels()

        print(f" Signal generation complete")
        print(f"  Added columns: Signal, Confidence, Strategy, etc.")

        return self.data

    def _select_strategies(self):
        """
        Select strategy for each row based on regime.
        Vectorized implementation.
        """
        market_regime = self.data['Market_Regime'].values
        vol_regime = self.data['Volatility_Regime'].values
        trend_persist = self.data['Trend_Persistence'].values

        # Default to trend-following
        strategy = np.full(len(self.data), Strategy.TREND_FOLLOWING.value)

        # Defensive in high volatility or choppy markets
        is_defensive = (
            (vol_regime == 'HIGH_VOLATILITY') |
            (market_regime == 'CHOPPY')
        )
        strategy[is_defensive] = Strategy.DEFENSIVE.value

        # Mean-reversion in sideways + mean-reverting conditions
        is_mean_revert = (
            (market_regime == 'SIDEWAYS') &
            (trend_persist == 'MEAN_REVERTING')
        )
        strategy[is_mean_revert] = Strategy.MEAN_REVERSION.value

        self.data['Strategy'] = strategy

    def _generate_indicator_signals(self):
        """
        Generate individual indicator signals.
        Each indicator votes: +1 (bullish), -1 (bearish), 0 (neutral)
        """
        # Get strategy for adaptive thresholds
        strategy = self.data['Strategy'].values

        # Initialize signal arrays
        n = len(self.data)

        # RSI Signal
        rsi = self.data['RSI'].values
        rsi_signal = np.zeros(n)

        # Adaptive RSI thresholds
        for strat, thresh in self.THRESHOLDS.items():
            mask = strategy == strat.upper()
            if strat == 'trending':
                mask = strategy == Strategy.TREND_FOLLOWING.value
            elif strat == 'mean_reverting':
                mask = strategy == Strategy.MEAN_REVERSION.value
            elif strat == 'defensive':
                mask = strategy == Strategy.DEFENSIVE.value

            rsi_signal[mask & (rsi < thresh['rsi_oversold'])] = 1  # Oversold = bullish
            rsi_signal[mask & (rsi > thresh['rsi_overbought'])] = -1  # Overbought = bearish

        self.data['RSI_Signal'] = rsi_signal

        # MACD Signal
        macd = self.data['MACD'].values
        macd_sig = self.data['MACD_Signal'].values
        macd_hist = self.data.get('MACD_Histogram', macd - macd_sig).values

        macd_signal = np.where(
            (macd > macd_sig) & (macd_hist > 0), 1,  # Bullish crossover + positive histogram
            np.where(
                (macd < macd_sig) & (macd_hist < 0), -1,  # Bearish crossover + negative histogram
                0
            )
        )
        self.data['MACD_Signal_Vote'] = macd_signal

        # Moving Average Signal (Price vs SMA 50/200)
        close = self.data['Close'].values
        ma_signal = np.zeros(n)

        if 'SMA_50' in self.data.columns and 'SMA_200' in self.data.columns:
            sma50 = self.data['SMA_50'].values
            sma200 = self.data['SMA_200'].values

            # Bullish: Price > SMA50 > SMA200 (uptrend)
            # Bearish: Price < SMA50 < SMA200 (downtrend)
            ma_signal = np.where(
                (close > sma50) & (sma50 > sma200), 1,
                np.where(
                    (close < sma50) & (sma50 < sma200), -1,
                    0
                )
            )
        self.data['MA_Signal'] = ma_signal

        # Bollinger Band Signal
        bb_signal = np.zeros(n)
        if 'BB_Percent_B' in self.data.columns:
            bb_pct = self.data['BB_Percent_B'].values

            # For trend-following: ride the bands
            # For mean-reversion: fade the bands
            for strat, thresh in self.THRESHOLDS.items():
                if strat == 'trending':
                    mask = strategy == Strategy.TREND_FOLLOWING.value
                    # In trends, breakout above upper band is bullish
                    bb_signal[mask & (bb_pct > thresh['bb_upper'])] = 1
                    bb_signal[mask & (bb_pct < thresh['bb_lower'])] = -1
                elif strat == 'mean_reverting':
                    mask = strategy == Strategy.MEAN_REVERSION.value
                    # In mean-reversion, fade the extremes
                    bb_signal[mask & (bb_pct > thresh['bb_upper'])] = -1  # Overbought, sell
                    bb_signal[mask & (bb_pct < thresh['bb_lower'])] = 1   # Oversold, buy
                elif strat == 'defensive':
                    mask = strategy == Strategy.DEFENSIVE.value
                    # Very conservative
                    bb_signal[mask & (bb_pct > thresh['bb_upper'])] = -1
                    bb_signal[mask & (bb_pct < thresh['bb_lower'])] = 1

        self.data['BB_Signal'] = bb_signal

        # ADX/DI Signal (if available)
        adx_signal = np.zeros(n)
        if 'ADX' in self.data.columns and 'Plus_DI' in self.data.columns:
            adx = self.data['ADX'].values
            plus_di = self.data['Plus_DI'].values
            minus_di = self.data['Minus_DI'].values

            # Strong trend (ADX > 25) + DI direction
            strong_trend = adx > 25
            adx_signal[strong_trend & (plus_di > minus_di)] = 1  # Bullish trend
            adx_signal[strong_trend & (minus_di > plus_di)] = -1  # Bearish trend

        self.data['ADX_Signal'] = adx_signal

        # Volume Confirmation (if available)
        vol_signal = np.zeros(n)
        if 'Volume_Ratio' in self.data.columns:
            vol_ratio = self.data['Volume_Ratio'].values
            daily_return = self.data['Close'].pct_change().values

            # High volume confirms price direction
            high_vol = vol_ratio > 1.5
            vol_signal[high_vol & (daily_return > 0)] = 1  # High vol + up = bullish
            vol_signal[high_vol & (daily_return < 0)] = -1  # High vol + down = bearish

        self.data['Volume_Signal'] = vol_signal

    def _calculate_confluence(self):
        """
        Calculate confluence score from all indicator signals.
        More indicators agreeing = higher confidence.
        """
        signal_cols = ['RSI_Signal', 'MACD_Signal_Vote', 'MA_Signal',
                       'BB_Signal', 'ADX_Signal', 'Volume_Signal']

        # Get available signal columns
        available = [col for col in signal_cols if col in self.data.columns]

        if not available:
            self.data['Confluence_Score'] = 0
            self.data['Bullish_Count'] = 0
            self.data['Bearish_Count'] = 0
            return

        # Stack signals
        signals = self.data[available].values

        # Count bullish and bearish
        bullish_count = (signals > 0).sum(axis=1)
        bearish_count = (signals < 0).sum(axis=1)
        total_signals = len(available)

        # Confluence score: net direction normalized by total indicators
        # Range: -1 (all bearish) to +1 (all bullish)
        confluence = (bullish_count - bearish_count) / total_signals

        self.data['Confluence_Score'] = confluence
        self.data['Bullish_Count'] = bullish_count
        self.data['Bearish_Count'] = bearish_count

    def _generate_final_signals(self):
        """
        Generate final BUY/SELL/HOLD signals based on confluence and regime.
        """
        confluence = self.data['Confluence_Score'].values
        strategy = self.data['Strategy'].values
        regime_conf = self.data['Regime_Confidence'].values if 'Regime_Confidence' in self.data.columns else np.ones(len(self.data)) * 0.5

        n = len(self.data)
        signal = np.full(n, Signal.HOLD.value, dtype=object)  # Use object dtype for variable-length strings
        confidence = np.zeros(n)

        # Trend-following strategy
        trend_mask = strategy == Strategy.TREND_FOLLOWING.value

        # Strong signals require high confluence
        signal[(trend_mask) & (confluence >= 0.5)] = Signal.STRONG_BUY.value
        signal[(trend_mask) & (confluence >= 0.25) & (confluence < 0.5)] = Signal.BUY.value
        signal[(trend_mask) & (confluence <= -0.5)] = Signal.STRONG_SELL.value
        signal[(trend_mask) & (confluence <= -0.25) & (confluence > -0.5)] = Signal.SELL.value

        # Mean-reversion strategy (opposite logic for extremes)
        mr_mask = strategy == Strategy.MEAN_REVERSION.value

        signal[(mr_mask) & (confluence >= 0.5)] = Signal.STRONG_BUY.value
        signal[(mr_mask) & (confluence >= 0.25) & (confluence < 0.5)] = Signal.BUY.value
        signal[(mr_mask) & (confluence <= -0.5)] = Signal.STRONG_SELL.value
        signal[(mr_mask) & (confluence <= -0.25) & (confluence > -0.5)] = Signal.SELL.value

        # Defensive strategy - very conservative, mostly hold
        def_mask = strategy == Strategy.DEFENSIVE.value

        signal[(def_mask) & (confluence >= 0.67)] = Signal.BUY.value  # Only strong signals
        signal[(def_mask) & (confluence <= -0.67)] = Signal.SELL.value

        # Calculate confidence
        # Base confidence from confluence strength
        base_confidence = np.abs(confluence)

        # Boost confidence when regime and signal align
        regime_boost = np.where(
            regime_conf > 0.5,
            0.1 * regime_conf,  # Up to 10% boost for high regime confidence
            0
        )

        # Reduce confidence in defensive mode
        strategy_modifier = np.where(
            strategy == Strategy.DEFENSIVE.value,
            0.7,  # 30% reduction in defensive mode
            1.0
        )

        confidence = np.clip(base_confidence + regime_boost, 0, 1) * strategy_modifier

        self.data['Signal'] = signal
        self.data['Signal_Confidence'] = confidence

    def _calculate_trade_levels(self):
        """
        Calculate stop loss and take profit levels based on ATR.
        """
        close = self.data['Close'].values

        # Use ATR for dynamic stops if available
        if 'ATR' in self.data.columns:
            atr = self.data['ATR'].values

            # Stop loss: 2x ATR from entry
            # Take profit: 3x ATR from entry (1.5:1 reward/risk)
            stop_distance = 2 * atr
            profit_distance = 3 * atr

        else:
            # Fallback: 2% stop, 3% target
            stop_distance = close * 0.02
            profit_distance = close * 0.03

        # Calculate levels based on signal direction
        signal = self.data['Signal'].values

        stop_loss = np.where(
            np.isin(signal, [Signal.STRONG_BUY.value, Signal.BUY.value]),
            close - stop_distance,  # Long: stop below
            np.where(
                np.isin(signal, [Signal.STRONG_SELL.value, Signal.SELL.value]),
                close + stop_distance,  # Short: stop above
                close  # Hold: no stop
            )
        )

        take_profit = np.where(
            np.isin(signal, [Signal.STRONG_BUY.value, Signal.BUY.value]),
            close + profit_distance,  # Long: target above
            np.where(
                np.isin(signal, [Signal.STRONG_SELL.value, Signal.SELL.value]),
                close - profit_distance,  # Short: target below
                close  # Hold: no target
            )
        )

        # Position size modifier based on volatility and confidence
        vol_regime = self.data['Volatility_Regime'].values
        confidence = self.data['Signal_Confidence'].values

        position_modifier = np.where(
            vol_regime == 'HIGH_VOLATILITY',
            0.5,  # Half size in high vol
            np.where(
                vol_regime == 'LOW_VOLATILITY',
                1.2,  # Can increase slightly in low vol
                1.0  # Normal
            )
        ) * confidence  # Scale by confidence

        self.data['Stop_Loss'] = stop_loss
        self.data['Take_Profit'] = take_profit
        self.data['Position_Modifier'] = np.clip(position_modifier, 0.1, 1.5)

    def get_current_signal(self) -> TradeSignal:
        """
        Get the most recent trading signal.

        Returns:
            TradeSignal dataclass with all signal details
        """
        if len(self.data) == 0:
            raise ValueError("No data available")

        latest = self.data.iloc[-1]

        # Build reasoning string
        bullish = int(latest.get('Bullish_Count', 0))
        bearish = int(latest.get('Bearish_Count', 0))
        regime = latest.get('Market_Regime', 'UNKNOWN')
        strategy = latest.get('Strategy', 'UNKNOWN')

        reasoning = (
            f"Regime: {regime}, Strategy: {strategy}. "
            f"Indicators: {bullish} bullish, {bearish} bearish. "
            f"Confluence: {latest.get('Confluence_Score', 0):.2f}"
        )

        return TradeSignal(
            signal=Signal(latest.get('Signal', 'HOLD')),
            confidence=float(latest.get('Signal_Confidence', 0)),
            strategy=Strategy(latest.get('Strategy', 'TREND_FOLLOWING')),
            entry_price=float(latest['Close']),
            stop_loss=float(latest.get('Stop_Loss', latest['Close'])),
            take_profit=float(latest.get('Take_Profit', latest['Close'])),
            position_size_modifier=float(latest.get('Position_Modifier', 1.0)),
            indicators_bullish=bullish,
            indicators_bearish=bearish,
            reasoning=reasoning
        )

    def get_signal_summary(self) -> str:
        """
        Generate human-readable summary of current signal.

        Returns:
            Formatted string summary
        """
        try:
            signal = self.get_current_signal()
        except ValueError:
            return "No signal data available"

        latest = self.data.iloc[-1]

        # Signal interpretation
        if signal.signal in [Signal.STRONG_BUY, Signal.BUY]:
            action = "LONG"
            direction = "bullish"
        elif signal.signal in [Signal.STRONG_SELL, Signal.SELL]:
            action = "SHORT"
            direction = "bearish"
        else:
            action = "WAIT"
            direction = "neutral"

        # Risk/reward calculation
        if signal.signal != Signal.HOLD:
            risk = abs(signal.entry_price - signal.stop_loss)
            reward = abs(signal.take_profit - signal.entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
        else:
            risk = reward = rr_ratio = 0

        summary = f"""
=== TRADING SIGNAL SUMMARY ===
Date: {self.data.index[-1].date() if hasattr(self.data.index[-1], 'date') else 'N/A'}
Price: ${signal.entry_price:.2f}

SIGNAL: {signal.signal.value}
  Confidence: {signal.confidence:.1%}
  Strategy: {signal.strategy.value}
  Direction: {direction.upper()}

INDICATOR CONFLUENCE:
  Bullish indicators: {signal.indicators_bullish}
  Bearish indicators: {signal.indicators_bearish}
  Net score: {latest.get('Confluence_Score', 0):.2f}

TRADE LEVELS:
  Entry: ${signal.entry_price:.2f}
  Stop Loss: ${signal.stop_loss:.2f} ({abs(signal.entry_price - signal.stop_loss) / signal.entry_price * 100:.1f}%)
  Take Profit: ${signal.take_profit:.2f} ({abs(signal.take_profit - signal.entry_price) / signal.entry_price * 100:.1f}%)
  Risk/Reward: 1:{rr_ratio:.1f}

POSITION SIZING:
  Modifier: {signal.position_size_modifier:.2f}x
  (Adjust base position by this factor)

REASONING:
  {signal.reasoning}
"""
        return summary

    def get_signal_history(self, last_n: int = 30) -> pd.DataFrame:
        """
        Get signal history for the last N periods.

        Args:
            last_n: Number of periods to return

        Returns:
            DataFrame with signal history
        """
        cols = ['Close', 'Signal', 'Signal_Confidence', 'Strategy',
                'Confluence_Score', 'Stop_Loss', 'Take_Profit']
        available_cols = [c for c in cols if c in self.data.columns]

        return self.data[available_cols].tail(last_n)

    def get_signal_statistics(self) -> Dict:
        """
        Calculate statistics about generated signals.

        Returns:
            Dictionary with signal statistics
        """
        if 'Signal' not in self.data.columns:
            return {}

        signals = self.data['Signal'].value_counts()
        total = len(self.data)

        # Count buy and sell signals properly
        buy_count = int(signals.get('STRONG_BUY', 0) + signals.get('BUY', 0))
        sell_count = int(signals.get('STRONG_SELL', 0) + signals.get('SELL', 0))
        hold_count = int(signals.get('HOLD', 0))

        stats = {
            'total_signals': total,
            'signal_distribution': {str(k): int(v) for k, v in signals.items()},
            'strong_buy': int(signals.get('STRONG_BUY', 0)),
            'buy': int(signals.get('BUY', 0)),
            'hold': hold_count,
            'sell': int(signals.get('SELL', 0)),
            'strong_sell': int(signals.get('STRONG_SELL', 0)),
            'total_buy_signals': buy_count,
            'total_sell_signals': sell_count,
            'avg_confidence': float(self.data['Signal_Confidence'].mean()),
            'strategy_distribution': {str(k): int(v) for k, v in self.data['Strategy'].value_counts().items()}
        }

        return stats


# =============================================================================
# TEST SCRIPT
# =============================================================================
if __name__ == "__main__":
    """
    Test script for signal generation.
    Usage: python src/signal_generator.py
    """
    from data_collector import DataCollector
    from technical_indicators import TechnicalIndicators
    from regime_detector import RegimeDetector

    print("=" * 60)
    print("SIGNAL GENERATOR TEST")
    print("=" * 60)

    # Load data
    collector = DataCollector()
    data = collector.get_data("AAPL", years=10)

    # Calculate technical indicators
    indicators = TechnicalIndicators(data)
    data = indicators.calculate_all()

    # Run regime detection
    detector = RegimeDetector(data)
    data = detector.detect_all_regimes()

    # Generate signals
    generator = SignalGenerator(data)
    data = generator.generate_signals()

    # Print current signal
    print(generator.get_signal_summary())

    # Show signal statistics
    print("\n" + "=" * 60)
    print("SIGNAL STATISTICS")
    print("=" * 60)
    stats = generator.get_signal_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Show recent signal history
    print("\n" + "=" * 60)
    print("RECENT SIGNAL HISTORY (Last 10 days)")
    print("=" * 60)
    history = generator.get_signal_history(10)
    print(history.to_string())
