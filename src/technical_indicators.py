"""
Technical Indicators Module
Calculates 6 key technical indicators for trading strategies:
1. RSI (Relative Strength Index) - Momentum oscillator
2. MACD (Moving Average Convergence Divergence) - Trend/momentum
3. Moving Averages (SMA 50/200) - Trend identification
4. Bollinger Bands - Volatility and mean reversion
5. ATR (Average True Range) - Volatility measure
6. Volume Moving Average - Volume confirmation
"""

import pandas as pd
import numpy as np


class TechnicalIndicators:
    """
    Calculates technical indicators and adds them as columns to price data.
    All indicators are widely used in professional trading systems.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with OHLCV price data.

        Args:
            data: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
                  Index should be DatetimeIndex
        """
        # Store a copy to avoid modifying original data
        self.data = data.copy()

        # Validate required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_cols if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def calculate_all(self) -> pd.DataFrame:
        """
        Calculate all 6 technical indicators at once.
        This is the main method to use - calculates everything in correct order.

        Returns:
            DataFrame with original OHLCV data plus all indicator columns
        """
        print("\n Calculating Technical Indicators...")

        # Calculate each indicator (order doesn't matter, they're independent)
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_moving_averages()
        self.calculate_bollinger_bands()
        self.calculate_atr()
        self.calculate_volume_ma()

        # Add some useful derived signals
        self._calculate_derived_signals()

        print(f" Added {len(self.data.columns) - 6} indicator columns")
        print(f"  Indicators: RSI, MACD, SMA_50, SMA_200, BB, ATR, Volume_MA")

        return self.data

    # =========================================================================
    # INDICATOR 1: RSI (Relative Strength Index)
    # =========================================================================
    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index).

        RSI measures momentum by comparing recent gains vs losses.
        - RSI > 70: Overbought (potentially overvalued, may fall)
        - RSI < 30: Oversold (potentially undervalued, may rise)
        - RSI = 50: Neutral momentum

        Formula:
            RSI = 100 - (100 / (1 + RS))
            RS = Average Gain / Average Loss (over 'period' days)

        Args:
            period: Lookback period (standard is 14 days)

        Returns:
            RSI Series (values 0-100)
        """
        # Calculate daily price changes (today's close - yesterday's close)
        delta = self.data['Close'].diff()

        # Separate gains (positive changes) from losses (negative changes)
        # Use .clip() to set floor/ceiling: gains keeps positives, losses keeps negatives
        gains = delta.clip(lower=0)  # Keep only positive values, replace negatives with 0
        losses = (-delta).clip(lower=0)  # Flip sign, keep only positives (these were losses)

        # Calculate exponential moving average of gains and losses
        # Using EMA (exponential) instead of SMA gives more weight to recent data
        # adjust=False uses recursive formula: EMA_t = alpha*x_t + (1-alpha)*EMA_(t-1)
        avg_gain = gains.ewm(span=period, adjust=False).mean()
        avg_loss = losses.ewm(span=period, adjust=False).mean()

        # Calculate Relative Strength (RS)
        # Add small epsilon to avoid division by zero when no losses
        rs = avg_gain / (avg_loss + 1e-10)

        # Convert RS to RSI (bounded 0-100)
        rsi = 100 - (100 / (1 + rs))

        # Store in DataFrame
        self.data['RSI'] = rsi

        return rsi

    # =========================================================================
    # INDICATOR 2: MACD (Moving Average Convergence Divergence)
    # =========================================================================
    def calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        MACD shows relationship between two moving averages of price.
        Used to identify trend direction and momentum.

        Components:
        - MACD Line: Fast EMA - Slow EMA (difference between short & long term trends)
        - Signal Line: EMA of MACD Line (smoothed version for crossover signals)
        - Histogram: MACD Line - Signal Line (momentum strength)

        Trading Signals:
        - MACD crosses above Signal: Bullish (buy signal)
        - MACD crosses below Signal: Bearish (sell signal)
        - Histogram growing: Momentum increasing
        - Histogram shrinking: Momentum decreasing

        Args:
            fast: Fast EMA period (standard: 12)
            slow: Slow EMA period (standard: 26)
            signal: Signal line EMA period (standard: 9)

        Returns:
            DataFrame with MACD, Signal, and Histogram columns
        """
        close = self.data['Close']

        # Calculate fast and slow EMAs
        # Fast EMA reacts quicker to price changes
        # Slow EMA is smoother, shows longer-term trend
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        # MACD Line = Fast EMA - Slow EMA
        # Positive: Short-term trend above long-term (bullish)
        # Negative: Short-term trend below long-term (bearish)
        macd_line = ema_fast - ema_slow

        # Signal Line = 9-day EMA of MACD Line
        # Smooths out MACD to identify trend changes
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        # Histogram = MACD Line - Signal Line
        # Shows momentum strength and direction
        histogram = macd_line - signal_line

        # Store all components in DataFrame
        self.data['MACD'] = macd_line
        self.data['MACD_Signal'] = signal_line
        self.data['MACD_Histogram'] = histogram

        return pd.DataFrame({
            'MACD': macd_line,
            'MACD_Signal': signal_line,
            'MACD_Histogram': histogram
        })

    # =========================================================================
    # INDICATOR 3: Moving Averages (SMA 50 & 200)
    # =========================================================================
    def calculate_moving_averages(self, short: int = 50, long: int = 200) -> pd.DataFrame:
        """
        Calculate Simple Moving Averages (SMA).

        Moving averages smooth out price data to identify trends.
        - SMA_50: Short-term trend (about 2.5 months)
        - SMA_200: Long-term trend (about 10 months)

        Trading Signals:
        - Golden Cross: SMA_50 crosses ABOVE SMA_200 (bullish - uptrend starting)
        - Death Cross: SMA_50 crosses BELOW SMA_200 (bearish - downtrend starting)
        - Price > SMA_200: Long-term uptrend
        - Price < SMA_200: Long-term downtrend

        Args:
            short: Short-term MA period (standard: 50)
            long: Long-term MA period (standard: 200)

        Returns:
            DataFrame with SMA columns
        """
        close = self.data['Close']

        # Simple Moving Average = Sum of last N prices / N
        # Uses rolling window to calculate at each point
        sma_short = close.rolling(window=short).mean()
        sma_long = close.rolling(window=long).mean()

        # Store in DataFrame with descriptive names
        self.data[f'SMA_{short}'] = sma_short
        self.data[f'SMA_{long}'] = sma_long

        # Also calculate EMAs (some strategies prefer these)
        # EMA gives more weight to recent prices
        self.data[f'EMA_{short}'] = close.ewm(span=short, adjust=False).mean()
        self.data[f'EMA_{long}'] = close.ewm(span=long, adjust=False).mean()

        return pd.DataFrame({
            f'SMA_{short}': sma_short,
            f'SMA_{long}': sma_long
        })

    # =========================================================================
    # INDICATOR 4: Bollinger Bands
    # =========================================================================
    def calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.

        Bollinger Bands show price volatility and potential overbought/oversold levels.
        - Middle Band: 20-day SMA (the trend)
        - Upper Band: Middle + 2 standard deviations
        - Lower Band: Middle - 2 standard deviations

        About 95% of price action should stay within the bands.

        Trading Signals:
        - Price touches Upper Band: Potentially overbought (may reverse down)
        - Price touches Lower Band: Potentially oversold (may reverse up)
        - Bands contracting: Low volatility, big move coming (squeeze)
        - Bands expanding: High volatility, trend in progress

        Args:
            period: Lookback period for SMA and std dev (standard: 20)
            std_dev: Number of standard deviations for bands (standard: 2)

        Returns:
            DataFrame with BB_Upper, BB_Middle, BB_Lower, BB_Width
        """
        close = self.data['Close']

        # Middle band = Simple Moving Average
        middle = close.rolling(window=period).mean()

        # Calculate rolling standard deviation (measures volatility)
        rolling_std = close.rolling(window=period).std()

        # Upper band = Middle + (std_dev * standard deviation)
        upper = middle + (std_dev * rolling_std)

        # Lower band = Middle - (std_dev * standard deviation)
        lower = middle - (std_dev * rolling_std)

        # Bandwidth = (Upper - Lower) / Middle
        # Measures volatility as percentage of price
        # Low bandwidth = squeeze (volatility compression)
        bandwidth = (upper - lower) / middle

        # %B = Where price is within the bands (0 = lower, 1 = upper)
        # %B > 1: Price above upper band
        # %B < 0: Price below lower band
        percent_b = (close - lower) / (upper - lower)

        # Store all components
        self.data['BB_Upper'] = upper
        self.data['BB_Middle'] = middle
        self.data['BB_Lower'] = lower
        self.data['BB_Width'] = bandwidth
        self.data['BB_Percent_B'] = percent_b

        return pd.DataFrame({
            'BB_Upper': upper,
            'BB_Middle': middle,
            'BB_Lower': lower,
            'BB_Width': bandwidth,
            'BB_Percent_B': percent_b
        })

    # =========================================================================
    # INDICATOR 5: ATR (Average True Range)
    # =========================================================================
    def calculate_atr(self, period: int = 14) -> pd.Series:
        """
        Calculate ATR (Average True Range).

        ATR measures market volatility by analyzing price ranges.
        Does NOT indicate direction, only how much price is moving.

        True Range = Max of:
        1. Current High - Current Low (today's range)
        2. |Current High - Previous Close| (gap up)
        3. |Current Low - Previous Close| (gap down)

        ATR = Moving average of True Range

        Uses:
        - Position sizing: Higher ATR = smaller position size
        - Stop losses: Set stops at 2-3x ATR from entry
        - Volatility filter: Don't trade when ATR too low/high

        Args:
            period: Lookback period (standard: 14)

        Returns:
            ATR Series
        """
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']

        # Get previous close (shifted by 1 day)
        prev_close = close.shift(1)

        # Calculate the three components of True Range
        # 1. Today's High - Low (intraday range)
        hl = high - low

        # 2. |Today's High - Yesterday's Close| (captures gap ups)
        hc = (high - prev_close).abs()

        # 3. |Today's Low - Yesterday's Close| (captures gap downs)
        lc = (low - prev_close).abs()

        # True Range = Maximum of the three
        true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)

        # ATR = Exponential Moving Average of True Range
        # Using EMA to give more weight to recent volatility
        atr = true_range.ewm(span=period, adjust=False).mean()

        # Also calculate ATR as percentage of price (normalized volatility)
        # Useful for comparing volatility across different priced stocks
        atr_percent = (atr / close) * 100

        # Store in DataFrame
        self.data['ATR'] = atr
        self.data['ATR_Percent'] = atr_percent

        return atr

    # =========================================================================
    # INDICATOR 6: Volume Moving Average
    # =========================================================================
    def calculate_volume_ma(self, period: int = 20) -> pd.Series:
        """
        Calculate Volume Moving Average and Volume Ratio.

        Volume confirms price movements:
        - Price up + High volume: Strong move, likely to continue
        - Price up + Low volume: Weak move, may reverse
        - Price down + High volume: Strong selling pressure
        - Price down + Low volume: Weak selling, may bounce

        Components:
        - Volume_MA: 20-day average volume
        - Volume_Ratio: Today's volume / Average volume
            > 1.5: High volume (significant)
            > 2.0: Very high volume (major event)

        Args:
            period: Lookback period for volume average (standard: 20)

        Returns:
            Volume MA Series
        """
        volume = self.data['Volume']

        # Simple moving average of volume
        volume_ma = volume.rolling(window=period).mean()

        # Volume ratio = Current volume / Average volume
        # Shows if today's volume is above or below average
        volume_ratio = volume / volume_ma

        # Store in DataFrame
        self.data['Volume_MA'] = volume_ma
        self.data['Volume_Ratio'] = volume_ratio

        return volume_ma

    # =========================================================================
    # DERIVED SIGNALS (Useful combinations)
    # =========================================================================
    def _calculate_derived_signals(self):
        """
        Calculate useful derived signals from the base indicators.
        These are commonly used in trading strategies.
        """
        # Price vs Moving Averages
        # Positive = above MA (bullish), Negative = below MA (bearish)
        if 'SMA_50' in self.data.columns:
            self.data['Price_vs_SMA50'] = (
                (self.data['Close'] - self.data['SMA_50']) / self.data['SMA_50'] * 100
            )

        if 'SMA_200' in self.data.columns:
            self.data['Price_vs_SMA200'] = (
                (self.data['Close'] - self.data['SMA_200']) / self.data['SMA_200'] * 100
            )

        # Trend Direction based on MA crossover
        # 1 = Bullish (SMA50 > SMA200), 0 = Bearish (SMA50 < SMA200)
        if 'SMA_50' in self.data.columns and 'SMA_200' in self.data.columns:
            self.data['Trend'] = np.where(
                self.data['SMA_50'] > self.data['SMA_200'], 1, -1
            )

        # RSI Zones
        # 1 = Overbought, -1 = Oversold, 0 = Neutral
        if 'RSI' in self.data.columns:
            self.data['RSI_Zone'] = np.select(
                [
                    self.data['RSI'] > 70,  # Overbought
                    self.data['RSI'] < 30,  # Oversold
                ],
                [1, -1],
                default=0  # Neutral
            )

        # MACD Signal
        # 1 = Bullish (MACD > Signal), -1 = Bearish (MACD < Signal)
        if 'MACD' in self.data.columns and 'MACD_Signal' in self.data.columns:
            self.data['MACD_Bullish'] = np.where(
                self.data['MACD'] > self.data['MACD_Signal'], 1, -1
            )

        # Daily Returns (useful for various calculations)
        self.data['Daily_Return'] = self.data['Close'].pct_change() * 100

        # 52-Week High/Low (for breakout strategy)
        self.data['52W_High'] = self.data['High'].rolling(window=252).max()
        self.data['52W_Low'] = self.data['Low'].rolling(window=252).min()

        # Distance from 52-week high (percentage)
        self.data['Pct_From_52W_High'] = (
            (self.data['Close'] - self.data['52W_High']) / self.data['52W_High'] * 100
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    def get_latest_signals(self) -> dict:
        """
        Get the most recent indicator values.
        Useful for generating current trading signals.

        Returns:
            Dictionary with latest indicator values
        """
        if len(self.data) == 0:
            return {}

        latest = self.data.iloc[-1]

        signals = {
            'date': str(self.data.index[-1].date()),
            'close': round(latest['Close'], 2),
            'rsi': round(latest.get('RSI', np.nan), 2),
            'macd': round(latest.get('MACD', np.nan), 4),
            'macd_signal': round(latest.get('MACD_Signal', np.nan), 4),
            'macd_histogram': round(latest.get('MACD_Histogram', np.nan), 4),
            'sma_50': round(latest.get('SMA_50', np.nan), 2),
            'sma_200': round(latest.get('SMA_200', np.nan), 2),
            'bb_upper': round(latest.get('BB_Upper', np.nan), 2),
            'bb_lower': round(latest.get('BB_Lower', np.nan), 2),
            'bb_percent_b': round(latest.get('BB_Percent_B', np.nan), 2),
            'atr': round(latest.get('ATR', np.nan), 2),
            'atr_percent': round(latest.get('ATR_Percent', np.nan), 2),
            'volume_ratio': round(latest.get('Volume_Ratio', np.nan), 2),
            'trend': int(latest.get('Trend', 0)),
            'rsi_zone': int(latest.get('RSI_Zone', 0)),
        }

        return signals

    def get_indicator_summary(self) -> str:
        """
        Generate a human-readable summary of current indicators.
        Useful for reports and LLM prompts.

        Returns:
            Formatted string summary
        """
        signals = self.get_latest_signals()

        if not signals:
            return "No data available"

        # Interpret RSI
        rsi_val = signals['rsi']
        if rsi_val > 70:
            rsi_interp = "OVERBOUGHT"
        elif rsi_val < 30:
            rsi_interp = "OVERSOLD"
        else:
            rsi_interp = "NEUTRAL"

        # Interpret Trend
        trend_interp = "BULLISH" if signals['trend'] == 1 else "BEARISH"

        # Interpret MACD
        macd_interp = "BULLISH" if signals['macd_histogram'] > 0 else "BEARISH"

        # Interpret Bollinger %B
        bb_val = signals['bb_percent_b']
        if bb_val > 1:
            bb_interp = "ABOVE UPPER BAND"
        elif bb_val < 0:
            bb_interp = "BELOW LOWER BAND"
        else:
            bb_interp = f"{bb_val:.0%} between bands"

        summary = f"""
=== TECHNICAL INDICATOR SUMMARY ===
Date: {signals['date']}
Price: ${signals['close']}

TREND INDICATORS:
  SMA 50:  ${signals['sma_50']} (Short-term trend)
  SMA 200: ${signals['sma_200']} (Long-term trend)
  Trend:   {trend_interp} (50 vs 200 SMA)

MOMENTUM INDICATORS:
  RSI:     {signals['rsi']} ({rsi_interp})
  MACD:    {signals['macd']:.4f}
  Signal:  {signals['macd_signal']:.4f}
  MACD:    {macd_interp}

VOLATILITY INDICATORS:
  ATR:     ${signals['atr']} ({signals['atr_percent']:.1f}% of price)
  BB Upper: ${signals['bb_upper']}
  BB Lower: ${signals['bb_lower']}
  Position: {bb_interp}

VOLUME:
  Volume Ratio: {signals['volume_ratio']:.2f}x average
"""
        return summary


# =============================================================================
# TEST SCRIPT
# =============================================================================
if __name__ == "__main__":
    """
    Test script to calculate indicators on sample data.
    Usage: python src/technical_indicators.py
    """
    from data_collector import DataCollector

    print("=" * 60)
    print("TECHNICAL INDICATORS TEST")
    print("=" * 60)

    # Load data using data collector
    collector = DataCollector()
    data = collector.get_data("AAPL", years=10)

    # Create indicator calculator
    indicators = TechnicalIndicators(data)

    # Calculate all indicators
    data_with_indicators = indicators.calculate_all()

    # Show latest signals
    print("\n" + "=" * 60)
    print("LATEST INDICATOR VALUES")
    print("=" * 60)

    signals = indicators.get_latest_signals()
    for key, value in signals.items():
        print(f"  {key}: {value}")

    # Print human-readable summary
    print(indicators.get_indicator_summary())

    # Show data structure
    print("\n" + "=" * 60)
    print("DATA STRUCTURE")
    print("=" * 60)
    print(f"Total columns: {len(data_with_indicators.columns)}")
    print(f"Columns: {list(data_with_indicators.columns)}")

    # Show sample of data with indicators
    print("\n" + "=" * 60)
    print("SAMPLE DATA (Last 5 rows, key columns)")
    print("=" * 60)

    key_cols = ['Close', 'RSI', 'MACD', 'SMA_50', 'SMA_200', 'ATR', 'Volume_Ratio']
    available_cols = [col for col in key_cols if col in data_with_indicators.columns]
    print(data_with_indicators[available_cols].tail())

    # Show NaN counts (first rows have NaN due to lookback periods)
    print("\n" + "=" * 60)
    print("NaN COUNTS (expected for initial rows)")
    print("=" * 60)
    nan_counts = data_with_indicators.isnull().sum()
    print(nan_counts[nan_counts > 0])
