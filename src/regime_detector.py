"""
Regime Detection Module
Statistical methods for identifying market regimes:
1. Hurst Exponent - Trend persistence measurement (R/S analysis)
2. Variance Change Detection - Volatility regime shifts
3. Parkinson Volatility - Volatility clustering detection
4. Trend Strength Classification - ADX + MA + momentum combined

Outputs 5 regime classifications:
- STRONG_BULL, MODERATE_BULL, SIDEWAYS, MODERATE_BEAR, STRONG_BEAR
Plus volatility states: HIGH_VOLATILITY, LOW_VOLATILITY, NORMAL_VOLATILITY
And trend persistence: TRENDING, MEAN_REVERTING, RANDOM_WALK
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    """Market direction regimes"""
    STRONG_BULL = "STRONG_BULL"
    MODERATE_BULL = "MODERATE_BULL"
    SIDEWAYS = "SIDEWAYS"
    MODERATE_BEAR = "MODERATE_BEAR"
    STRONG_BEAR = "STRONG_BEAR"
    CHOPPY = "CHOPPY"


class VolatilityRegime(Enum):
    """Volatility state regimes"""
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    NORMAL_VOLATILITY = "NORMAL_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"


class TrendPersistence(Enum):
    """Trend persistence based on Hurst exponent"""
    TRENDING = "TRENDING"           # H > 0.55 - persistent trends
    RANDOM_WALK = "RANDOM_WALK"     # 0.45 < H < 0.55 - random
    MEAN_REVERTING = "MEAN_REVERTING"  # H < 0.45 - mean reverting


@dataclass
class RegimeState:
    """Complete regime state at a point in time"""
    market_regime: MarketRegime
    volatility_regime: VolatilityRegime
    trend_persistence: TrendPersistence
    hurst_exponent: float
    trend_strength: float  # 0-100 based on ADX
    volatility_percentile: float  # 0-100
    regime_confidence: float  # 0-1 confidence in classification

    def to_dict(self) -> dict:
        return {
            'market_regime': self.market_regime.value,
            'volatility_regime': self.volatility_regime.value,
            'trend_persistence': self.trend_persistence.value,
            'hurst_exponent': self.hurst_exponent,
            'trend_strength': self.trend_strength,
            'volatility_percentile': self.volatility_percentile,
            'regime_confidence': self.regime_confidence
        }


class RegimeDetector:
    """
    Regime detection using multiple statistical methods.

    Methods:
    1. Hurst Exponent (R/S Analysis) - Measures trend persistence
       - H > 0.5: Trending (momentum strategies work)
       - H = 0.5: Random walk (no edge)
       - H < 0.5: Mean reverting (contrarian strategies work)

    2. Variance Change Detection - Identifies volatility regime shifts
       - Uses rolling variance with statistical significance testing
       - Detects sustained changes vs temporary spikes

    3. Parkinson Volatility - High/Low based volatility estimator
       - More efficient than close-to-close volatility
       - Better at detecting intraday volatility clustering

    4. Trend Strength Classification - Multi-indicator trend measurement
       - Combines ADX, MA relationships, and price momentum
       - Classifies into 5 market regimes
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with OHLCV data that has technical indicators calculated.

        Args:
            data: DataFrame with OHLCV + technical indicators
                  Required: Open, High, Low, Close, Volume
                  Optional but recommended: ADX, SMA_50, SMA_200, ATR
        """
        self.data = data.copy()
        self._validate_data()

    def _validate_data(self):
        """Validate required columns exist"""
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    # =========================================================================
    # METHOD 1: HURST EXPONENT (R/S Analysis)
    # =========================================================================
    def calculate_hurst_exponent(self, window: int = 200, min_window: int = 20) -> pd.Series:
        """
        Calculate rolling Hurst Exponent using Rescaled Range (R/S) analysis.

        The Hurst exponent measures the long-term memory of a time series:
        - H > 0.5: Persistent/trending behavior (trends tend to continue)
        - H = 0.5: Random walk (no predictable pattern)
        - H < 0.5: Anti-persistent/mean-reverting (trends tend to reverse)

        R/S Analysis Method:
        1. For each sub-period of length n:
           - Calculate mean of returns
           - Calculate cumulative deviation from mean
           - R = max(cumulative) - min(cumulative) (Range)
           - S = standard deviation of returns
           - R/S ratio for that period
        2. Regress log(R/S) against log(n) for various n
        3. Hurst exponent H = slope of regression

        Args:
            window: Lookback window for Hurst calculation (default 100 days)
            min_window: Minimum sub-window size for R/S calculation

        Returns:
            Series of rolling Hurst exponent values
        """
        returns = self.data['Close'].pct_change().dropna()
        hurst_values = pd.Series(index=self.data.index, dtype=float)

        for i in range(window, len(returns) + 1):
            # Get the window of returns
            window_returns = returns.iloc[i-window:i].values

            # Calculate Hurst for this window
            h = self._calculate_hurst_rs(window_returns, min_window)
            hurst_values.iloc[i] = h

        self.data['Hurst_Exponent'] = hurst_values
        return hurst_values

    def _calculate_hurst_rs(self, returns: np.ndarray, min_window: int = 20) -> float:
        """
        Calculate Hurst exponent for a single window using R/S analysis.

        Args:
            returns: Array of returns
            min_window: Minimum sub-window size

        Returns:
            Hurst exponent value
        """
        n = len(returns)
        if n < min_window * 2:
            return np.nan

        # Generate different window sizes (powers of 2 work well)
        max_k = int(np.log2(n // min_window))
        if max_k < 1:
            return np.nan

        window_sizes = [min_window * (2 ** i) for i in range(max_k + 1) if min_window * (2 ** i) <= n // 2]

        if len(window_sizes) < 2:
            return np.nan

        rs_values = []

        for window_size in window_sizes:
            # Number of non-overlapping windows
            num_windows = n // window_size
            rs_window = []

            for j in range(num_windows):
                start = j * window_size
                end = start + window_size
                sub_returns = returns[start:end]

                # Calculate mean-adjusted cumulative sum
                mean_return = np.mean(sub_returns)
                cumulative_dev = np.cumsum(sub_returns - mean_return)

                # Range (R)
                R = np.max(cumulative_dev) - np.min(cumulative_dev)

                # Standard deviation (S)
                S = np.std(sub_returns, ddof=1)

                if S > 0:
                    rs_window.append(R / S)

            if rs_window:
                rs_values.append((window_size, np.mean(rs_window)))

        if len(rs_values) < 2:
            return np.nan

        # Linear regression of log(R/S) vs log(n)
        log_n = np.log([x[0] for x in rs_values])
        log_rs = np.log([x[1] for x in rs_values])

        # Simple linear regression: H = slope
        slope, _ = np.polyfit(log_n, log_rs, 1)

        # Hurst should be between 0 and 1
        return np.clip(slope, 0, 1)

    def get_trend_persistence(self, hurst: float) -> TrendPersistence:
        """
        Classify trend persistence based on Hurst exponent.

        Args:
            hurst: Hurst exponent value

        Returns:
            TrendPersistence enum value
        """
        if np.isnan(hurst):
            return TrendPersistence.RANDOM_WALK
        elif hurst > 0.55:
            return TrendPersistence.TRENDING
        elif hurst < 0.45:
            return TrendPersistence.MEAN_REVERTING
        else:
            return TrendPersistence.RANDOM_WALK

    # =========================================================================
    # METHOD 2: VARIANCE CHANGE DETECTION
    # =========================================================================
    def detect_variance_changes(self, short_window: int = 20, long_window: int = 60,
                                threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect volatility regime shifts using rolling variance analysis.

        Compares short-term variance to long-term variance to identify
        statistically significant changes in volatility regime.
  
        Method:
        1. Calculate short-term rolling variance (recent volatility)
        2. Calculate long-term rolling variance (baseline volatility)
        3. Variance ratio = short / long
        4. High ratio (>1.5) = volatility expansion
        5. Low ratio (<0.67) = volatility contraction

        Args:
            short_window: Short-term variance window (default 20 days)
            long_window: Long-term variance window (default 60 days)
            threshold: Ratio threshold for regime change (default 1.5)

        Returns:
            DataFrame with variance metrics
        """
        returns = self.data['Close'].pct_change()

        # Calculate rolling variances
        short_var = returns.rolling(window=short_window).var()
        long_var = returns.rolling(window=long_window).var()

        # Variance ratio
        var_ratio = short_var / long_var

        # Detect regime changes
        # High variance regime when short-term vol >> long-term vol
        high_vol_signal = var_ratio > threshold
        low_vol_signal = var_ratio < (1 / threshold)

        # Z-score of variance for statistical significance
        var_zscore = (short_var - long_var.rolling(window=long_window).mean()) / \
                     long_var.rolling(window=long_window).std()

        # Store results
        self.data['Variance_Ratio'] = var_ratio
        self.data['Variance_ZScore'] = var_zscore
        self.data['High_Vol_Signal'] = high_vol_signal.astype(int)
        self.data['Low_Vol_Signal'] = low_vol_signal.astype(int)

        return pd.DataFrame({
            'Variance_Ratio': var_ratio,
            'Variance_ZScore': var_zscore,
            'High_Vol_Signal': high_vol_signal,
            'Low_Vol_Signal': low_vol_signal
        })

    # =========================================================================
    # METHOD 3: PARKINSON VOLATILITY (High-Low Estimator)
    # =========================================================================
    def calculate_parkinson_volatility(self, window: int = 20) -> pd.Series:
        """
        Calculate Parkinson volatility estimator using High-Low range.

        Parkinson (1980) volatility is more efficient than close-to-close
        volatility because it uses intraday price range information.

        Formula:
            σ² = (1 / 4ln(2)) * (ln(H/L))²

        This estimator:
        - Is 5x more efficient than close-to-close for continuous processes
        - Better captures intraday volatility
        - Detects volatility clustering more effectively

        Args:
            window: Rolling window for volatility calculation

        Returns:
            Series of Parkinson volatility values (annualized)
        """
        high = self.data['High']
        low = self.data['Low']

        # Parkinson volatility formula
        log_hl = np.log(high / low)
        parkinson_var = (1 / (4 * np.log(2))) * (log_hl ** 2)

        # Rolling average and annualize (sqrt(252) for daily data)
        parkinson_vol = np.sqrt(parkinson_var.rolling(window=window).mean() * 252)

        # Calculate percentile rank for regime classification
        vol_percentile = parkinson_vol.rolling(window=252, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
        )

        self.data['Parkinson_Volatility'] = parkinson_vol
        self.data['Volatility_Percentile'] = vol_percentile

        return parkinson_vol

    def get_volatility_regime(self, vol_percentile: float) -> VolatilityRegime:
        """
        Classify volatility regime based on percentile rank.

        Args:
            vol_percentile: Volatility percentile (0-100)

        Returns:
            VolatilityRegime enum value
        """
        if np.isnan(vol_percentile):
            return VolatilityRegime.NORMAL_VOLATILITY
        elif vol_percentile > 75:
            return VolatilityRegime.HIGH_VOLATILITY
        elif vol_percentile < 25:
            return VolatilityRegime.LOW_VOLATILITY
        else:
            return VolatilityRegime.NORMAL_VOLATILITY

    # =========================================================================
    # METHOD 4: TREND STRENGTH CLASSIFICATION
    # =========================================================================
    def calculate_trend_strength(self, ma_short: int = 50, ma_long: int = 200,
                                  momentum_window: int = 20) -> pd.DataFrame:
        """
        Calculate comprehensive trend strength using multiple indicators.

        Combines:
        1. ADX (if available) - Direct trend strength measurement
        2. MA Relationship - Price position relative to moving averages
        3. MA Slope - Direction and steepness of trend
        4. Price Momentum - Rate of change

        Outputs a composite trend score from -100 (strong bear) to +100 (strong bull)

        Args:
            ma_short: Short-term MA period (default 50)
            ma_long: Long-term MA period (default 200)
            momentum_window: Momentum calculation window (default 20)

        Returns:
            DataFrame with trend strength metrics
        """
        close = self.data['Close']

        # Calculate MAs if not present
        if f'SMA_{ma_short}' not in self.data.columns:
            self.data[f'SMA_{ma_short}'] = close.rolling(window=ma_short).mean()
        if f'SMA_{ma_long}' not in self.data.columns:
            self.data[f'SMA_{ma_long}'] = close.rolling(window=ma_long).mean()

        sma_short = self.data[f'SMA_{ma_short}']
        sma_long = self.data[f'SMA_{ma_long}']

        # 1. Price vs MA score (-1 to +1 for each)
        price_vs_short = np.where(close > sma_short, 1, -1)
        price_vs_long = np.where(close > sma_long, 1, -1)

        # 2. MA crossover score
        ma_crossover = np.where(sma_short > sma_long, 1, -1)

        # 3. MA slope (rate of change of long MA)
        ma_slope = sma_long.pct_change(periods=momentum_window) * 100
        ma_slope_normalized = np.clip(ma_slope / 5, -1, 1)  # Normalize to -1 to +1

        # 4. Price momentum (rate of change)
        momentum = close.pct_change(periods=momentum_window) * 100
        momentum_normalized = np.clip(momentum / 10, -1, 1)  # Normalize to -1 to +1

        # 5. ADX score (if available)
        if 'ADX' in self.data.columns:
            adx = self.data['ADX']
            adx_factor = np.clip(adx / 50, 0, 1)  # 0 to 1 based on ADX

            # Direction from +DI/-DI if available
            if 'Plus_DI' in self.data.columns and 'Minus_DI' in self.data.columns:
                di_direction = np.where(
                    self.data['Plus_DI'] > self.data['Minus_DI'], 1, -1
                )
            else:
                di_direction = ma_crossover
        else:
            adx_factor = np.ones(len(close)) * 0.5
            di_direction = ma_crossover

        # Composite trend score (-100 to +100)
        # Weight: ADX direction (25%), Price vs MAs (15%), MA crossover (15%), Momentum (20%)
        trend_direction = (
            0.25 * di_direction +
            0.15 * price_vs_short +
            0.15 * price_vs_long +
            0.15 * ma_slope_normalized +
            0.15 * ma_crossover +
            0.15 * momentum_normalized
        )

        # Scale by ADX strength (stronger ADX = more confident in direction)
        trend_score = trend_direction * (0.5 + 0.5 * adx_factor) * 100

        # Store results
        self.data['Trend_Score'] = trend_score
        self.data['Trend_Direction'] = np.sign(trend_score)

        return pd.DataFrame({
            'Trend_Score': trend_score,
            'Price_vs_Short_MA': price_vs_short,
            'Price_vs_Long_MA': price_vs_long,
            'MA_Crossover': ma_crossover,
            'Momentum': momentum,
            'ADX_Factor': adx_factor
        })

    def get_market_regime(self, trend_score: float, volatility_regime: VolatilityRegime,
                          hurst: float) -> MarketRegime:
        """
        Classify market regime for a SINGLE observation.
    
        Note: For bulk classification of entire DataFrame, use _classify_regimes()
        which is 100x faster via vectorization. This function is useful for:
        - Unit testing
        - Manual classification
        - Real-time single-point classification
        
        Args:
            trend_score: Composite trend score (-100 to +100)
            volatility_regime: Current volatility regime
            hurst: Hurst exponent
        
        Returns:
            MarketRegime enum value
        """
        if np.isnan(trend_score):
            return MarketRegime.SIDEWAYS

        # Check for choppy market (low Hurst + high volatility)
        if hurst < 0.45 and volatility_regime == VolatilityRegime.HIGH_VOLATILITY:
            return MarketRegime.CHOPPY

        # Classify based on trend score
        if trend_score > 50:
            return MarketRegime.STRONG_BULL
        elif trend_score > 20:
            return MarketRegime.MODERATE_BULL
        elif trend_score < -50:
            return MarketRegime.STRONG_BEAR
        elif trend_score < -20:
            return MarketRegime.MODERATE_BEAR
        else:
            return MarketRegime.SIDEWAYS

    # =========================================================================
    # MAIN DETECTION METHOD
    # =========================================================================
    def detect_all_regimes(self, hurst_window: int = 100,
                           variance_short: int = 20,
                           variance_long: int = 60,
                           parkinson_window: int = 20) -> pd.DataFrame:
        """
        Run all regime detection methods and combine results.

        This is the main method to call - it calculates all regime indicators
        and adds them to the DataFrame.

        Args:
            hurst_window: Window for Hurst exponent calculation
            variance_short: Short window for variance change detection
            variance_long: Long window for variance change detection
            parkinson_window: Window for Parkinson volatility

        Returns:
            DataFrame with all regime indicators added
        """
        print("\n Detecting Market Regimes...")

        # 1. Calculate Hurst Exponent
        print("  Calculating Hurst Exponent (R/S analysis)...")
        self.calculate_hurst_exponent(window=hurst_window)

        # 2. Detect Variance Changes
        print("  Detecting variance regime shifts...")
        self.detect_variance_changes(short_window=variance_short,
                                     long_window=variance_long)

        # 3. Calculate Parkinson Volatility
        print("  Calculating Parkinson volatility...")
        self.calculate_parkinson_volatility(window=parkinson_window)

        # 4. Calculate Trend Strength
        print("  Calculating trend strength...")
        self.calculate_trend_strength()

        # 5. Combine into final regime classifications
        print("  Combining regime classifications...")
        self._classify_regimes()

        print(f" Regime detection complete")
        print(f"  Added columns: Hurst_Exponent, Parkinson_Volatility, Trend_Score, etc.")

        return self.data

    def _classify_regimes(self):
        """
        Combine all indicators into final regime classifications.
        Uses vectorised operations for speed.
        """
        # Get all values as arrays (much faster than row-by-row)
        hurst = self.data['Hurst_Exponent'].values
        vol_pct = self.data['Volatility_Percentile'].values
        trend_score = self.data['Trend_Score'].values if 'Trend_Score' in self.data.columns else np.zeros(len(self.data))
        
        # 1. Trend Persistence (vectorised)
        trend_persistence = np.where(
            hurst > 0.55, TrendPersistence.TRENDING.value,
            np.where(hurst < 0.45, TrendPersistence.MEAN_REVERTING.value,
                    TrendPersistence.RANDOM_WALK.value)
        )
        
        # 2. Volatility Regime (vectorised)
        volatility_regime = np.where(
            vol_pct > 75, VolatilityRegime.HIGH_VOLATILITY.value,
            np.where(vol_pct < 25, VolatilityRegime.LOW_VOLATILITY.value,
                    VolatilityRegime.NORMAL_VOLATILITY.value)
        )
        
        # 3. Market Regime (vectorised)
        # First check for CHOPPY (low Hurst + high volatility)
        is_choppy = (hurst < 0.45) & (vol_pct > 75)
        
        market_regime = np.where(
            is_choppy, MarketRegime.CHOPPY.value,
            np.where(trend_score > 50, MarketRegime.STRONG_BULL.value,
            np.where(trend_score > 20, MarketRegime.MODERATE_BULL.value,
            np.where(trend_score < -50, MarketRegime.STRONG_BEAR.value,
            np.where(trend_score < -20, MarketRegime.MODERATE_BEAR.value,
                    MarketRegime.SIDEWAYS.value))))
        )
        
        # 4. Confidence (vectorised)
        confidence = self._calculate_confidence_vectorised()
        
        # Assign all at once (fast)
        self.data['Market_Regime'] = market_regime
        self.data['Volatility_Regime'] = volatility_regime
        self.data['Trend_Persistence'] = trend_persistence
        self.data['Regime_Confidence'] = confidence

    def _calculate_confidence_vectorised(self) -> np.ndarray:
        """
        Calculate confidence scores for all rows at once.
        """
        confidence_sum = np.zeros(len(self.data))
        confidence_count = np.zeros(len(self.data))
        
        # ADX confidence
        if 'ADX' in self.data.columns:
            adx = self.data['ADX'].values
            adx_conf = np.clip(adx / 40, 0, 1)
            valid = ~np.isnan(adx)
            confidence_sum[valid] += adx_conf[valid]
            confidence_count[valid] += 1
        
        # Hurst confidence
        hurst = self.data['Hurst_Exponent'].values
        hurst_conf = np.abs(hurst - 0.5) * 2
        valid = ~np.isnan(hurst)
        confidence_sum[valid] += hurst_conf[valid]
        confidence_count[valid] += 1
        
        # Variance ratio confidence
        if 'Variance_Ratio' in self.data.columns:
            var_ratio = self.data['Variance_Ratio'].values
            var_conf = np.minimum(np.abs(np.log(var_ratio)) / 1.5, 1)
            valid = ~np.isnan(var_ratio) & (var_ratio > 0)
            confidence_sum[valid] += var_conf[valid]
            confidence_count[valid] += 1
        
        # Trend score confidence
        if 'Trend_Score' in self.data.columns:
            trend = self.data['Trend_Score'].values
            trend_conf = np.abs(trend) / 100
            valid = ~np.isnan(trend)
            confidence_sum[valid] += trend_conf[valid]
            confidence_count[valid] += 1
        
        # Average (avoid division by zero)
        confidence_count = np.maximum(confidence_count, 1)
        return confidence_sum / confidence_count

    # =========================================================================
    # OUTPUT METHODS
    # =========================================================================
    def get_current_regime(self) -> RegimeState:
        """
        Get the current (most recent) regime state.

        Returns:
            RegimeState dataclass with all regime information
        """
        if len(self.data) == 0:
            raise ValueError("No data available")

        latest = self.data.iloc[-1]

        return RegimeState(
            market_regime=MarketRegime(latest.get('Market_Regime', 'SIDEWAYS')),
            volatility_regime=VolatilityRegime(latest.get('Volatility_Regime', 'NORMAL_VOLATILITY')),
            trend_persistence=TrendPersistence(latest.get('Trend_Persistence', 'RANDOM_WALK')),
            hurst_exponent=latest.get('Hurst_Exponent', 0.5),
            trend_strength=abs(latest.get('Trend_Score', 0)),
            volatility_percentile=latest.get('Volatility_Percentile', 50),
            regime_confidence=latest.get('Regime_Confidence', 0.5)
        )

    def get_regime_summary(self) -> str:
        """
        Generate a human-readable summary of the current regime.

        Returns:
            Formatted string summary
        """
        try:
            state = self.get_current_regime()
        except ValueError:
            return "No regime data available"

        # Interpret Hurst
        if state.hurst_exponent > 0.6:
            hurst_interp = "Strong trending behavior - momentum strategies favored"
        elif state.hurst_exponent > 0.55:
            hurst_interp = "Moderate trending - trend-following may work"
        elif state.hurst_exponent < 0.4:
            hurst_interp = "Strong mean reversion - contrarian strategies favored"
        elif state.hurst_exponent < 0.45:
            hurst_interp = "Moderate mean reversion - fade extremes"
        else:
            hurst_interp = "Random walk - no clear edge"

        # Strategy recommendation
        if state.market_regime in [MarketRegime.STRONG_BULL, MarketRegime.MODERATE_BULL]:
            if state.trend_persistence == TrendPersistence.TRENDING:
                strategy = "TREND-FOLLOWING (Long bias with momentum)"
            else:
                strategy = "CAUTIOUS LONG (Bull market but choppy)"
        elif state.market_regime in [MarketRegime.STRONG_BEAR, MarketRegime.MODERATE_BEAR]:
            if state.trend_persistence == TrendPersistence.TRENDING:
                strategy = "TREND-FOLLOWING (Short bias or defensive)"
            else:
                strategy = "DEFENSIVE (Bear market, watch for bounces)"
        elif state.market_regime == MarketRegime.CHOPPY:
            strategy = "REDUCE EXPOSURE (High vol, no trend)"
        else:  # SIDEWAYS
            if state.trend_persistence == TrendPersistence.MEAN_REVERTING:
                strategy = "MEAN-REVERSION (Range trading)"
            else:
                strategy = "NEUTRAL (Wait for clearer signal)"

        # Volatility adjustment
        if state.volatility_regime == VolatilityRegime.HIGH_VOLATILITY:
            vol_adjust = "REDUCE POSITION SIZE (High volatility)"
        elif state.volatility_regime == VolatilityRegime.LOW_VOLATILITY:
            vol_adjust = "Can increase position size (Low volatility)"
        else:
            vol_adjust = "Normal position sizing"

        summary = f"""
            === REGIME DETECTION SUMMARY ===
            Date: {self.data.index[-1].date() if hasattr(self.data.index[-1], 'date') else 'N/A'}

            MARKET REGIME: {state.market_regime.value}
            Trend Strength: {state.trend_strength:.1f}/100
            Confidence: {state.regime_confidence:.1%}

            VOLATILITY REGIME: {state.volatility_regime.value}
            Volatility Percentile: {state.volatility_percentile:.1f}%

            TREND PERSISTENCE: {state.trend_persistence.value}
            Hurst Exponent: {state.hurst_exponent:.3f}
            Interpretation: {hurst_interp}

            STRATEGY RECOMMENDATION:
            Primary: {strategy}
            Position Sizing: {vol_adjust}
            """
        return summary

    def get_regime_probabilities(self) -> Dict[str, float]:
        """
        Get deterministic probability distribution across regimes.

        Based on trend score distribution, returns probability-like
        weights for each regime.

        Returns:
            Dictionary with regime probabilities
        """
        if 'Trend_Score' not in self.data.columns:
            return {}

        trend_score = self.data['Trend_Score'].iloc[-1]

        if np.isnan(trend_score):
            return {regime.value: 0.2 for regime in MarketRegime}

        # Convert trend score to probability distribution using softmax-like logic
        # Center points for each regime
        regime_centers = {
            MarketRegime.STRONG_BULL: 75,
            MarketRegime.MODERATE_BULL: 35,
            MarketRegime.SIDEWAYS: 0,
            MarketRegime.MODERATE_BEAR: -35,
            MarketRegime.STRONG_BEAR: -75,
        }

        # Calculate distance-based weights
        weights = {}
        for regime, center in regime_centers.items():
            distance = abs(trend_score - center)
            weights[regime.value] = np.exp(-distance / 30)  # Decay factor

        # Normalize to sum to 1
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def get_regime_history(self, last_n: int = 30) -> pd.DataFrame:
        """
        Get regime history for the last N periods.

        Args:
            last_n: Number of periods to return

        Returns:
            DataFrame with regime history
        """
        cols = ['Market_Regime', 'Volatility_Regime', 'Trend_Persistence',
                'Hurst_Exponent', 'Trend_Score', 'Regime_Confidence']
        available_cols = [c for c in cols if c in self.data.columns]

        return self.data[available_cols].tail(last_n)


# =============================================================================
# TEST SCRIPT
# =============================================================================
if __name__ == "__main__":
    """
    Test script for regime detection.
    Usage: python src/regime_detector.py
    """
    from data_collector import DataCollector
    from technical_indicators import TechnicalIndicators

    print("=" * 60)
    print("REGIME DETECTOR TEST")
    print("=" * 60)

    # Load data
    collector = DataCollector()
    data = collector.get_data("AAPL", years=10)

    # Calculate technical indicators first (regime detector uses ADX)
    indicators = TechnicalIndicators(data)
    data_with_indicators = indicators.calculate_all()

    # Create regime detector
    detector = RegimeDetector(data_with_indicators)

    # Run all regime detection
    data_with_regimes = detector.detect_all_regimes()

    # Print current regime summary
    print(detector.get_regime_summary())

    # Show regime probabilities
    print("\n" + "=" * 60)
    print("REGIME PROBABILITIES")
    print("=" * 60)
    probs = detector.get_regime_probabilities()
    for regime, prob in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"  {regime}: {prob:.1%}")

    # Show recent regime history
    print("\n" + "=" * 60)
    print("RECENT REGIME HISTORY (Last 10 days)")
    print("=" * 60)
    history = detector.get_regime_history(10)
    print(history.to_string())

    # Show data structure
    print("\n" + "=" * 60)
    print("NEW COLUMNS ADDED")
    print("=" * 60)
    new_cols = ['Hurst_Exponent', 'Parkinson_Volatility', 'Volatility_Percentile',
                'Variance_Ratio', 'Trend_Score', 'Market_Regime', 'Volatility_Regime',
                'Trend_Persistence', 'Regime_Confidence']
    for col in new_cols:
        if col in data_with_regimes.columns:
            latest = data_with_regimes[col].iloc[-1]
            print(f"  {col}: {latest}")
