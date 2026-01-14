"""
Strategy Comparison Module
Multi-strategy framework for running, comparing, and combining trading strategies.

Features:
1. Multi-Strategy Framework - Runs 4 strategies in parallel:
   - Momentum (RSI + MACD)
   - Trend-Following (MA crossover)
   - Mean-Reversion (Bollinger Bands)
   - Breakout (52-week high + volume)

2. Walk-Forward Testing - Independent validation per strategy
3. Strategy Ranking - By Sharpe, CAGR, Max Drawdown, Win Rate
4. Ensemble Strategy - Combines top 2 strategies with dynamic weighting
5. Regime-Specific Performance - Shows which strategy works best per regime
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import vectorbt as vbt

# Import thresholds from signal_generator for consistent indicator logic
from signal_generator import SignalGenerator

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

# Use same thresholds as signal_generator for consistency
THRESHOLDS = SignalGenerator.THRESHOLDS


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================
class StrategyType(Enum):
    """Available trading strategy types."""
    MOMENTUM = "momentum"           # RSI + MACD
    TREND_FOLLOWING = "trend_following"  # MA crossover
    MEAN_REVERSION = "mean_reversion"    # Bollinger Bands
    BREAKOUT = "breakout"           # 52-week high + volume


@dataclass
class StrategyConfig:
    """Configuration for individual strategy parameters."""
    # Momentum strategy parameters
    momentum_rsi_oversold: int = 30
    momentum_rsi_overbought: int = 70
    momentum_macd_threshold: float = 0.0

    # Trend-following parameters
    trend_fast_ma: int = 20
    trend_slow_ma: int = 50
    trend_adx_threshold: float = 25.0

    # Mean-reversion parameters
    mr_bb_lower: float = 0.0    # Buy below lower band
    mr_bb_upper: float = 1.0    # Sell above upper band
    mr_rsi_oversold: int = 25
    mr_rsi_overbought: int = 75

    # Breakout parameters
    breakout_lookback: int = 252  # 52 weeks
    breakout_volume_mult: float = 1.5  # Volume must be 1.5x average
    breakout_atr_mult: float = 0.5  # Entry above high by 0.5 ATR


@dataclass
class StrategyResult:
    """Results from a single strategy backtest."""
    strategy_type: StrategyType

    # Performance metrics
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float

    # Trade statistics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_duration: float

    # Equity and signals
    equity_curve: pd.Series = field(default_factory=pd.Series)
    signals: pd.Series = field(default_factory=pd.Series)

    # Walk-forward results (if run)
    wf_efficiency: Optional[float] = None
    oos_sharpe: Optional[float] = None
    is_robust: Optional[bool] = None

    # Regime performance (dict of regime -> metrics)
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'strategy': self.strategy_type.value,
            'total_return': self.total_return,
            'cagr': self.cagr,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_trade_duration': self.avg_trade_duration,
            'wf_efficiency': self.wf_efficiency,
            'oos_sharpe': self.oos_sharpe,
            'is_robust': self.is_robust,
            'regime_performance': self.regime_performance
        }


@dataclass
class EnsembleResult:
    """Results from ensemble strategy combining multiple strategies."""
    # Component strategies
    strategy_1: StrategyType
    strategy_2: StrategyType
    weights: Tuple[float, float]
    weighting_method: str

    # Combined performance
    total_return: float
    cagr: float
    sharpe_ratio: float
    max_drawdown: float

    # Comparison to components
    improvement_vs_best: float  # % improvement over best single strategy

    # Equity curve
    equity_curve: pd.Series = field(default_factory=pd.Series)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategies': [self.strategy_1.value, self.strategy_2.value],
            'weights': list(self.weights),
            'weighting_method': self.weighting_method,
            'total_return': self.total_return,
            'cagr': self.cagr,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'improvement_vs_best': self.improvement_vs_best
        }


@dataclass
class ComparisonResult:
    """Complete comparison results across all strategies."""
    # Individual strategy results
    strategy_results: Dict[StrategyType, StrategyResult]

    # Rankings by different metrics
    ranking_by_sharpe: List[StrategyType]
    ranking_by_cagr: List[StrategyType]
    ranking_by_max_dd: List[StrategyType]
    ranking_by_win_rate: List[StrategyType]

    # Best strategies per regime
    best_by_regime: Dict[str, StrategyType]

    # Ensemble result
    ensemble: Optional[EnsembleResult] = None

    # Overall recommendation
    recommended_strategy: Optional[StrategyType] = None
    recommendation_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'individual_results': {k.value: v.to_dict() for k, v in self.strategy_results.items()},
            'rankings': {
                'by_sharpe': [s.value for s in self.ranking_by_sharpe],
                'by_cagr': [s.value for s in self.ranking_by_cagr],
                'by_max_drawdown': [s.value for s in self.ranking_by_max_dd],
                'by_win_rate': [s.value for s in self.ranking_by_win_rate]
            },
            'best_by_regime': {k: v.value for k, v in self.best_by_regime.items()},
            'ensemble': self.ensemble.to_dict() if self.ensemble else None,
            'recommendation': {
                'strategy': self.recommended_strategy.value if self.recommended_strategy else None,
                'reason': self.recommendation_reason
            }
        }


# =============================================================================
# INDIVIDUAL STRATEGY IMPLEMENTATIONS
# =============================================================================
class BaseStrategy:
    """Base class for trading strategies."""

    def __init__(self, data: pd.DataFrame, config: Optional[StrategyConfig] = None):
        """
        Initialize strategy with data and configuration.

        Args:
            data: DataFrame with OHLCV and technical indicators
            config: Strategy configuration parameters
        """
        self.data = data.copy()
        self.config = config or StrategyConfig()
        self.signals = None

    def generate_signals(self) -> pd.Series:
        """Generate trading signals. Override in subclasses."""
        raise NotImplementedError

    def _validate_columns(self, required: List[str]) -> bool:
        """Check if required columns exist."""
        missing = [col for col in required if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy using RSI and MACD.
    Uses same indicator logic as signal_generator.py for consistency.

    Entry conditions:
    - RSI in oversold territory (using THRESHOLDS from signal_generator)
    - MACD bullish crossover with positive histogram

    Exit conditions:
    - RSI in overbought territory
    - MACD bearish crossover with negative histogram
    """

    def generate_signals(self) -> pd.Series:
        """Generate momentum-based trading signals using signal_generator logic."""
        self._validate_columns(['RSI', 'MACD', 'MACD_Signal', 'Close'])

        rsi = self.data['RSI'].values
        macd = self.data['MACD'].values
        macd_signal = self.data['MACD_Signal'].values
        macd_hist = macd - macd_signal

        n = len(self.data)
        signals = np.full(n, 'HOLD', dtype=object)

        # Use trending thresholds (momentum is a trending strategy)
        thresh = THRESHOLDS['trending']

        # RSI Signal (same logic as signal_generator)
        rsi_oversold = rsi < thresh['rsi_oversold']  # 25
        rsi_overbought = rsi > thresh['rsi_overbought']  # 75
        rsi_signal = np.zeros(n)
        rsi_signal[rsi_oversold] = 1  # Bullish
        rsi_signal[rsi_overbought] = -1  # Bearish

        # MACD Signal (same logic as signal_generator)
        macd_vote = np.where(
            (macd > macd_signal) & (macd_hist > 0), 1,  # Bullish crossover + positive histogram
            np.where(
                (macd < macd_signal) & (macd_hist < 0), -1,  # Bearish crossover + negative histogram
                0
            )
        )

        # Confluence: Both RSI and MACD must agree
        confluence = rsi_signal + macd_vote

        # Generate signals based on confluence
        # Buy when both indicators bullish (confluence >= 1)
        buy_signal = confluence >= 1

        # Sell when both indicators bearish (confluence <= -1)
        sell_signal = confluence <= -1

        signals[buy_signal] = 'BUY'
        signals[sell_signal] = 'SELL'

        # Strong signals when both strongly agree (confluence = 2 or -2)
        strong_buy = confluence >= 2
        strong_sell = confluence <= -2

        signals[strong_buy] = 'STRONG_BUY'
        signals[strong_sell] = 'STRONG_SELL'

        self.signals = pd.Series(signals, index=self.data.index)
        return self.signals


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend-Following Strategy using Moving Average crossovers and ADX.
    Uses same indicator logic as signal_generator.py for consistency.

    Entry conditions:
    - Price > SMA50 > SMA200 (uptrend alignment)
    - ADX > 25 with Plus_DI > Minus_DI (strong bullish trend)

    Exit conditions:
    - Price < SMA50 < SMA200 (downtrend alignment)
    - ADX > 25 with Minus_DI > Plus_DI (strong bearish trend)
    """

    def generate_signals(self) -> pd.Series:
        """Generate trend-following signals using signal_generator logic."""
        required = ['Close']
        self._validate_columns(required)

        close = self.data['Close'].values
        n = len(self.data)

        # Get SMA 50 and SMA 200 (same as signal_generator)
        if 'SMA_50' in self.data.columns:
            sma50 = self.data['SMA_50'].values
        else:
            sma50 = pd.Series(close).rolling(50).mean().values

        if 'SMA_200' in self.data.columns:
            sma200 = self.data['SMA_200'].values
        else:
            sma200 = pd.Series(close).rolling(200).mean().values

        signals = np.full(n, 'HOLD', dtype=object)

        # MA Signal (same logic as signal_generator)
        # Bullish: Price > SMA50 > SMA200 (uptrend)
        # Bearish: Price < SMA50 < SMA200 (downtrend)
        ma_signal = np.where(
            (close > sma50) & (sma50 > sma200), 1,
            np.where(
                (close < sma50) & (sma50 < sma200), -1,
                0
            )
        )

        # ADX/DI Signal (same logic as signal_generator)
        adx_signal = np.zeros(n)
        if 'ADX' in self.data.columns and 'Plus_DI' in self.data.columns and 'Minus_DI' in self.data.columns:
            adx = self.data['ADX'].values
            plus_di = self.data['Plus_DI'].values
            minus_di = self.data['Minus_DI'].values

            # Strong trend (ADX > 25) + DI direction
            strong_trend = adx > 25
            adx_signal[strong_trend & (plus_di > minus_di)] = 1  # Bullish trend
            adx_signal[strong_trend & (minus_di > plus_di)] = -1  # Bearish trend

        # Confluence: MA and ADX must agree
        confluence = ma_signal + adx_signal

        # Generate signals based on confluence
        buy_signal = confluence >= 1
        sell_signal = confluence <= -1

        signals[buy_signal] = 'BUY'
        signals[sell_signal] = 'SELL'

        # Strong signals when both strongly agree
        strong_buy = confluence >= 2
        strong_sell = confluence <= -2

        signals[strong_buy] = 'STRONG_BUY'
        signals[strong_sell] = 'STRONG_SELL'

        self.signals = pd.Series(signals, index=self.data.index)
        return self.signals


class MeanReversionStrategy(BaseStrategy):
    """
    Mean-Reversion Strategy using Bollinger Bands and RSI.
    Uses same indicator logic as signal_generator.py for consistency.

    Entry conditions:
    - BB %B <= 0 (at or below lower band) - using mean_reverting thresholds
    - RSI in oversold territory (confirmation)

    Exit conditions:
    - BB %B >= 1 (at or above upper band)
    - RSI in overbought territory
    """

    def generate_signals(self) -> pd.Series:
        """Generate mean-reversion signals using signal_generator logic."""
        required = ['Close']
        self._validate_columns(required)

        close = self.data['Close'].values
        n = len(self.data)

        # Use mean_reverting thresholds (this is a mean-reversion strategy)
        thresh = THRESHOLDS['mean_reverting']

        # Get Bollinger Band %B
        if 'BB_Percent_B' in self.data.columns:
            bb_pct = self.data['BB_Percent_B'].values
        else:
            # Calculate BB if not present
            sma = pd.Series(close).rolling(20).mean().values
            std = pd.Series(close).rolling(20).std().values
            upper = sma + 2 * std
            lower = sma - 2 * std
            bb_pct = (close - lower) / (upper - lower)

        # BB Signal (same logic as signal_generator for mean-reversion)
        bb_signal = np.zeros(n)
        bb_signal[bb_pct > thresh['bb_upper']] = -1  # Overbought, sell
        bb_signal[bb_pct < thresh['bb_lower']] = 1   # Oversold, buy

        # RSI Signal (using mean_reverting thresholds)
        rsi_signal = np.zeros(n)
        if 'RSI' in self.data.columns:
            rsi = self.data['RSI'].values
            rsi_signal[rsi < thresh['rsi_oversold']] = 1  # Oversold = bullish
            rsi_signal[rsi > thresh['rsi_overbought']] = -1  # Overbought = bearish

        # Confluence: BB and RSI signals
        confluence = bb_signal + rsi_signal

        signals = np.full(n, 'HOLD', dtype=object)

        # Generate signals based on confluence
        buy_signal = confluence >= 1
        sell_signal = confluence <= -1

        signals[buy_signal] = 'BUY'
        signals[sell_signal] = 'SELL'

        # Strong signals when both strongly agree
        strong_buy = confluence >= 2
        strong_sell = confluence <= -2

        signals[strong_buy] = 'STRONG_BUY'
        signals[strong_sell] = 'STRONG_SELL'

        self.signals = pd.Series(signals, index=self.data.index)
        return self.signals


class BreakoutStrategy(BaseStrategy):
    """
    Breakout Strategy using 52-week high and volume confirmation.

    Entry conditions:
    - Price breaks above 52-week (252-day) high
    - Volume is significantly above average (confirmation)
    - ATR expansion (volatility confirmation)

    Exit conditions:
    - Price falls below recent support (20-day low)
    - OR trailing stop triggered (2 ATR)
    """

    def generate_signals(self) -> pd.Series:
        """Generate breakout signals."""
        required = ['Close', 'High', 'Low', 'Volume']
        self._validate_columns(required)

        close = self.data['Close'].values
        high = self.data['High'].values
        low = self.data['Low'].values
        volume = self.data['Volume'].values

        n = len(self.data)
        lookback = self.config.breakout_lookback

        signals = np.full(n, 'HOLD', dtype=object)

        # Calculate 52-week high/low
        rolling_high = pd.Series(high).rolling(lookback, min_periods=lookback).max().values
        rolling_low = pd.Series(low).rolling(20).min().values  # 20-day support

        # Volume confirmation
        avg_volume = pd.Series(volume).rolling(50).mean().values
        high_volume = volume > (avg_volume * self.config.breakout_volume_mult)

        # ATR for volatility expansion
        if 'ATR' in self.data.columns:
            atr = self.data['ATR'].values
            atr_ma = pd.Series(atr).rolling(20).mean().values
            vol_expansion = atr > atr_ma
        else:
            vol_expansion = np.ones(n, dtype=bool)

        # Breakout detection
        # New high breakout
        new_high = np.zeros(n, dtype=bool)
        for i in range(lookback, n):
            if close[i] > rolling_high[i-1]:  # Close above previous 52w high
                new_high[i] = True

        # Breakdown detection (price below 20-day low)
        breakdown = np.zeros(n, dtype=bool)
        for i in range(20, n):
            if close[i] < rolling_low[i-1]:
                breakdown[i] = True

        # Generate signals
        # Buy on breakout with volume confirmation
        buy_signal = new_high & high_volume

        # Sell on breakdown or loss of support
        sell_signal = breakdown

        signals[buy_signal] = 'BUY'
        signals[sell_signal] = 'SELL'

        # Strong signals with volatility expansion
        strong_buy = new_high & high_volume & vol_expansion
        strong_sell = breakdown & high_volume

        signals[strong_buy] = 'STRONG_BUY'
        signals[strong_sell] = 'STRONG_SELL'

        self.signals = pd.Series(signals, index=self.data.index)
        return self.signals


# =============================================================================
# STRATEGY COMPARISON ENGINE
# =============================================================================
class StrategyComparison:
    """
    Engine for comparing multiple trading strategies.

    Features:
    - Runs all 4 strategies in parallel
    - Walk-forward tests each strategy independently
    - Ranks strategies by multiple metrics
    - Creates ensemble strategy from top performers
    - Analyzes regime-specific performance
    """

    STRATEGY_CLASSES = {
        StrategyType.MOMENTUM: MomentumStrategy,
        StrategyType.TREND_FOLLOWING: TrendFollowingStrategy,
        StrategyType.MEAN_REVERSION: MeanReversionStrategy,
        StrategyType.BREAKOUT: BreakoutStrategy
    }

    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[StrategyConfig] = None,
        initial_capital: float = 100_000.0,
        commission_pct: float = 0.001
    ):
        """
        Initialize strategy comparison engine.

        Args:
            data: DataFrame with OHLCV and technical indicators
            config: Strategy configuration parameters
            initial_capital: Starting capital for backtests
            commission_pct: Commission percentage per trade
        """
        self.data = data.copy()
        self.config = config or StrategyConfig()
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct

        self._validate_data()

        # Results storage
        self.strategy_results: Dict[StrategyType, StrategyResult] = {}
        self.comparison_result: Optional[ComparisonResult] = None

    def _validate_data(self):
        """Validate required columns exist."""
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _run_single_strategy(
        self,
        strategy_type: StrategyType,
        run_walk_forward: bool = False
    ) -> StrategyResult:
        """
        Run backtest for a single strategy.

        Args:
            strategy_type: Which strategy to run
            run_walk_forward: Whether to run walk-forward analysis

        Returns:
            StrategyResult with performance metrics
        """
        # Generate signals
        strategy_class = self.STRATEGY_CLASSES[strategy_type]
        strategy = strategy_class(self.data, self.config)
        signals = strategy.generate_signals()

        # Prepare data with signals
        data_with_signals = self.data.copy()
        data_with_signals['Signal'] = signals

        # Run VectorBT backtest
        close = data_with_signals['Close']

        # Convert signals to entries/exits
        entries, exits = self._signals_to_entries_exits(signals.values)

        # ATR-based stops if available
        sl_stop = None
        tp_stop = None
        if 'ATR' in data_with_signals.columns:
            atr = data_with_signals['ATR'].values
            sl_stop = 2.0 * atr / close.values
            tp_stop = 3.0 * atr / close.values

        # Run portfolio simulation
        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=pd.Series(entries, index=close.index),
            exits=pd.Series(exits, index=close.index),
            size=1.0,
            size_type='percent',
            init_cash=self.initial_capital,
            fees=self.commission_pct,
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            freq='1D'
        )

        # Extract metrics
        returns = portfolio.returns()
        total_return = float(portfolio.total_return())

        # Calculate CAGR
        years = len(self.data) / 252
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Risk-adjusted metrics
        rf_daily = (1 + 0.05) ** (1/252) - 1
        excess_returns = returns - rf_daily
        sharpe = (excess_returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        downside_returns = returns[returns < 0]
        sortino = (excess_returns.mean() * 252) / (downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0

        max_dd = float(portfolio.max_drawdown())
        volatility = float(returns.std() * np.sqrt(252))

        # Trade statistics
        trades = portfolio.trades.records_readable if hasattr(portfolio.trades, 'records_readable') else pd.DataFrame()
        total_trades = len(trades) if len(trades) > 0 else 0

        if total_trades > 0:
            winning = trades[trades['PnL'] > 0]
            losing = trades[trades['PnL'] < 0]
            win_rate = len(winning) / total_trades

            gross_profit = winning['PnL'].sum() if len(winning) > 0 else 0
            gross_loss = abs(losing['PnL'].sum()) if len(losing) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            if 'Exit Timestamp' in trades.columns and 'Entry Timestamp' in trades.columns:
                durations = (pd.to_datetime(trades['Exit Timestamp']) -
                           pd.to_datetime(trades['Entry Timestamp'])).dt.days
                avg_duration = float(durations.mean())
            else:
                avg_duration = 0
        else:
            win_rate = profit_factor = avg_duration = 0

        # Calculate regime-specific performance
        regime_performance = self._calculate_regime_performance(
            returns, data_with_signals
        )

        # Walk-forward analysis (optional)
        wf_efficiency = None
        oos_sharpe = None
        is_robust = None

        if run_walk_forward and total_trades >= 10:
            try:
                wf_result = self._run_walk_forward(signals)
                wf_efficiency = wf_result['efficiency']
                oos_sharpe = wf_result['oos_sharpe']
                is_robust = wf_result['is_robust']
            except Exception as e:
                print(f"  Walk-forward failed for {strategy_type.value}: {e}")

        return StrategyResult(
            strategy_type=strategy_type,
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            volatility=volatility,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_duration=avg_duration,
            equity_curve=portfolio.value(),
            signals=signals,
            wf_efficiency=wf_efficiency,
            oos_sharpe=oos_sharpe,
            is_robust=is_robust,
            regime_performance=regime_performance
        )

    def _signals_to_entries_exits(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert signal strings to boolean entry/exit arrays."""
        n = len(signals)
        entries = np.zeros(n, dtype=bool)
        exits = np.zeros(n, dtype=bool)

        buy_signals = (signals == 'STRONG_BUY') | (signals == 'BUY')
        sell_signals = (signals == 'STRONG_SELL') | (signals == 'SELL')

        in_position = False
        for i in range(n):
            if not in_position and buy_signals[i]:
                entries[i] = True
                in_position = True
            elif in_position and sell_signals[i]:
                exits[i] = True
                in_position = False

        return entries, exits

    def _calculate_regime_performance(
        self,
        returns: pd.Series,
        data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Calculate strategy performance in each market regime."""
        regime_perf = {}

        if 'Market_Regime' not in data.columns:
            return regime_perf

        regimes = data['Market_Regime'].unique()

        for regime in regimes:
            mask = data['Market_Regime'] == regime
            regime_returns = returns[mask]

            if len(regime_returns) < 20:
                continue

            # Calculate metrics for this regime
            rf_daily = (1 + 0.05) ** (1/252) - 1
            excess = regime_returns - rf_daily
            sharpe = (excess.mean() * 252) / (regime_returns.std() * np.sqrt(252)) if regime_returns.std() > 0 else 0

            total_ret = (1 + regime_returns).prod() - 1

            regime_perf[regime] = {
                'total_return': float(total_ret),
                'sharpe_ratio': float(sharpe),
                'avg_daily_return': float(regime_returns.mean()),
                'periods': int(len(regime_returns))
            }

        return regime_perf

    def _run_walk_forward(self, signals: pd.Series, n_folds: int = 4) -> Dict[str, Any]:
        """Run simplified walk-forward analysis on a strategy."""
        n = len(signals)
        fold_size = n // n_folds

        is_sharpes = []
        oos_sharpes = []

        close = self.data['Close']

        for fold in range(1, n_folds):
            # In-sample: all data before this fold
            is_end = fold * fold_size
            is_start = 0

            # Out-of-sample: this fold
            oos_start = is_end
            oos_end = min((fold + 1) * fold_size, n)

            if oos_end - oos_start < 63:  # Min 3 months OOS
                continue

            # Calculate IS Sharpe
            is_signals = signals.iloc[is_start:is_end]
            entries_is, exits_is = self._signals_to_entries_exits(is_signals.values)

            try:
                pf_is = vbt.Portfolio.from_signals(
                    close=close.iloc[is_start:is_end],
                    entries=pd.Series(entries_is, index=close.index[is_start:is_end]),
                    exits=pd.Series(exits_is, index=close.index[is_start:is_end]),
                    size=1.0, size_type='percent',
                    init_cash=self.initial_capital,
                    fees=self.commission_pct, freq='1D'
                )
                is_ret = pf_is.returns()
                rf_daily = (1 + 0.05) ** (1/252) - 1
                is_sharpe = ((is_ret.mean() - rf_daily) * 252) / (is_ret.std() * np.sqrt(252)) if is_ret.std() > 0 else 0
                is_sharpes.append(is_sharpe)
            except:
                continue

            # Calculate OOS Sharpe
            oos_signals = signals.iloc[oos_start:oos_end]
            entries_oos, exits_oos = self._signals_to_entries_exits(oos_signals.values)

            try:
                pf_oos = vbt.Portfolio.from_signals(
                    close=close.iloc[oos_start:oos_end],
                    entries=pd.Series(entries_oos, index=close.index[oos_start:oos_end]),
                    exits=pd.Series(exits_oos, index=close.index[oos_start:oos_end]),
                    size=1.0, size_type='percent',
                    init_cash=self.initial_capital,
                    fees=self.commission_pct, freq='1D'
                )
                oos_ret = pf_oos.returns()
                oos_sharpe = ((oos_ret.mean() - rf_daily) * 252) / (oos_ret.std() * np.sqrt(252)) if oos_ret.std() > 0 else 0
                oos_sharpes.append(oos_sharpe)
            except:
                continue

        if not is_sharpes or not oos_sharpes:
            return {'efficiency': 0, 'oos_sharpe': 0, 'is_robust': False}

        avg_is = np.mean(is_sharpes)
        avg_oos = np.mean(oos_sharpes)
        efficiency = avg_oos / avg_is if avg_is > 0 else 0

        return {
            'efficiency': efficiency,
            'oos_sharpe': avg_oos,
            'is_robust': efficiency > 0.5 and avg_oos > 0.3
        }

    def run_all_strategies(
        self,
        strategies: Optional[List[StrategyType]] = None,
        run_walk_forward: bool = True,
        parallel: bool = True
    ) -> Dict[StrategyType, StrategyResult]:
        """
        Run all strategies and collect results.

        Args:
            strategies: List of strategies to run (default: all 4)
            run_walk_forward: Whether to run walk-forward analysis
            parallel: Whether to run strategies in parallel

        Returns:
            Dictionary mapping strategy type to results
        """
        if strategies is None:
            strategies = list(StrategyType)

        print("\n Running Strategy Comparison...")
        print(f"  Strategies: {[s.value for s in strategies]}")
        print(f"  Walk-forward: {run_walk_forward}")

        results = {}

        if parallel and len(strategies) > 1:
            # Run in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(
                        self._run_single_strategy,
                        strategy_type,
                        run_walk_forward
                    ): strategy_type
                    for strategy_type in strategies
                }

                for future in as_completed(futures):
                    strategy_type = futures[future]
                    try:
                        result = future.result()
                        results[strategy_type] = result
                        print(f"  {strategy_type.value}: Sharpe={result.sharpe_ratio:.2f}, CAGR={result.cagr:.1%}")
                    except Exception as e:
                        print(f"  {strategy_type.value}: FAILED - {e}")
        else:
            # Run sequentially
            for strategy_type in strategies:
                try:
                    result = self._run_single_strategy(strategy_type, run_walk_forward)
                    results[strategy_type] = result
                    print(f"  {strategy_type.value}: Sharpe={result.sharpe_ratio:.2f}, CAGR={result.cagr:.1%}")
                except Exception as e:
                    print(f"  {strategy_type.value}: FAILED - {e}")

        self.strategy_results = results
        return results

    def rank_strategies(self) -> Dict[str, List[StrategyType]]:
        """
        Rank strategies by multiple metrics.

        Returns:
            Dictionary with rankings by each metric
        """
        if not self.strategy_results:
            raise ValueError("Run strategies first")

        results_list = list(self.strategy_results.items())

        # Rank by Sharpe (higher is better)
        by_sharpe = sorted(results_list, key=lambda x: x[1].sharpe_ratio, reverse=True)

        # Rank by CAGR (higher is better)
        by_cagr = sorted(results_list, key=lambda x: x[1].cagr, reverse=True)

        # Rank by Max Drawdown (less negative is better)
        by_max_dd = sorted(results_list, key=lambda x: x[1].max_drawdown, reverse=True)

        # Rank by Win Rate (higher is better)
        by_win_rate = sorted(results_list, key=lambda x: x[1].win_rate, reverse=True)

        rankings = {
            'by_sharpe': [s[0] for s in by_sharpe],
            'by_cagr': [s[0] for s in by_cagr],
            'by_max_drawdown': [s[0] for s in by_max_dd],
            'by_win_rate': [s[0] for s in by_win_rate]
        }

        print("\n Strategy Rankings:")
        print(f"  By Sharpe: {[s.value for s in rankings['by_sharpe']]}")
        print(f"  By CAGR: {[s.value for s in rankings['by_cagr']]}")
        print(f"  By Max DD: {[s.value for s in rankings['by_max_drawdown']]}")
        print(f"  By Win Rate: {[s.value for s in rankings['by_win_rate']]}")

        return rankings

    def get_best_by_regime(self) -> Dict[str, StrategyType]:
        """
        Find the best strategy for each market regime.

        Returns:
            Dictionary mapping regime to best strategy
        """
        if not self.strategy_results:
            raise ValueError("Run strategies first")

        # Collect all regimes
        all_regimes = set()
        for result in self.strategy_results.values():
            all_regimes.update(result.regime_performance.keys())

        best_by_regime = {}

        for regime in all_regimes:
            best_strategy = None
            best_sharpe = float('-inf')

            for strategy_type, result in self.strategy_results.items():
                if regime in result.regime_performance:
                    regime_sharpe = result.regime_performance[regime]['sharpe_ratio']
                    if regime_sharpe > best_sharpe:
                        best_sharpe = regime_sharpe
                        best_strategy = strategy_type

            if best_strategy:
                best_by_regime[regime] = best_strategy

        print("\n Best Strategy by Regime:")
        for regime, strategy in best_by_regime.items():
            print(f"  {regime}: {strategy.value}")

        return best_by_regime

    def create_ensemble(
        self,
        strategies: Optional[Tuple[StrategyType, StrategyType]] = None,
        weighting: str = 'equal'
    ) -> EnsembleResult:
        """
        Create ensemble strategy combining top 2 strategies.

        Args:
            strategies: Tuple of 2 strategies to combine (default: top 2 by Sharpe)
            weighting: 'equal', 'sharpe_weighted', or 'inverse_vol'

        Returns:
            EnsembleResult with combined strategy performance
        """
        if not self.strategy_results:
            raise ValueError("Run strategies first")

        # Select top 2 strategies by Sharpe if not specified
        if strategies is None:
            rankings = self.rank_strategies()
            strategies = (rankings['by_sharpe'][0], rankings['by_sharpe'][1])

        strategy_1, strategy_2 = strategies
        result_1 = self.strategy_results[strategy_1]
        result_2 = self.strategy_results[strategy_2]

        print(f"\n Creating Ensemble: {strategy_1.value} + {strategy_2.value}")
        print(f"  Weighting method: {weighting}")

        # Calculate weights
        if weighting == 'equal':
            weights = (0.5, 0.5)
        elif weighting == 'sharpe_weighted':
            # Handle negative Sharpe ratios by shifting to positive space
            # This preserves the relative ranking while allowing proper weighting
            sharpe_1 = result_1.sharpe_ratio
            sharpe_2 = result_2.sharpe_ratio
            min_sharpe = min(sharpe_1, sharpe_2)

            if min_sharpe < 0:
                # Shift both to positive space (add abs(min) + small buffer)
                shift = abs(min_sharpe) + 0.1
                sharpe_1 += shift
                sharpe_2 += shift

            total_sharpe = sharpe_1 + sharpe_2
            if total_sharpe > 0:
                w1 = sharpe_1 / total_sharpe
                w2 = sharpe_2 / total_sharpe
                weights = (w1, w2)
            else:
                weights = (0.5, 0.5)
        elif weighting == 'inverse_vol':
            vol_1 = result_1.volatility if result_1.volatility > 0 else 0.01
            vol_2 = result_2.volatility if result_2.volatility > 0 else 0.01
            total_inv_vol = (1/vol_1) + (1/vol_2)
            w1 = (1/vol_1) / total_inv_vol
            w2 = (1/vol_2) / total_inv_vol
            weights = (w1, w2)
        else:
            weights = (0.5, 0.5)

        print(f"  Weights: {weights[0]:.1%} / {weights[1]:.1%}")

        # Combine equity curves
        equity_1 = result_1.equity_curve
        equity_2 = result_2.equity_curve

        # Normalize to same starting point
        norm_1 = equity_1 / equity_1.iloc[0]
        norm_2 = equity_2 / equity_2.iloc[0]

        # Weighted combination
        ensemble_equity = (norm_1 * weights[0] + norm_2 * weights[1]) * self.initial_capital

        # Calculate ensemble metrics
        ensemble_returns = ensemble_equity.pct_change().dropna()
        total_return = float(ensemble_equity.iloc[-1] / ensemble_equity.iloc[0] - 1)

        years = len(ensemble_returns) / 252
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        rf_daily = (1 + 0.05) ** (1/252) - 1
        excess = ensemble_returns - rf_daily
        sharpe = (excess.mean() * 252) / (ensemble_returns.std() * np.sqrt(252)) if ensemble_returns.std() > 0 else 0

        max_dd = float((ensemble_equity / ensemble_equity.cummax() - 1).min())

        # Improvement vs best single strategy
        best_single_return = max(result_1.total_return, result_2.total_return)
        improvement = (total_return - best_single_return) / abs(best_single_return) if best_single_return != 0 else 0

        ensemble_result = EnsembleResult(
            strategy_1=strategy_1,
            strategy_2=strategy_2,
            weights=weights,
            weighting_method=weighting,
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            improvement_vs_best=improvement,
            equity_curve=ensemble_equity
        )

        print(f"  Ensemble Sharpe: {sharpe:.2f}")
        print(f"  Ensemble CAGR: {cagr:.1%}")
        print(f"  Improvement vs best: {improvement:.1%}")

        return ensemble_result

    def run_full_comparison(
        self,
        run_walk_forward: bool = True,
        create_ensemble: bool = True
    ) -> ComparisonResult:
        """
        Run complete strategy comparison analysis.

        Args:
            run_walk_forward: Whether to run walk-forward analysis
            create_ensemble: Whether to create ensemble strategy

        Returns:
            ComparisonResult with all analysis
        """
        print("\n" + "="*60)
        print("STRATEGY COMPARISON ANALYSIS")
        print("="*60)

        # Run all strategies
        self.run_all_strategies(run_walk_forward=run_walk_forward)

        # Rank strategies
        rankings = self.rank_strategies()

        # Get best by regime
        best_by_regime = self.get_best_by_regime()

        # Create ensemble
        ensemble = None
        if create_ensemble and len(self.strategy_results) >= 2:
            # Try different weighting methods and pick best
            best_ensemble = None
            best_ensemble_sharpe = float('-inf')

            for weighting in ['equal', 'sharpe_weighted', 'inverse_vol']:
                try:
                    ens = self.create_ensemble(weighting=weighting)
                    if ens.sharpe_ratio > best_ensemble_sharpe:
                        best_ensemble_sharpe = ens.sharpe_ratio
                        best_ensemble = ens
                except:
                    continue

            ensemble = best_ensemble

        # Determine recommendation
        recommended = rankings['by_sharpe'][0] if rankings['by_sharpe'] else None

        # Build recommendation reason
        if recommended and recommended in self.strategy_results:
            result = self.strategy_results[recommended]
            reason_parts = [f"Highest Sharpe ratio ({result.sharpe_ratio:.2f})"]

            if result.is_robust:
                reason_parts.append("passed walk-forward robustness test")
            if result.cagr > 0.10:
                reason_parts.append(f"strong CAGR ({result.cagr:.1%})")
            if result.max_drawdown > -0.20:
                reason_parts.append(f"acceptable drawdown ({result.max_drawdown:.1%})")

            reason = ", ".join(reason_parts)
        else:
            reason = "Insufficient data for recommendation"

        self.comparison_result = ComparisonResult(
            strategy_results=self.strategy_results,
            ranking_by_sharpe=rankings['by_sharpe'],
            ranking_by_cagr=rankings['by_cagr'],
            ranking_by_max_dd=rankings['by_max_drawdown'],
            ranking_by_win_rate=rankings['by_win_rate'],
            best_by_regime=best_by_regime,
            ensemble=ensemble,
            recommended_strategy=recommended,
            recommendation_reason=reason
        )

        print("\n" + "="*60)
        print("RECOMMENDATION")
        print("="*60)
        print(f"  Strategy: {recommended.value if recommended else 'N/A'}")
        print(f"  Reason: {reason}")

        return self.comparison_result

    def get_summary(self) -> str:
        """Get formatted summary of comparison results."""
        if not self.comparison_result:
            return "No comparison results. Run full_comparison first."

        cr = self.comparison_result

        lines = [
            "\n" + "="*60,
            "STRATEGY COMPARISON SUMMARY",
            "="*60,
            "",
            "INDIVIDUAL STRATEGY PERFORMANCE:",
            "-" * 40
        ]

        for strategy_type, result in cr.strategy_results.items():
            lines.append(f"\n{strategy_type.value.upper()}:")
            lines.append(f"  Sharpe: {result.sharpe_ratio:.2f}")
            lines.append(f"  CAGR: {result.cagr:.1%}")
            lines.append(f"  Max Drawdown: {result.max_drawdown:.1%}")
            lines.append(f"  Win Rate: {result.win_rate:.1%}")
            lines.append(f"  Total Trades: {result.total_trades}")
            if result.is_robust is not None:
                lines.append(f"  Robust: {'Yes' if result.is_robust else 'No'}")

        lines.extend([
            "",
            "RANKINGS:",
            "-" * 40,
            f"  By Sharpe: {' > '.join([s.value for s in cr.ranking_by_sharpe])}",
            f"  By CAGR: {' > '.join([s.value for s in cr.ranking_by_cagr])}",
            f"  By Max DD: {' > '.join([s.value for s in cr.ranking_by_max_dd])}",
            f"  By Win Rate: {' > '.join([s.value for s in cr.ranking_by_win_rate])}"
        ])

        if cr.best_by_regime:
            lines.extend([
                "",
                "BEST BY REGIME:",
                "-" * 40
            ])
            for regime, strategy in cr.best_by_regime.items():
                lines.append(f"  {regime}: {strategy.value}")

        if cr.ensemble:
            lines.extend([
                "",
                "ENSEMBLE STRATEGY:",
                "-" * 40,
                f"  Components: {cr.ensemble.strategy_1.value} + {cr.ensemble.strategy_2.value}",
                f"  Weights: {cr.ensemble.weights[0]:.0%} / {cr.ensemble.weights[1]:.0%}",
                f"  Method: {cr.ensemble.weighting_method}",
                f"  Sharpe: {cr.ensemble.sharpe_ratio:.2f}",
                f"  CAGR: {cr.ensemble.cagr:.1%}",
                f"  Improvement vs Best: {cr.ensemble.improvement_vs_best:.1%}"
            ])

        lines.extend([
            "",
            "RECOMMENDATION:",
            "-" * 40,
            f"  Strategy: {cr.recommended_strategy.value if cr.recommended_strategy else 'N/A'}",
            f"  Reason: {cr.recommendation_reason}",
            ""
        ])

        return "\n".join(lines)

    def save_results(self, ticker: str, output_dir: Optional[str] = None) -> str:
        """
        Save comparison results to JSON file.

        Args:
            ticker: Stock ticker symbol
            output_dir: Directory to save results (defaults to outputs/)

        Returns:
            Path to saved file
        """
        import json
        import os
        from datetime import datetime

        if not self.comparison_result:
            raise ValueError("No comparison results. Run run_full_comparison first.")

        output = self.comparison_result.to_dict()

        # Add metadata
        output['metadata'] = {
            'ticker': ticker,
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'generated_by': 'StrategyComparison'
        }

        # Determine output directory
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')

        os.makedirs(output_dir, exist_ok=True)

        # Custom JSON encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        # Save to file
        output_path = os.path.join(output_dir, f'{ticker}_strategy_comparison.json')
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, cls=NumpyEncoder)

        return output_path


# =============================================================================
# TEST SCRIPT
# =============================================================================
if __name__ == "__main__":
    from data_collector import DataCollector
    from technical_indicators import TechnicalIndicators
    from regime_detector import RegimeDetector

    print("="*60)
    print("STRATEGY COMPARISON TEST")
    print("="*60)

    # Load and prepare data
    print("\nLoading data...")
    collector = DataCollector()
    data = collector.get_data('AAPL', years=10)

    print("Calculating indicators...")
    ti = TechnicalIndicators(data)
    data = ti.calculate_all()

    print("Detecting regimes...")
    rd = RegimeDetector(data)
    data = rd.detect_all_regimes()

    # Run strategy comparison
    comparison = StrategyComparison(data)
    result = comparison.run_full_comparison(
        run_walk_forward=True,
        create_ensemble=True
    )

    # Print summary
    print(comparison.get_summary())

    # Save results to JSON
    output_path = comparison.save_results('AAPL')
    print(f"\nResults saved to: {output_path}")
