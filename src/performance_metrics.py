"""
Performance Metrics Module - Hedge Fund Grade Analytics

Comprehensive performance measurement including:
1. Basic Metrics: CAGR, Sharpe, Max Drawdown, Win Rate
2. Advanced Metrics: Sortino, Calmar, Omega, Profit Factor
3. Risk Metrics: VaR, CVaR, Probabilistic/Deflated Sharpe
4. Statistical Tests: T-tests, confidence intervals, rolling metrics

"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from enum import Enum
import warnings


class MetricCategory(Enum):
    """Categories for organising metrics."""
    RETURN = "return"
    RISK = "risk"
    RISK_ADJUSTED = "risk_adjusted"
    TRADE = "trade"
    STATISTICAL = "statistical"


@dataclass
class MetricResult:
    """Container for a single metric result with metadata."""
    name: str
    value: float
    category: MetricCategory
    description: str
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    is_significant: Optional[bool] = None

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'value': self.value,
            'category': self.category.value,
            'description': self.description,
            'confidence_interval': self.confidence_interval,
            'p_value': self.p_value,
            'is_significant': self.is_significant
        }


@dataclass
class PerformanceReport:
    """Complete performance report with all metrics."""
    
    # Basic metrics
    total_return: float          # Total cumulative return (e.g., 2.39 = 239%)
    cagr: float                  # Compound Annual Growth Rate - smoothed yearly return
    volatility: float            # Annualised standard deviation of returns 
    sharpe_ratio: float          # Return per unit of risk: (return - risk_free) / volatility
    max_drawdown: float          # Largest peak-to-trough decline (worst loss from high)

    #Other risk-adjusted metrics
    sortino_ratio: float         # Like Sharpe but only penalises downside volatility
    calmar_ratio: float          # CAGR / Max Drawdown - return per unit of drawdown risk
    omega_ratio: float           # Probability-weighted gains / losses 
    profit_factor: float         # Gross profits / Gross losses (>1 = profitable, >2 = very good)

    # Risk metrics
    var_95: float                # Value at Risk 95% - max daily loss with 95% confidence
    var_99: float                # Value at Risk 99% - max daily loss with 99% confidence
    cvar_95: float               # Conditional VaR - expected loss in worst 5% of days
    cvar_975: float              # Conditional VaR - expected loss in worst 2.5% of days

    # Probabilistic metrics
    probabilistic_sharpe: float  # Probability that true Sharpe > 0 (accounts for estimation error)
    deflated_sharpe: float       # Sharpe adjusted for multiple testing bias (luck vs skill)
    sharpe_confidence_interval: Tuple[float, float]  # 95% CI for Sharpe (lower, upper)

    # Trade metrics
    total_trades: int            # Number of completed trades
    win_rate: float              # Percentage of profitable trades
    avg_win: float               # Average profit on winning trades
    avg_loss: float              # Average loss on losing trades
    best_trade: float            # Largest single trade profit
    worst_trade: float           # Largest single trade loss
    avg_trade_duration: float    # Average days per trade

    # Statistical tests
    cagr_tstat: float            # T-statistic testing if CAGR > 0
    cagr_pvalue: float           # P-value (< 0.05 = statistically significant returns)
    returns_skewness: float      # Distribution asymmetry (0 = normal, negative = left tail)
    returns_kurtosis: float      # Tail thickness (3 = normal, higher = fat tails)
    jarque_bera_stat: float      # Test statistic for normality
    jarque_bera_pvalue: float    # P-value (< 0.05 = returns NOT normally distributed)

    # Rolling metrics (time series)
    rolling_sharpe_30d: pd.Series = field(default_factory=pd.Series)   # 30-day rolling Sharpe
    rolling_sharpe_90d: pd.Series = field(default_factory=pd.Series)   # 90-day rolling Sharpe
    rolling_sharpe_252d: pd.Series = field(default_factory=pd.Series)  # 1-year rolling Sharpe
    rolling_volatility_30d: pd.Series = field(default_factory=pd.Series)  # 30-day rolling vol

    # Time analysis
    start_date: str = ""         # Backtest start date
    end_date: str = ""           # Backtest end date
    trading_days: int = 0        # Total trading days analyzed
    years: float = 0.0           # Total years (trading_days / 252)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding Series objects."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, pd.Series):
                continue  # Skip rolling series
            elif isinstance(value, tuple):
                result[key] = value
            else:
                result[key] = value
        return result

    def get_summary(self) -> str:
                """Generate human-readable summary."""
                return f"""
        ╔══════════════════════════════════════════════════════════════════╗
        ║                    PERFORMANCE REPORT                            ║
        ║                  {self.start_date} to {self.end_date}            ║
        ╠══════════════════════════════════════════════════════════════════╣
        ║  RETURN METRICS                                                  ║
        ║  ─────────────────────────────────────────────────────────────── ║
        ║  Total Return:        {self.total_return:>10.2%}                 ║
        ║  CAGR:                {self.cagr:>10.2%}  (p={self.cagr_pvalue:.3f})║
        ║  Annualized Vol:      {self.volatility:>10.2%}                   ║
        ║                                                                  ║
        ║  RISK-ADJUSTED METRICS                                           ║
        ║  ─────────────────────────────────────────────────────────────── ║
        ║  Sharpe Ratio:        {self.sharpe_ratio:>10.2f}  [{self.sharpe_confidence_interval[0]:.2f}, {self.sharpe_confidence_interval[1]:.2f}]            ║
        ║  Sortino Ratio:       {self.sortino_ratio:>10.2f}  (downside only)║
        ║  Calmar Ratio:        {self.calmar_ratio:>10.2f}  (return/maxDD)  ║
        ║  Omega Ratio:         {self.omega_ratio:>10.2f}  (gain/loss probability)║
        ║  Profit Factor:       {self.profit_factor:>10.2f}  (gross profit/loss)║
        ║                                                                  ║
        ║  RISK METRICS                                                    ║
        ║  ─────────────────────────────────────────────────────────────── ║
        ║  Max Drawdown:        {self.max_drawdown:>10.2%}                 ║
        ║  VaR (95%):           {self.var_95:>10.2%}  (daily)              ║
        ║  VaR (99%):           {self.var_99:>10.2%}  (daily)              ║
        ║  CVaR (95%):          {self.cvar_95:>10.2%}  (expected shortfall)║
        ║  CVaR (97.5%):        {self.cvar_975:>10.2%}  (tail risk)        ║
        ║                                                                  ║
        ║  PROBABILISTIC METRICS                                           ║
        ║  ─────────────────────────────────────────────────────────────── ║
        ║  Prob. Sharpe (PSR):  {self.probabilistic_sharpe:>10.2%}  (confidence Sharpe>0)║
        ║  Deflated Sharpe:     {self.deflated_sharpe:>10.2f}  (adjusted for trials)║
        ║                                                                  ║
        ║  TRADE STATISTICS                                                ║
        ║  ─────────────────────────────────────────────────────────────── ║
        ║  Total Trades:        {self.total_trades:>10d}                   ║
        ║  Win Rate:            {self.win_rate:>10.2%}                     ║
        ║  Avg Win:             {self.avg_win:>10.2%}                      ║
        ║  Avg Loss:            {self.avg_loss:>10.2%}                     ║
        ║  Best Trade:          {self.best_trade:>10.2%}                   ║
        ║  Worst Trade:         {self.worst_trade:>10.2%}                  ║
        ║                                                                  ║
        ║  STATISTICAL TESTS                                               ║
        ║  ─────────────────────────────────────────────────────────────── ║
        ║  CAGR t-stat:         {self.cagr_tstat:>10.2f}  (p={self.cagr_pvalue:.4f})║
        ║  Skewness:            {self.returns_skewness:>10.2f}  (0=normal)║
        ║  Kurtosis:            {self.returns_kurtosis:>10.2f}  (3=normal)║
        ║  Jarque-Bera:         {self.jarque_bera_stat:>10.2f}  (p={self.jarque_bera_pvalue:.4f})               ║
        ╚══════════════════════════════════════════════════════════════════╝
        """


class PerformanceAnalyser:
    """
    Comprehensive performance analysis with hedge fund-grade metrics.

    Usage:
        analyser = PerformanceAnalyser(returns, trades_df)
        report = analyser.generate_report()
        print(report.get_summary())
    """

    TRADING_DAYS_PER_YEAR = 252
    RISK_FREE_RATE = 0.05  # 5% annual risk-free rate (adjustable)
    # 5 percent based on the average historical return of 10-year US Treasury bonds

    def __init__(self,
                 returns: pd.Series,
                 trades: Optional[pd.DataFrame] = None,
                 benchmark_returns: Optional[pd.Series] = None,
                 risk_free_rate: float = 0.05,
                 n_trials: int = 1):
        """
        Initialize performance analyser.

        Args:
            returns: Daily returns series (not cumulative)
            trades: DataFrame with trade records (optional)
                   Expected columns: 'PnL', 'Return', 'Duration' (or similar)
            benchmark_returns: Benchmark returns for comparison (optional)
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            n_trials: Number of strategy variations tested (for deflated Sharpe)
        """
        self.returns = returns.dropna()
        self.trades = trades
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        self.n_trials = n_trials

        # Daily risk-free rate since trading strategy generates daily returns
        self.daily_rf = (1 + risk_free_rate) ** (1/self.TRADING_DAYS_PER_YEAR) - 1

        # Excess returns
        self.excess_returns = self.returns - self.daily_rf

        # Validate
        # 30 data points minimum for reliable stats based on the Central Limit Theorem
        if len(self.returns) < 30:
            warnings.warn(f"Only {len(self.returns)} data points. Results may be unreliable.")

    # =========================================================================
    # BASIC METRICS
    # =========================================================================
    def calculate_total_return(self) -> float:
        """Calculate total cumulative return."""
        return (1 + self.returns).prod() - 1

    def calculate_cagr(self) -> float:
        """
        Calculate Compound Annual Growth Rate.

        CAGR = (Final Value / Initial Value)^(1/years) - 1
        """
        total_return = self.calculate_total_return()
        years = len(self.returns) / self.TRADING_DAYS_PER_YEAR

        if years <= 0 or total_return <= -1:
            return 0.0

        return (1 + total_return) ** (1 / years) - 1

    def calculate_volatility(self, annualize: bool = True) -> float:
        """
        Calculate return volatility (standard deviation).

        Args:
            annualize: If True, annualise the volatility
        """
        vol = self.returns.std()
        if annualize:
            vol *= np.sqrt(self.TRADING_DAYS_PER_YEAR)
        return vol

    def calculate_sharpe_ratio(self) -> float:
        """
        Calculate annualized Sharpe ratio.

        Sharpe = (Mean Return - Risk Free Rate) / Std Dev
        """
        if self.returns.std() == 0:
            return 0.0

        excess_return = self.returns.mean() - self.daily_rf
        sharpe = excess_return / self.returns.std()

        # Annualize
        return sharpe * np.sqrt(self.TRADING_DAYS_PER_YEAR)

    def calculate_max_drawdown(self) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate maximum drawdown and its dates.

        Returns:
            Tuple of (max_drawdown, peak_date, trough_date)
        """
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        max_dd = drawdown.min()
        trough_idx = drawdown.idxmin()
        peak_idx = cumulative[:trough_idx].idxmax()

        return max_dd, peak_idx, trough_idx

    def calculate_win_rate(self) -> float:
        """Calculate percentage of positive returns (or winning trades if available)."""
        if self.trades is not None and len(self.trades) > 0:
            # Use trade-level win rate
            if 'PnL' in self.trades.columns:
                wins = (self.trades['PnL'] > 0).sum()
                return wins / len(self.trades)
            elif 'Return' in self.trades.columns:
                wins = (self.trades['Return'] > 0).sum()
                return wins / len(self.trades)

        # Fall back to daily win rate
        return (self.returns > 0).mean()

    # =========================================================================
    # ADVANCED RISK-ADJUSTED METRICS
    # =========================================================================
    def calculate_sortino_ratio(self, target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio (downside risk-adjusted return).

        Unlike Sharpe, only penalises downside volatility.

        Sortino = (Mean Return - Target) / Downside Deviation
        """
        # Calculate excess returns (strategy return - risk-free rate)
        excess = self.returns - self.daily_rf

        # Isolate only the losing days (returns below target, usually 0%)
        # Sortino only cares about "bad" volatility, not upside swings
        downside_returns = self.returns[self.returns < target_return]

        if len(downside_returns) == 0:
            return float('inf')  # No downside

        # Calculate downside deviation (volatility of only the losing days)
        # This is the "semideviation" - standard deviation of negative returns
        downside_std = np.sqrt(np.mean(downside_returns ** 2))

        # Edge case: if downside returns exist but have zero volatility
        if downside_std == 0:
            return float('inf')

         # Sortino = average excess return / downside risk
        sortino = excess.mean() / downside_std
        
        # Annualise: convert daily Sortino to annual Sortino
        # Use sqrt(252) because volatility scales with square root of time
        return sortino * np.sqrt(self.TRADING_DAYS_PER_YEAR)

    def calculate_calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio (CAGR / Max Drawdown).

        Higher is better. Measures return per unit of drawdown risk.
        """
        
        # Get annual return rate
        cagr = self.calculate_cagr()
        # Get worst peak-to-trough decline
        max_dd, _, _ = self.calculate_max_drawdown()

         # If no drawdown occurred, strategy has perfect downside protection
        if max_dd == 0:
            return float('inf')
        
        # Calmar = annual return / max drawdown
        # Higher Calmar = better returns for given drawdown risk
        return cagr / abs(max_dd)

    def calculate_omega_ratio(self, threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio.

        Omega = Sum of gains above threshold / Sum of losses below threshold

        Unlike Sharpe, captures all moments of the distribution.
        """
        
        # Calculate gains above threshold 
        gains = self.returns[self.returns > threshold] - threshold
        # Calculate losses below threshold
        losses = threshold - self.returns[self.returns < threshold]

        if losses.sum() == 0:
            return float('inf')

        # Omega = total gains / total losses 
        # Omega > 1 means more gains than losses
        # Omega = 2 means you gain $2 for every $1 lost
        return gains.sum() / losses.sum()

    def calculate_profit_factor(self) -> float:
        """
        Calculate profit factor (gross profits / gross losses).

        PF > 1 means profitable, PF > 2 is considered very good.
        """
        
        # Try to use trade-level PnL if available
        if self.trades is not None and len(self.trades) > 0:
            if 'PnL' in self.trades.columns:
                # Sum all winning trades
                gross_profit = self.trades[self.trades['PnL'] > 0]['PnL'].sum()
                # Sum all losing trades (convert to positive number)
                gross_loss = abs(self.trades[self.trades['PnL'] < 0]['PnL'].sum())
            else:
                # No trade data available, use daily returns
                gross_profit = self.returns[self.returns > 0].sum()
                gross_loss = abs(self.returns[self.returns < 0].sum())
        else:
            gross_profit = self.returns[self.returns > 0].sum()
            gross_loss = abs(self.returns[self.returns < 0].sum())

        if gross_loss == 0:
            return float('inf')

         # Profit Factor = total $ won / total $ lost
        return gross_profit / gross_loss

    # =========================================================================
    # RISK METRICS (VaR, CVaR)
    # =========================================================================
    def calculate_var(self, confidence: float = 0.95, method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR).

        VaR answers: "What is the maximum loss at X% confidence?"

        Args:
            confidence: Confidence level (0.95 = 95%)
            method: 'historical', 'parametric', or 'cornish_fisher'

        Returns:
            VaR as a negative percentage (loss)
        """
        if method == 'historical':
            # Historical simulation: use actual past returns
            # For 95% confidence, find the 5th percentile (bottom 5% of returns)
            return np.percentile(self.returns, (1 - confidence) * 100)

        elif method == 'parametric':
            # Parametric VaR: assumes returns follow a normal distribution (bell curve)
            # Get z-score (standard deviations from mean for given confidence)
            z_score = stats.norm.ppf(1 - confidence)
            # VaR = mean return + (z-score × volatility)
            return self.returns.mean() + z_score * self.returns.std()

        elif method == 'cornish_fisher':
            # Cornish-Fisher VaR: adjusts for non-normal distributions
            # Real returns often have fat tails (extreme events) and skew (asymmetry)
            # Get base z-score for normal distribution
            z = stats.norm.ppf(1 - confidence)
            s = stats.skew(self.returns)
            k = stats.kurtosis(self.returns)

            # Cornish-Fisher expansion: adjusts z-score for skewness and kurtosis
            # This makes VaR more accurate for non-normal returns
            z_cf = (z + (z**2 - 1) * s / 6 +        # Skewness adjustment
                    (z**3 - 3*z) * (k - 3) / 24 -   # Kurtosis adjustment
                    (2*z**3 - 5*z) * s**2 / 36)     # Second-order skewness adjustment

            # Calculate VaR using adjusted z-score
            return self.returns.mean() + z_cf * self.returns.std()

        else:
            raise ValueError(f"Unknown VaR method: {method}")

    def calculate_cvar(self, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall).

        CVaR answers: "What is the expected loss in the worst X% of cases?"

        More conservative than VaR - captures tail risk.
        """
        
        # First calculate VaR (the threshold loss at given confidence)
        var = self.calculate_var(confidence, method='historical')

         # Find all returns that are worse than (or equal to) VaR
        # These are the "tail events" - the worst 5% of days
        tail_returns = self.returns[self.returns <= var]

        if len(tail_returns) == 0:
            return var

        # CVaR = average of all returns in the tail
        return tail_returns.mean()

    # =========================================================================
    # PROBABILISTIC METRICS
    # =========================================================================
    def calculate_probabilistic_sharpe_ratio(self, benchmark_sharpe: float = 0.0) -> float:
        """
        Calculate Probabilistic Sharpe Ratio (PSR).

        PSR measures the probability that the true Sharpe ratio exceeds
        a benchmark Sharpe ratio, accounting for estimation error.
        Args:
            benchmark_sharpe: Sharpe ratio to compare against (default 0)

        Returns:
            Probability that true Sharpe > benchmark_sharpe
        """
        # Gather inputs for PSR calculation
        n = len(self.returns) # Sample size (number of trading days)
        sharpe = self.calculate_sharpe_ratio()  # Observed Sharpe from our data
        skew = stats.skew(self.returns) # Asymmetry measure (0 = symmetric)
        kurt = stats.kurtosis(self.returns) # Fat tails measure (0 = normal distribution)

        # Calculate standard error (uncertainty) of our Sharpe estimate
        #  "How much might the true Sharpe differ from what we measured?"
        # Formula accounts for:
        # - Sample size (n): more data = less uncertainty
        # - Non-normality: skewness and kurtosis increase uncertainty
        sharpe_std = np.sqrt(
            (1 + 0.5 * sharpe**2 - skew * sharpe + (kurt - 3) / 4 * sharpe**2) / (n - 1)
        )
        # Edge case: if no uncertainty exists (shouldn't happen in practice)
        if sharpe_std == 0:
            # If observed Sharpe > benchmark, we're 100% confident, else 0% confident
            return 1.0 if sharpe > benchmark_sharpe else 0.0

        # Calculate z-score: "How many standard errors is our Sharpe above the benchmark?"
        z = (sharpe - benchmark_sharpe) / sharpe_std

        # Convert z-score to probability using normal distribution
        return stats.norm.cdf(z)

    def calculate_deflated_sharpe_ratio(self) -> float:
        """
        Calculate Deflated Sharpe Ratio (DSR).

        Adjusts Sharpe ratio for multiple testing bias.
        When you test many strategies, some will look good by chance.
        DSR penalises for the number of trials.
        Returns:
            Deflated Sharpe ratio accounting for multiple testing
        """
        n = len(self.returns)
        sharpe = self.calculate_sharpe_ratio()
        skew = stats.skew(self.returns)
        kurt = stats.kurtosis(self.returns)

        # Expected maximum Sharpe under null (pure luck)
        # Approximation for multiple testing
        if self.n_trials <= 1:
            expected_max_sharpe = 0
        else:
            # E[max(SR)] under null ~ sqrt(2 * log(n_trials))
            expected_max_sharpe = np.sqrt(2 * np.log(self.n_trials))

            # Adjust for Euler-Mascheroni constant
            euler_mascheroni = 0.5772156649
            expected_max_sharpe *= (1 - euler_mascheroni / np.log(self.n_trials))

        # Standard error of Sharpe
        sharpe_std = np.sqrt(
            (1 + 0.5 * sharpe**2 - skew * sharpe + (kurt - 3) / 4 * sharpe**2) / (n - 1)
        )

        if sharpe_std == 0:
            return sharpe

        # Deflated Sharpe: compare observed vs expected under luck
        z = (sharpe - expected_max_sharpe) / sharpe_std

        # Return as a Sharpe number (can be negative if overfitted)
        # This represents the Sharpe ratio adjusted for selection bias
        return stats.norm.cdf(z)

    def calculate_sharpe_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for Sharpe ratio.

        Args:
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        n = len(self.returns)
        sharpe = self.calculate_sharpe_ratio()
        skew = stats.skew(self.returns)
        kurt = stats.kurtosis(self.returns)

        # Standard error (Lo 2002, adjusted for non-normality)
        se = np.sqrt(
            (1 + 0.5 * sharpe**2 - skew * sharpe + (kurt - 3) / 4 * sharpe**2) / (n - 1)
        )

        z = stats.norm.ppf((1 + confidence) / 2)

        return (sharpe - z * se, sharpe + z * se)

    # =========================================================================
    # STATISTICAL TESTS
    # =========================================================================
    def test_cagr_significance(self) -> Tuple[float, float]:
        """
        Test if CAGR is significantly different from zero.

        Uses t-test on log returns.

        Returns:
            Tuple of (t_statistic, p_value)
        """
        # Use log returns for proper statistical testing
        log_returns = np.log(1 + self.returns)

        # One-sample t-test: is mean > 0?
        t_stat, p_value = stats.ttest_1samp(log_returns, 0)

        # One-tailed p-value (we care if returns > 0)
        p_value_one_tailed = p_value / 2 if t_stat > 0 else 1 - p_value / 2

        return t_stat, p_value_one_tailed

    def test_normality(self) -> Tuple[float, float, float, float]:
        """
        Test return distribution normality using Jarque-Bera test.

        Returns:
            Tuple of (skewness, kurtosis, jb_statistic, jb_p_value)
        """
        skewness = stats.skew(self.returns)
        kurtosis = stats.kurtosis(self.returns)  # Excess kurtosis (normal = 0)

        jb_stat, jb_pvalue = stats.jarque_bera(self.returns)

        return skewness, kurtosis, jb_stat, jb_pvalue

    # =========================================================================
    # TRADE STATISTICS
    # =========================================================================
    def calculate_trade_statistics(self) -> Dict[str, float]:
        """Calculate detailed trade-level statistics."""
        
        #Handle case where no trade data is available
        # Return zeros for all metrics if trades don't exist
        if self.trades is None or len(self.trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'best_trade': 0.0,
                'worst_trade': 0.0,
                'avg_duration': 0.0
            }

        # Determine which column has PnL data
        if 'Return' in self.trades.columns:
            pnl_col = 'Return'
        elif 'PnL' in self.trades.columns:
            pnl_col = 'PnL'
        else:
            # No PnL data
            # Return zeros for all metrics
            return {
                'total_trades': len(self.trades),
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'best_trade': 0.0,
                'worst_trade': 0.0,
                'avg_duration': 0.0
            }

        # Extract profit/losses and separate winners/losers
        trades_pnl = self.trades[pnl_col]
        winners = trades_pnl[trades_pnl > 0]
        losers = trades_pnl[trades_pnl < 0]

        # Duration - calculate from timestamps if available (VectorBT format)
        avg_duration = 0.0
        if 'Duration' in self.trades.columns:
            avg_duration = self.trades['Duration'].mean()
        elif 'Entry Timestamp' in self.trades.columns and 'Exit Timestamp' in self.trades.columns:
            # VectorBT uses 'Entry Timestamp' and 'Exit Timestamp'
            try:
                entry_times = pd.to_datetime(self.trades['Entry Timestamp'])
                exit_times = pd.to_datetime(self.trades['Exit Timestamp'])
                durations = (exit_times - entry_times).dt.days
                avg_duration = float(durations.mean()) if len(durations) > 0 else 0.0
            except:
                avg_duration = 0.0

        return {
            'total_trades': len(self.trades),
            'win_rate': len(winners) / len(self.trades) if len(self.trades) > 0 else 0.0,
            'avg_win': winners.mean() if len(winners) > 0 else 0.0,
            'avg_loss': losers.mean() if len(losers) > 0 else 0.0,
            'best_trade': trades_pnl.max(),
            'worst_trade': trades_pnl.min(),
            'avg_duration': avg_duration
        }

    # =========================================================================
    # ROLLING METRICS
    # =========================================================================
    def calculate_rolling_sharpe(self, window: int = 252) -> pd.Series:
        """
        Calculate rolling Sharpe ratio.

        Args:
            window: Rolling window in trading days

        Returns:
            Series of rolling Sharpe ratios
        """
        
        rolling_mean = self.excess_returns.rolling(window=window).mean()
        #For each day, calculate volatility over past 'window' days
        rolling_std = self.returns.rolling(window=window).std()
        # Sharpe ratio at each point in time - annualised
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(self.TRADING_DAYS_PER_YEAR)

        return rolling_sharpe

    def calculate_rolling_volatility(self, window: int = 30) -> pd.Series:
        """
        Calculate rolling volatility.

        Args:
            window: Rolling window in trading days

        Returns:
            Series of annualized rolling volatility
        """
        return self.returns.rolling(window=window).std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)

    def calculate_rolling_drawdown(self) -> pd.Series:
        """Calculate rolling drawdown from peak."""
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.cummax()
        return (cumulative - running_max) / running_max

    # =========================================================================
    # GENERATE COMPLETE REPORT
    # =========================================================================
    def generate_report(self) -> PerformanceReport:
        """
        Generate comprehensive performance report with all metrics.

        Returns:
            PerformanceReport dataclass with all metrics
        """
        # Basic metrics
        total_return = self.calculate_total_return()
        cagr = self.calculate_cagr()
        volatility = self.calculate_volatility()
        sharpe = self.calculate_sharpe_ratio()
        max_dd, _, _ = self.calculate_max_drawdown()

        # Advanced metrics
        sortino = self.calculate_sortino_ratio()
        calmar = self.calculate_calmar_ratio()
        omega = self.calculate_omega_ratio()
        profit_factor = self.calculate_profit_factor()

        # Risk metrics
        var_95 = self.calculate_var(0.95)
        var_99 = self.calculate_var(0.99)
        cvar_95 = self.calculate_cvar(0.95)
        cvar_975 = self.calculate_cvar(0.975)

        # Probabilistic metrics
        psr = self.calculate_probabilistic_sharpe_ratio()
        dsr = self.calculate_deflated_sharpe_ratio()
        sharpe_ci = self.calculate_sharpe_confidence_interval()

        # Trade statistics
        trade_stats = self.calculate_trade_statistics()

        # Statistical tests
        t_stat, p_value = self.test_cagr_significance()
        skew, kurt, jb_stat, jb_pvalue = self.test_normality()

        # Rolling metrics
        rolling_sharpe_30 = self.calculate_rolling_sharpe(30)
        rolling_sharpe_90 = self.calculate_rolling_sharpe(90)
        rolling_sharpe_252 = self.calculate_rolling_sharpe(252)
        rolling_vol_30 = self.calculate_rolling_volatility(30)

        # Time info
        years = len(self.returns) / self.TRADING_DAYS_PER_YEAR

        return PerformanceReport(
            # Basic
            total_return=total_return,
            cagr=cagr,
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,

            # Advanced
            sortino_ratio=sortino if not np.isinf(sortino) else 999.99,
            calmar_ratio=calmar if not np.isinf(calmar) else 999.99,
            omega_ratio=omega if not np.isinf(omega) else 999.99,
            profit_factor=profit_factor if not np.isinf(profit_factor) else 999.99,

            # Risk
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_975=cvar_975,

            # Probabilistic
            probabilistic_sharpe=psr,
            deflated_sharpe=dsr,
            sharpe_confidence_interval=sharpe_ci,

            # Trade
            total_trades=trade_stats['total_trades'],
            win_rate=trade_stats['win_rate'],
            avg_win=trade_stats['avg_win'],
            avg_loss=trade_stats['avg_loss'],
            best_trade=trade_stats['best_trade'],
            worst_trade=trade_stats['worst_trade'],
            avg_trade_duration=trade_stats['avg_duration'],

            # Statistical
            cagr_tstat=t_stat,
            cagr_pvalue=p_value,
            returns_skewness=skew,
            returns_kurtosis=kurt,
            jarque_bera_stat=jb_stat,
            jarque_bera_pvalue=jb_pvalue,

            # Rolling
            rolling_sharpe_30d=rolling_sharpe_30,
            rolling_sharpe_90d=rolling_sharpe_90,
            rolling_sharpe_252d=rolling_sharpe_252,
            rolling_volatility_30d=rolling_vol_30,

            # Time
            start_date=str(self.returns.index[0].date()) if hasattr(self.returns.index[0], 'date') else str(self.returns.index[0]),
            end_date=str(self.returns.index[-1].date()) if hasattr(self.returns.index[-1], 'date') else str(self.returns.index[-1]),
            trading_days=len(self.returns),
            years=years
        )

    # =========================================================================
    # COMPARISON METHODS
    # =========================================================================
    def compare_to_benchmark(self, benchmark_returns: pd.Series) -> Dict[str, float]:
        """
        Compare strategy to benchmark.

        Args:
            benchmark_returns: Benchmark daily returns

        Returns:
            Dictionary with comparison metrics
        """
        # Align returns
        common_idx = self.returns.index.intersection(benchmark_returns.index)
        strat_ret = self.returns.loc[common_idx]
        bench_ret = benchmark_returns.loc[common_idx]

        # Benchmark metrics
        bench_analyzer = PerformanceAnalyser(bench_ret)
        bench_sharpe = bench_analyzer.calculate_sharpe_ratio()
        bench_cagr = bench_analyzer.calculate_cagr()
        bench_max_dd, _, _ = bench_analyzer.calculate_max_drawdown()

        # Strategy metrics
        strat_sharpe = self.calculate_sharpe_ratio()
        strat_cagr = self.calculate_cagr()
        strat_max_dd, _, _ = self.calculate_max_drawdown()

        # Alpha and Beta (CAPM)
        if len(common_idx) > 1 and bench_ret.var() > 0:
            cov_matrix = np.cov(strat_ret, bench_ret)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1]
            alpha = (strat_ret.mean() - beta * bench_ret.mean()) * self.TRADING_DAYS_PER_YEAR
        else:
            beta = 1.0
            alpha = 0.0

        # Information Ratio (alpha / tracking error)
        tracking_error = (strat_ret - bench_ret).std() * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        information_ratio = alpha / tracking_error if tracking_error > 0 else 0.0

        # Treynor Ratio (excess return / beta)
        treynor = (strat_cagr - self.risk_free_rate) / beta if beta != 0 else 0.0

        return {
            'strategy_sharpe': strat_sharpe,
            'benchmark_sharpe': bench_sharpe,
            'strategy_cagr': strat_cagr,
            'benchmark_cagr': bench_cagr,
            'strategy_max_dd': strat_max_dd,
            'benchmark_max_dd': bench_max_dd,
            'alpha': alpha,
            'beta': beta,
            'information_ratio': information_ratio,
            'treynor_ratio': treynor,
            'tracking_error': tracking_error,
            'excess_return': strat_cagr - bench_cagr,
            'excess_sharpe': strat_sharpe - bench_sharpe
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def analyze_from_backtest(portfolio, trades_df: Optional[pd.DataFrame] = None,
                          risk_free_rate: float = 0.05,
                          n_trials: int = 1) -> PerformanceReport:
    """
    Convenience function to analyze a VectorBT portfolio.

    Args:
        portfolio: VectorBT Portfolio object
        trades_df: Optional trades DataFrame
        risk_free_rate: Annual risk-free rate
        n_trials: Number of strategy variations tested

    Returns:
        PerformanceReport
    """
    returns = portfolio.returns()

    # Get trades from VectorBT if not provided
    if trades_df is None and hasattr(portfolio, 'trades'):
        try:
            trades_df = portfolio.trades.records_readable
        except:
            trades_df = None

    analyzer = PerformanceAnalyser(
        returns=returns,
        trades=trades_df,
        risk_free_rate=risk_free_rate,
        n_trials=n_trials
    )

    return analyzer.generate_report()


def quick_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate quick summary metrics without full report.

    Args:
        returns: Daily returns series

    Returns:
        Dictionary with key metrics
    """
    analyzer = PerformanceAnalyser(returns)

    max_dd, _, _ = analyzer.calculate_max_drawdown()

    return {
        'total_return': analyzer.calculate_total_return(),
        'cagr': analyzer.calculate_cagr(),
        'sharpe': analyzer.calculate_sharpe_ratio(),
        'sortino': analyzer.calculate_sortino_ratio(),
        'max_drawdown': max_dd,
        'volatility': analyzer.calculate_volatility(),
        'win_rate': analyzer.calculate_win_rate()
    }


# =============================================================================
# TEST SCRIPT
# =============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).rsplit('/', 1)[0])

    from data_collector import DataCollector
    from technical_indicators import TechnicalIndicators
    from regime_detector import RegimeDetector
    from signal_generator import SignalGenerator
    from backtest_engine import BacktestEngine, BacktestConfig

    print("=" * 70)
    print("PERFORMANCE METRICS MODULE TEST")
    print("=" * 70)

    # Load data
    collector = DataCollector(use_parquet=True)
    data = collector.get_data('AAPL', years=10)

    # Run full pipeline
    indicators = TechnicalIndicators(data)
    data = indicators.calculate_all()

    detector = RegimeDetector(data)
    data = detector.detect_all_regimes()

    generator = SignalGenerator(data)
    data = generator.generate_signals()

    # Run backtest
    config = BacktestConfig()
    engine = BacktestEngine(data, config)
    results = engine.run_backtest()

    # Get returns and trades from VectorBT portfolio
    returns = engine.portfolio.returns()

    try:
        trades_df = engine.portfolio.trades.records_readable
    except:
        trades_df = None

    print(f"\nAnalyzing {len(returns)} trading days of returns...")
    print(f"Trades found: {len(trades_df) if trades_df is not None else 0}")

    # Create analyzer
    analyzer = PerformanceAnalyser(
        returns=returns,
        trades=trades_df,
        n_trials=1  # Single strategy test
    )

    # Generate full report
    report = analyzer.generate_report()

    # Print summary
    print(report.get_summary())

    # Test individual metrics
    print("\n" + "=" * 70)
    print("INDIVIDUAL METRIC TESTS")
    print("=" * 70)

    print(f"\nBasic Metrics:")
    print(f"  Total Return: {analyzer.calculate_total_return():.2%}")
    print(f"  CAGR: {analyzer.calculate_cagr():.2%}")
    print(f"  Volatility: {analyzer.calculate_volatility():.2%}")
    print(f"  Sharpe: {analyzer.calculate_sharpe_ratio():.2f}")

    max_dd, peak, trough = analyzer.calculate_max_drawdown()
    print(f"  Max Drawdown: {max_dd:.2%} (peak: {peak.date()}, trough: {trough.date()})")

    print(f"\nRisk Metrics:")
    print(f"  VaR 95% (Historical): {analyzer.calculate_var(0.95, 'historical'):.2%}")
    print(f"  VaR 95% (Parametric): {analyzer.calculate_var(0.95, 'parametric'):.2%}")
    print(f"  VaR 95% (Cornish-Fisher): {analyzer.calculate_var(0.95, 'cornish_fisher'):.2%}")
    print(f"  CVaR 95%: {analyzer.calculate_cvar(0.95):.2%}")
    print(f"  CVaR 97.5%: {analyzer.calculate_cvar(0.975):.2%}")

    print(f"\nProbabilistic Metrics:")
    print(f"  PSR (prob Sharpe > 0): {analyzer.calculate_probabilistic_sharpe_ratio():.2%}")
    print(f"  Deflated Sharpe: {analyzer.calculate_deflated_sharpe_ratio():.2f}")
    ci = analyzer.calculate_sharpe_confidence_interval()
    print(f"  Sharpe 95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")

    print(f"\nStatistical Tests:")
    t_stat, p_val = analyzer.test_cagr_significance()
    print(f"  CAGR t-test: t={t_stat:.2f}, p={p_val:.4f}")
    skew, kurt, jb, jb_p = analyzer.test_normality()
    print(f"  Skewness: {skew:.2f}")
    print(f"  Excess Kurtosis: {kurt:.2f}")
    print(f"  Jarque-Bera: stat={jb:.2f}, p={jb_p:.4f}")
    print(f"  Returns are {'NOT ' if jb_p < 0.05 else ''}normally distributed")

    # Quick metrics test
    print(f"\nQuick Metrics Test:")
    quick = quick_metrics(returns)
    for key, value in quick.items():
        if key in ['total_return', 'cagr', 'max_drawdown', 'win_rate', 'volatility']:
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value:.2f}")

    # Compare to buy & hold
    print(f"\n" + "=" * 70)
    print("BENCHMARK COMPARISON")
    print("=" * 70)
    
    

    # Get buy & hold returns
    bh_returns = data['Close'].pct_change().dropna()
    comparison = analyzer.compare_to_benchmark(bh_returns)

    print(f"\nStrategy vs Buy & Hold:")
    print(f"  Strategy Sharpe: {comparison['strategy_sharpe']:.2f}")
    print(f"  Benchmark Sharpe: {comparison['benchmark_sharpe']:.2f}")
    print(f"  Alpha: {comparison['alpha']:.2%}")
    print(f"  Beta: {comparison['beta']:.2f}")
    print(f"  Information Ratio: {comparison['information_ratio']:.2f}")
    print(f"  Treynor Ratio: {comparison['treynor_ratio']:.2f}")

    print("\n Performance metrics module test complete!")
