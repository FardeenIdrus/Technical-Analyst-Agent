"""
Monte Carlo Simulation Module - Statistically Rigorous Strategy Validation

Implements block bootstrap simulation to validate strategy performance:
1. Block Bootstrap: Preserves autocorrelation in returns (realistic market behavior)
2. 10,000 simulations for statistical significance
3. Confidence intervals on all metrics (CAGR, Sharpe, Max DD)
4. Probability of ruin at various drawdown thresholds
5. Statistical significance testing (skill vs luck)

Key Features:
- Block size = 20 days (captures realistic return patterns)
- Preserves serial correlation unlike simple bootstrap
- Full probability distributions of outcomes
- Risk metrics: P(loss > X%), P(ruin), etc.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""

    # Simulation parameters
    n_simulations: int
    block_size: int
    original_days: int

    # Distribution of metrics (arrays of simulated values)
    cagr_distribution: np.ndarray = field(default_factory=lambda: np.array([]))
    sharpe_distribution: np.ndarray = field(default_factory=lambda: np.array([]))
    max_dd_distribution: np.ndarray = field(default_factory=lambda: np.array([]))
    total_return_distribution: np.ndarray = field(default_factory=lambda: np.array([]))
    volatility_distribution: np.ndarray = field(default_factory=lambda: np.array([]))

    # Confidence intervals (95%)
    cagr_ci: Tuple[float, float] = (0.0, 0.0)
    sharpe_ci: Tuple[float, float] = (0.0, 0.0)
    max_dd_ci: Tuple[float, float] = (0.0, 0.0)
    total_return_ci: Tuple[float, float] = (0.0, 0.0)

    # Risk metrics
    prob_loss_10pct: float = 0.0  # P(total return < -10%)
    prob_loss_20pct: float = 0.0  # P(total return < -20%)
    prob_negative_cagr: float = 0.0  # P(CAGR < 0%)
    prob_sharpe_below_zero: float = 0.0  # P(Sharpe < 0)
    prob_sharpe_below_one: float = 0.0  # P(Sharpe < 1)

    # Probability of ruin (max drawdown exceeds threshold)
    prob_ruin_10pct: float = 0.0  # P(Max DD > 10%)
    prob_ruin_20pct: float = 0.0  # P(Max DD > 20%)
    prob_ruin_30pct: float = 0.0  # P(Max DD > 30%)
    prob_ruin_50pct: float = 0.0  # P(Max DD > 50%)

    # Best/worst case scenarios (5th and 95th percentile)
    best_case_cagr: float = 0.0
    worst_case_cagr: float = 0.0
    best_case_sharpe: float = 0.0
    worst_case_sharpe: float = 0.0
    best_case_max_dd: float = 0.0  # Smallest drawdown
    worst_case_max_dd: float = 0.0  # Largest drawdown

    # Statistical significance
    actual_cagr: float = 0.0
    actual_sharpe: float = 0.0
    actual_max_dd: float = 0.0
    cagr_percentile: float = 0.0  # Where actual falls in distribution
    sharpe_percentile: float = 0.0
    is_statistically_significant: bool = False  # Actual in top 5%

    # Summary statistics
    median_cagr: float = 0.0
    median_sharpe: float = 0.0
    median_max_dd: float = 0.0
    std_cagr: float = 0.0
    std_sharpe: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding large arrays)."""
        return {
            'n_simulations': self.n_simulations,
            'block_size': self.block_size,
            'original_days': self.original_days,
            'cagr_ci': self.cagr_ci,
            'sharpe_ci': self.sharpe_ci,
            'max_dd_ci': self.max_dd_ci,
            'total_return_ci': self.total_return_ci,
            'prob_loss_10pct': self.prob_loss_10pct,
            'prob_loss_20pct': self.prob_loss_20pct,
            'prob_negative_cagr': self.prob_negative_cagr,
            'prob_sharpe_below_zero': self.prob_sharpe_below_zero,
            'prob_sharpe_below_one': self.prob_sharpe_below_one,
            'prob_ruin_10pct': self.prob_ruin_10pct,
            'prob_ruin_20pct': self.prob_ruin_20pct,
            'prob_ruin_30pct': self.prob_ruin_30pct,
            'prob_ruin_50pct': self.prob_ruin_50pct,
            'best_case_cagr': self.best_case_cagr,
            'worst_case_cagr': self.worst_case_cagr,
            'best_case_sharpe': self.best_case_sharpe,
            'worst_case_sharpe': self.worst_case_sharpe,
            'best_case_max_dd': self.best_case_max_dd,
            'worst_case_max_dd': self.worst_case_max_dd,
            'actual_cagr': self.actual_cagr,
            'actual_sharpe': self.actual_sharpe,
            'actual_max_dd': self.actual_max_dd,
            'cagr_percentile': self.cagr_percentile,
            'sharpe_percentile': self.sharpe_percentile,
            'is_statistically_significant': self.is_statistically_significant,
            'median_cagr': self.median_cagr,
            'median_sharpe': self.median_sharpe,
            'median_max_dd': self.median_max_dd,
            'std_cagr': self.std_cagr,
            'std_sharpe': self.std_sharpe
        }

    def get_summary(self) -> str:
        """Create readable summary."""
        significance = "YES - Strategy shows skill" if self.is_statistically_significant else "NO - Could be luck"

        return f"""
╔══════════════════════════════════════════════════════════════════╗
║                 MONTE CARLO SIMULATION RESULTS                   ║
║                   {self.n_simulations:,} Simulations             ║
╠══════════════════════════════════════════════════════════════════╣
║  CONFIDENCE INTERVALS (95%)                                      ║
║  ─────────────────────────────────────────────────────────────── ║
║  CAGR:         [{self.cagr_ci[0]:>7.2%}, {self.cagr_ci[1]:>7.2%}]  (median: {self.median_cagr:>7.2%})║
║  Sharpe:       [{self.sharpe_ci[0]:>7.2f}, {self.sharpe_ci[1]:>7.2f}]  (median: {self.median_sharpe:>7.2f})║
║  Max Drawdown: [{self.max_dd_ci[0]:>7.2%}, {self.max_dd_ci[1]:>7.2%}]  (median: {self.median_max_dd:>7.2%})║
║                                                                  ║
║  RISK PROBABILITIES                                              ║
║  ─────────────────────────────────────────────────────────────── ║
║  P(Loss > 10%):     {self.prob_loss_10pct:>7.1%}                 ║
║  P(Loss > 20%):     {self.prob_loss_20pct:>7.1%}                 ║
║  P(Negative CAGR):  {self.prob_negative_cagr:>7.1%}              ║
║  P(Sharpe < 0):     {self.prob_sharpe_below_zero:>7.1%}          ║
║  P(Sharpe < 1):     {self.prob_sharpe_below_one:>7.1%}           ║
║                                                                  ║
║  PROBABILITY OF RUIN (Max DD exceeds threshold)                  ║
║  ─────────────────────────────────────────────────────────────── ║
║  P(Max DD > 10%):   {self.prob_ruin_10pct:>7.1%}                 ║
║  P(Max DD > 20%):   {self.prob_ruin_20pct:>7.1%}                 ║
║  P(Max DD > 30%):   {self.prob_ruin_30pct:>7.1%}                 ║
║  P(Max DD > 50%):   {self.prob_ruin_50pct:>7.1%}                 ║
║                                                                  ║
║  BEST/WORST CASE SCENARIOS (5th/95th percentile)                 ║
║  ─────────────────────────────────────────────────────────────── ║
║  CAGR:         Best: {self.best_case_cagr:>7.2%}    Worst: {self.worst_case_cagr:>7.2%}        ║
║  Sharpe:       Best: {self.best_case_sharpe:>7.2f}    Worst: {self.worst_case_sharpe:>7.2f}        ║
║  Max DD:       Best: {self.best_case_max_dd:>7.2%}    Worst: {self.worst_case_max_dd:>7.2%}        ║
║                                                                  ║
║  STATISTICAL SIGNIFICANCE                                        ║
║  ─────────────────────────────────────────────────────────────── ║
║  Actual CAGR:       {self.actual_cagr:>7.2%}  (percentile: {self.cagr_percentile:>5.1f}%)       ║
║  Actual Sharpe:     {self.actual_sharpe:>7.2f}  (percentile: {self.sharpe_percentile:>5.1f}%)       ║
║  Significant:       {significance:<40}║
╚══════════════════════════════════════════════════════════════════╝
"""


class MonteCarloSimulator:
    """
    Monte Carlo simulation using block bootstrap for strategy validation.

    Block bootstrap preserves the autocorrelation structure of returns,
    making simulations more realistic than simple random sampling.

    Usage:
        simulator = MonteCarloSimulator(returns, trades_df)
        result = simulator.run_simulation(n_simulations=10000)
        print(result.get_summary())
    """

    TRADING_DAYS_PER_YEAR = 252

    def __init__(
        self,
        returns: pd.Series,
        trades: Optional[pd.DataFrame] = None,
        block_size: int = 20,
        risk_free_rate: float = 0.05
    ):
        """
        Initialise Monte Carlo simulator.

        Args:
            returns: Daily returns series from strategy
            trades: Optional DataFrame with trade records
            block_size: Size of blocks for bootstrap (default 20 days)
                       Larger blocks preserve more autocorrelation
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.returns = returns.dropna().values
        self.trades = trades
        self.block_size = block_size
        self.risk_free_rate = risk_free_rate
        self.daily_rf = (1 + risk_free_rate) ** (1/self.TRADING_DAYS_PER_YEAR) - 1

        self.n_days = len(self.returns)

        if self.n_days < block_size * 2:
            warnings.warn(f"Only {self.n_days} days of data. Block size reduced to {self.n_days // 4}")
            self.block_size = max(5, self.n_days // 4)

    # =========================================================================
    # BLOCK BOOTSTRAP METHODS
    # =========================================================================

    def _generate_block_bootstrap_sample(self, target_length: Optional[int] = None) -> np.ndarray:
        """
        Generate a single bootstrap sample using block bootstrap.

        Block bootstrap randomly samples contiguous blocks of returns,
        preserving the autocorrelation structure within each block.

        Args:
            target_length: Length of output sample (default: same as original)

        Returns:
            Bootstrapped returns array
        """
        if target_length is None:
            target_length = self.n_days

        # # Calculate how many 20-day blocks needed to reach target length
        #  2,520 days / 20 = 126 blocks
        n_blocks = int(np.ceil(target_length / self.block_size))

        # Latest day where a full 20-day block can start
        # Example: 2,520 total days - 20 block size = can start up to day 2,500
        max_start = self.n_days - self.block_size

        if max_start <= 0:
            # Data too short for proper blocking, use simple bootstrap
            indices = np.random.randint(0, self.n_days, size=target_length)
            return self.returns[indices]

        # Randomly select block starting positions
        # Key point: Blocks can overlap - Sample with replacement
        block_starts = np.random.randint(0, max_start + 1, size=n_blocks)

        # Build bootstrapped sample from blocks
        bootstrapped = []
        for start in block_starts:
            block = self.returns[start:start + self.block_size]
            bootstrapped.extend(block)

        # Trim to exact target length
        return np.array(bootstrapped[:target_length])

    def _calculate_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics for a single simulation.

        Args:
            returns: Array of daily returns

        Returns:
            Dictionary with CAGR, Sharpe, Max DD, etc.
        """
        n = len(returns)
        years = n / self.TRADING_DAYS_PER_YEAR

        # Total return: Compound all daily returns
        cumulative = np.cumprod(1 + returns)
        total_return = cumulative[-1] - 1 if len(cumulative) > 0 else 0

        # CAGR: Annualised return rate
        # CAGR = (Ending Value / Beginning Value)^(1/Years) - 1
        
        if total_return > -1 and years > 0:
            cagr = (1 + total_return) ** (1 / years) - 1
        else:
            cagr = -1.0  # Edge case: complete loss (portfolio went to zero)
            
        # Volatility and Sharpe
        if len(returns) > 1:
            # Volatility = annualised std dev of daily returns
            # Daily volatility * sqrt(252) = annual volatility
            volatility = np.std(returns) * np.sqrt(self.TRADING_DAYS_PER_YEAR)
            #Excess returns = strategy returns - risk-free rate
            excess_returns = returns - self.daily_rf
            # Sharpe = (Mean Excess Return / Volatility) * sqrt(252)
            # Measure risk-adjusted return
            if volatility > 0:
                sharpe = (np.mean(excess_returns) / np.std(returns)) * np.sqrt(self.TRADING_DAYS_PER_YEAR)
            else:
                sharpe = 0.0 #No volatility means no risk, Sharpe undefined
        else:
            # Not enough data to calculate volatility/sharpe
            volatility = 0.0
            sharpe = 0.0

        # Max Drawdown: Worst peak-to-trough decline
        if len(cumulative) > 0:
            
            # Track highest point reached so far at each day
            # E.g : [2, 1, 1, 2, 3] -> [2, 2, 2, 2, 3]
            running_max = np.maximum.accumulate(cumulative)
            
            # Calculate drawdonws at each point : (current - peak) / peak
            drawdowns = (cumulative - running_max) / running_max
            max_dd = np.min(drawdowns)  # Most negative value
        else:
            max_dd = 0.0

        return {
            'total_return': total_return,   # Total return over period
            'cagr': cagr,                   # Compound Annual Growth Rate (annualised return)
            'sharpe': sharpe,               # Sharpe Ratio (risk-adjusted return)
            'volatility': volatility,       # Annualised volatility
            'max_dd': max_dd                # Maximum Drawdown (worst peak-to-trough)
        }

    def _run_single_simulation(self, seed: Optional[int] = None) -> Dict[str, float]:
        """Run a single Monte Carlo simulation."""
        # Set random seed for reproducibility (if provided)
        if seed is not None:
            np.random.seed(seed)

        # Calls the block boostrap function
        # Create one shuffled version of returns
        # Result: Array of daily returns in new random order 
        bootstrap_returns = self._generate_block_bootstrap_sample()

        # Calculate metrics
        return self._calculate_metrics(bootstrap_returns)

    # =========================================================================
    # MAIN SIMULATION
    # =========================================================================

    def run_simulation(
        self,
        n_simulations: int = 10000,
        confidence_level: float = 0.95,
        parallel: bool = False, # Not used because often slower for this
        verbose: bool = True
    ) -> MonteCarloResult:
        """
        Run full Monte Carlo simulation.

        Args:
            n_simulations: Number of simulations (default 10,000)
            confidence_level: Confidence level for intervals (default 95%)
            parallel: Use parallel processing (default False - often slower for this)
            verbose: Print progress updates

        Returns:
            MonteCarloResult with full analysis
        """
        if verbose:
            print(f"\n Running Monte Carlo Simulation...")
            print(f"  Simulations: {n_simulations:,}")
            print(f"  Block size: {self.block_size} days")
            print(f"  Original data: {self.n_days} days")

        # Pre-allocate arrays to store results from all simulations
        cagrs = np.zeros(n_simulations) # Array to hold CAGR results
        sharpes = np.zeros(n_simulations)  # Array to hold Sharpe results
        max_dds = np.zeros(n_simulations)   # Array to hold Max Drawdown results
        total_returns = np.zeros(n_simulations)     # Array to hold Total Return results
        volatilities = np.zeros(n_simulations)      # Array to hold Volatility results

        # Run simulations
        for i in range(n_simulations):
            if verbose and (i + 1) % 2000 == 0:
                print(f"  Progress: {i + 1:,}/{n_simulations:,} ({100 * (i + 1) / n_simulations:.0f}%)")

             # Run one simulation: shuffle returns and calculate metrics
            result = self._run_single_simulation()
            cagrs[i] = result['cagr']
            sharpes[i] = result['sharpe']
            max_dds[i] = result['max_dd']
            total_returns[i] = result['total_return']
            volatilities[i] = result['volatility']

        # Calculate actual strategy metrics
        actual_metrics = self._calculate_metrics(self.returns)

        # Calculate 95% confidence intervals
        alpha = 1 - confidence_level    # 0.05 for 95% CI
        lower_pct = alpha / 2 * 100     # 2.5th percentile
        upper_pct = (1 - alpha / 2) * 100   # 97.5th percentile

        # Calculate confidence intervals for each metric
        cagr_ci = (np.percentile(cagrs, lower_pct), np.percentile(cagrs, upper_pct))
        sharpe_ci = (np.percentile(sharpes, lower_pct), np.percentile(sharpes, upper_pct))
        # Note: Reversed for max drawdown (less negative = better)
        max_dd_ci = (np.percentile(max_dds, upper_pct), np.percentile(max_dds, lower_pct))  
        total_return_ci = (np.percentile(total_returns, lower_pct), np.percentile(total_returns, upper_pct))

        # Calculate risk probabilities
        # Proportion of simulations where bad outcome occured
        prob_loss_10pct = np.mean(total_returns < -0.10) # P(total return < -10%)
        prob_loss_20pct = np.mean(total_returns < -0.20) # P(total return < -20%)
        prob_negative_cagr = np.mean(cagrs < 0) # P(CAGR < 0%)
        prob_sharpe_below_zero = np.mean(sharpes < 0)   # P(Sharpe < 0)
        prob_sharpe_below_one = np.mean(sharpes < 1)  # P(Sharpe < 1)

        # Probability of ruin (max DD exceeds threshold)
        prob_ruin_10pct = np.mean(max_dds < -0.10)  # P(Max DD > 10%)
        prob_ruin_20pct = np.mean(max_dds < -0.20)  # P(Max DD > 20%)
        prob_ruin_30pct = np.mean(max_dds < -0.30)  # P(Max DD > 30%)
        prob_ruin_50pct = np.mean(max_dds < -0.50)  # P(Max DD > 50%)

          # 95th percentile = top 5% (optimistic outcome)
        best_case_cagr = np.percentile(cagrs, 95)
        best_case_sharpe = np.percentile(sharpes, 95)
        best_case_max_dd = np.percentile(max_dds, 95)  # Least negative drawdown
        
        # 5th percentile = bottom 5% (pessimistic outcome)
        worst_case_cagr = np.percentile(cagrs, 5)
        worst_case_sharpe = np.percentile(sharpes, 5)
        worst_case_max_dd = np.percentile(max_dds, 5)  # Most negative drawdown

        # Statistical significance (skill vs luck)
        
        cagr_percentile = stats.percentileofscore(cagrs, actual_metrics['cagr'])
        sharpe_percentile = stats.percentileofscore(sharpes, actual_metrics['sharpe'])
        # Consider significant if actual in top 5% for either metric
        is_significant = cagr_percentile >= 95 or sharpe_percentile >= 95

        # Summary statistics
        # Median = 50th percentile
        median_cagr = np.median(cagrs)
        median_sharpe = np.median(sharpes)
        median_max_dd = np.median(max_dds)
        # Standard deviation = spread of results (uncertainty measure)
        std_cagr = np.std(cagrs)
        std_sharpe = np.std(sharpes)

        if verbose:
            print(f" Simulation complete!")
            print(f"  Median CAGR: {median_cagr:.2%}")
            print(f"  Median Sharpe: {median_sharpe:.2f}")
            print(f"  Actual CAGR percentile: {cagr_percentile:.1f}%")

        # Package all results into MonteCarloResult dataclass
        return MonteCarloResult(
            # Simulation parameters
            n_simulations=n_simulations,
            block_size=self.block_size,
            original_days=self.n_days,
            # Full distributions (all 10,000 values for each metric)
            cagr_distribution=cagrs,
            sharpe_distribution=sharpes,
            max_dd_distribution=max_dds,
            total_return_distribution=total_returns,
            volatility_distribution=volatilities,
            # 95% confidence intervals
            cagr_ci=cagr_ci,
            sharpe_ci=sharpe_ci,
            max_dd_ci=max_dd_ci,
            total_return_ci=total_return_ci,
            # Risk probabilities
            prob_loss_10pct=prob_loss_10pct,
            prob_loss_20pct=prob_loss_20pct,
            prob_negative_cagr=prob_negative_cagr,
            prob_sharpe_below_zero=prob_sharpe_below_zero,
            prob_sharpe_below_one=prob_sharpe_below_one,
            # Probability of ruin
            prob_ruin_10pct=prob_ruin_10pct,
            prob_ruin_20pct=prob_ruin_20pct,
            prob_ruin_30pct=prob_ruin_30pct,
            prob_ruin_50pct=prob_ruin_50pct,
            # Best/worst case scenarios
            best_case_cagr=best_case_cagr,
            worst_case_cagr=worst_case_cagr,
            best_case_sharpe=best_case_sharpe,
            worst_case_sharpe=worst_case_sharpe,
            best_case_max_dd=best_case_max_dd,
            worst_case_max_dd=worst_case_max_dd,
            # Actual backtest results
            actual_cagr=actual_metrics['cagr'],
            actual_sharpe=actual_metrics['sharpe'],
            actual_max_dd=actual_metrics['max_dd'],
             # Statistical significance
            cagr_percentile=cagr_percentile,
            sharpe_percentile=sharpe_percentile,
            is_statistically_significant=is_significant,
            # Summary statistics
            median_cagr=median_cagr,
            median_sharpe=median_sharpe,
            median_max_dd=median_max_dd,
            std_cagr=std_cagr,
            std_sharpe=std_sharpe
        )

    # =========================================================================
    # TRADE-LEVEL BOOTSTRAP
    # =========================================================================

    def run_trade_bootstrap(
        self,
        n_simulations: int = 10000,
        verbose: bool = True
    ) -> Optional[MonteCarloResult]:
        """
        Run Monte Carlo using trade-level bootstrap.

        Instead of bootstrapping daily returns, this resamples trades
        to understand the distribution of possible outcomes.

        Useful when you have discrete trades rather than continuous returns.

        Args:
            n_simulations: Number of simulations
            verbose: Print progress

        Returns:
            MonteCarloResult or None if no trades available
        """
        if self.trades is None or len(self.trades) < 10:
            if verbose:
                print("Insufficient trade data for trade bootstrap (need >= 10 trades)")
            return None

        # Get trade returns
        if 'Return' in self.trades.columns:
            trade_returns = self.trades['Return'].values
        elif 'PnL' in self.trades.columns:
            # Convert PnL to returns (assuming you need entry values)
            trade_returns = self.trades['PnL'].values / 10000  # Approximate
        else:
            if verbose:
                print("No Return or PnL column in trades DataFrame")
            return None

        n_trades = len(trade_returns)

        if verbose:
            print(f"\n Running Trade Bootstrap Simulation...")
            print(f"  Simulations: {n_simulations:,}")
            print(f"  Trades: {n_trades}")

        # Storage
        total_returns = np.zeros(n_simulations)
        sharpes = np.zeros(n_simulations)
        max_dds = np.zeros(n_simulations)

        for i in range(n_simulations):
            if verbose and (i + 1) % 2000 == 0:
                print(f"  Progress: {i + 1:,}/{n_simulations:,}")

            # Resample trades with replacement
            bootstrap_trades = np.random.choice(trade_returns, size=n_trades, replace=True)

            # Total return (compounded)
            total_returns[i] = np.prod(1 + bootstrap_trades) - 1

            # Sharpe (approximation)
            if np.std(bootstrap_trades) > 0:
                sharpes[i] = np.mean(bootstrap_trades) / np.std(bootstrap_trades) * np.sqrt(n_trades)
            else:
                sharpes[i] = 0

            # Max drawdown (sequential)
            cumulative = np.cumprod(1 + bootstrap_trades)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            max_dds[i] = np.min(drawdowns)

        # Calculate percentiles and probabilities
        actual_total_return = np.prod(1 + trade_returns) - 1
        actual_sharpe = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(n_trades) if np.std(trade_returns) > 0 else 0

        # Build result (simplified)
        result = MonteCarloResult(
            n_simulations=n_simulations,
            block_size=1,  # Trade-level
            original_days=n_trades,
            total_return_distribution=total_returns,
            sharpe_distribution=sharpes,
            max_dd_distribution=max_dds,
            total_return_ci=(np.percentile(total_returns, 2.5), np.percentile(total_returns, 97.5)),
            sharpe_ci=(np.percentile(sharpes, 2.5), np.percentile(sharpes, 97.5)),
            max_dd_ci=(np.percentile(max_dds, 97.5), np.percentile(max_dds, 2.5)),
            prob_loss_10pct=np.mean(total_returns < -0.10),
            prob_loss_20pct=np.mean(total_returns < -0.20),
            prob_negative_cagr=np.mean(total_returns < 0),
            prob_sharpe_below_zero=np.mean(sharpes < 0),
            prob_sharpe_below_one=np.mean(sharpes < 1),
            prob_ruin_10pct=np.mean(max_dds < -0.10),
            prob_ruin_20pct=np.mean(max_dds < -0.20),
            prob_ruin_30pct=np.mean(max_dds < -0.30),
            prob_ruin_50pct=np.mean(max_dds < -0.50),
            best_case_cagr=np.percentile(total_returns, 95),
            worst_case_cagr=np.percentile(total_returns, 5),
            best_case_sharpe=np.percentile(sharpes, 95),
            worst_case_sharpe=np.percentile(sharpes, 5),
            best_case_max_dd=np.percentile(max_dds, 95),
            worst_case_max_dd=np.percentile(max_dds, 5),
            actual_cagr=actual_total_return,
            actual_sharpe=actual_sharpe,
            cagr_percentile=stats.percentileofscore(total_returns, actual_total_return),
            sharpe_percentile=stats.percentileofscore(sharpes, actual_sharpe),
            is_statistically_significant=stats.percentileofscore(total_returns, actual_total_return) >= 95,
            median_cagr=np.median(total_returns),
            median_sharpe=np.median(sharpes),
            median_max_dd=np.median(max_dds),
            std_cagr=np.std(total_returns),
            std_sharpe=np.std(sharpes)
        )

        if verbose:
            print(f" Trade bootstrap complete!")

        return result

    # =========================================================================
    # STRESS TESTING
    # =========================================================================

    def run_stress_test(
        self,
        n_simulations: int = 1000,
        stress_factor: float = 2.0,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run stress test with amplified negative returns.

        Simulates how strategy would perform if market conditions
        were significantly worse than historical.

        Args:
            n_simulations: Number of simulations
            stress_factor: Multiply negative returns by this factor
            verbose: Print progress

        Returns:
            Dictionary with stress test results
        """
        if verbose:
            print(f"\n Running Stress Test (factor={stress_factor}x)...")

        # Create stressed returns
        stressed_returns = self.returns.copy()
        stressed_returns[stressed_returns < 0] *= stress_factor

        # Run simulations with stressed returns
        stressed_cagrs = np.zeros(n_simulations)
        stressed_max_dds = np.zeros(n_simulations)

        for i in range(n_simulations):
            # Bootstrap from stressed returns
            n_blocks = int(np.ceil(self.n_days / self.block_size))
            max_start = len(stressed_returns) - self.block_size

            if max_start > 0:
                block_starts = np.random.randint(0, max_start + 1, size=n_blocks)
                bootstrap = []
                for start in block_starts:
                    bootstrap.extend(stressed_returns[start:start + self.block_size])
                bootstrap = np.array(bootstrap[:self.n_days])
            else:
                indices = np.random.randint(0, len(stressed_returns), size=self.n_days)
                bootstrap = stressed_returns[indices]

            # Calculate metrics
            metrics = self._calculate_metrics(bootstrap)
            stressed_cagrs[i] = metrics['cagr']
            stressed_max_dds[i] = metrics['max_dd']

        results = {
            'stress_factor': stress_factor,
            'n_simulations': n_simulations,
            'median_stressed_cagr': np.median(stressed_cagrs),
            'median_stressed_max_dd': np.median(stressed_max_dds),
            'prob_loss_under_stress': np.mean(stressed_cagrs < 0),
            'prob_ruin_20pct_under_stress': np.mean(stressed_max_dds < -0.20),
            'prob_ruin_50pct_under_stress': np.mean(stressed_max_dds < -0.50),
            'worst_case_cagr': np.percentile(stressed_cagrs, 5),
            'worst_case_max_dd': np.percentile(stressed_max_dds, 5),
            'stressed_cagr_ci': (np.percentile(stressed_cagrs, 2.5), np.percentile(stressed_cagrs, 97.5)),
            'stressed_max_dd_ci': (np.percentile(stressed_max_dds, 97.5), np.percentile(stressed_max_dds, 2.5))
        }

        if verbose:
            print(f"  Median CAGR under stress: {results['median_stressed_cagr']:.2%}")
            print(f"  Median Max DD under stress: {results['median_stressed_max_dd']:.2%}")
            print(f"  P(Loss) under stress: {results['prob_loss_under_stress']:.1%}")

        return results

    # =========================================================================
    # SEQUENCE OF RETURNS RISK
    # =========================================================================

    def analyze_sequence_risk(
        self,
        n_simulations: int = 1000,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze sequence of returns risk.

        Same returns in different order can produce very different outcomes.
        This shuffles the return sequence to understand path dependency.

        Args:
            n_simulations: Number of simulations
            verbose: Print progress

        Returns:
            Dictionary with sequence risk analysis
        """
        if verbose:
            print(f"\n Analyzing Sequence of Returns Risk...")

        # Original metrics
        original_metrics = self._calculate_metrics(self.returns)

        # Shuffle returns and recalculate
        shuffled_cagrs = np.zeros(n_simulations)
        shuffled_max_dds = np.zeros(n_simulations)

        for i in range(n_simulations):
            # Randomly permute returns (destroys autocorrelation)
            shuffled = np.random.permutation(self.returns)
            metrics = self._calculate_metrics(shuffled)
            shuffled_cagrs[i] = metrics['cagr']
            shuffled_max_dds[i] = metrics['max_dd']

        # Note: CAGR should be identical (geometric mean is order-independent)
        # But max drawdown varies significantly with sequence

        results = {
            'n_simulations': n_simulations,
            'original_max_dd': original_metrics['max_dd'],
            'median_shuffled_max_dd': np.median(shuffled_max_dds),
            'max_dd_range': (np.percentile(shuffled_max_dds, 5), np.percentile(shuffled_max_dds, 95)),
            'max_dd_std': np.std(shuffled_max_dds),
            'sequence_risk_factor': np.std(shuffled_max_dds) / abs(original_metrics['max_dd']) if original_metrics['max_dd'] != 0 else 0,
            'worst_sequence_max_dd': np.min(shuffled_max_dds),
            'best_sequence_max_dd': np.max(shuffled_max_dds),
            'original_percentile': stats.percentileofscore(shuffled_max_dds, original_metrics['max_dd'])
        }

        if verbose:
            print(f"  Original Max DD: {results['original_max_dd']:.2%}")
            print(f"  Median (shuffled): {results['median_shuffled_max_dd']:.2%}")
            print(f"  Range: [{results['max_dd_range'][0]:.2%}, {results['max_dd_range'][1]:.2%}]")
            print(f"  Original percentile: {results['original_percentile']:.1f}%")

        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_monte_carlo(
    returns: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    n_simulations: int = 10000,
    block_size: int = 20,
    verbose: bool = True
) -> MonteCarloResult:
    """
    Convenience function to run Monte Carlo simulation.

    Args:
        returns: Daily returns series
        trades: Optional trades DataFrame
        n_simulations: Number of simulations
        block_size: Block size for bootstrap
        verbose: Print progress

    Returns:
        MonteCarloResult
    """
    simulator = MonteCarloSimulator(returns, trades, block_size)
    return simulator.run_simulation(n_simulations, verbose=verbose)


def quick_monte_carlo(returns: pd.Series, n_simulations: int = 1000) -> Dict[str, float]:
    """
    Quick Monte Carlo with minimal output.

    Args:
        returns: Daily returns series
        n_simulations: Number of simulations (default 1000 for speed)

    Returns:
        Dictionary with key results
    """
    simulator = MonteCarloSimulator(returns)
    result = simulator.run_simulation(n_simulations, verbose=False)

    return {
        'cagr_ci_lower': result.cagr_ci[0],
        'cagr_ci_upper': result.cagr_ci[1],
        'sharpe_ci_lower': result.sharpe_ci[0],
        'sharpe_ci_upper': result.sharpe_ci[1],
        'prob_loss_20pct': result.prob_loss_20pct,
        'prob_ruin_30pct': result.prob_ruin_30pct,
        'is_significant': result.is_statistically_significant,
        'actual_cagr_percentile': result.cagr_percentile
    }


# =============================================================================
# TEST SCRIPT
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    from data_collector import DataCollector
    from technical_indicators import TechnicalIndicators
    from regime_detector import RegimeDetector
    from signal_generator import SignalGenerator
    from backtest_engine import BacktestEngine, BacktestConfig

    print("=" * 70)
    print("MONTE CARLO SIMULATION TEST")
    print("=" * 70)

    # Load data and run backtest
    print("\nLoading data and running backtest...")
    collector = DataCollector()
    data = collector.get_data('AAPL', years=10)

    ti = TechnicalIndicators(data)
    data = ti.calculate_all()

    rd = RegimeDetector(data)
    data = rd.detect_all_regimes()

    sg = SignalGenerator(data)
    data = sg.generate_signals()

    # Run backtest (disable stops for higher returns)
    config = BacktestConfig(
        use_stop_loss=False,
        use_take_profit=False,
        use_position_sizer=False
    )
    engine = BacktestEngine(data, config)
    results = engine.run_backtest()

    print(f"\nBacktest Results:")
    print(f"  Total Return: {results.total_return:.2%}")
    print(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {results.max_drawdown:.2%}")

    # Get returns from portfolio
    returns = engine.portfolio.returns()

    # Get trades
    try:
        trades_df = engine.portfolio.trades.records_readable
    except:
        trades_df = None

    print(f"\nData for Monte Carlo:")
    print(f"  Trading days: {len(returns)}")
    print(f"  Trades: {len(trades_df) if trades_df is not None else 0}")

    # Run Monte Carlo simulation
    print("\n" + "=" * 70)
    print("BLOCK BOOTSTRAP MONTE CARLO")
    print("=" * 70)

    simulator = MonteCarloSimulator(returns, trades_df, block_size=20)
    mc_result = simulator.run_simulation(n_simulations=10000)

    print(mc_result.get_summary())

    # Run stress test
    print("\n" + "=" * 70)
    print("STRESS TEST")
    print("=" * 70)
    stress_results = simulator.run_stress_test(n_simulations=1000, stress_factor=2.0)

    print(f"\nStress Test Results (2x negative returns):")
    print(f"  Median CAGR under stress: {stress_results['median_stressed_cagr']:.2%}")
    print(f"  Median Max DD under stress: {stress_results['median_stressed_max_dd']:.2%}")
    print(f"  P(Loss) under stress: {stress_results['prob_loss_under_stress']:.1%}")
    print(f"  P(Ruin 20%) under stress: {stress_results['prob_ruin_20pct_under_stress']:.1%}")

    # Sequence risk analysis
    print("\n" + "=" * 70)
    print("SEQUENCE OF RETURNS RISK")
    print("=" * 70)
    sequence_results = simulator.analyze_sequence_risk(n_simulations=1000)

    print(f"\nSequence Risk Analysis:")
    print(f"  Original Max DD: {sequence_results['original_max_dd']:.2%}")
    print(f"  Max DD range (shuffled): [{sequence_results['max_dd_range'][0]:.2%}, {sequence_results['max_dd_range'][1]:.2%}]")
    print(f"  Sequence risk factor: {sequence_results['sequence_risk_factor']:.2f}")

    # Trade bootstrap (if trades available)
    if trades_df is not None and len(trades_df) >= 10:
        print("\n" + "=" * 70)
        print("TRADE-LEVEL BOOTSTRAP")
        print("=" * 70)
        trade_result = simulator.run_trade_bootstrap(n_simulations=5000)
        if trade_result:
            print(f"\nTrade Bootstrap Results:")
            print(f"  Total Return CI: [{trade_result.total_return_ci[0]:.2%}, {trade_result.total_return_ci[1]:.2%}]")
            print(f"  P(Loss > 20%): {trade_result.prob_loss_20pct:.1%}")

    # Quick Monte Carlo
    print("\n" + "=" * 70)
    print("QUICK MONTE CARLO")
    print("=" * 70)
    quick_result = quick_monte_carlo(returns, n_simulations=1000)
    print(f"\nQuick Results:")
    for key, value in quick_result.items():
        if isinstance(value, float):
            if 'pct' in key or 'percentile' in key:
                print(f"  {key}: {value:.1f}%")
            elif 'ci' in key:
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("All Monte Carlo tests completed!")
    print("=" * 70)
