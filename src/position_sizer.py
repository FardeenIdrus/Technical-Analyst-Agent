"""
Position Sizer Module - Advanced Position Sizing for Trading Strategies

Implements multiple position sizing methods:
1. Kelly Criterion with Quarter-Kelly for safety
2. GARCH(1,1) Volatility Forecasting
3. Volatility Targeting (15% annualized default)
4. Risk Budgeting (portfolio heat and per-trade limits)

Author: Technical Analyst Agent
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union
from scipy.optimize import minimize
import warnings


class PositionSizer:
    """
    Advanced position sizing combining multiple methods for optimal risk-adjusted sizing.

    Methods:
    - Kelly Criterion: Optimal fraction based on win rate and payoff ratio
    - GARCH Volatility: Forward-looking volatility estimates
    - Volatility Targeting: Scale positions to target portfolio volatility
    - Risk Budgeting: Enforce portfolio heat and per-trade risk limits
    """

    def __init__(
        self,
        target_volatility: float = 0.30,  # 30% annualized target (matches typical stock vol)
        max_portfolio_heat: float = 0.10,  # 10% total capital at risk
        max_per_trade_risk: float = 0.02,  # 2% per trade
        kelly_fraction: float = 0.25,  # Quarter-Kelly
        lookback_trades: int = 100,  # Rolling window for Kelly stats
        garch_lookback: int = 252,  # 1 year of daily data for GARCH
        drawdown_scaling: bool = True,
        max_drawdown_threshold: float = 0.10  # Start scaling at 10% DD
    ):
        """
        Initialize the position sizer.

        Parameters:
        -----------
        target_volatility : float
            Target annualized portfolio volatility (default 15%)
        max_portfolio_heat : float
            Maximum total capital at risk across all positions (default 10%)
        max_per_trade_risk : float
            Maximum risk per individual trade (default 2%)
        kelly_fraction : float
            Fraction of Kelly to use (0.25 = Quarter-Kelly for safety)
        lookback_trades : int
            Number of recent trades for Kelly statistics
        garch_lookback : int
            Number of days for GARCH estimation
        drawdown_scaling : bool
            Whether to reduce positions during drawdowns
        max_drawdown_threshold : float
            Drawdown level at which to start reducing positions
        """
        self.target_volatility = target_volatility
        self.max_portfolio_heat = max_portfolio_heat
        self.max_per_trade_risk = max_per_trade_risk
        self.kelly_fraction = kelly_fraction
        self.lookback_trades = lookback_trades
        self.garch_lookback = garch_lookback
        self.drawdown_scaling = drawdown_scaling
        self.max_drawdown_threshold = max_drawdown_threshold

        # GARCH parameters (will be estimated)
        self.garch_omega = None
        self.garch_alpha = None
        self.garch_beta = None

        # Trade history for Kelly
        self.trade_history = []

    # =========================================================================
    # Kelly Criterion Methods
    # =========================================================================

    def calculate_kelly_fraction(
        self,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        trades: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Calculate Kelly Criterion optimal fraction.

        Formula: f* = (p * b - q) / b
        Where:
            p = probability of winning
            q = probability of losing (1 - p)
            b = ratio of average win to average loss (payoff ratio)

        Parameters:
        -----------
        win_rate : float, optional
            Win probability (0 to 1)
        avg_win : float, optional
            Average winning trade return
        avg_loss : float, optional
            Average losing trade return (positive number)
        trades : pd.DataFrame, optional
            DataFrame with 'pnl' or 'return' column to calculate stats

        Returns:
        --------
        dict with kelly metrics
        """
        # Calculate from trade history if provided
        if trades is not None and len(trades) > 0:
            if 'pnl' in trades.columns:
                returns = trades['pnl']
            elif 'return' in trades.columns:
                returns = trades['return']
            else:
                raise ValueError("trades must have 'pnl' or 'return' column")

            wins = returns[returns > 0]
            losses = returns[returns < 0]

            if len(wins) == 0 or len(losses) == 0:
                return {
                    'full_kelly': 0.0,
                    'fractional_kelly': 0.0,
                    'win_rate': len(wins) / len(returns) if len(returns) > 0 else 0.0,
                    'payoff_ratio': 0.0,
                    'edge': 0.0,
                    'num_trades': len(returns)
                }

            win_rate = len(wins) / len(returns)
            avg_win = wins.mean()
            avg_loss = abs(losses.mean())

        # Validate inputs
        if win_rate is None or avg_win is None or avg_loss is None:
            raise ValueError("Must provide either trades DataFrame or win_rate/avg_win/avg_loss")

        if avg_loss == 0:
            return {
                'full_kelly': 0.0,
                'fractional_kelly': 0.0,
                'win_rate': win_rate,
                'payoff_ratio': float('inf'),
                'edge': 0.0,
                'num_trades': len(trades) if trades is not None else 0
            }

        # Calculate Kelly
        q = 1 - win_rate
        b = avg_win / avg_loss  # Payoff ratio

        # Kelly formula: f* = (p * b - q) / b
        full_kelly = (win_rate * b - q) / b

        # Edge = expected value per unit risked
        edge = win_rate * b - q

        # Apply fractional Kelly
        fractional_kelly = max(0, full_kelly * self.kelly_fraction)

        return {
            'full_kelly': full_kelly,
            'fractional_kelly': fractional_kelly,
            'win_rate': win_rate,
            'payoff_ratio': b,
            'edge': edge,
            'num_trades': len(trades) if trades is not None else 0
        }

    def update_trade_history(self, trade_return: float):
        """Add a trade to history for rolling Kelly calculation."""
        self.trade_history.append(trade_return)

        # Keep only lookback_trades most recent
        if len(self.trade_history) > self.lookback_trades:
            self.trade_history = self.trade_history[-self.lookback_trades:]

    def get_rolling_kelly(self) -> Dict[str, float]:
        """Calculate Kelly based on rolling trade history."""
        if len(self.trade_history) < 10:
            return {
                'full_kelly': 0.0,
                'fractional_kelly': 0.0,
                'win_rate': 0.0,
                'payoff_ratio': 0.0,
                'edge': 0.0,
                'num_trades': len(self.trade_history),
                'warning': 'Insufficient trade history (need at least 10 trades)'
            }

        trades = pd.DataFrame({'return': self.trade_history})
        return self.calculate_kelly_fraction(trades=trades)

    # =========================================================================
    # GARCH Volatility Forecasting
    # =========================================================================

    def fit_garch(self, returns: pd.Series, verbose: bool = False) -> Dict[str, float]:
        """
        Fit GARCH(1,1) model to returns.

        Model: σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}

        Parameters:
        -----------
        returns : pd.Series
            Historical return series
        verbose : bool
            Print fitting information

        Returns:
        --------
        dict with GARCH parameters
        """
        returns = returns.dropna()

        if len(returns) < 50:
            warnings.warn("Insufficient data for GARCH estimation, using sample variance")
            sample_var = returns.var()
            self.garch_omega = sample_var * 0.1
            self.garch_alpha = 0.1
            self.garch_beta = 0.8
            return {
                'omega': self.garch_omega,
                'alpha': self.garch_alpha,
                'beta': self.garch_beta,
                'long_run_var': sample_var,
                'persistence': 0.9,
                'warning': 'Used default parameters due to insufficient data'
            }

        # Demean returns
        returns = returns - returns.mean()

        def garch_likelihood(params, returns):
            """Negative log-likelihood for GARCH(1,1)."""
            omega, alpha, beta = params

            # Constraints
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10

            n = len(returns)
            sigma2 = np.zeros(n)
            sigma2[0] = returns.var()  # Initialize with sample variance

            for t in range(1, n):
                sigma2[t] = omega + alpha * returns.iloc[t-1]**2 + beta * sigma2[t-1]
                sigma2[t] = max(sigma2[t], 1e-10)  # Floor to avoid numerical issues

            # Log-likelihood (ignoring constant)
            ll = -0.5 * np.sum(np.log(sigma2) + returns**2 / sigma2)

            return -ll  # Negative for minimization

        # Initial guesses
        sample_var = returns.var()
        init_params = [sample_var * 0.05, 0.1, 0.85]

        # Bounds
        bounds = [(1e-10, sample_var), (0, 0.5), (0, 0.99)]

        # Optimize
        try:
            result = minimize(
                garch_likelihood,
                init_params,
                args=(returns,),
                method='L-BFGS-B',
                bounds=bounds
            )

            self.garch_omega = result.x[0]
            self.garch_alpha = result.x[1]
            self.garch_beta = result.x[2]

        except Exception as e:
            if verbose:
                print(f"GARCH optimization failed: {e}, using defaults")
            self.garch_omega = sample_var * 0.05
            self.garch_alpha = 0.1
            self.garch_beta = 0.85

        # Long-run variance
        persistence = self.garch_alpha + self.garch_beta
        if persistence < 1:
            long_run_var = self.garch_omega / (1 - persistence)
        else:
            long_run_var = sample_var

        if verbose:
            print(f"GARCH(1,1) fitted: ω={self.garch_omega:.6f}, α={self.garch_alpha:.4f}, β={self.garch_beta:.4f}")
            print(f"Persistence: {persistence:.4f}, Long-run vol: {np.sqrt(long_run_var * 252):.2%}")

        return {
            'omega': self.garch_omega,
            'alpha': self.garch_alpha,
            'beta': self.garch_beta,
            'long_run_var': long_run_var,
            'persistence': persistence
        }

    def forecast_volatility(
        self,
        returns: pd.Series,
        horizon: int = 1,
        annualize: bool = True
    ) -> Dict[str, float]:
        """
        Forecast volatility using fitted GARCH model.

        Parameters:
        -----------
        returns : pd.Series
            Recent returns (need at least last value)
        horizon : int
            Forecast horizon in days
        annualize : bool
            Return annualized volatility

        Returns:
        --------
        dict with volatility forecasts
        """
        if self.garch_omega is None:
            self.fit_garch(returns)

        returns = returns.dropna()
        last_return = returns.iloc[-1] - returns.mean()

        # Calculate current variance estimate
        # Use exponentially weighted estimate for last variance
        ewma_var = returns.ewm(span=20).var().iloc[-1]

        # One-step ahead forecast
        sigma2_forecast = (self.garch_omega +
                          self.garch_alpha * last_return**2 +
                          self.garch_beta * ewma_var)

        # Multi-step forecast
        persistence = self.garch_alpha + self.garch_beta
        long_run_var = self.garch_omega / (1 - persistence) if persistence < 1 else ewma_var

        # h-step ahead forecast
        if horizon > 1:
            sigma2_h = (long_run_var +
                       (persistence ** (horizon - 1)) * (sigma2_forecast - long_run_var))
        else:
            sigma2_h = sigma2_forecast

        daily_vol = np.sqrt(sigma2_h)
        annualized_vol = daily_vol * np.sqrt(252) if annualize else daily_vol

        return {
            'daily_volatility': daily_vol,
            'annualized_volatility': daily_vol * np.sqrt(252),
            'forecast_horizon': horizon,
            'current_variance': ewma_var,
            'long_run_volatility': np.sqrt(long_run_var * 252)
        }

    # =========================================================================
    # Volatility Targeting
    # =========================================================================

    def calculate_vol_target_size(
        self,
        forecasted_vol: float,
        current_position: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate position size to target specific volatility.

        Position = Target Vol / Forecasted Vol

        Parameters:
        -----------
        forecasted_vol : float
            Forecasted annualized volatility
        current_position : float
            Current position size (for calculating adjustment)

        Returns:
        --------
        dict with sizing information
        """
        if forecasted_vol <= 0:
            warnings.warn("Forecasted volatility <= 0, using target volatility")
            forecasted_vol = self.target_volatility

        # Raw position size based on volatility
        target_position = (self.target_volatility / forecasted_vol)

        # Cap at 2x leverage
        target_position = min(target_position, 2.0)

        # Floor at 0.25 (minimum 25% position)
        target_position = max(target_position, 0.25)

        adjustment = target_position - current_position

        return {
            'target_position': target_position,
            'current_position': current_position,
            'adjustment': adjustment,
            'target_volatility': self.target_volatility,
            'forecasted_volatility': forecasted_vol,
            'vol_ratio': self.target_volatility / forecasted_vol
        }

    # =========================================================================
    # Risk Budgeting
    # =========================================================================

    def calculate_risk_budget_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        portfolio_value: float,
        current_heat: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate position size based on risk budget.

        Parameters:
        -----------
        entry_price : float
            Entry price for the trade
        stop_loss_price : float
            Stop loss price
        portfolio_value : float
            Current portfolio value
        current_heat : float
            Current portfolio heat (total risk across positions)

        Returns:
        --------
        dict with position sizing info
        """
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        risk_pct = risk_per_share / entry_price

        # Available risk budget
        available_heat = max(0, self.max_portfolio_heat - current_heat)

        # Risk this trade can take
        trade_risk = min(self.max_per_trade_risk, available_heat)

        # Dollar amount to risk
        dollar_risk = portfolio_value * trade_risk

        # Number of shares
        if risk_per_share > 0:
            shares = int(dollar_risk / risk_per_share)
            position_value = shares * entry_price
            position_pct = position_value / portfolio_value
        else:
            shares = 0
            position_value = 0
            position_pct = 0

        return {
            'shares': shares,
            'position_value': position_value,
            'position_pct': position_pct,
            'risk_per_share': risk_per_share,
            'risk_pct': risk_pct,
            'dollar_risk': shares * risk_per_share if shares > 0 else 0,
            'trade_risk_pct': trade_risk,
            'available_heat': available_heat,
            'remaining_heat': available_heat - (shares * risk_per_share / portfolio_value if shares > 0 else 0)
        }

    # =========================================================================
    # Drawdown Scaling
    # =========================================================================

    def calculate_drawdown_scalar(
        self,
        current_drawdown: float
    ) -> float:
        """
        Calculate position scaling factor based on current drawdown.

        Reduces position size as drawdown increases beyond threshold.
        At 2x threshold, position is reduced to 50%.
        At 3x threshold, position is reduced to 25%.

        Parameters:
        -----------
        current_drawdown : float
            Current portfolio drawdown (positive number, e.g., 0.15 = 15%)

        Returns:
        --------
        float: Scaling factor (0 to 1)
        """
        if not self.drawdown_scaling or current_drawdown <= self.max_drawdown_threshold:
            return 1.0

        # Linear reduction beyond threshold
        excess_dd = current_drawdown - self.max_drawdown_threshold

        # Scale from 1.0 to 0.25 as DD goes from threshold to 3x threshold
        scaling_range = 2 * self.max_drawdown_threshold
        reduction = min(excess_dd / scaling_range, 0.75)

        return max(0.25, 1.0 - reduction)

    # =========================================================================
    # Combined Position Sizing
    # =========================================================================

    def calculate_position_size(
        self,
        returns: pd.Series,
        entry_price: float,
        stop_loss_price: float,
        portfolio_value: float,
        current_heat: float = 0.0,
        current_drawdown: float = 0.0,
        trade_history: Optional[pd.DataFrame] = None,
        method: str = 'combined'
    ) -> Dict[str, any]:
        """
        Calculate optimal position size using specified method.

        Parameters:
        -----------
        returns : pd.Series
            Historical returns for volatility calculation
        entry_price : float
            Entry price
        stop_loss_price : float
            Stop loss price
        portfolio_value : float
            Current portfolio value
        current_heat : float
            Current portfolio heat
        current_drawdown : float
            Current drawdown
        trade_history : pd.DataFrame, optional
            Trade history for Kelly calculation
        method : str
            'kelly', 'volatility', 'risk_budget', or 'combined'

        Returns:
        --------
        dict with comprehensive position sizing
        """
        result = {
            'method': method,
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price,
            'portfolio_value': portfolio_value
        }

        # 1. GARCH Volatility Forecast
        vol_forecast = self.forecast_volatility(returns)
        result['volatility_forecast'] = vol_forecast

        # 2. Volatility Target Size
        vol_target = self.calculate_vol_target_size(vol_forecast['annualized_volatility'])
        result['vol_target_position'] = vol_target['target_position']

        # 3. Risk Budget Size
        risk_budget = self.calculate_risk_budget_size(
            entry_price, stop_loss_price, portfolio_value, current_heat
        )
        result['risk_budget'] = risk_budget

        # 4. Kelly Fraction (if trade history available)
        if trade_history is not None and len(trade_history) >= 10:
            kelly = self.calculate_kelly_fraction(trades=trade_history)
            result['kelly'] = kelly
            kelly_size = kelly['fractional_kelly']
        else:
            kelly_size = self.max_per_trade_risk  # Default to max per-trade risk
            result['kelly'] = {'note': 'Using default - insufficient trade history'}

        # 5. Drawdown Scaling
        dd_scalar = self.calculate_drawdown_scalar(current_drawdown)
        result['drawdown_scalar'] = dd_scalar

        # Calculate final position based on method
        if method == 'kelly':
            base_position_pct = kelly_size
        elif method == 'volatility':
            base_position_pct = vol_target['target_position']
        elif method == 'risk_budget':
            base_position_pct = risk_budget['position_pct']
        elif method == 'combined':
            # Combined: Use minimum of all methods for safety
            # This ensures we don't exceed any single limit
            positions = [
                vol_target['target_position'],
                risk_budget['position_pct'],
                kelly_size
            ]
            # Filter out zeros and take minimum
            valid_positions = [p for p in positions if p > 0]
            base_position_pct = min(valid_positions) if valid_positions else 0.0
        else:
            raise ValueError(f"Unknown method: {method}")

        # Apply drawdown scaling
        final_position_pct = base_position_pct * dd_scalar

        # Apply absolute limits
        final_position_pct = min(final_position_pct, 1.0)  # Max 100% of portfolio
        final_position_pct = max(final_position_pct, 0.0)  # No negative

        # Calculate shares
        position_value = portfolio_value * final_position_pct
        shares = int(position_value / entry_price)
        actual_position_value = shares * entry_price
        actual_position_pct = actual_position_value / portfolio_value

        result['final_position'] = {
            'position_pct': final_position_pct,
            'actual_position_pct': actual_position_pct,
            'shares': shares,
            'position_value': actual_position_value,
            'base_position_pct': base_position_pct,
            'drawdown_adjustment': dd_scalar
        }

        return result

    def get_position_size_pct(
        self,
        returns: pd.Series,
        current_drawdown: float = 0.0,
        trade_history: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Simplified method to get position size as percentage.

        For integration with BacktestEngine.

        Parameters:
        -----------
        returns : pd.Series
            Historical returns
        current_drawdown : float
            Current drawdown
        trade_history : pd.DataFrame, optional
            Trade history for Kelly

        Returns:
        --------
        float: Position size as fraction (0.0 to 1.0)
        """
        # Get volatility forecast
        vol_forecast = self.forecast_volatility(returns)
        forecasted_vol = vol_forecast['annualized_volatility']

        # Volatility target position
        vol_position = min(self.target_volatility / forecasted_vol, 2.0)

        # Kelly position (if available)
        if trade_history is not None and len(trade_history) >= 10:
            kelly = self.calculate_kelly_fraction(trades=trade_history)
            kelly_position = kelly['fractional_kelly']
        else:
            kelly_position = 1.0  # No constraint if no history

        # Take minimum
        base_position = min(vol_position, kelly_position, 1.0)

        # Apply drawdown scaling
        dd_scalar = self.calculate_drawdown_scalar(current_drawdown)
        final_position = base_position * dd_scalar

        return max(0.0, min(1.0, final_position))

    # =========================================================================
    # Reporting
    # =========================================================================

    def generate_sizing_report(
        self,
        returns: pd.Series,
        trade_history: Optional[pd.DataFrame] = None
    ) -> str:
        """Generate a text report of position sizing analysis."""
        report = []
        report.append("=" * 60)
        report.append("POSITION SIZING ANALYSIS REPORT")
        report.append("=" * 60)

        # GARCH Analysis
        report.append("\n[GARCH VOLATILITY MODEL]")
        garch_params = self.fit_garch(returns, verbose=False)
        vol_forecast = self.forecast_volatility(returns)

        report.append(f"  Parameters:")
        report.append(f"    ω (omega):  {garch_params['omega']:.6f}")
        report.append(f"    α (alpha):  {garch_params['alpha']:.4f}")
        report.append(f"    β (beta):   {garch_params['beta']:.4f}")
        report.append(f"  Persistence:  {garch_params['persistence']:.4f}")
        report.append(f"  Long-run vol: {np.sqrt(garch_params['long_run_var'] * 252):.2%}")
        report.append(f"  Forecast vol: {vol_forecast['annualized_volatility']:.2%}")

        # Volatility Targeting
        report.append("\n[VOLATILITY TARGETING]")
        vol_target = self.calculate_vol_target_size(vol_forecast['annualized_volatility'])
        report.append(f"  Target vol:     {self.target_volatility:.2%}")
        report.append(f"  Forecasted vol: {vol_forecast['annualized_volatility']:.2%}")
        report.append(f"  Position size:  {vol_target['target_position']:.2%}")

        # Kelly Criterion
        report.append("\n[KELLY CRITERION]")
        if trade_history is not None and len(trade_history) >= 10:
            kelly = self.calculate_kelly_fraction(trades=trade_history)
            report.append(f"  Trades analyzed: {kelly['num_trades']}")
            report.append(f"  Win rate:        {kelly['win_rate']:.2%}")
            report.append(f"  Payoff ratio:    {kelly['payoff_ratio']:.2f}")
            report.append(f"  Edge:            {kelly['edge']:.4f}")
            report.append(f"  Full Kelly:      {kelly['full_kelly']:.2%}")
            report.append(f"  {self.kelly_fraction:.0%} Kelly:     {kelly['fractional_kelly']:.2%}")
        else:
            report.append("  Insufficient trade history for Kelly calculation")

        # Risk Budgeting Parameters
        report.append("\n[RISK BUDGETING]")
        report.append(f"  Max portfolio heat: {self.max_portfolio_heat:.2%}")
        report.append(f"  Max per-trade risk: {self.max_per_trade_risk:.2%}")
        report.append(f"  Drawdown scaling:   {'Enabled' if self.drawdown_scaling else 'Disabled'}")
        report.append(f"  DD threshold:       {self.max_drawdown_threshold:.2%}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    print("Testing Position Sizer Module")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    returns = pd.Series(np.random.normal(0.0005, 0.02, 500), index=dates)

    # Add some volatility clustering
    for i in range(1, len(returns)):
        if abs(returns.iloc[i-1]) > 0.03:
            returns.iloc[i] *= 1.5

    # Create sample trade history
    trade_returns = np.random.choice(
        [-0.02, -0.015, -0.01, 0.02, 0.025, 0.03, 0.035],
        size=50,
        p=[0.1, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1]
    )
    trade_history = pd.DataFrame({'return': trade_returns})

    # Initialize sizer
    sizer = PositionSizer(
        target_volatility=0.15,
        max_portfolio_heat=0.10,
        max_per_trade_risk=0.02,
        kelly_fraction=0.25
    )

    # Test 1: GARCH Fitting
    print("\n[Test 1: GARCH Model]")
    garch = sizer.fit_garch(returns, verbose=True)

    # Test 2: Volatility Forecast
    print("\n[Test 2: Volatility Forecast]")
    vol = sizer.forecast_volatility(returns)
    print(f"Daily vol: {vol['daily_volatility']:.4f}")
    print(f"Annual vol: {vol['annualized_volatility']:.2%}")
    print(f"Long-run vol: {vol['long_run_volatility']:.2%}")

    # Test 3: Kelly Criterion
    print("\n[Test 3: Kelly Criterion]")
    kelly = sizer.calculate_kelly_fraction(trades=trade_history)
    print(f"Win rate: {kelly['win_rate']:.2%}")
    print(f"Payoff ratio: {kelly['payoff_ratio']:.2f}")
    print(f"Full Kelly: {kelly['full_kelly']:.2%}")
    print(f"Quarter Kelly: {kelly['fractional_kelly']:.2%}")

    # Test 4: Volatility Targeting
    print("\n[Test 4: Volatility Targeting]")
    vol_target = sizer.calculate_vol_target_size(vol['annualized_volatility'])
    print(f"Target position: {vol_target['target_position']:.2%}")

    # Test 5: Risk Budget
    print("\n[Test 5: Risk Budget]")
    risk = sizer.calculate_risk_budget_size(
        entry_price=150.0,
        stop_loss_price=145.0,
        portfolio_value=100000.0,
        current_heat=0.03
    )
    print(f"Shares: {risk['shares']}")
    print(f"Position value: ${risk['position_value']:,.2f}")
    print(f"Position %: {risk['position_pct']:.2%}")
    print(f"Dollar risk: ${risk['dollar_risk']:,.2f}")

    # Test 6: Drawdown Scaling
    print("\n[Test 6: Drawdown Scaling]")
    for dd in [0.05, 0.10, 0.15, 0.20, 0.30]:
        scalar = sizer.calculate_drawdown_scalar(dd)
        print(f"  {dd:.0%} drawdown → {scalar:.2%} position")

    # Test 7: Combined Position Sizing
    print("\n[Test 7: Combined Position Sizing]")
    combined = sizer.calculate_position_size(
        returns=returns,
        entry_price=150.0,
        stop_loss_price=145.0,
        portfolio_value=100000.0,
        current_heat=0.02,
        current_drawdown=0.08,
        trade_history=trade_history,
        method='combined'
    )
    print(f"Final position: {combined['final_position']['position_pct']:.2%}")
    print(f"Shares: {combined['final_position']['shares']}")
    print(f"Value: ${combined['final_position']['position_value']:,.2f}")

    # Test 8: Full Report
    print("\n[Test 8: Full Report]")
    report = sizer.generate_sizing_report(returns, trade_history)
    print(report)

    print("\n✓ All tests completed successfully!")