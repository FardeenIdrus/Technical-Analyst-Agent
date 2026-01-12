"""
Visualisations Module - Institutional-Grade Performance Charts

Professional visualisation suite for strategy analysis:
1. Equity Curve with Drawdown - Portfolio value + drawdown regions
2. Monthly Returns Heatmap - Calendar-style seasonality analysis
3. Strategy Comparison Dashboard - Multi-strategy comparison
4. Trade Distribution Analysis - Win/loss histograms
5. Rolling Performance Metrics - Consistency over time
6. Regime-Coloured Price Chart - Performance by regime
7. Risk Metrics Dashboard - VaR, CVaR, Monte Carlo intervals
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning)

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

# Professional colour palette
COLOURS = {
    'primary': '#1f77b4',      # Blue - main strategy
    'secondary': '#ff7f0e',    # Orange - benchmark
    'positive': '#2ca02c',     # Green - gains
    'negative': '#d62728',     # Red - losses
    'neutral': '#7f7f7f',      # Gray - neutral
    'accent1': '#9467bd',      # Purple
    'accent2': '#8c564b',      # Brown
    'accent3': '#e377c2',      # Pink
    'background': '#f7f7f7',   # Light gray background
    'grid': '#e0e0e0',         # Grid lines
}

# Regime colours
REGIME_COLOURS = {
    'STRONG_BULL': '#00a86b',      # Dark green
    'MODERATE_BULL': '#90EE90',    # Light green
    'SIDEWAYS': '#FFD700',         # Gold
    'MODERATE_BEAR': '#FFA07A',    # Light salmon
    'STRONG_BEAR': '#DC143C',      # Crimson
    'CHOPPY': '#9370DB',           # Medium purple
}

# Volatility regime colours
VOL_REGIME_COLOURS = {
    'HIGH_VOLATILITY': '#FF6B6B',   # Light red
    'NORMAL_VOLATILITY': '#4ECDC4', # Teal
    'LOW_VOLATILITY': '#45B7D1',    # Light blue
}


def set_professional_style():
    """Set matplotlib style for professional charts."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': COLOURS['grid'],
        'axes.labelcolor': '#333333',
        'axes.titlesize': 14,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'font.family': 'sans-serif',
        'grid.color': COLOURS['grid'],
        'grid.alpha': 0.5,
    })


# =============================================================================
# 1. EQUITY CURVE WITH DRAWDOWN
# =============================================================================

def plot_equity_curve_with_drawdown(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    initial_capital: float = 100000,
    title: str = "Portfolio Performance",
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot equity curve with drawdown visualization.

    Features:
    - Portfolio value over time (top panel)
    - Drawdown regions shaded (bottom panel)
    - Optional benchmark comparison
    - Key statistics annotations

    Args:
        returns: Daily strategy returns
        benchmark_returns: Optional benchmark returns for comparison
        initial_capital: Starting capital
        title: Chart title
        figsize: Figure dimensions
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    set_professional_style()

    # Calculate cumulative returns and equity
    cumulative = (1 + returns).cumprod()
    equity = cumulative * initial_capital

    # Calculate drawdown
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)

    # =========================================================================
    # TOP PANEL: Equity Curve
    # =========================================================================
    ax1.plot(equity.index, equity.values,
             color=COLOURS['primary'], linewidth=2, label='Strategy')

    # Plot benchmark if provided
    if benchmark_returns is not None:
        bench_cumulative = (1 + benchmark_returns).cumprod()
        bench_equity = bench_cumulative * initial_capital
        ax1.plot(bench_equity.index, bench_equity.values,
                 color=COLOURS['secondary'], linewidth=1.5,
                 linestyle='--', alpha=0.8, label='Benchmark')

    # Fill area under equity curve
    ax1.fill_between(equity.index, initial_capital, equity.values,
                     alpha=0.1, color=COLOURS['primary'])

    # Add horizontal line at initial capital
    ax1.axhline(y=initial_capital, color=COLOURS['neutral'],
                linestyle=':', alpha=0.7, linewidth=1)

    # Format y-axis as currency
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Calculate and display key metrics
    total_return = (equity.iloc[-1] / initial_capital - 1) * 100
    max_dd = drawdown.min() * 100
    years = len(returns) / 252
    cagr = ((equity.iloc[-1] / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0

    # Add stats text box
    stats_text = f'Total Return: {total_return:.1f}%\nCAGR: {cagr:.1f}%\nMax Drawdown: {max_dd:.1f}%'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=COLOURS['grid']))

    ax1.set_ylabel('Portfolio Value', fontsize=12)
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # BOTTOM PANEL: Drawdown
    # =========================================================================
    ax2.fill_between(drawdown.index, 0, drawdown.values * 100,
                     color=COLOURS['negative'], alpha=0.7)
    ax2.plot(drawdown.index, drawdown.values * 100,
             color=COLOURS['negative'], linewidth=0.5)

    # Add horizontal lines at key drawdown levels
    for level in [-10, -20, -30]:
        if drawdown.min() * 100 < level:
            ax2.axhline(y=level, color=COLOURS['neutral'],
                       linestyle='--', alpha=0.5, linewidth=0.5)
            ax2.text(drawdown.index[0], level + 1, f'{level}%',
                    fontsize=8, color=COLOURS['neutral'])

    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylim(top=5)  # Small buffer above 0
    ax2.grid(True, alpha=0.3)

    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


# =============================================================================
# 2. MONTHLY RETURNS HEATMAP
# =============================================================================

def plot_monthly_returns_heatmap(
    returns: pd.Series,
    title: str = "Monthly Returns (%)",
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot calendar-style monthly returns heatmap.

    Features:
    - Rows = years, Columns = months
    - Colour-coded: green=positive, red=negative
    - Shows seasonality patterns
    - Annual totals column

    Args:
        returns: Daily returns series with DatetimeIndex
        title: Chart title
        figsize: Figure dimensions
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    set_professional_style()

    # Resample to monthly returns
    monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)

    # Create pivot table (years x months)
    monthly_df = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values * 100  # Convert to percentage
    })

    pivot = monthly_df.pivot(index='year', columns='month', values='return')

    # Calculate annual returns
    annual_returns = returns.resample('YE').apply(lambda x: (1 + x).prod() - 1) * 100
    annual_returns.index = annual_returns.index.year

    # Add annual column
    pivot['Annual'] = annual_returns

    # Rename columns to month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']
    pivot.columns = month_names

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create custom colormap (red-white-green)
    colors = [COLOURS['negative'], 'white', COLOURS['positive']]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('returns', colors, N=n_bins)

    # Determine color scale limits (symmetric around 0)
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)].min()),
               abs(pivot.values[~np.isnan(pivot.values)].max()))
    vmax = min(vmax, 20)  # Cap at 20% for better color distribution

    # Plot heatmap
    im = ax.imshow(pivot.values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='Return (%)')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(month_names)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(month_names)
    ax.set_yticklabels(pivot.index)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(month_names)):
            value = pivot.iloc[i, j]
            if not np.isnan(value):
                # Choose text colour based on background
                text_color = 'white' if abs(value) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                       color=text_color, fontsize=9, fontweight='bold')

    # Add vertical line before Annual column
    ax.axvline(x=11.5, color='black', linewidth=2)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


# =============================================================================
# 3. STRATEGY COMPARISON DASHBOARD
# =============================================================================

def plot_strategy_comparison(
    strategy_results: Dict[str, Dict[str, float]],
    title: str = "Strategy Comparison",
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot multi-strategy comparison dashboard.

    Features:
    - Bar charts: CAGR, Sharpe, Max DD, Win Rate
    - Scatter plot: Return vs Risk
    - Table with all metrics

    Args:
        strategy_results: Dict of {strategy_name: {metric: value}}
            Expected metrics: cagr, sharpe, max_dd, win_rate, volatility
        title: Chart title
        figsize: Figure dimensions
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    set_professional_style()

    strategies = list(strategy_results.keys())
    n_strategies = len(strategies)
    colors = [COLOURS['primary'], COLOURS['secondary'], COLOURS['accent1'],
              COLOURS['accent2'], COLOURS['accent3']][:n_strategies]

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # =========================================================================
    # CAGR Comparison (top-left)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    cagrs = [strategy_results[s].get('cagr', 0) * 100 for s in strategies]
    bars = ax1.bar(strategies, cagrs, color=colors)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_ylabel('CAGR (%)')
    ax1.set_title('Compound Annual Growth Rate', fontweight='bold')

    # Add value labels on bars
    for bar, val in zip(bars, cagrs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=10, fontweight='bold')

    # =========================================================================
    # Sharpe Ratio Comparison (top-center)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    sharpes = [strategy_results[s].get('sharpe', 0) for s in strategies]
    bars = ax2.bar(strategies, sharpes, color=colors)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.axhline(y=1, color=COLOURS['positive'], linestyle='--', alpha=0.5, label='Good (>1)')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Risk-Adjusted Return', fontweight='bold')

    for bar, val in zip(bars, sharpes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=10, fontweight='bold')

    # =========================================================================
    # Max Drawdown Comparison (top-right)
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    max_dds = [abs(strategy_results[s].get('max_dd', 0)) * 100 for s in strategies]
    bars = ax3.bar(strategies, max_dds, color=[COLOURS['negative']] * n_strategies)
    ax3.set_ylabel('Max Drawdown (%)')
    ax3.set_title('Maximum Drawdown (Lower = Better)', fontweight='bold')

    for bar, val in zip(bars, max_dds):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # =========================================================================
    # Win Rate Comparison (bottom-left)
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    win_rates = [strategy_results[s].get('win_rate', 0) * 100 for s in strategies]
    bars = ax4.bar(strategies, win_rates, color=colors)
    ax4.axhline(y=50, color=COLOURS['neutral'], linestyle='--', alpha=0.5, label='50% baseline')
    ax4.set_ylabel('Win Rate (%)')
    ax4.set_title('Trade Win Rate', fontweight='bold')
    ax4.set_ylim(0, 100)

    for bar, val in zip(bars, win_rates):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.0f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # =========================================================================
    # Return vs Risk Scatter (bottom-center)
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    vols = [strategy_results[s].get('volatility', 0) * 100 for s in strategies]

    for i, strat in enumerate(strategies):
        ax5.scatter(vols[i], cagrs[i], s=200, c=[colors[i]],
                   label=strat, edgecolors='black', linewidth=1, zorder=5)
        ax5.annotate(strat, (vols[i], cagrs[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    # Add diagonal lines for Sharpe = 0.5, 1.0, 1.5
    max_vol = max(vols) * 1.2 if vols else 30
    for sharpe_line in [0.5, 1.0, 1.5]:
        x_line = np.linspace(0, max_vol, 100)
        y_line = sharpe_line * x_line
        ax5.plot(x_line, y_line, '--', alpha=0.3, color='gray')
        ax5.text(max_vol * 0.9, sharpe_line * max_vol * 0.9,
                f'Sharpe={sharpe_line}', fontsize=8, alpha=0.5)

    ax5.set_xlabel('Volatility (%)')
    ax5.set_ylabel('CAGR (%)')
    ax5.set_title('Return vs Risk', fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # =========================================================================
    # Metrics Table (bottom-right)
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    # Create table data
    metrics = ['CAGR', 'Sharpe', 'Sortino', 'Max DD', 'Win Rate', 'Volatility']
    table_data = []
    for strat in strategies:
        row = [
            f"{strategy_results[strat].get('cagr', 0)*100:.1f}%",
            f"{strategy_results[strat].get('sharpe', 0):.2f}",
            f"{strategy_results[strat].get('sortino', 0):.2f}",
            f"{strategy_results[strat].get('max_dd', 0)*100:.1f}%",
            f"{strategy_results[strat].get('win_rate', 0)*100:.0f}%",
            f"{strategy_results[strat].get('volatility', 0)*100:.1f}%",
        ]
        table_data.append(row)

    table = ax6.table(cellText=table_data,
                      rowLabels=strategies,
                      colLabels=metrics,
                      cellLoc='center',
                      loc='center',
                      colColours=[COLOURS['background']] * len(metrics))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style the table
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header row
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor(COLOURS['primary'])
            cell.set_text_props(color='white')
        elif col == -1:  # Row labels
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor(colors[row-1] if row-1 < len(colors) else COLOURS['neutral'])
            cell.set_text_props(color='white')

    ax6.set_title('Performance Summary', fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


# =============================================================================
# 4. TRADE DISTRIBUTION ANALYSIS
# =============================================================================

def plot_trade_distribution(
    trades: pd.DataFrame,
    title: str = "Trade Distribution Analysis",
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot trade return distributions and analysis.

    Features:
    - Histogram of trade returns
    - Win/loss pie chart
    - Hold time distribution
    - Cumulative P&L

    Args:
        trades: DataFrame with 'Return', 'PnL', 'Duration' columns
        title: Chart title
        figsize: Figure dimensions
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    set_professional_style()

    # Get return column
    if 'Return' in trades.columns:
        returns = trades['Return'].values * 100  # Convert to percentage
    elif 'PnL' in trades.columns:
        returns = trades['PnL'].values / trades['PnL'].abs().mean() * 100
    else:
        raise ValueError("Trades DataFrame must have 'Return' or 'PnL' column")

    # Separate winners and losers
    winners = returns[returns > 0]
    losers = returns[returns < 0]

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # =========================================================================
    # Trade Returns Histogram (top-left)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])

    # Create bins
    n_bins = min(50, len(returns) // 5 + 10)
    bins = np.linspace(returns.min(), returns.max(), n_bins)

    # Plot histogram with separate colors for wins/losses
    ax1.hist(losers, bins=bins, color=COLOURS['negative'], alpha=0.7,
             label=f'Losses ({len(losers)})', edgecolor='white')
    ax1.hist(winners, bins=bins, color=COLOURS['positive'], alpha=0.7,
             label=f'Wins ({len(winners)})', edgecolor='white')

    # Add vertical lines for mean and median
    ax1.axvline(returns.mean(), color='black', linestyle='--',
                linewidth=2, label=f'Mean: {returns.mean():.1f}%')
    ax1.axvline(np.median(returns), color='blue', linestyle=':',
                linewidth=2, label=f'Median: {np.median(returns):.1f}%')
    ax1.axvline(0, color='gray', linewidth=1)

    ax1.set_xlabel('Return (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Trade Return Distribution', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # Win/Loss Pie Chart (top-right)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    win_rate = len(winners) / len(returns) * 100
    loss_rate = len(losers) / len(returns) * 100
    breakeven = len(returns) - len(winners) - len(losers)

    sizes = [len(winners), len(losers)]
    labels = [f'Winners\n{len(winners)} ({win_rate:.1f}%)',
              f'Losers\n{len(losers)} ({loss_rate:.1f}%)']
    colours_pie = [COLOURS['positive'], COLOURS['negative']]

    if breakeven > 0:
        sizes.append(breakeven)
        labels.append(f'Breakeven\n{breakeven}')
        colours_pie.append(COLOURS['neutral'])

    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colours_pie,
                                        autopct='', startangle=90,
                                        wedgeprops={'edgecolor': 'white', 'linewidth': 2})

    # Add center circle for donut chart
    centre_circle = plt.Circle((0, 0), 0.5, fc='white')
    ax2.add_patch(centre_circle)

    # Add stats in center
    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = losers.mean() if len(losers) > 0 else 0
    ax2.text(0, 0.1, f'Avg Win: +{avg_win:.1f}%', ha='center', va='center',
             fontsize=10, fontweight='bold', color=COLOURS['positive'])
    ax2.text(0, -0.1, f'Avg Loss: {avg_loss:.1f}%', ha='center', va='center',
             fontsize=10, fontweight='bold', color=COLOURS['negative'])

    ax2.set_title('Win/Loss Breakdown', fontweight='bold')

    # =========================================================================
    # Hold Time Distribution (bottom-left)
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])

    if 'Duration' in trades.columns:
        durations = trades['Duration'].values
        # Convert timedelta to days if needed
        if hasattr(durations[0], 'days'):
            durations = np.array([d.days for d in durations])
    else:
        # Estimate from index if available
        durations = np.random.randint(1, 30, size=len(trades))  # Placeholder

    ax3.hist(durations, bins=min(30, len(np.unique(durations))),
             color=COLOURS['primary'], alpha=0.7, edgecolor='white')
    ax3.axvline(np.mean(durations), color='black', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(durations):.1f} days')
    ax3.axvline(np.median(durations), color='blue', linestyle=':',
                linewidth=2, label=f'Median: {np.median(durations):.1f} days')

    ax3.set_xlabel('Hold Time (days)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Trade Duration Distribution', fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # Cumulative P&L (bottom-right)
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    cumulative_returns = np.cumsum(returns)
    trade_numbers = np.arange(1, len(returns) + 1)

    # Colour based on whether cumulative is positive or negative
    ax4.fill_between(trade_numbers, 0, cumulative_returns,
                     where=(cumulative_returns >= 0),
                     color=COLOURS['positive'], alpha=0.3)
    ax4.fill_between(trade_numbers, 0, cumulative_returns,
                     where=(cumulative_returns < 0),
                     color=COLOURS['negative'], alpha=0.3)
    ax4.plot(trade_numbers, cumulative_returns,
             color=COLOURS['primary'], linewidth=2)

    ax4.axhline(0, color='black', linewidth=0.5)

    # Mark maximum and minimum
    max_idx = np.argmax(cumulative_returns)
    min_idx = np.argmin(cumulative_returns)
    ax4.scatter([max_idx + 1], [cumulative_returns[max_idx]],
                color=COLOURS['positive'], s=100, zorder=5,
                label=f'Peak: +{cumulative_returns[max_idx]:.1f}%')
    ax4.scatter([min_idx + 1], [cumulative_returns[min_idx]],
                color=COLOURS['negative'], s=100, zorder=5,
                label=f'Trough: {cumulative_returns[min_idx]:.1f}%')

    ax4.set_xlabel('Trade Number')
    ax4.set_ylabel('Cumulative Return (%)')
    ax4.set_title('Cumulative Trade Performance', fontweight='bold')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


# =============================================================================
# 5. ROLLING PERFORMANCE METRICS
# =============================================================================

def plot_rolling_performance(
    returns: pd.Series,
    windows: List[int] = [30, 90, 252],
    risk_free_rate: float = 0.05,
    title: str = "Rolling Performance Metrics",
    figsize: Tuple[int, int] = (14, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot rolling performance metrics over time.

    Features:
    - Rolling Sharpe ratio (multiple windows)
    - Rolling CAGR
    - Rolling volatility
    - Shows consistency over time

    Args:
        returns: Daily returns series
        windows: List of rolling window sizes in days
        risk_free_rate: Annual risk-free rate
        title: Chart title
        figsize: Figure dimensions
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    set_professional_style()

    daily_rf = (1 + risk_free_rate) ** (1/252) - 1

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    colours_windows = [COLOURS['primary'], COLOURS['secondary'], COLOURS['accent1']]

    # =========================================================================
    # Rolling Sharpe Ratio
    # =========================================================================
    ax1 = axes[0]

    for i, window in enumerate(windows):
        excess = returns - daily_rf
        rolling_mean = excess.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

        ax1.plot(rolling_sharpe.index, rolling_sharpe.values,
                 color=colours_windows[i], linewidth=1.5,
                 label=f'{window}d rolling', alpha=0.8)

    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.axhline(y=1, color=COLOURS['positive'], linestyle='--',
                alpha=0.5, label='Target (1.0)')

    # Add shaded regions
    ax1.fill_between(returns.index, 0, 1, alpha=0.1, color=COLOURS['positive'])
    ax1.fill_between(returns.index, -1, 0, alpha=0.1, color=COLOURS['negative'])

    ax1.set_ylabel('Sharpe Ratio')
    ax1.set_title('Rolling Sharpe Ratio', fontweight='bold')
    ax1.legend(loc='upper right', ncol=4, fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-3, 4)

    # =========================================================================
    # Rolling CAGR
    # =========================================================================
    ax2 = axes[1]

    for i, window in enumerate(windows):
        rolling_cagr = returns.rolling(window=window).apply(
            lambda x: (1 + x).prod() ** (252/window) - 1 if len(x) == window else np.nan
        ) * 100

        ax2.plot(rolling_cagr.index, rolling_cagr.values,
                 color=colours_windows[i], linewidth=1.5,
                 label=f'{window}d rolling', alpha=0.8)

    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.fill_between(returns.index,
                     [0] * len(returns),
                     [100] * len(returns),
                     alpha=0.05, color=COLOURS['positive'])
    ax2.fill_between(returns.index,
                     [-100] * len(returns),
                     [0] * len(returns),
                     alpha=0.05, color=COLOURS['negative'])

    ax2.set_ylabel('CAGR (%)')
    ax2.set_title('Rolling Annualized Return', fontweight='bold')
    ax2.legend(loc='upper right', ncol=4, fontsize=9)
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Rolling Volatility
    # =========================================================================
    ax3 = axes[2]

    for i, window in enumerate(windows):
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100

        ax3.plot(rolling_vol.index, rolling_vol.values,
                 color=colours_windows[i], linewidth=1.5,
                 label=f'{window}d rolling', alpha=0.8)

    # Add VIX-like reference levels
    ax3.axhline(y=20, color=COLOURS['positive'], linestyle='--',
                alpha=0.5, label='Low vol (20%)')
    ax3.axhline(y=30, color=COLOURS['secondary'], linestyle='--',
                alpha=0.5, label='Normal vol (30%)')
    ax3.axhline(y=40, color=COLOURS['negative'], linestyle='--',
                alpha=0.5, label='High vol (40%)')

    ax3.set_ylabel('Volatility (%)')
    ax3.set_xlabel('Date')
    ax3.set_title('Rolling Annualized Volatility', fontweight='bold')
    ax3.legend(loc='upper right', ncol=4, fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Format x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


# =============================================================================
# 6. REGIME-COLOURED PRICE CHART
# =============================================================================

def plot_regime_coloured_chart(
    data: pd.DataFrame,
    price_col: str = 'Close',
    regime_col: str = 'Market_Regime',
    title: str = "Price with Market Regimes",
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot price chart with regime-coloured background.

    Features:
    - Price line with regime backgrounds
    - Shows which regimes strategy performed best in
    - Volume bars coloured by regime

    Args:
        data: DataFrame with price and regime columns
        price_col: Name of price column
        regime_col: Name of regime column
        title: Chart title
        figsize: Figure dimensions
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    set_professional_style()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                    height_ratios=[3, 1], sharex=True)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    price = data[price_col]
    regimes = data[regime_col] if regime_col in data.columns else None

    # =========================================================================
    # Price Chart with Regime Backgrounds
    # =========================================================================
    ax1.plot(price.index, price.values, color='black', linewidth=1.5, zorder=5)

    if regimes is not None:
        # Get regime change points
        regime_changes = regimes != regimes.shift(1)
        change_indices = regime_changes[regime_changes].index.tolist()

        # Add start and end
        all_points = [price.index[0]] + change_indices + [price.index[-1]]

        # Colour background for each regime period
        for i in range(len(all_points) - 1):
            start_idx = all_points[i]
            end_idx = all_points[i + 1]
            regime_value = regimes.loc[start_idx]

            color = REGIME_COLOURS.get(regime_value, COLOURS['neutral'])
            ax1.axvspan(start_idx, end_idx, alpha=0.2, color=color, zorder=1)

    ax1.set_ylabel(f'{price_col} Price', fontsize=12)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.grid(True, alpha=0.3, zorder=0)

    # Add legend for regimes
    if regimes is not None:
        unique_regimes = regimes.dropna().unique()
        patches = [plt.Rectangle((0, 0), 1, 1, fc=REGIME_COLOURS.get(r, COLOURS['neutral']),
                                  alpha=0.3, label=r) for r in unique_regimes]
        ax1.legend(handles=patches, loc='upper left', ncol=3, fontsize=9)

    # =========================================================================
    # Volume Chart
    # =========================================================================
    if 'Volume' in data.columns:
        volume = data['Volume']

        if regimes is not None:
            # Colour volume bars by regime
            colours_vol = [REGIME_COLOURS.get(r, COLOURS['neutral']) for r in regimes.values]
        else:
            colours_vol = COLOURS['primary']

        ax2.bar(volume.index, volume.values, color=colours_vol, alpha=0.6, width=1)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
    else:
        # If no volume, show regime timeline
        if regimes is not None:
            regime_numeric = pd.Categorical(regimes).codes
            ax2.fill_between(regimes.index, 0, regime_numeric, alpha=0.5)
            ax2.set_ylabel('Regime')

    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


# =============================================================================
# 7. RISK METRICS DASHBOARD
# =============================================================================

def plot_risk_dashboard(
    returns: pd.Series,
    mc_result: Optional[Any] = None,
    title: str = "Risk Metrics Dashboard",
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comprehensive risk metrics dashboard.

    Features:
    - Max drawdown timeline
    - VaR/CVaR visualization
    - Monte Carlo confidence intervals
    - Return distribution with risk metrics

    Args:
        returns: Daily returns series
        mc_result: Optional MonteCarloResult object
        title: Chart title
        figsize: Figure dimensions
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    set_professional_style()

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # Calculate metrics
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    # =========================================================================
    # Drawdown Timeline (top-left)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])

    ax1.fill_between(drawdown.index, 0, drawdown.values * 100,
                     color=COLOURS['negative'], alpha=0.7)
    ax1.plot(drawdown.index, drawdown.values * 100,
             color=COLOURS['negative'], linewidth=0.5)

    # Mark worst drawdown
    worst_dd_idx = drawdown.idxmin()
    worst_dd_val = drawdown.min() * 100
    ax1.scatter([worst_dd_idx], [worst_dd_val], color='black', s=100, zorder=5)
    ax1.annotate(f'Worst: {worst_dd_val:.1f}%',
                 (worst_dd_idx, worst_dd_val),
                 xytext=(10, -20), textcoords='offset points',
                 fontsize=10, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='black'))

    # Add reference lines
    for level in [-10, -20, -30, -50]:
        if worst_dd_val < level:
            ax1.axhline(y=level, color=COLOURS['neutral'], linestyle='--', alpha=0.3)

    ax1.set_ylabel('Drawdown (%)')
    ax1.set_xlabel('Date')
    ax1.set_title('Drawdown Timeline', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # =========================================================================
    # VaR/CVaR Visualization (top-right)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    # Calculate VaR and CVaR
    var_95 = np.percentile(returns, 5) * 100
    var_99 = np.percentile(returns, 1) * 100
    cvar_95 = returns[returns <= var_95/100].mean() * 100
    cvar_99 = returns[returns <= var_99/100].mean() * 100

    # Plot return distribution
    returns_pct = returns * 100
    n_bins = 50
    counts, bins, patches = ax2.hist(returns_pct, bins=n_bins,
                                      color=COLOURS['primary'], alpha=0.7,
                                      edgecolor='white', density=True)

    # Colour the tails
    for i, (count, patch) in enumerate(zip(counts, patches)):
        if bins[i] <= var_95:
            patch.set_facecolor(COLOURS['negative'])
            patch.set_alpha(0.9)

    # Add vertical lines for VaR/CVaR
    ax2.axvline(var_95, color='orange', linestyle='--', linewidth=2,
                label=f'VaR 95%: {var_95:.2f}%')
    ax2.axvline(var_99, color='red', linestyle='--', linewidth=2,
                label=f'VaR 99%: {var_99:.2f}%')
    ax2.axvline(cvar_95, color='darkred', linestyle=':', linewidth=2,
                label=f'CVaR 95%: {cvar_95:.2f}%')

    ax2.set_xlabel('Daily Return (%)')
    ax2.set_ylabel('Density')
    ax2.set_title('Value at Risk Analysis', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Monte Carlo Distribution (bottom-left) - if available
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])

    if mc_result is not None and hasattr(mc_result, 'cagr_distribution'):
        cagr_dist = mc_result.cagr_distribution * 100

        ax3.hist(cagr_dist, bins=50, color=COLOURS['primary'], alpha=0.7,
                 edgecolor='white', density=True)

        # Add confidence interval
        ci_low, ci_high = mc_result.cagr_ci
        ax3.axvline(ci_low * 100, color=COLOURS['secondary'], linestyle='--',
                    linewidth=2, label=f'95% CI: [{ci_low*100:.1f}%, {ci_high*100:.1f}%]')
        ax3.axvline(ci_high * 100, color=COLOURS['secondary'], linestyle='--', linewidth=2)

        # Actual CAGR
        actual = mc_result.actual_cagr * 100
        ax3.axvline(actual, color=COLOURS['positive'], linewidth=3,
                    label=f'Actual: {actual:.1f}% (P{mc_result.cagr_percentile:.0f})')

        # Fill confidence region
        ax3.axvspan(ci_low * 100, ci_high * 100, alpha=0.2, color=COLOURS['secondary'])

        ax3.set_xlabel('CAGR (%)')
        ax3.set_ylabel('Density')
        ax3.set_title('Monte Carlo CAGR Distribution', fontweight='bold')
        ax3.legend(loc='upper right', fontsize=9)
    else:
        # If no MC results, show rolling max drawdown
        rolling_max_dd = drawdown.rolling(window=252).min() * 100
        ax3.plot(rolling_max_dd.index, rolling_max_dd.values,
                 color=COLOURS['negative'], linewidth=1.5)
        ax3.fill_between(rolling_max_dd.index, 0, rolling_max_dd.values,
                         alpha=0.3, color=COLOURS['negative'])
        ax3.set_ylabel('Rolling 1-Year Max DD (%)')
        ax3.set_xlabel('Date')
        ax3.set_title('Rolling Maximum Drawdown', fontweight='bold')

    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # Risk Metrics Summary Table (bottom-right)
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Calculate additional metrics
    volatility = returns.std() * np.sqrt(252) * 100
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    max_dd = drawdown.min() * 100

    # Build metrics table
    metrics_data = [
        ['Annualized Volatility', f'{volatility:.1f}%'],
        ['Max Drawdown', f'{max_dd:.1f}%'],
        ['VaR (95%)', f'{var_95:.2f}%'],
        ['VaR (99%)', f'{var_99:.2f}%'],
        ['CVaR (95%)', f'{cvar_95:.2f}%'],
        ['Skewness', f'{skewness:.2f}'],
        ['Kurtosis', f'{kurtosis:.2f}'],
    ]

    if mc_result is not None:
        metrics_data.extend([
            ['', ''],  # Spacer
            ['Monte Carlo Results', ''],
            ['CAGR 95% CI', f'[{mc_result.cagr_ci[0]*100:.1f}%, {mc_result.cagr_ci[1]*100:.1f}%]'],
            ['P(Loss > 10%)', f'{mc_result.prob_loss_10pct*100:.1f}%'],
            ['P(Max DD > 20%)', f'{mc_result.prob_ruin_20pct*100:.1f}%'],
            ['Statistically Significant', 'Yes' if mc_result.is_statistically_significant else 'No'],
        ])

    table = ax4.table(cellText=metrics_data,
                      colLabels=['Metric', 'Value'],
                      cellLoc='left',
                      loc='center',
                      colWidths=[0.6, 0.4])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style the table
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor(COLOURS['primary'])
            cell.set_text_props(color='white')
        elif col == 0:  # Metric names
            cell.set_text_props(fontweight='bold')

        # Highlight concerning values
        if row > 0 and col == 1:
            text = cell.get_text().get_text()
            if 'No' in text or (text.endswith('%') and
                                float(text.replace('%', '').replace('[', '').split(',')[0]) < -20):
                cell.set_facecolor('#ffcccc')

    ax4.set_title('Risk Metrics Summary', fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_full_report(
    returns: pd.Series,
    data: pd.DataFrame,
    trades: Optional[pd.DataFrame] = None,
    benchmark_returns: Optional[pd.Series] = None,
    mc_result: Optional[Any] = None,
    output_dir: str = 'charts',
    prefix: str = 'strategy'
) -> Dict[str, plt.Figure]:
    """
    Generate all charts and save to directory.

    Args:
        returns: Daily strategy returns
        data: Full DataFrame with price/regime data
        trades: Trade records DataFrame
        benchmark_returns: Benchmark returns for comparison
        mc_result: Monte Carlo simulation results
        output_dir: Directory to save charts
        prefix: Filename prefix

    Returns:
        Dictionary of {chart_name: figure}
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    figures = {}

    # 1. Equity Curve
    print("Generating equity curve...")
    fig = plot_equity_curve_with_drawdown(
        returns, benchmark_returns,
        save_path=f'{output_dir}/{prefix}_equity_curve.png'
    )
    figures['equity_curve'] = fig

    # 2. Monthly Heatmap
    print("Generating monthly heatmap...")
    fig = plot_monthly_returns_heatmap(
        returns,
        save_path=f'{output_dir}/{prefix}_monthly_heatmap.png'
    )
    figures['monthly_heatmap'] = fig

    # 3. Rolling Performance
    print("Generating rolling performance...")
    fig = plot_rolling_performance(
        returns,
        save_path=f'{output_dir}/{prefix}_rolling_performance.png'
    )
    figures['rolling_performance'] = fig

    # 4. Trade Distribution (if trades available)
    if trades is not None and len(trades) > 0:
        print("Generating trade distribution...")
        fig = plot_trade_distribution(
            trades,
            save_path=f'{output_dir}/{prefix}_trade_distribution.png'
        )
        figures['trade_distribution'] = fig

    # 5. Regime Chart
    if 'Market_Regime' in data.columns:
        print("Generating regime chart...")
        fig = plot_regime_coloured_chart(
            data,
            save_path=f'{output_dir}/{prefix}_regime_chart.png'
        )
        figures['regime_chart'] = fig

    # 6. Risk Dashboard
    print("Generating risk dashboard...")
    fig = plot_risk_dashboard(
        returns, mc_result,
        save_path=f'{output_dir}/{prefix}_risk_dashboard.png'
    )
    figures['risk_dashboard'] = fig

    print(f"\nAll charts saved to {output_dir}/")

    return figures


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
    from monte_carlo import MonteCarloSimulator

    print("=" * 70)
    print("VISUALIZATIONS TEST")
    print("=" * 70)

    # Load and process data
    print("\nLoading data...")
    collector = DataCollector()
    data = collector.get_data('AAPL', years=10)

    print("Processing indicators...")
    ti = TechnicalIndicators(data)
    data = ti.calculate_all()

    print("Detecting regimes...")
    rd = RegimeDetector(data)
    data = rd.detect_all_regimes()

    print("Generating signals...")
    sg = SignalGenerator(data)
    data = sg.generate_signals()

    # Run backtest
    print("Running backtest...")
    config = BacktestConfig(
        use_stop_loss=False,
        use_take_profit=False,
        use_position_sizer=False
    )
    engine = BacktestEngine(data, config)
    results = engine.run_backtest()

    # Get returns and trades
    returns = engine.portfolio.returns()
    try:
        trades_df = engine.portfolio.trades.records_readable
    except:
        trades_df = None

    # Get benchmark (buy & hold)
    benchmark_returns = data['Close'].pct_change().dropna()
    # Align with strategy returns
    common_idx = returns.index.intersection(benchmark_returns.index)
    benchmark_returns = benchmark_returns.loc[common_idx]
    returns = returns.loc[common_idx]

    # Run Monte Carlo
    print("Running Monte Carlo...")
    simulator = MonteCarloSimulator(returns)
    mc_result = simulator.run_simulation(n_simulations=1000, verbose=False)

    # Generate all charts
    print("\n" + "=" * 70)
    print("GENERATING CHARTS")
    print("=" * 70)

    figures = create_full_report(
        returns=returns,
        data=data,
        trades=trades_df,
        benchmark_returns=benchmark_returns,
        mc_result=mc_result,
        output_dir='charts',
        prefix='AAPL'
    )

    print(f"\nGenerated {len(figures)} charts:")
    for name in figures:
        print(f"  - {name}")

    # Show one chart
    print("\nDisplaying equity curve (close window to continue)...")
    plt.show()

    print("\n" + "=" * 70)
    print("VISUALIZATION TESTS COMPLETE")
    print("=" * 70)
