# Technical Analyst Agent

An AI-powered quantitative analysis agent that performs comprehensive technical analysis on publicly traded securities and generates professional investment reports with actionable trading recommendations.

## Overview

This agent simulates the role of a quantitative technical analyst by:

1. **Collecting Data**: Ingesting 10 years of OHLCV data from Yahoo Finance
2. **Computing Indicators**: Calculating 10 technical indicators across momentum, trend, volatility, and volume
3. **Detecting Market Regimes**: Classifying market conditions using Hurst exponent and volatility regime analysis
4. **Generating Signals**: Producing BUY/SELL/HOLD signals via multi-indicator confluence scoring
5. **Backtesting**: Validating strategy performance using vectorized simulation over historical data
6. **Monte Carlo Simulation**: Assessing statistical significance through bootstrap resampling (1000 simulations)
7. **Position Sizing**: Computing optimal allocation using Kelly criterion with GARCH volatility forecasting
8. **Generating Reports**: Creating LLM-powered investment reports with scenario analysis and component scoring

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd technical_analyst_agent

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Run full analysis (default: AAPL)
cd src
python llm_agent.py

# To analyze a different ticker, edit llm_agent.py line 2562:
# TICKER = 'NVDA'  # Change to desired ticker
```

### Output Files

After running, the agent generates three files in the `outputs/` directory:

| Format | File | Description |
|--------|------|-------------|
| PDF | `{TICKER}_analysis.pdf` | Professional report with executive summary, technical analysis, and recommendations |
| JSON | `{TICKER}_analysis.json` | Structured data containing all analysis results for programmatic use |
| TXT | `{TICKER}_analysis.txt` | Plain text summary for quick review |

## Sample Output

The JSON output contains comprehensive analysis data:

```json
{
  "metadata": {
    "ticker": "AAPL",
    "analysis_date": "2026-01-20",
    "current_price": 246.70
  },
  "recommendation": {
    "action": "BUY",
    "confidence": 0.39,
    "time_horizon": "swing",
    "rationale": "BUY signal generated due to extremely oversold RSI reading of 8.6..."
  },
  "trade_specifications": {
    "entry_price": 246.70,
    "stop_loss": 241.41,
    "stop_loss_pct": -2.14,
    "take_profit": 266.00,
    "target_1": 256.00,
    "target_2": 266.00,
    "target_3": 276.00,
    "risk_reward_ratio": 2.5,
    "position_size_pct": 0.05
  },
  "technical_analysis": {
    "momentum": { "rsi": 8.63, "rsi_signal": "oversold", "macd": -5.13, "macd_histogram": -1.63 },
    "trend": { "sma_50": 271.03, "sma_200": 233.87, "adx": 41.95, "trend_strength": "strong" },
    "volatility": { "atr": 5.29, "atr_pct": 2.15, "bb_percent_b": -0.09 },
    "volume_ratio": 1.69
  },
  "regime_analysis": {
    "market_regime": "SIDEWAYS",
    "volatility_regime": "LOW_VOLATILITY",
    "trend_persistence": "MEAN_REVERTING",
    "hurst_exponent": 0.436,
    "regime_confidence": 0.33
  },
  "scenario_analysis": {
    "bull_case": { "probability": 0.30, "target_price": 276.00, "return_pct": 11.89 },
    "base_case": { "probability": 0.45, "target_price": 266.00, "return_pct": 7.82 },
    "bear_case": { "probability": 0.25, "target_price": 241.41, "return_pct": -2.14 },
    "expected_value_pct": 5.89
  },
  "component_scores": {
    "overall": 73.7,
    "momentum": 70.1,
    "trend": 64.0,
    "volatility": 82.8,
    "volume": 84.4
  },
  "backtest_metrics": {
    "return_metrics": { "total_return": 1.086, "cagr": 0.077, "volatility": 0.097 },
    "risk_adjusted_metrics": { "sharpe_ratio": 0.31, "sortino_ratio": 0.20, "profit_factor": 1.81 },
    "trade_statistics": { "total_trades": 85, "win_rate": 0.47, "avg_trade_duration": 14.7 },
    "risk_metrics": { "max_drawdown": -0.126, "var_95": -0.008 }
  },
  "monte_carlo": {
    "cagr_percentile": 44.2,
    "is_statistically_significant": false,
    "probability_of_loss": 0.006
  },
  "position_sizing": {
    "full_kelly": 0.225,
    "fractional_kelly": 0.056,
    "garch_volatility_forecast": 0.252
  }
}
```

## Project Structure

```
technical_analyst_agent/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env                         # API keys (create this)
│
├── src/                         # Source modules
│   ├── __init__.py              # Package initialization
│   ├── data_collector.py        # Step 1: Price data ingestion from Yahoo Finance
│   ├── technical_indicators.py  # Step 2: Technical indicator calculation
│   ├── regime_detector.py       # Step 3: Market regime classification
│   ├── signal_generator.py      # Step 4: Signal generation with confluence scoring
│   ├── backtest_engine.py       # Step 5: Vectorized backtesting with VectorBT
│   ├── performance_metrics.py   # Step 6: Risk-adjusted performance analytics
│   ├── monte_carlo.py           # Step 7: Bootstrap simulation for significance testing
│   ├── position_sizer.py        # Step 8: Kelly criterion and GARCH-based sizing
│   ├── strategy_comparison.py   # Multi-strategy performance comparison
│   ├── visualisations.py        # Chart and visualization generation
│   └── llm_agent.py             # Main orchestration, LLM integration, report generation
│
├── data/
│   └── raw/                     # Cached price data in parquet format
│
└── outputs/                     # Generated analysis reports
    ├── {TICKER}_analysis.json
    ├── {TICKER}_analysis.txt
    └── {TICKER}_analysis.pdf
```

## Module Descriptions

### Step 1: data_collector.py

**Purpose**: Ingest and cache historical price data from Yahoo Finance.

**Functionality**:
- Fetches OHLCV (Open, High, Low, Close, Volume) data via yfinance API
- Configurable time periods (default: 10 years of daily data)
- Local caching in parquet format for faster subsequent runs
- Automatic handling of missing data, stock splits, and market holidays

**Output**: DataFrame with columns `[Open, High, Low, Close, Volume, Adj Close]` indexed by date.

---

### Step 2: technical_indicators.py

**Purpose**: Calculate technical indicators across four categories.

**Indicators Calculated**:

| Category | Indicator | Parameters | Interpretation |
|----------|-----------|------------|----------------|
| Momentum | RSI | 14-period | <30 oversold, >70 overbought |
| Momentum | MACD | 12/26/9 | Histogram shows momentum direction |
| Trend | SMA | 50, 200-period | Price vs MA indicates trend |
| Trend | EMA | 50, 200-period | Faster response to price changes |
| Trend | ADX | 14-period | >25 trending, <20 ranging |
| Trend | +DI / -DI | 14-period | Directional movement |
| Volatility | ATR | 14-period | Average daily price range |
| Volatility | Bollinger Bands | 20-period, 2 std | %B shows position within bands |
| Volume | Volume Ratio | 20-period MA | >1.5 indicates high volume |

**Derived Metrics**:
- 52-week high/low and percent from high
- Daily returns
- Price vs SMA50/SMA200 percentage
- Trend direction (SMA50 > SMA200)
- RSI zones (overbought/neutral/oversold)
- MACD crossover signals

**Output**: Original DataFrame augmented with ~30 indicator columns.

---

### Step 3: regime_detector.py

**Purpose**: Classify current market conditions to select appropriate trading strategy.

**Regime Detection Methods**:

| Method | Calculation | Output |
|--------|-------------|--------|
| Hurst Exponent | Rescaled Range (R/S) analysis over rolling window | H > 0.5: trending, H < 0.5: mean-reverting, H = 0.5: random walk |
| Volatility Regime | Parkinson volatility vs historical percentiles | LOW (<25th), NORMAL (25-75th), HIGH (>75th) |
| Trend Strength | ADX combined with price momentum | STRONG, MODERATE, WEAK |
| Regime Confidence | Agreement between multiple classification methods | 0-1 score |

**Strategy Mapping**:
```
Hurst > 0.5 + Low/Normal Volatility  → TREND_FOLLOWING
Hurst < 0.5 + Low/Normal Volatility  → MEAN_REVERSION
High Volatility (any Hurst)          → DEFENSIVE
```

**Output**: Columns `[Market_Regime, Volatility_Regime, Trend_Persistence, Hurst_Exponent, Regime_Confidence, Strategy]`

---

### Step 4: signal_generator.py

**Purpose**: Generate trading signals based on multi-indicator confluence and regime-adaptive rules.

**Confluence Scoring**:
- 6 indicators vote independently: RSI, MACD, MA Crossover, Bollinger Bands, ADX/DI, Volume
- Each indicator votes: +1 (bullish), -1 (bearish), or 0 (neutral)
- Confluence Score = Sum of votes / Number of indicators (range: -1 to +1)

**Signal Generation Rules**:

```
TREND_FOLLOWING Strategy:
  confluence >= 0.35  → STRONG_BUY
  confluence >= 0.15  → BUY
  confluence <= -0.15 → SELL
  confluence <= -0.35 → STRONG_SELL

MEAN_REVERSION Strategy (inverted logic):
  confluence <= -0.35 → STRONG_BUY  (fade extreme bearishness)
  confluence <= -0.15 → BUY
  confluence >= 0.25  → SELL        (fade extreme bullishness)
  confluence >= 0.50  → STRONG_SELL

DEFENSIVE Strategy:
  |confluence| >= 0.50 → Signal (requires strong agreement)
  Otherwise           → HOLD
```

**Confidence Calculation** (4-component system):
```
Confidence = 0.40 × Threshold_Margin    (how far past trigger threshold)
           + 0.30 × RSI_Extremity       (how extreme is RSI reading)
           + 0.20 × Regime_Confidence   (confidence in regime classification)
           + 0.10 × Base_Floor          (minimum for any triggered signal)
```

**Output**: Columns `[Signal, Signal_Confidence, Confluence_Score, Strategy, Stop_Loss, Take_Profit]`

---

### Step 5: backtest_engine.py

**Purpose**: Validate strategy performance through historical simulation using VectorBT.

**Functionality**:
- Vectorized portfolio simulation for performance
- Converts signals to entry/exit arrays
- Models transaction costs (commission and slippage)
- Implements risk management rules

**Risk Management**:
```
Stop Loss    = Entry Price - (2 × ATR)   → ~2% risk per trade
Take Profit  = Entry Price + (3 × ATR)   → 1.5:1 reward/risk ratio
```

**Position Sizing Integration**:
- Uses PositionSizer module for dynamic allocation
- Respects maximum position limits and portfolio heat constraints

**Output**: `BacktestResult` object containing returns series, trade records, and portfolio value history.

---

### Step 6: performance_metrics.py

**Purpose**: Calculate comprehensive risk-adjusted performance statistics.

**Metrics Calculated**:

| Category | Metrics |
|----------|---------|
| Return | Total Return, CAGR, Annualized Volatility |
| Risk-Adjusted | Sharpe Ratio, Sortino Ratio, Calmar Ratio, Omega Ratio |
| Drawdown | Maximum Drawdown, Drawdown Duration, Recovery Time |
| Trade Statistics | Win Rate, Profit Factor, Avg Win/Loss, Avg Trade Duration |
| Risk | VaR (95%, 99%), CVaR/Expected Shortfall, Tail Ratio |
| Statistical | T-statistic, P-value, Jarque-Bera test, Skewness, Kurtosis |

**Key Formulas**:
```
Sharpe Ratio  = (Return - Risk_Free_Rate) / Volatility
Sortino Ratio = (Return - Risk_Free_Rate) / Downside_Deviation
Profit Factor = Gross_Profits / Gross_Losses
Calmar Ratio  = CAGR / |Max_Drawdown|
```

**Output**: `PerformanceReport` dataclass with all metrics.

---

### Step 7: monte_carlo.py

**Purpose**: Assess statistical significance of backtest results through bootstrap simulation.

**Methodology**:
1. Take actual daily returns from backtest
2. Generate 1000 simulated equity curves by randomly resampling returns with replacement
3. Calculate CAGR for each simulation
4. Compare actual CAGR to distribution of simulated CAGRs

**Statistical Tests**:
```
CAGR Percentile     = Rank of actual CAGR in simulated distribution
Statistically Significant = True if percentile > 95
Probability of Loss = % of simulations with negative terminal value
```

**Distribution Outputs**:
- 5th, 25th, 50th, 75th, 95th percentile CAGRs
- Median Sharpe and Max Drawdown from simulations
- Confidence intervals for performance metrics

**Output**: `MonteCarloResult` with significance assessment and distribution statistics.

---

### Step 8: position_sizer.py

**Purpose**: Calculate optimal position sizes using quantitative methods.

**Methods Implemented**:

| Method | Formula | Description |
|--------|---------|-------------|
| Kelly Criterion | f* = (p × b - q) / b | Optimal fraction based on edge and odds |
| Fractional Kelly | f = 0.25 × Kelly | Conservative sizing (25% of optimal) |
| GARCH(1,1) | σ²_t = ω + α×ε²_{t-1} + β×σ²_{t-1} | Volatility forecasting |
| Volatility Targeting | Size = Target_Vol / Forecast_Vol | Normalize position to target volatility |

**Risk Constraints**:
- Maximum position size cap
- Maximum portfolio heat (total capital at risk)
- Volatility-adjusted sizing

**Output**: Recommended position size as percentage of portfolio with supporting calculations.

---

### Step 9: llm_agent.py (Main Orchestration)

**Purpose**: Coordinate analysis pipeline, generate AI-powered narratives, and produce multi-format reports.

**Pipeline Orchestration**:
1. Execute Steps 1-8 in sequence
2. Aggregate results into `AnalysisContext` object
3. Generate LLM-powered analysis and recommendations
4. Export to JSON, TXT, and PDF formats

**LLM Integration**:
- Supports Claude (Anthropic) and GPT-4 (OpenAI)
- Structured prompts for consistent output format
- Generates: investment thesis, technical summary, risk factors, scenario analysis

**Scenario Analysis Generation**:
```
Bull Case: Probability-weighted upside target with drivers
Base Case: Most likely outcome with key assumptions
Bear Case: Downside scenario with risk factors
Expected Value = Σ(Probability × Return) for all scenarios
```

**Component Scoring** (0-100 scale):
- Momentum Score: Based on RSI position and MACD histogram
- Trend Score: Based on price vs MAs and ADX strength
- Volatility Score: Based on ATR percentile and BB position
- Volume Score: Based on volume ratio confirmation

**Report Generation**:
- PDF: Professional format with sections for executive summary, technical analysis, regime analysis, scenario analysis, performance metrics, and recommendations
- JSON: Structured data for programmatic consumption
- TXT: Plain text summary

## Key Features

### Regime-Adaptive Strategy Selection

Different market conditions require different trading approaches:

- **Trending Markets (Hurst > 0.5)**: Momentum strategies outperform; system follows trends
- **Mean-Reverting Markets (Hurst < 0.5)**: Contrarian strategies outperform; system fades extremes
- **Volatile Markets**: Defensive positioning; system requires stronger signal confirmation

### Multi-Indicator Confluence

No single indicator is reliable in isolation. The system requires agreement:

- 6 indicators vote independently on market direction
- Signals only trigger when sufficient indicators agree
- Confluence score quantifies the level of agreement
- Higher confluence correlates with higher-probability trades

### Statistical Validation

Backtesting alone is insufficient due to overfitting risk:

- Monte Carlo simulation generates 1000 alternative scenarios
- Actual performance compared against random distribution
- Statistical significance requires outperforming 95% of simulations
- Provides probability estimates for various loss scenarios

### AI-Powered Analysis

LLM integration provides institutional-quality narratives:

- Professional investment thesis generation
- Scenario analysis with probability-weighted outcomes
- Risk factor identification with mitigation strategies
- Component scoring with plain-English interpretation

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# At least one API key required for LLM-powered reports
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
```

### Customizable Parameters

**Signal Thresholds** (`signal_generator.py` lines 362-390):
```python
# Adjust confluence thresholds for signal sensitivity
confluence >= 0.35  # STRONG_BUY (default)
confluence >= 0.15  # BUY (default)
```

**Risk Management** (`backtest_engine.py` lines 60-80):
```python
stop_loss_atr_mult: float = 2.0    # Stop loss = 2 × ATR
take_profit_atr_mult: float = 3.0  # Take profit = 3 × ATR
kelly_fraction: float = 0.25       # Use 25% of Kelly optimal
```

## Usage Examples

### Basic Usage

```python
import sys
sys.path.insert(0, 'src')

from data_collector import DataCollector
from technical_indicators import TechnicalIndicators
from regime_detector import RegimeDetector
from signal_generator import SignalGenerator

# Step 1: Collect data
collector = DataCollector()
data = collector.get_data('AAPL', years=10)

# Step 2: Calculate indicators
ti = TechnicalIndicators(data)
data = ti.calculate_all()

# Step 3: Detect regime
rd = RegimeDetector(data)
data = rd.detect_all_regimes()

# Step 4: Generate signals
sg = SignalGenerator(data)
data = sg.generate_signals()

# Access results
latest = data.iloc[-1]
print(f"Signal: {latest['Signal']}")
print(f"Confidence: {latest['Signal_Confidence']:.1%}")
print(f"Regime: {latest['Market_Regime']}")
print(f"Hurst: {latest['Hurst_Exponent']:.3f}")
```

### Full Analysis with Backtest and Report

```bash
# From src/ directory - runs complete pipeline
python llm_agent.py
```

## Dependencies

```
yfinance>=0.2.32     # Yahoo Finance API
pandas>=2.1.3        # DataFrame operations
numpy>=1.26.2        # Numerical computing
scipy>=1.11.4        # Statistical functions
vectorbt>=0.28.2     # Vectorized backtesting
matplotlib>=3.8.2    # Visualization
reportlab>=4.0.0     # PDF generation
anthropic>=0.18.0    # Claude API (optional)
openai>=1.3.7        # GPT-4 API (optional)
python-dotenv>=1.0.0 # Environment management
pyarrow>=14.0.0      # Parquet file support
```

## Troubleshooting

**1. No data returned for ticker**
- Verify ticker symbol is valid (e.g., AAPL, MSFT, GOOGL)
- Check internet connection
- Confirm ticker is available on Yahoo Finance

**2. Import errors**
- Ensure you're in the project root directory
- Activate virtual environment: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

**3. PDF generation fails**
- Install reportlab: `pip install reportlab`
- Check write permissions for `outputs/` directory

**4. LLM reports not generating**
- Verify API key is correctly set in `.env`
- Check API key validity and available credits
- System requires at least one valid API key

**5. Backtest returns errors**
- Ensure minimum 252 trading days of data
- Verify all indicator columns exist before backtesting

## Disclaimer

This software is for educational and research purposes only. It does not constitute financial advice. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.
