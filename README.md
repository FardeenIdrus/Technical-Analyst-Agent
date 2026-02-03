# Technical Analyst Agent

An AI-powered quantitative analysis agent that performs comprehensive technical analysis on publicly traded securities and generates professional investment reports with actionable trading recommendations.

## Overview

This agent simulates the role of a quantitative technical analyst by:

1. **Collecting Data**: Ingesting 10 years of OHLCV data from Yahoo Finance
2. **Computing Indicators**: Calculating 10 technical indicators across momentum, trend, volatility, and volume categories
3. **Detecting Market Regimes**: Classifying market conditions using Hurst exponent and volatility regime analysis
4. **Generating Signals**: Producing BUY/SELL/HOLD signals via multi-indicator confluence scoring with regime-adaptive strategy selection
5. **Backtesting**: Validating strategy performance using vectorized simulation over 10 years of historical data
6. **Calculating Performance Metrics**: Computing 25+ risk-adjusted metrics including Probabilistic Sharpe and statistical significance tests
7. **Running Monte Carlo Simulation**: Assessing statistical significance through 1000 bootstrap simulations
8. **Position Sizing**: Computing optimal allocation using Kelly criterion with GARCH(1,1) volatility forecasting
9. **Generating AI Reports**: Creating LLM-powered investment reports with scenario analysis, risk factors, catalysts, and actionable recommendations

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
| PDF | `{TICKER}_analysis.pdf` | Professional report with executive summary, technical analysis, scenario analysis, and recommendations |
| JSON | `{TICKER}_analysis.json` | Structured data containing all analysis results for programmatic use |
| TXT | `{TICKER}_analysis.txt` | Plain text summary for quick review |

## Sample Output

The JSON output contains comprehensive analysis across 15 sections:

```json
{
  "metadata": {
    "ticker": "AAPL",
    "analysis_date": "2026-01-29",
    "current_price": 257.64,
    "generated_by": "LLMAgent",
    "model": "gpt-4o"
  },
  "recommendation": {
    "action": "HOLD",
    "confidence": 0.067,
    "time_horizon": "swing",
    "rationale": "The HOLD signal was generated due to a confluence of neutral technical indicators and a sideways market regime. The RSI at 53.4 indicates neutral momentum, neither overbought nor oversold. The MACD histogram is slightly positive, but the lack of crossover indicates insufficient strength. The confluence score of 0.00 reflects the offsetting nature of these indicators."
  },
  "trade_specifications": {
    "entry_price": 257.64,
    "stop_loss": 252.17,
    "stop_loss_pct": -2.12,
    "take_profit": 268.44,
    "take_profit_pct": 4.19,
    "risk_reward_ratio": 1.5,
    "target_1": 262.00,
    "target_2": 268.44,
    "target_3": 275.00,
    "position_size_pct": 0.05
  },
  "technical_analysis": {
    "momentum": {
      "rsi": 53.42,
      "rsi_signal": "neutral",
      "macd": -4.13,
      "macd_signal": -4.74,
      "macd_histogram": 0.61
    },
    "trend": {
      "sma_50": 268.44,
      "sma_200": 236.09,
      "price_vs_sma50_pct": -4.02,
      "price_vs_sma200_pct": 9.13,
      "adx": 35.54,
      "trend_strength": "strong"
    },
    "volatility": {
      "atr": 5.47,
      "atr_pct": 2.12,
      "bb_percent_b": 0.48
    },
    "levels": {
      "bb_upper": 272.26,
      "bb_middle": 258.24,
      "bb_lower": 244.21
    },
    "volume_ratio": 0.57,
    "summary": "AAPL's technical structure reveals a stock in neutral territory with RSI at 53.4. Price action at $257.64 is 4% below SMA 50, suggesting short-term weakness, yet 9.1% above SMA 200, indicating longer-term strength. ADX at 35.5 indicates a strong trend, yet the sideways market regime suggests this strength lacks directional clarity."
  },
  "regime_analysis": {
    "market_regime": "SIDEWAYS",
    "volatility_regime": "NORMAL_VOLATILITY",
    "trend_persistence": "RANDOM_WALK",
    "hurst_exponent": 0.533,
    "hurst_interpretation": "trending",
    "regime_confidence": 0.336
  },
  "signal_system": {
    "current_signal": "HOLD",
    "signal_confidence": 0.067,
    "confluence_score": 0.0,
    "active_strategy": "TREND_FOLLOWING"
  },
  "scenario_analysis": {
    "bull_case": {
      "probability": 0.30,
      "target_price": 275.00,
      "return_pct": 6.73,
      "drivers": ["Strong earnings report", "Positive market sentiment"]
    },
    "base_case": {
      "probability": 0.45,
      "target_price": 268.44,
      "return_pct": 4.19,
      "drivers": ["Stable market conditions", "Neutral technical indicators"]
    },
    "bear_case": {
      "probability": 0.25,
      "target_price": 252.17,
      "return_pct": -2.12,
      "drivers": ["Market downturn", "Negative macroeconomic data"]
    },
    "expected_value_pct": 2.98
  },
  "key_levels": {
    "support": [252.17, 236.09],
    "resistance": [268.44, 275.00]
  },
  "component_scores": {
    "overall": 49.4,
    "momentum": 50.9,
    "trend": 59.5,
    "volatility": 53.0,
    "volume": 28.4
  },
  "risk_factors": [
    "Unexpected macroeconomic data causing volatility; mitigation through dynamic hedging",
    "Geopolitical tensions impacting tech sector; mitigation through diversified portfolio",
    "Earnings report surprise; mitigation through options strategies",
    "Regulatory changes affecting tech companies; mitigation through sector rotation"
  ],
  "catalysts": [
    "Bullish catalyst: Positive earnings surprise",
    "Bullish catalyst: New product launch",
    "Bearish catalyst: Regulatory scrutiny",
    "Bearish catalyst: Supply chain disruptions"
  ],
  "investment_thesis": "The HOLD recommendation for AAPL is compelling due to the current sideways market regime and neutral technical indicators, which suggest limited directional bias. This stance allows for flexibility in response to potential market shifts, making it an institutional-quality opportunity as it balances risk and reward.",
  "performance_analysis": {
    "overall_assessment": "fair",
    "confidence_in_edge": 0.4,
    "summary": "The strategy exhibits a moderate CAGR of 7.4% with a strong profit factor, but its risk-adjusted performance and statistical significance are lacking.",
    "strengths": [
      "Strong profit factor of 1.78",
      "Low probability of significant loss (0.2%)"
    ],
    "weaknesses": [
      "Low Sharpe ratio of 0.28",
      "CAGR percentile at 44th, indicating underperformance against random baseline"
    ],
    "warnings": [
      "Statistically insignificant results",
      "Low win rate of 47.7%"
    ],
    "suggestions": [
      "Improve risk management to enhance Sharpe ratio",
      "Increase trade frequency or improve entry criteria to boost win rate"
    ]
  },
  "backtest_metrics": {
    "return_metrics": {
      "total_return": 1.029,
      "cagr": 0.0735,
      "volatility": 0.0974
    },
    "risk_adjusted_metrics": {
      "sharpe_ratio": 0.276,
      "sortino_ratio": 0.184,
      "calmar_ratio": 0.584,
      "omega_ratio": 1.257,
      "profit_factor": 1.776
    },
    "risk_metrics": {
      "max_drawdown": -0.126,
      "var_95": -0.008,
      "var_99": -0.018,
      "cvar_95": -0.015
    },
    "probabilistic_metrics": {
      "probabilistic_sharpe": 1.0,
      "deflated_sharpe": 1.0,
      "sharpe_confidence_interval": { "lower": 0.234, "upper": 0.317 }
    },
    "trade_statistics": {
      "total_trades": 86,
      "win_rate": 0.477,
      "avg_win": 0.054,
      "avg_loss": -0.026,
      "best_trade": 0.127,
      "worst_trade": -0.092,
      "avg_trade_duration": 14.57
    },
    "statistical_tests": {
      "cagr_tstat": 2.305,
      "cagr_pvalue": 0.011,
      "returns_skewness": 0.672,
      "returns_kurtosis": 17.45
    },
    "time_period": {
      "start_date": "2016-02-01",
      "end_date": "2026-01-29",
      "trading_days": 2514,
      "years": 9.98
    }
  },
  "monte_carlo": {
    "statistical_significance": {
      "cagr_percentile": 44.0,
      "is_statistically_significant": false,
      "probability_of_loss": 0.002
    },
    "cagr_distribution": {
      "percentile_5th": 0.027,
      "percentile_25th": 0.055,
      "percentile_50th_median": 0.078,
      "percentile_75th": 0.100,
      "percentile_95th": 0.142
    },
    "other_metrics": {
      "sharpe_median": 0.321,
      "max_drawdown_median": -0.157
    }
  },
  "position_sizing": {
    "full_kelly": 0.221,
    "fractional_kelly": 0.055,
    "kelly_multiplier": 0.25,
    "garch_volatility_forecast": 0.202,
    "volatility_adjusted_size": 1.486
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
│   ├── technical_indicators.py  # Step 2: Technical indicator calculation (10 indicators, 30 columns)
│   ├── regime_detector.py       # Step 3: Market regime classification via Hurst exponent
│   ├── signal_generator.py      # Step 4: Signal generation with confluence scoring
│   ├── backtest_engine.py       # Step 5: Vectorized backtesting with VectorBT
│   ├── performance_metrics.py   # Step 6: 25+ risk-adjusted performance metrics
│   ├── monte_carlo.py           # Step 7: Bootstrap simulation (1000 runs) for significance testing
│   ├── position_sizer.py        # Step 8: Kelly criterion and GARCH-based position sizing
│   ├── strategy_comparison.py   # Multi-strategy performance comparison
│   ├── visualisations.py        # Chart and visualization generation
│   └── llm_agent.py             # Step 9: LLM integration, report generation, pipeline orchestration
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
| Momentum | MACD | 12/26/9 | Histogram direction indicates momentum |
| Trend | SMA | 50, 200-period | Price vs MA indicates trend direction |
| Trend | EMA | 50, 200-period | Faster response to recent price changes |
| Trend | ADX | 14-period | >25 trending, <20 ranging |
| Trend | +DI / -DI | 14-period | Directional movement indicators |
| Volatility | ATR | 14-period | Average True Range for position sizing |
| Volatility | Bollinger Bands | 20-period, 2 std | %B shows position within bands |
| Volume | Volume Ratio | 20-period MA | >1.5 indicates high relative volume |

**Derived Metrics**:
- 52-week high/low and percent from high
- Daily returns and cumulative returns
- Price vs SMA50/SMA200 percentage deviation
- Trend direction (Golden Cross / Death Cross detection)
- RSI zones (overbought/neutral/oversold)
- MACD crossover signals

**Output**: Original DataFrame augmented with 30 indicator columns.

---

### Step 3: regime_detector.py

**Purpose**: Classify current market conditions to select appropriate trading strategy.

**Regime Detection Methods**:

| Method | Calculation | Output |
|--------|-------------|--------|
| Hurst Exponent | Rescaled Range (R/S) analysis over rolling window | H > 0.5: trending, H < 0.5: mean-reverting, H = 0.5: random walk |
| Volatility Regime | Parkinson volatility vs historical percentiles | LOW (<25th), NORMAL (25-75th), HIGH (>75th) |
| Trend Strength | ADX combined with directional movement | STRONG, MODERATE, WEAK |
| Regime Confidence | Weighted agreement between classification methods | 0-1 confidence score |

**Strategy Mapping**:
```
Hurst > 0.5 + Low/Normal Volatility  -> TREND_FOLLOWING
Hurst < 0.5 + Low/Normal Volatility  -> MEAN_REVERSION
High Volatility (any Hurst)          -> DEFENSIVE
```

**Output**: Columns `[Market_Regime, Volatility_Regime, Trend_Persistence, Hurst_Exponent, Regime_Confidence, Strategy]`

---

### Step 4: signal_generator.py

**Purpose**: Generate trading signals based on multi-indicator confluence and regime-adaptive rules.

**Confluence Scoring System**:
- 6 indicators vote independently: RSI, MACD, MA Crossover, Bollinger Bands, ADX/DI, Volume
- Each indicator casts a vote: +1 (bullish), -1 (bearish), or 0 (neutral)
- Confluence Score = Sum of votes / Number of indicators (range: -1 to +1)

**Signal Generation Rules**:

```
TREND_FOLLOWING Strategy:
  confluence >= 0.35  -> STRONG_BUY
  confluence >= 0.15  -> BUY
  confluence <= -0.15 -> SELL
  confluence <= -0.35 -> STRONG_SELL

MEAN_REVERSION Strategy (inverted logic to fade extremes):
  confluence <= -0.35 -> STRONG_BUY  (fade extreme bearishness)
  confluence <= -0.15 -> BUY
  confluence >= 0.25  -> SELL        (fade extreme bullishness)
  confluence >= 0.50  -> STRONG_SELL

DEFENSIVE Strategy:
  |confluence| >= 0.50 -> Signal (requires strong agreement)
  Otherwise           -> HOLD
```

**Confidence Calculation** (4-component weighted system):
```
Confidence = 0.40 x Threshold_Margin    (distance past trigger threshold)
           + 0.30 x RSI_Extremity       (how extreme is RSI reading)
           + 0.20 x Regime_Confidence   (confidence in regime classification)
           + 0.10 x Base_Floor          (minimum for any triggered signal)
```

**Output**: Columns `[Signal, Signal_Confidence, Confluence_Score, Strategy, Stop_Loss, Take_Profit]`

---

### Step 5: backtest_engine.py

**Purpose**: Validate strategy performance through historical simulation using VectorBT.

**Functionality**:
- High-performance vectorized portfolio simulation
- Converts signals to entry/exit boolean arrays
- Models realistic transaction costs (commission and slippage)
- Implements ATR-based risk management rules
- Integrates with PositionSizer for dynamic allocation

**Risk Management**:
```
Stop Loss    = Entry Price - (2 x ATR)   -> Approximately 2% risk per trade
Take Profit  = Entry Price + (3 x ATR)   -> 1.5:1 reward-to-risk ratio
```

**Output**: `BacktestResult` object containing:
- Daily returns series
- Trade records with entry/exit prices, P&L, duration
- Portfolio equity curve
- Drawdown series

---

### Step 6: performance_metrics.py

**Purpose**: Calculate comprehensive risk-adjusted performance statistics.

**Metrics Calculated (25+ metrics)**:

| Category | Metrics |
|----------|---------|
| Return | Total Return, CAGR, Annualized Volatility |
| Risk-Adjusted | Sharpe Ratio, Sortino Ratio, Calmar Ratio, Omega Ratio, Profit Factor |
| Probabilistic | Probabilistic Sharpe Ratio, Deflated Sharpe Ratio, Sharpe Confidence Interval |
| Drawdown | Maximum Drawdown, Drawdown Duration, Recovery Time |
| Trade Statistics | Win Rate, Avg Win/Loss, Best/Worst Trade, Avg Trade Duration, Total Trades |
| Risk | VaR (95%, 99%), CVaR/Expected Shortfall (95%, 97.5%), Tail Ratio |
| Statistical | T-statistic, P-value, Jarque-Bera test, Skewness, Kurtosis |

**Key Formulas**:
```
Sharpe Ratio        = (Return - Risk_Free_Rate) / Volatility
Sortino Ratio       = (Return - Risk_Free_Rate) / Downside_Deviation
Profit Factor       = Gross_Profits / Gross_Losses
Calmar Ratio        = CAGR / |Max_Drawdown|
Probabilistic Sharpe = Probability that true Sharpe > 0 given sample
Deflated Sharpe     = Sharpe adjusted for multiple testing bias
```

**Output**: `PerformanceReport` dataclass with all metrics organized by category.

---

### Step 7: monte_carlo.py

**Purpose**: Assess statistical significance of backtest results through bootstrap simulation.

**Methodology**:
1. Extract actual daily returns from backtest
2. Generate 1000 simulated equity curves by randomly resampling returns with replacement
3. Calculate CAGR, Sharpe, and Max Drawdown for each simulation
4. Compare actual performance to distribution of simulated outcomes

**Statistical Tests**:
```
CAGR Percentile         = Rank of actual CAGR in simulated distribution
Statistically Significant = True if CAGR percentile > 95
Probability of Loss     = Percentage of simulations with negative terminal value
```

**Distribution Outputs**:
- 5th, 25th, 50th (median), 75th, 95th percentile CAGRs
- Median Sharpe Ratio from simulations
- Median Maximum Drawdown from simulations
- Confidence intervals for all key metrics

**Output**: `MonteCarloResult` with significance assessment and full distribution statistics.

---

### Step 8: position_sizer.py

**Purpose**: Calculate optimal position sizes using quantitative methods.

**Methods Implemented**:

| Method | Formula | Description |
|--------|---------|-------------|
| Kelly Criterion | f* = (p x b - q) / b | Optimal fraction based on win rate and payoff ratio |
| Fractional Kelly | f = 0.25 x Kelly | Conservative sizing at 25% of optimal to reduce risk of ruin |
| GARCH(1,1) Forecast | sigma_t = omega + alpha*epsilon_{t-1} + beta*sigma_{t-1} | Predicts near-term volatility |
| Volatility Targeting | Size = Target_Vol / Forecast_Vol | Normalizes position to target volatility level |

**Risk Constraints**:
- Maximum position size cap (default: 100% of portfolio)
- Maximum portfolio heat (total capital at risk across all positions)
- Volatility-adjusted sizing based on GARCH forecast

**Output**: Recommended position size as percentage of portfolio with full Kelly and GARCH calculations.

---

### Step 9: llm_agent.py (Main Orchestration)

**Purpose**: Coordinate analysis pipeline, generate AI-powered narratives, and produce multi-format reports.

**Pipeline Orchestration**:
1. Execute Steps 1-8 in sequence
2. Aggregate all results into unified `AnalysisContext` object
3. Generate LLM-powered analysis via structured prompts
4. Export to JSON, TXT, and PDF formats

**LLM Integration**:
- Supports Claude (Anthropic) and GPT-4 (OpenAI)
- Structured prompts ensure consistent, parseable output format
- Generates institutional-quality narratives

**AI-Generated Content**:

| Section | Description |
|---------|-------------|
| Recommendation Rationale | Detailed explanation of why the signal was generated |
| Technical Summary | Plain-English interpretation of all indicator readings |
| Investment Thesis | Professional narrative suitable for investment memos |
| Risk Factors | Identified risks with specific mitigation strategies |
| Catalysts | Bullish and bearish catalysts that could move the stock |
| Scenario Analysis | Bull/base/bear cases with probabilities, targets, and drivers |
| Performance Analysis | Strengths, weaknesses, warnings, and actionable suggestions |

**Scenario Analysis Generation**:
```
Bull Case:  Probability-weighted upside target with specific drivers
Base Case:  Most likely outcome with key assumptions
Bear Case:  Downside scenario with risk factors
Expected Value = Sum(Probability x Return) across all scenarios
```

**Component Scoring** (0-100 scale):
- Momentum Score: Based on RSI position relative to extremes and MACD histogram
- Trend Score: Based on price vs moving averages and ADX strength
- Volatility Score: Based on ATR percentile and Bollinger Band position
- Volume Score: Based on volume ratio confirmation of price moves
- Overall Score: Weighted combination of all component scores

**Report Generation**:
- PDF: Professional format with executive summary, technical analysis, regime analysis, scenario analysis, performance metrics, risk factors, and recommendations
- JSON: Complete structured data for programmatic consumption and integration
- TXT: Plain text summary for quick review

## Key Features

### Regime-Adaptive Strategy Selection

Different market conditions require different trading approaches:

- **Trending Markets (Hurst > 0.5)**: Momentum strategies outperform; system follows trends
- **Mean-Reverting Markets (Hurst < 0.5)**: Contrarian strategies outperform; system fades extremes
- **Volatile Markets**: Defensive positioning; system requires stronger signal confirmation

The Hurst exponent is calculated using Rescaled Range (R/S) analysis, a robust statistical method for detecting long-term memory in time series.

### Multi-Indicator Confluence

No single indicator is reliable in isolation. The system requires agreement:

- 6 indicators vote independently on market direction
- Signals only trigger when sufficient indicators agree (configurable thresholds)
- Confluence score quantifies the level of agreement (-1 to +1)
- Higher confluence correlates with higher-probability trades

### Statistical Validation

Backtesting alone is insufficient due to overfitting risk:

- Monte Carlo simulation generates 1000 alternative scenarios via bootstrap resampling
- Actual performance is ranked against simulated distribution
- Statistical significance requires outperforming 95% of random simulations
- Probabilistic Sharpe Ratio accounts for estimation uncertainty
- Deflated Sharpe Ratio adjusts for multiple testing bias

### AI-Powered Analysis

LLM integration provides institutional-quality narratives:

- Professional investment thesis generation
- Scenario analysis with probability-weighted expected value calculation
- Risk factor identification with specific mitigation strategies
- Bullish and bearish catalyst identification
- Performance analysis with actionable improvement suggestions
- Component-by-component scoring with plain-English interpretation

### Comprehensive Risk Management

Multiple layers of risk controls:

- ATR-based stop losses and take profit targets
- Kelly criterion position sizing with fractional adjustment
- GARCH volatility forecasting for dynamic position adjustment
- VaR and CVaR calculations at multiple confidence levels
- Maximum drawdown tracking and analysis

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
confluence >= 0.35  # STRONG_BUY threshold
confluence >= 0.15  # BUY threshold
```

**Risk Management** (`backtest_engine.py` lines 60-80):
```python
stop_loss_atr_mult: float = 2.0    # Stop loss = 2 x ATR
take_profit_atr_mult: float = 3.0  # Take profit = 3 x ATR
kelly_fraction: float = 0.25       # Use 25% of Kelly optimal
```

**Monte Carlo Settings** (`monte_carlo.py`):
```python
n_simulations: int = 1000  # Number of bootstrap simulations
confidence_level: float = 0.95  # Threshold for statistical significance
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

# Access current recommendation
latest = data.iloc[-1]
print(f"Signal: {latest['Signal']}")
print(f"Confidence: {latest['Signal_Confidence']:.1%}")
print(f"Regime: {latest['Market_Regime']}")
print(f"Hurst Exponent: {latest['Hurst_Exponent']:.3f}")
print(f"Strategy: {latest['Strategy']}")
```

### Full Analysis with Backtest and Report

```bash
# From src/ directory - runs complete 9-step pipeline
python llm_agent.py
```

This generates PDF, JSON, and TXT reports in the `outputs/` directory.

## Dependencies

```
yfinance>=0.2.32     # Yahoo Finance API for market data
pandas>=2.1.3        # DataFrame operations and time series
numpy>=1.26.2        # Numerical computing
scipy>=1.11.4        # Statistical functions and tests
vectorbt>=0.28.2     # High-performance vectorized backtesting
matplotlib>=3.8.2    # Visualization and charting
reportlab>=4.0.0     # Professional PDF report generation
anthropic>=0.18.0    # Claude API integration
openai>=1.3.7        # GPT-4 API integration
python-dotenv>=1.0.0 # Environment variable management
pyarrow>=14.0.0      # Parquet file format support
```

## Troubleshooting

**1. No data returned for ticker**
- Verify ticker symbol is valid (e.g., AAPL, MSFT, GOOGL)
- Check internet connection
- Confirm ticker is available on Yahoo Finance

**2. Import errors**
- Ensure you are in the project root directory
- Activate virtual environment: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

**3. PDF generation fails**
- Install reportlab: `pip install reportlab`
- Check write permissions for `outputs/` directory

**4. LLM reports not generating**
- Verify API key is correctly set in `.env` file
- Check API key validity and available credits
- System requires at least one valid API key (Anthropic or OpenAI)

**5. Backtest returns errors**
- Ensure minimum 252 trading days of data available
- Verify all indicator columns exist before running backtest
- Check for NaN values in price data

## Disclaimer

This software is for educational and research purposes only. It does not constitute financial advice. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.
