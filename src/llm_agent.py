"""
LLM Agent Module - AI-Powered Trading Analysis

Advanced Claude API integration for intelligent trade analysis:
1. Structured Outputs: Pydantic models for guaranteed JSON format
2. Tool-Calling: Dynamic data requests for deeper analysis
3. Chain-of-Thought: Step-by-step reasoning transparency
4. Multi-Stage Analysis: Technical -> Regime -> Risk -> Recommendation

Outputs professional trade notes with:
- Executive summary (BUY/SELL/HOLD with confidence)
- Technical analysis section
- Regime context
- Risk factors
- Trade specifications
- Performance attribution
"""

import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from datetime import datetime
import warnings

# Try to import Anthropic SDK
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    warnings.warn("Anthropic SDK not installed. Install with: pip install anthropic")

# Try to import OpenAI as fallback
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class Recommendation(Enum):
    """Trading recommendation types."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class RiskLevel(Enum):
    """Risk assessment levels."""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


@dataclass
class TradeRecommendation:
    """Structured trade recommendation from LLM."""
    recommendation: Recommendation
    confidence: float  # 0-1 scale
    rationale: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float
    risk_reward_ratio: float
    risks: List[str]
    catalysts: List[str]
    time_horizon: str  # "intraday", "swing", "position"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'recommendation': self.recommendation.value,
            'confidence': self.confidence,
            'rationale': self.rationale,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size_pct': self.position_size_pct,
            'risk_reward_ratio': self.risk_reward_ratio,
            'risks': self.risks,
            'catalysts': self.catalysts,
            'time_horizon': self.time_horizon
        }


@dataclass
class PerformanceAnalysis:
    """Structured performance analysis from LLM."""
    summary: str
    strengths: List[str]
    weaknesses: List[str]
    attribution: Dict[str, str]  # Factor -> explanation
    warnings: List[str]
    suggestions: List[str]
    overall_assessment: str  # "excellent", "good", "fair", "poor"
    confidence_in_edge: float  # 0-1, confidence strategy has real edge

    def to_dict(self) -> Dict[str, Any]:
        return {
            'summary': self.summary,
            'strengths': self.strengths,
            'weaknesses': self.weaknesses,
            'attribution': self.attribution,
            'warnings': self.warnings,
            'suggestions': self.suggestions,
            'overall_assessment': self.overall_assessment,
            'confidence_in_edge': self.confidence_in_edge
        }


@dataclass
class AnalysisContext:
    """Context data for LLM analysis."""
    ticker: str
    current_price: float
    current_date: str

    # Technical indicators
    rsi: float
    macd: float
    macd_signal: float
    sma_50: float
    sma_200: float
    bb_percent_b: float
    atr: float
    adx: float

    # Regime information
    market_regime: str
    volatility_regime: str
    trend_persistence: str
    hurst_exponent: float
    regime_confidence: float

    # Signal information
    signal: str
    signal_confidence: float
    confluence_score: float
    strategy: str

    # === BASIC PERFORMANCE METRICS ===
    total_return: Optional[float] = None
    cagr: Optional[float] = None
    volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None

    # === ADVANCED RISK-ADJUSTED METRICS ===
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    omega_ratio: Optional[float] = None
    profit_factor: Optional[float] = None

    # === RISK METRICS (VaR/CVaR) ===
    var_95: Optional[float] = None
    var_99: Optional[float] = None
    cvar_95: Optional[float] = None
    cvar_975: Optional[float] = None

    # === PROBABILISTIC METRICS ===
    probabilistic_sharpe: Optional[float] = None
    deflated_sharpe: Optional[float] = None
    sharpe_ci_lower: Optional[float] = None
    sharpe_ci_upper: Optional[float] = None

    # === TRADE STATISTICS ===
    total_trades: Optional[int] = None
    win_rate: Optional[float] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    best_trade: Optional[float] = None
    worst_trade: Optional[float] = None
    avg_trade_duration: Optional[float] = None

    # === STATISTICAL TESTS ===
    cagr_tstat: Optional[float] = None
    cagr_pvalue: Optional[float] = None
    returns_skewness: Optional[float] = None
    returns_kurtosis: Optional[float] = None
    jarque_bera_stat: Optional[float] = None
    jarque_bera_pvalue: Optional[float] = None

    # === MONTE CARLO RESULTS ===
    mc_cagr_percentile: Optional[float] = None
    mc_is_significant: Optional[bool] = None
    mc_prob_loss: Optional[float] = None
    mc_cagr_5th: Optional[float] = None
    mc_cagr_25th: Optional[float] = None
    mc_cagr_50th: Optional[float] = None
    mc_cagr_75th: Optional[float] = None
    mc_cagr_95th: Optional[float] = None
    mc_sharpe_median: Optional[float] = None
    mc_max_dd_median: Optional[float] = None

    # === POSITION SIZING (from PositionSizer) ===
    kelly_fraction: Optional[float] = None
    garch_volatility: Optional[float] = None
    vol_adjusted_size: Optional[float] = None

    # === TIME PERIOD ===
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    trading_days: Optional[int] = None
    years: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# =============================================================================
# TOOL DEFINITIONS FOR LLM
# =============================================================================

AVAILABLE_TOOLS = [
    {
        "name": "fetch_indicator_history",
        "description": "Fetch historical values for a specific technical indicator",
        "input_schema": {
            "type": "object",
            "properties": {
                "indicator": {
                    "type": "string",
                    "description": "Indicator name (RSI, MACD, SMA_50, etc.)"
                },
                "periods": {
                    "type": "integer",
                    "description": "Number of periods to fetch"
                }
            },
            "required": ["indicator", "periods"]
        }
    },
    {
        "name": "calculate_correlation",
        "description": "Calculate correlation between the asset and a benchmark",
        "input_schema": {
            "type": "object",
            "properties": {
                "benchmark": {
                    "type": "string",
                    "description": "Benchmark ticker (SPY, QQQ, etc.)"
                },
                "periods": {
                    "type": "integer",
                    "description": "Number of periods for correlation"
                }
            },
            "required": ["benchmark", "periods"]
        }
    },
    {
        "name": "get_regime_history",
        "description": "Get historical regime classifications",
        "input_schema": {
            "type": "object",
            "properties": {
                "periods": {
                    "type": "integer",
                    "description": "Number of periods to fetch"
                }
            },
            "required": ["periods"]
        }
    },
    {
        "name": "get_recent_trades",
        "description": "Get recent trade history with P&L",
        "input_schema": {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Number of recent trades to fetch"
                }
            },
            "required": ["count"]
        }
    }
]


# =============================================================================
# LLM AGENT CLASS
# =============================================================================

class LLMAgent:
    """
    AI-powered trading analysis agent using Claude API.

    Features:
    - Structured outputs with Pydantic-style validation
    - Tool calling for dynamic data retrieval
    - Chain-of-thought reasoning
    - Multi-stage analysis pipeline

    Usage:
        agent = LLMAgent(data_df)
        recommendation = agent.generate_trade_recommendation(context)
        analysis = agent.analyse_performance(metrics, mc_results)
        report = agent.generate_full_report(context, metrics)
    """

    # System prompts for different analysis types
    TRADE_ANALYSIS_PROMPT = """You are a senior quantitative analyst at a hedge fund.
Your task is to analyse technical indicators and market regime data to generate
actionable trading recommendations.

ANALYSIS FRAMEWORK:
1. Technical Indicator Analysis
   - Evaluate momentum (RSI, MACD)
   - Assess trend (SMA crossovers, ADX)
   - Check volatility (ATR, Bollinger Bands)

2. Regime Context
   - Consider market regime (bull/bear/sideways)
   - Account for volatility regime
   - Factor in trend persistence (Hurst exponent)

3. Risk Assessment
   - Identify key risks
   - Set appropriate stop loss
   - Calculate risk/reward ratio

4. Recommendation Synthesis
   - Combine all factors
   - Assign confidence level
   - Specify entry, stop, target

CHAIN OF THOUGHT:
Always explain your reasoning step by step:
- "RSI at X indicates..."
- "Combined with MACD showing..."
- "Given the STRONG_BULL regime..."
- "Therefore, my recommendation is..."

OUTPUT FORMAT:
Provide your analysis in this exact JSON structure:
{
    "recommendation": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",
    "confidence": 0.0-1.0,
    "rationale": "detailed explanation",
    "entry_price": float,
    "stop_loss": float,
    "take_profit": float,
    "position_size_pct": 0.0-1.0,
    "risk_reward_ratio": float,
    "risks": ["risk1", "risk2"],
    "catalysts": ["catalyst1", "catalyst2"],
    "time_horizon": "intraday|swing|position",
    "reasoning_chain": ["step1", "step2", "step3"]
}"""

    PERFORMANCE_ANALYSIS_PROMPT = """You are a senior portfolio manager reviewing strategy performance.
Analyse the backtest results and provide institutional-quality assessment.

ANALYSIS FRAMEWORK:
1. Return Analysis
   - Is CAGR attractive vs benchmarks?
   - Is total return meaningful given the period?

2. Risk-Adjusted Performance
   - Sharpe ratio quality (>1 good, >2 excellent)
   - Sortino for downside focus
   - Calmar for drawdown efficiency

3. Statistical Significance
   - Monte Carlo percentile (want >60th)
   - Probability of loss
   - Is the edge real or luck?

4. Trade Quality
   - Win rate assessment
   - Profit factor (want >1.5)
   - Average trade quality

OUTPUT FORMAT:
{
    "summary": "one paragraph executive summary",
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "attribution": {
        "factor": "explanation of contribution"
    },
    "warnings": ["warning1", "warning2"],
    "suggestions": ["improvement1", "improvement2"],
    "overall_assessment": "excellent|good|fair|poor",
    "confidence_in_edge": 0.0-1.0
}"""

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        model: str = "claude-sonnet-4-20250514",
        fallback_model: str = "gpt-4o",
        max_tokens: int = 4096,
        temperature: float = 0.3
    ):
        """
        Initialise LLM agent.

        Args:
            data: DataFrame with price and indicator data (for tool calls)
            model: Claude model to use
            fallback_model: OpenAI model if Claude unavailable
            max_tokens: Maximum response tokens
            temperature: Sampling temperature (lower = more deterministic)
        """
        self.data = data
        self.model = model
        self.fallback_model = fallback_model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Initialise API clients
        self.anthropic_client = None
        self.openai_client = None

        # Try Anthropic first
        if ANTHROPIC_AVAILABLE:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                self.anthropic_client = Anthropic(api_key=api_key)
                print(" LLM Agent initialised with Claude API")
            else:
                warnings.warn("ANTHROPIC_API_KEY not found in environment")

        # Fallback to OpenAI
        if self.anthropic_client is None and OPENAI_AVAILABLE:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
                print(" LLM Agent initialised with OpenAI API (fallback)")
            else:
                warnings.warn("OPENAI_API_KEY not found in environment")

        if self.anthropic_client is None and self.openai_client is None:
            warnings.warn("No LLM API available. Analysis will return placeholder results.")

        # Tool handlers
        self.tool_handlers: Dict[str, Callable] = {
            'fetch_indicator_history': self._fetch_indicator_history,
            'calculate_correlation': self._calculate_correlation,
            'get_regime_history': self._get_regime_history,
            'get_recent_trades': self._get_recent_trades
        }

        # Trade history (can be set externally)
        self.trades_df: Optional[pd.DataFrame] = None

    # =========================================================================
    # TOOL IMPLEMENTATIONS
    # =========================================================================

    def _fetch_indicator_history(self, indicator: str, periods: int) -> Dict[str, Any]:
        """Fetch historical indicator values."""
        if self.data is None or indicator not in self.data.columns:
            return {"error": f"Indicator {indicator} not available"}

        values = self.data[indicator].tail(periods).tolist()
        dates = self.data.index[-periods:].strftime('%Y-%m-%d').tolist()

        return {
            "indicator": indicator,
            "periods": periods,
            "values": values,
            "dates": dates,
            "current": values[-1] if values else None,
            "mean": np.mean(values) if values else None,
            "std": np.std(values) if values else None
        }

    def _calculate_correlation(self, benchmark: str, periods: int) -> Dict[str, Any]:
        """Calculate correlation with benchmark."""
        if self.data is None:
            return {"error": "No data available"}

        # For now, return placeholder - would need benchmark data
        returns = self.data['Close'].pct_change().tail(periods)

        return {
            "benchmark": benchmark,
            "periods": periods,
            "correlation": 0.7,  # Placeholder
            "note": "Benchmark data not loaded - showing placeholder"
        }

    def _get_regime_history(self, periods: int) -> Dict[str, Any]:
        """Get regime classification history."""
        if self.data is None or 'Market_Regime' not in self.data.columns:
            return {"error": "Regime data not available"}

        regimes = self.data['Market_Regime'].tail(periods).tolist()
        dates = self.data.index[-periods:].strftime('%Y-%m-%d').tolist()

        # Count regime distribution
        from collections import Counter
        distribution = dict(Counter(regimes))

        return {
            "periods": periods,
            "regimes": regimes,
            "dates": dates,
            "distribution": distribution,
            "current": regimes[-1] if regimes else None
        }

    def _get_recent_trades(self, count: int) -> Dict[str, Any]:
        """Get recent trade history."""
        if self.trades_df is None or len(self.trades_df) == 0:
            return {"error": "No trade history available"}

        recent = self.trades_df.tail(count)

        trades_list = []
        for _, row in recent.iterrows():
            trade = {
                'return': float(row.get('Return', 0)),
                'pnl': float(row.get('PnL', 0)) if 'PnL' in row else None
            }
            trades_list.append(trade)

        return {
            "count": len(trades_list),
            "trades": trades_list,
            "win_rate": sum(1 for t in trades_list if t['return'] > 0) / len(trades_list) if trades_list else 0
        }

    def _handle_tool_call(self, tool_name: str, tool_input: Dict) -> str:
        """Handle a tool call from the LLM."""
        if tool_name in self.tool_handlers:
            result = self.tool_handlers[tool_name](**tool_input)
            return json.dumps(result)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    # =========================================================================
    # CLAUDE API METHODS
    # =========================================================================

    def _call_claude(
        self,
        system_prompt: str,
        user_message: str,
        use_tools: bool = False
    ) -> str:
        """
        Call Claude API with optional tool use.

        Args:
            system_prompt: System instructions
            user_message: User query
            use_tools: Whether to enable tool calling

        Returns:
            Model response text
        """
        if self.anthropic_client is None:
            return self._call_openai_fallback(system_prompt, user_message)

        messages = [{"role": "user", "content": user_message}]

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": messages
        }

        if use_tools:
            kwargs["tools"] = AVAILABLE_TOOLS

        # Initial call
        response = self.anthropic_client.messages.create(**kwargs)

        # Handle tool use loop
        while response.stop_reason == "tool_use":
            # Process tool calls
            tool_results = []
            assistant_content = response.content

            for block in response.content:
                if block.type == "tool_use":
                    tool_result = self._handle_tool_call(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_result
                    })

            # Continue conversation with tool results
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

            kwargs["messages"] = messages
            response = self.anthropic_client.messages.create(**kwargs)

        # Extract text from response
        for block in response.content:
            if hasattr(block, 'text'):
                return block.text

        return ""

    def _call_openai_fallback(self, system_prompt: str, user_message: str) -> str:
        """Fallback to OpenAI if Claude unavailable."""
        if self.openai_client is None:
            return self._generate_placeholder_response()

        response = self.openai_client.chat.completions.create(
            model=self.fallback_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        return response.choices[0].message.content

    def _generate_placeholder_response(self) -> str:
        """Generate placeholder when no API available."""
        return json.dumps({
            "error": "No LLM API available",
            "recommendation": "HOLD",
            "confidence": 0.0,
            "rationale": "Unable to generate analysis - API keys not configured"
        })

    # =========================================================================
    # PUBLIC ANALYSIS METHODS
    # =========================================================================

    def generate_trade_recommendation(
        self,
        context: AnalysisContext,
        use_tools: bool = True
    ) -> TradeRecommendation:
        """
        Generate trade recommendation from context.

        Args:
            context: AnalysisContext with all relevant data
            use_tools: Whether to allow tool calls for additional data

        Returns:
            TradeRecommendation object
        """
        # Build user message with context
        user_message = f"""
Analyse the following market data and provide a trading recommendation:

TICKER: {context.ticker}
CURRENT PRICE: ${context.current_price:.2f}
DATE: {context.current_date}

TECHNICAL INDICATORS:
- RSI: {context.rsi:.1f}
- MACD: {context.macd:.4f}
- MACD Signal: {context.macd_signal:.4f}
- MACD Histogram: {context.macd - context.macd_signal:.4f}
- SMA 50: ${context.sma_50:.2f}
- SMA 200: ${context.sma_200:.2f}
- Price vs SMA50: {(context.current_price / context.sma_50 - 1) * 100:.1f}%
- Price vs SMA200: {(context.current_price / context.sma_200 - 1) * 100:.1f}%
- BB %B: {context.bb_percent_b:.2f}
- ATR: ${context.atr:.2f} ({context.atr / context.current_price * 100:.1f}%)
- ADX: {context.adx:.1f}

REGIME ANALYSIS:
- Market Regime: {context.market_regime}
- Volatility Regime: {context.volatility_regime}
- Trend Persistence: {context.trend_persistence}
- Hurst Exponent: {context.hurst_exponent:.3f}
- Regime Confidence: {context.regime_confidence:.1%}

SIGNAL SYSTEM:
- Current Signal: {context.signal}
- Signal Confidence: {context.signal_confidence:.1%}
- Confluence Score: {context.confluence_score:.2f}
- Active Strategy: {context.strategy}

Provide your analysis following the chain-of-thought framework and output in JSON format.
"""

        response = self._call_claude(
            self.TRADE_ANALYSIS_PROMPT,
            user_message,
            use_tools=use_tools
        )

        # Parse response
        return self._parse_trade_recommendation(response, context)

    def _parse_trade_recommendation(
        self,
        response: str,
        context: AnalysisContext
    ) -> TradeRecommendation:
        """Parse LLM response into TradeRecommendation."""
        try:
            # Try to extract JSON from response
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            data = json.loads(json_str)

            # Map recommendation string to enum
            rec_map = {
                'STRONG_BUY': Recommendation.STRONG_BUY,
                'BUY': Recommendation.BUY,
                'HOLD': Recommendation.HOLD,
                'SELL': Recommendation.SELL,
                'STRONG_SELL': Recommendation.STRONG_SELL
            }

            return TradeRecommendation(
                recommendation=rec_map.get(data.get('recommendation', 'HOLD'), Recommendation.HOLD),
                confidence=float(data.get('confidence', 0.5)),
                rationale=data.get('rationale', ''),
                entry_price=float(data.get('entry_price', context.current_price)),
                stop_loss=float(data.get('stop_loss', context.current_price * 0.95)),
                take_profit=float(data.get('take_profit', context.current_price * 1.10)),
                position_size_pct=float(data.get('position_size_pct', 0.1)),
                risk_reward_ratio=float(data.get('risk_reward_ratio', 2.0)),
                risks=data.get('risks', []),
                catalysts=data.get('catalysts', []),
                time_horizon=data.get('time_horizon', 'swing')
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Return default recommendation on parse error
            warnings.warn(f"Failed to parse LLM response: {e}")
            return TradeRecommendation(
                recommendation=Recommendation.HOLD,
                confidence=0.0,
                rationale=f"Parse error: {str(e)}. Raw response: {response[:500]}",
                entry_price=context.current_price,
                stop_loss=context.current_price * 0.95,
                take_profit=context.current_price * 1.05,
                position_size_pct=0.0,
                risk_reward_ratio=1.0,
                risks=["Unable to parse LLM response"],
                catalysts=[],
                time_horizon="swing"
            )

    def analyse_performance(
        self,
        context: AnalysisContext,
        performance_report: Optional[Any] = None,
        mc_result: Optional[Any] = None
    ) -> PerformanceAnalysis:
        """
        Analyse strategy performance.

        Args:
            context: AnalysisContext with performance metrics
            performance_report: Optional PerformanceReport object
            mc_result: Optional MonteCarloResult object

        Returns:
            PerformanceAnalysis object
        """
        # Build performance summary
        perf_section = ""
        if context.cagr is not None:
            perf_section = f"""
PERFORMANCE METRICS:
- CAGR: {context.cagr:.1%}
- Sharpe Ratio: {context.sharpe_ratio:.2f}
- Max Drawdown: {context.max_drawdown:.1%}
- Win Rate: {context.win_rate:.1%}
- Profit Factor: {context.profit_factor:.2f}
- Total Trades: {context.total_trades}
"""

        mc_section = ""
        if context.mc_cagr_percentile is not None:
            mc_section = f"""
MONTE CARLO ANALYSIS:
- CAGR Percentile: {context.mc_cagr_percentile:.0f}th
- Statistically Significant: {"Yes" if context.mc_is_significant else "No"}
- Probability of 10%+ Loss: {context.mc_prob_loss:.1%}
"""

        user_message = f"""
Analyse the following strategy performance:

{perf_section}
{mc_section}

CURRENT MARKET CONTEXT:
- Market Regime: {context.market_regime}
- Volatility Regime: {context.volatility_regime}
- Active Strategy: {context.strategy}

Provide institutional-quality assessment in JSON format.
"""

        response = self._call_claude(
            self.PERFORMANCE_ANALYSIS_PROMPT,
            user_message,
            use_tools=False
        )

        return self._parse_performance_analysis(response)

    def _parse_performance_analysis(self, response: str) -> PerformanceAnalysis:
        """Parse LLM response into PerformanceAnalysis."""
        try:
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            data = json.loads(json_str)

            return PerformanceAnalysis(
                summary=data.get('summary', ''),
                strengths=data.get('strengths', []),
                weaknesses=data.get('weaknesses', []),
                attribution=data.get('attribution', {}),
                warnings=data.get('warnings', []),
                suggestions=data.get('suggestions', []),
                overall_assessment=data.get('overall_assessment', 'fair'),
                confidence_in_edge=float(data.get('confidence_in_edge', 0.5))
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            warnings.warn(f"Failed to parse performance analysis: {e}")
            return PerformanceAnalysis(
                summary=f"Parse error: {str(e)}",
                strengths=[],
                weaknesses=["Unable to parse LLM response"],
                attribution={},
                warnings=["Analysis may be incomplete"],
                suggestions=[],
                overall_assessment="unknown",
                confidence_in_edge=0.0
            )

    def generate_full_report(
        self,
        context: AnalysisContext,
        include_performance: bool = True
    ) -> str:
        """
        Generate comprehensive trade report.

        Args:
            context: AnalysisContext with all data
            include_performance: Whether to include performance analysis

        Returns:
            Formatted markdown report
        """
        # Get trade recommendation
        recommendation = self.generate_trade_recommendation(context)

        # Get performance analysis if requested and data available
        performance = None
        if include_performance and context.cagr is not None:
            performance = self.analyse_performance(context)

        # Build report
        report = self._format_report(context, recommendation, performance)

        return report

    def _format_report(
        self,
        context: AnalysisContext,
        recommendation: TradeRecommendation,
        performance: Optional[PerformanceAnalysis]
    ) -> str:
        """Format analysis into professional report."""

        # Recommendation colour coding
        rec_emoji = {
            Recommendation.STRONG_BUY: "+++",
            Recommendation.BUY: "++",
            Recommendation.HOLD: "~",
            Recommendation.SELL: "--",
            Recommendation.STRONG_SELL: "---"
        }

        report = f"""
================================================================================
                        TRADE ANALYSIS REPORT
                        {context.ticker} | {context.current_date}
================================================================================

EXECUTIVE SUMMARY
--------------------------------------------------------------------------------
Recommendation: {recommendation.recommendation.value} {rec_emoji.get(recommendation.recommendation, '')}
Confidence: {recommendation.confidence:.0%}
Time Horizon: {recommendation.time_horizon.upper()}

{recommendation.rationale}

TRADE SPECIFICATIONS
--------------------------------------------------------------------------------
Entry Price:     ${recommendation.entry_price:,.2f}
Stop Loss:       ${recommendation.stop_loss:,.2f} ({(recommendation.stop_loss/recommendation.entry_price - 1)*100:+.1f}%)
Take Profit:     ${recommendation.take_profit:,.2f} ({(recommendation.take_profit/recommendation.entry_price - 1)*100:+.1f}%)
Position Size:   {recommendation.position_size_pct:.0%} of portfolio
Risk/Reward:     1:{recommendation.risk_reward_ratio:.1f}

TECHNICAL ANALYSIS
--------------------------------------------------------------------------------
Momentum:
  - RSI: {context.rsi:.1f} {'(Oversold)' if context.rsi < 30 else '(Overbought)' if context.rsi > 70 else '(Neutral)'}
  - MACD: {context.macd:.4f} vs Signal: {context.macd_signal:.4f}
  - MACD Histogram: {context.macd - context.macd_signal:+.4f}

Trend:
  - Price vs SMA50: {(context.current_price/context.sma_50 - 1)*100:+.1f}%
  - Price vs SMA200: {(context.current_price/context.sma_200 - 1)*100:+.1f}%
  - ADX: {context.adx:.1f} {'(Strong Trend)' if context.adx > 25 else '(Weak Trend)'}

Volatility:
  - ATR: ${context.atr:.2f} ({context.atr/context.current_price*100:.1f}% of price)
  - BB %B: {context.bb_percent_b:.2f}

REGIME CONTEXT
--------------------------------------------------------------------------------
Market Regime:     {context.market_regime}
Volatility Regime: {context.volatility_regime}
Trend Persistence: {context.trend_persistence}
Hurst Exponent:    {context.hurst_exponent:.3f} {'(Trending)' if context.hurst_exponent > 0.5 else '(Mean-Reverting)'}
Confidence:        {context.regime_confidence:.0%}

RISK FACTORS
--------------------------------------------------------------------------------
"""
        for i, risk in enumerate(recommendation.risks, 1):
            report += f"  {i}. {risk}\n"

        if recommendation.catalysts:
            report += """
POTENTIAL CATALYSTS
--------------------------------------------------------------------------------
"""
            for i, catalyst in enumerate(recommendation.catalysts, 1):
                report += f"  {i}. {catalyst}\n"

        if performance:
            report += f"""
PERFORMANCE ANALYSIS
--------------------------------------------------------------------------------
Overall Assessment: {performance.overall_assessment.upper()}
Confidence in Edge: {performance.confidence_in_edge:.0%}

{performance.summary}

Strengths:
"""
            for strength in performance.strengths:
                report += f"  + {strength}\n"

            report += "\nWeaknesses:\n"
            for weakness in performance.weaknesses:
                report += f"  - {weakness}\n"

            if performance.warnings:
                report += "\nWarnings:\n"
                for warning in performance.warnings:
                    report += f"  ! {warning}\n"

            if performance.suggestions:
                report += "\nSuggestions:\n"
                for suggestion in performance.suggestions:
                    report += f"  > {suggestion}\n"

        report += """
================================================================================
                        END OF REPORT
================================================================================
"""

        return report

    def generate_json_report(
        self,
        context: AnalysisContext,
        include_performance: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive trade analysis as structured JSON.

        This is the primary output method for feeding into a master agent.
        Returns a complete JSON-serializable dictionary with all analysis data.

        Args:
            context: AnalysisContext with all data
            include_performance: Whether to include performance analysis

        Returns:
            Dictionary with complete analysis, ready for JSON serialization
        """
        # Get trade recommendation
        recommendation = self.generate_trade_recommendation(context)

        # Get performance analysis if requested and data available
        performance = None
        if include_performance and context.cagr is not None:
            performance = self.analyse_performance(context)

        # Build structured output
        output = {
            "metadata": {
                "ticker": context.ticker,
                "analysis_date": context.current_date,
                "current_price": context.current_price,
                "generated_by": "LLMAgent",
                "model": self.model if self.anthropic_client else self.fallback_model
            },
            "recommendation": {
                "action": recommendation.recommendation.value,
                "confidence": recommendation.confidence,
                "time_horizon": recommendation.time_horizon,
                "rationale": recommendation.rationale
            },
            "trade_specifications": {
                "entry_price": recommendation.entry_price,
                "stop_loss": recommendation.stop_loss,
                "stop_loss_pct": (recommendation.stop_loss / recommendation.entry_price - 1) * 100,
                "take_profit": recommendation.take_profit,
                "take_profit_pct": (recommendation.take_profit / recommendation.entry_price - 1) * 100,
                "position_size_pct": recommendation.position_size_pct,
                "risk_reward_ratio": recommendation.risk_reward_ratio
            },
            "technical_analysis": {
                "momentum": {
                    "rsi": context.rsi,
                    "rsi_signal": "oversold" if context.rsi < 30 else "overbought" if context.rsi > 70 else "neutral",
                    "macd": context.macd,
                    "macd_signal": context.macd_signal,
                    "macd_histogram": context.macd - context.macd_signal
                },
                "trend": {
                    "sma_50": context.sma_50,
                    "sma_200": context.sma_200,
                    "price_vs_sma50_pct": (context.current_price / context.sma_50 - 1) * 100,
                    "price_vs_sma200_pct": (context.current_price / context.sma_200 - 1) * 100,
                    "adx": context.adx,
                    "trend_strength": "strong" if context.adx > 25 else "weak"
                },
                "volatility": {
                    "atr": context.atr,
                    "atr_pct": context.atr / context.current_price * 100,
                    "bb_percent_b": context.bb_percent_b
                }
            },
            "regime_analysis": {
                "market_regime": context.market_regime,
                "volatility_regime": context.volatility_regime,
                "trend_persistence": context.trend_persistence,
                "hurst_exponent": context.hurst_exponent,
                "hurst_interpretation": "trending" if context.hurst_exponent > 0.5 else "mean_reverting",
                "regime_confidence": context.regime_confidence
            },
            "signal_system": {
                "current_signal": context.signal,
                "signal_confidence": context.signal_confidence,
                "confluence_score": context.confluence_score,
                "active_strategy": context.strategy
            },
            "risk_factors": recommendation.risks,
            "catalysts": recommendation.catalysts
        }

        # Add performance analysis if available
        if performance:
            output["performance_analysis"] = {
                "overall_assessment": performance.overall_assessment,
                "confidence_in_edge": performance.confidence_in_edge,
                "summary": performance.summary,
                "strengths": performance.strengths,
                "weaknesses": performance.weaknesses,
                "attribution": performance.attribution,
                "warnings": performance.warnings,
                "suggestions": performance.suggestions
            }

        # Add comprehensive backtest metrics if available
        if context.cagr is not None:
            output["backtest_metrics"] = {
                "return_metrics": {
                    "total_return": context.total_return,
                    "cagr": context.cagr,
                    "volatility": context.volatility
                },
                "risk_adjusted_metrics": {
                    "sharpe_ratio": context.sharpe_ratio,
                    "sortino_ratio": context.sortino_ratio,
                    "calmar_ratio": context.calmar_ratio,
                    "omega_ratio": context.omega_ratio,
                    "profit_factor": context.profit_factor
                },
                "risk_metrics": {
                    "max_drawdown": context.max_drawdown,
                    "var_95": context.var_95,
                    "var_99": context.var_99,
                    "cvar_95": context.cvar_95,
                    "cvar_975": context.cvar_975
                },
                "probabilistic_metrics": {
                    "probabilistic_sharpe": context.probabilistic_sharpe,
                    "deflated_sharpe": context.deflated_sharpe,
                    "sharpe_confidence_interval": {
                        "lower": context.sharpe_ci_lower,
                        "upper": context.sharpe_ci_upper
                    }
                },
                "trade_statistics": {
                    "total_trades": context.total_trades,
                    "win_rate": context.win_rate,
                    "avg_win": context.avg_win,
                    "avg_loss": context.avg_loss,
                    "best_trade": context.best_trade,
                    "worst_trade": context.worst_trade,
                    "avg_trade_duration": context.avg_trade_duration
                },
                "statistical_tests": {
                    "cagr_tstat": context.cagr_tstat,
                    "cagr_pvalue": context.cagr_pvalue,
                    "returns_skewness": context.returns_skewness,
                    "returns_kurtosis": context.returns_kurtosis,
                    "jarque_bera_stat": context.jarque_bera_stat,
                    "jarque_bera_pvalue": context.jarque_bera_pvalue,
                    "returns_normally_distributed": context.jarque_bera_pvalue > 0.05 if context.jarque_bera_pvalue else None
                },
                "time_period": {
                    "start_date": context.start_date,
                    "end_date": context.end_date,
                    "trading_days": context.trading_days,
                    "years": context.years
                }
            }

        # Add comprehensive Monte Carlo results if available
        if context.mc_cagr_percentile is not None:
            output["monte_carlo"] = {
                "statistical_significance": {
                    "cagr_percentile": context.mc_cagr_percentile,
                    "is_statistically_significant": context.mc_is_significant,
                    "probability_of_loss": context.mc_prob_loss
                },
                "cagr_distribution": {
                    "percentile_5th": context.mc_cagr_5th,
                    "percentile_25th": context.mc_cagr_25th,
                    "percentile_50th_median": context.mc_cagr_50th,
                    "percentile_75th": context.mc_cagr_75th,
                    "percentile_95th": context.mc_cagr_95th
                },
                "other_metrics": {
                    "sharpe_median": context.mc_sharpe_median,
                    "max_drawdown_median": context.mc_max_dd_median
                }
            }

        # Add position sizing metrics if available
        if context.kelly_fraction is not None or context.garch_volatility is not None:
            output["position_sizing"] = {
                "kelly_fraction": context.kelly_fraction,
                "garch_volatility_forecast": context.garch_volatility,
                "volatility_adjusted_size": context.vol_adjusted_size
            }

        return output

    def generate_json_string(
        self,
        context: AnalysisContext,
        include_performance: bool = True,
        indent: int = 2
    ) -> str:
        """
        Generate JSON string output (convenience wrapper).

        Args:
            context: AnalysisContext with all data
            include_performance: Whether to include performance analysis
            indent: JSON indentation level (None for compact)

        Returns:
            JSON string
        """
        data = self.generate_json_report(context, include_performance)
        return json.dumps(data, indent=indent, default=str)

    # =========================================================================
    # FILE OUTPUT METHODS
    # =========================================================================

    def save_all_outputs(
        self,
        context: AnalysisContext,
        output_dir: str = "outputs",
        analyst_name: str = "Fardeen Idrus",
        agent_name: str = "Technical Analyst Agent",
        include_performance: bool = True
    ) -> Dict[str, str]:
        """
        Save analysis to JSON, TXT, and PDF files.

        Args:
            context: AnalysisContext with all data
            output_dir: Directory to save outputs (default: "outputs")
            analyst_name: Name of the analyst for PDF header
            agent_name: Name of the agent for PDF header
            include_performance: Whether to include performance analysis

        Returns:
            Dictionary with paths to saved files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Use ticker as filename (overwrites existing files)
        base_filename = f"{context.ticker}_analysis"

        # Generate the analysis data once
        json_data = self.generate_json_report(context, include_performance)
        text_report = self.generate_full_report(context, include_performance)

        # Save files
        paths = {}

        # Save JSON
        json_path = os.path.join(output_dir, f"{base_filename}.json")
        paths['json'] = self._save_json(json_data, json_path)

        # Save TXT
        txt_path = os.path.join(output_dir, f"{base_filename}.txt")
        paths['txt'] = self._save_txt(text_report, txt_path, analyst_name, agent_name)

        # Save PDF
        pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
        paths['pdf'] = self._save_pdf(
            json_data, text_report, pdf_path,
            analyst_name=analyst_name,
            agent_name=agent_name
        )

        print(f"\nOutputs saved to {output_dir}/:")
        print(f"  - JSON: {os.path.basename(paths['json'])}")
        print(f"  - TXT:  {os.path.basename(paths['txt'])}")
        print(f"  - PDF:  {os.path.basename(paths['pdf'])}")

        return paths

    def _save_json(self, data: Dict[str, Any], filepath: str) -> str:
        """Save JSON data to file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        return filepath

    def _save_txt(
        self,
        report: str,
        filepath: str,
        analyst_name: str,
        agent_name: str
    ) -> str:
        """Save text report to file with header."""
        header = f"""
{'=' * 80}
{agent_name.upper()}
{'=' * 80}
Analyst: {analyst_name}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'=' * 80}
"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write(report)
        return filepath

    def _save_pdf(
        self,
        json_data: Dict[str, Any],
        text_report: str,
        filepath: str,
        analyst_name: str = "Fardeen Idrus",
        agent_name: str = "Technical Analyst Agent"
    ) -> str:
        """
        Generate and save professionally formatted PDF report.

        Args:
            json_data: Structured analysis data
            text_report: Text version of report
            filepath: Output path for PDF
            analyst_name: Name of the analyst
            agent_name: Name of the agent

        Returns:
            Path to saved PDF
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch, cm
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                PageBreak, HRFlowable
            )
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        except ImportError:
            warnings.warn("reportlab not installed. PDF generation skipped. Install with: pip install reportlab")
            # Fall back to saving text as a simple file
            with open(filepath.replace('.pdf', '_fallback.txt'), 'w') as f:
                f.write(text_report)
            return filepath.replace('.pdf', '_fallback.txt')

        # Create document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=A4,
            rightMargin=1*inch,
            leftMargin=1*inch,
            topMargin=1*inch,
            bottomMargin=1*inch
        )

        # Styles
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=6,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1a365d')
        )

        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#4a5568')
        )

        section_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#2d3748'),
            borderPadding=5
        )

        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            leading=14
        )

        # Build document content
        content = []

        # Header
        content.append(Paragraph(agent_name, title_style))
        content.append(Paragraph(
            f"Analysis by: <b>{analyst_name}</b>",
            subtitle_style
        ))
        content.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ParagraphStyle('DateStyle', parent=body_style, alignment=TA_CENTER)
        ))

        # Horizontal line
        content.append(Spacer(1, 10))
        content.append(HRFlowable(
            width="100%",
            thickness=2,
            color=colors.HexColor('#1a365d'),
            spaceBefore=5,
            spaceAfter=15
        ))

        # Executive Summary
        rec = json_data['recommendation']
        content.append(Paragraph("EXECUTIVE SUMMARY", section_style))

        # Recommendation box
        rec_color = {
            'STRONG_BUY': colors.HexColor('#22543d'),
            'BUY': colors.HexColor('#276749'),
            'HOLD': colors.HexColor('#744210'),
            'SELL': colors.HexColor('#9b2c2c'),
            'STRONG_SELL': colors.HexColor('#742a2a')
        }.get(rec['action'], colors.gray)

        summary_data = [
            ['Ticker', json_data['metadata']['ticker']],
            ['Current Price', f"${json_data['metadata']['current_price']:,.2f}"],
            ['Recommendation', rec['action']],
            ['Confidence', f"{rec['confidence']:.0%}"],
            ['Time Horizon', rec['time_horizon'].upper()]
        ]

        summary_table = Table(summary_data, colWidths=[2*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2d3748')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        content.append(summary_table)

        content.append(Spacer(1, 15))
        content.append(Paragraph(f"<b>Rationale:</b> {rec['rationale']}", body_style))

        # Trade Specifications
        content.append(Paragraph("TRADE SPECIFICATIONS", section_style))

        specs = json_data['trade_specifications']
        spec_data = [
            ['Entry Price', f"${specs['entry_price']:,.2f}"],
            ['Stop Loss', f"${specs['stop_loss']:,.2f} ({specs['stop_loss_pct']:+.1f}%)"],
            ['Take Profit', f"${specs['take_profit']:,.2f} ({specs['take_profit_pct']:+.1f}%)"],
            ['Position Size', f"{specs['position_size_pct']:.0%} of portfolio"],
            ['Risk/Reward', f"1:{specs['risk_reward_ratio']:.1f}"]
        ]

        spec_table = Table(spec_data, colWidths=[2*inch, 3*inch])
        spec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        content.append(spec_table)

        # Technical Analysis
        content.append(Paragraph("TECHNICAL ANALYSIS", section_style))

        tech = json_data['technical_analysis']
        tech_data = [
            ['Indicator', 'Value', 'Signal'],
            ['RSI', f"{tech['momentum']['rsi']:.1f}", tech['momentum']['rsi_signal'].upper()],
            ['MACD', f"{tech['momentum']['macd']:.4f}", 'BULLISH' if tech['momentum']['macd_histogram'] > 0 else 'BEARISH'],
            ['ADX', f"{tech['trend']['adx']:.1f}", tech['trend']['trend_strength'].upper()],
            ['Price vs SMA50', f"{tech['trend']['price_vs_sma50_pct']:+.1f}%", 'ABOVE' if tech['trend']['price_vs_sma50_pct'] > 0 else 'BELOW'],
            ['Price vs SMA200', f"{tech['trend']['price_vs_sma200_pct']:+.1f}%", 'ABOVE' if tech['trend']['price_vs_sma200_pct'] > 0 else 'BELOW'],
            ['ATR', f"{tech['volatility']['atr_pct']:.1f}%", '-'],
            ['BB %B', f"{tech['volatility']['bb_percent_b']:.2f}", '-']
        ]

        tech_table = Table(tech_data, colWidths=[1.5*inch, 1.5*inch, 2*inch])
        tech_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a365d')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')])
        ]))
        content.append(tech_table)

        # Regime Analysis
        content.append(Paragraph("REGIME ANALYSIS", section_style))

        regime = json_data['regime_analysis']
        regime_data = [
            ['Market Regime', regime['market_regime']],
            ['Volatility Regime', regime['volatility_regime']],
            ['Trend Persistence', regime['trend_persistence']],
            ['Hurst Exponent', f"{regime['hurst_exponent']:.3f} ({regime['hurst_interpretation'].upper()})"],
            ['Regime Confidence', f"{regime['regime_confidence']:.0%}"]
        ]

        regime_table = Table(regime_data, colWidths=[2*inch, 3*inch])
        regime_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f7fafc')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        content.append(regime_table)

        # Risk Factors
        if json_data['risk_factors']:
            content.append(Paragraph("RISK FACTORS", section_style))
            for i, risk in enumerate(json_data['risk_factors'], 1):
                content.append(Paragraph(f"{i}. {risk}", body_style))

        # Catalysts
        if json_data['catalysts']:
            content.append(Paragraph("POTENTIAL CATALYSTS", section_style))
            for i, catalyst in enumerate(json_data['catalysts'], 1):
                content.append(Paragraph(f"{i}. {catalyst}", body_style))

        # Performance Analysis (if available)
        if 'performance_analysis' in json_data:
            content.append(PageBreak())
            content.append(Paragraph("PERFORMANCE ANALYSIS", section_style))

            perf = json_data['performance_analysis']
            content.append(Paragraph(
                f"<b>Overall Assessment:</b> {perf['overall_assessment'].upper()}",
                body_style
            ))
            content.append(Paragraph(
                f"<b>Confidence in Edge:</b> {perf['confidence_in_edge']:.0%}",
                body_style
            ))
            content.append(Spacer(1, 10))
            content.append(Paragraph(perf['summary'], body_style))

            if perf['strengths']:
                content.append(Spacer(1, 10))
                content.append(Paragraph("<b>Strengths:</b>", body_style))
                for s in perf['strengths']:
                    content.append(Paragraph(f"• {s}", body_style))

            if perf['weaknesses']:
                content.append(Spacer(1, 10))
                content.append(Paragraph("<b>Weaknesses:</b>", body_style))
                for w in perf['weaknesses']:
                    content.append(Paragraph(f"• {w}", body_style))

        # Comprehensive Backtest Metrics (if available)
        if 'backtest_metrics' in json_data:
            bt = json_data['backtest_metrics']

            # Return Metrics Section
            content.append(Paragraph("RETURN METRICS", section_style))
            ret = bt.get('return_metrics', {})
            ret_data = [
                ['Metric', 'Value'],
                ['Total Return', f"{ret.get('total_return', 0):.1%}" if ret.get('total_return') else 'N/A'],
                ['CAGR', f"{ret.get('cagr', 0):.1%}" if ret.get('cagr') else 'N/A'],
                ['Volatility (Ann.)', f"{ret.get('volatility', 0):.1%}" if ret.get('volatility') else 'N/A']
            ]
            ret_table = Table(ret_data, colWidths=[2.5*inch, 2.5*inch])
            ret_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a365d')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            content.append(ret_table)

            # Risk-Adjusted Metrics Section
            content.append(Paragraph("RISK-ADJUSTED METRICS", section_style))
            risk_adj = bt.get('risk_adjusted_metrics', {})
            risk_adj_data = [
                ['Metric', 'Value', 'Interpretation'],
                ['Sharpe Ratio', f"{risk_adj.get('sharpe_ratio', 0):.2f}" if risk_adj.get('sharpe_ratio') else 'N/A',
                 'Good' if risk_adj.get('sharpe_ratio', 0) > 1 else 'Fair' if risk_adj.get('sharpe_ratio', 0) > 0.5 else 'Poor'],
                ['Sortino Ratio', f"{risk_adj.get('sortino_ratio', 0):.2f}" if risk_adj.get('sortino_ratio') else 'N/A',
                 'Downside-adjusted'],
                ['Calmar Ratio', f"{risk_adj.get('calmar_ratio', 0):.2f}" if risk_adj.get('calmar_ratio') else 'N/A',
                 'Return/MaxDD'],
                ['Omega Ratio', f"{risk_adj.get('omega_ratio', 0):.2f}" if risk_adj.get('omega_ratio') else 'N/A',
                 'Gain/Loss prob'],
                ['Profit Factor', f"{risk_adj.get('profit_factor', 0):.2f}" if risk_adj.get('profit_factor') else 'N/A',
                 'Good' if risk_adj.get('profit_factor', 0) > 1.5 else 'Fair']
            ]
            risk_adj_table = Table(risk_adj_data, colWidths=[1.5*inch, 1.5*inch, 2*inch])
            risk_adj_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a365d')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            content.append(risk_adj_table)

            # Risk Metrics (VaR/CVaR) Section
            content.append(Paragraph("RISK METRICS (VaR / CVaR)", section_style))
            risk = bt.get('risk_metrics', {})
            risk_data = [
                ['Metric', 'Value', 'Description'],
                ['Max Drawdown', f"{risk.get('max_drawdown', 0):.1%}" if risk.get('max_drawdown') else 'N/A', 'Worst peak-to-trough'],
                ['VaR (95%)', f"{risk.get('var_95', 0):.2%}" if risk.get('var_95') else 'N/A', 'Daily loss 95% conf'],
                ['VaR (99%)', f"{risk.get('var_99', 0):.2%}" if risk.get('var_99') else 'N/A', 'Daily loss 99% conf'],
                ['CVaR (95%)', f"{risk.get('cvar_95', 0):.2%}" if risk.get('cvar_95') else 'N/A', 'Expected shortfall'],
                ['CVaR (97.5%)', f"{risk.get('cvar_975', 0):.2%}" if risk.get('cvar_975') else 'N/A', 'Tail risk']
            ]
            risk_table = Table(risk_data, colWidths=[1.5*inch, 1.5*inch, 2*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#742a2a')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            content.append(risk_table)

            # Trade Statistics Section
            content.append(Paragraph("TRADE STATISTICS", section_style))
            trades = bt.get('trade_statistics', {})
            trades_data = [
                ['Metric', 'Value'],
                ['Total Trades', str(trades.get('total_trades', 0))],
                ['Win Rate', f"{trades.get('win_rate', 0):.1%}" if trades.get('win_rate') else 'N/A'],
                ['Avg Win', f"{trades.get('avg_win', 0):.2%}" if trades.get('avg_win') else 'N/A'],
                ['Avg Loss', f"{trades.get('avg_loss', 0):.2%}" if trades.get('avg_loss') else 'N/A'],
                ['Best Trade', f"{trades.get('best_trade', 0):.2%}" if trades.get('best_trade') else 'N/A'],
                ['Worst Trade', f"{trades.get('worst_trade', 0):.2%}" if trades.get('worst_trade') else 'N/A']
            ]
            trades_table = Table(trades_data, colWidths=[2.5*inch, 2.5*inch])
            trades_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a365d')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            content.append(trades_table)

            # Statistical Tests Section
            content.append(Paragraph("STATISTICAL SIGNIFICANCE", section_style))
            stats = bt.get('statistical_tests', {})
            prob = bt.get('probabilistic_metrics', {})
            stats_data = [
                ['Test', 'Value', 'Interpretation'],
                ['CAGR p-value', f"{stats.get('cagr_pvalue', 0):.4f}" if stats.get('cagr_pvalue') else 'N/A',
                 'Significant' if stats.get('cagr_pvalue', 1) < 0.05 else 'Not significant'],
                ['Prob. Sharpe', f"{prob.get('probabilistic_sharpe', 0):.1%}" if prob.get('probabilistic_sharpe') else 'N/A',
                 'P(true Sharpe > 0)'],
                ['Deflated Sharpe', f"{prob.get('deflated_sharpe', 0):.2f}" if prob.get('deflated_sharpe') else 'N/A',
                 'Adjusted for trials'],
                ['Skewness', f"{stats.get('returns_skewness', 0):.2f}" if stats.get('returns_skewness') else 'N/A',
                 'Normal=0'],
                ['Kurtosis', f"{stats.get('returns_kurtosis', 0):.2f}" if stats.get('returns_kurtosis') else 'N/A',
                 'Normal=3'],
                ['Jarque-Bera p', f"{stats.get('jarque_bera_pvalue', 0):.4f}" if stats.get('jarque_bera_pvalue') else 'N/A',
                 'Normal' if stats.get('jarque_bera_pvalue', 0) > 0.05 else 'Non-normal']
            ]
            stats_table = Table(stats_data, colWidths=[1.5*inch, 1.5*inch, 2*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a365d')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            content.append(stats_table)

        # Comprehensive Monte Carlo Section (if available)
        if 'monte_carlo' in json_data:
            content.append(PageBreak())
            content.append(Paragraph("MONTE CARLO SIMULATION ANALYSIS", section_style))

            mc = json_data['monte_carlo']
            sig = mc.get('statistical_significance', {})
            dist = mc.get('cagr_distribution', {})
            other = mc.get('other_metrics', {})

            # Significance table
            sig_data = [
                ['Statistical Significance', 'Value'],
                ['CAGR Percentile', f"{sig.get('cagr_percentile', 0):.0f}th" if sig.get('cagr_percentile') else 'N/A'],
                ['Statistically Significant', 'Yes' if sig.get('is_statistically_significant') else 'No'],
                ['Probability of 10%+ Loss', f"{sig.get('probability_of_loss', 0):.1%}" if sig.get('probability_of_loss') is not None else 'N/A']
            ]
            sig_table = Table(sig_data, colWidths=[2.5*inch, 2.5*inch])
            sig_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#276749')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            content.append(sig_table)

            # CAGR Distribution table
            if dist:
                content.append(Spacer(1, 15))
                content.append(Paragraph("<b>CAGR Distribution (Bootstrap Simulations)</b>", body_style))
                dist_data = [
                    ['Percentile', '5th', '25th', '50th (Median)', '75th', '95th'],
                    ['CAGR',
                     f"{dist.get('percentile_5th', 0):.1%}" if dist.get('percentile_5th') else 'N/A',
                     f"{dist.get('percentile_25th', 0):.1%}" if dist.get('percentile_25th') else 'N/A',
                     f"{dist.get('percentile_50th_median', 0):.1%}" if dist.get('percentile_50th_median') else 'N/A',
                     f"{dist.get('percentile_75th', 0):.1%}" if dist.get('percentile_75th') else 'N/A',
                     f"{dist.get('percentile_95th', 0):.1%}" if dist.get('percentile_95th') else 'N/A']
                ]
                dist_table = Table(dist_data, colWidths=[1*inch, 0.9*inch, 0.9*inch, 1.2*inch, 0.9*inch, 0.9*inch])
                dist_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a5568')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                    ('TOPPADDING', (0, 0), (-1, -1), 5),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ]))
                content.append(dist_table)

        # Position Sizing Section (if available)
        if 'position_sizing' in json_data:
            content.append(Paragraph("POSITION SIZING ANALYSIS", section_style))
            ps = json_data['position_sizing']
            ps_data = [
                ['Metric', 'Value'],
                ['Kelly Fraction', f"{ps.get('kelly_fraction', 0):.1%}" if ps.get('kelly_fraction') else 'N/A'],
                ['GARCH Vol Forecast', f"{ps.get('garch_volatility_forecast', 0):.1%}" if ps.get('garch_volatility_forecast') else 'N/A'],
                ['Vol-Adjusted Size', f"{ps.get('volatility_adjusted_size', 0):.1%}" if ps.get('volatility_adjusted_size') else 'N/A']
            ]
            ps_table = Table(ps_data, colWidths=[2.5*inch, 2.5*inch])
            ps_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a365d')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            content.append(ps_table)

        # Footer
        content.append(Spacer(1, 30))
        content.append(HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor('#e2e8f0'),
            spaceBefore=10,
            spaceAfter=10
        ))
        content.append(Paragraph(
            f"<i>Report generated by {agent_name} | Analyst: {analyst_name}</i>",
            ParagraphStyle('Footer', parent=body_style, alignment=TA_CENTER, fontSize=8, textColor=colors.gray)
        ))

        # Build PDF
        doc.build(content)

        return filepath

    def get_quick_signal_explanation(self, context: AnalysisContext) -> str:
        """
        Get a quick one-paragraph explanation of the current signal.

        Args:
            context: AnalysisContext

        Returns:
            Brief explanation string
        """
        prompt = f"""In 2-3 sentences, explain why the signal is {context.signal} given:
- RSI: {context.rsi:.1f}
- MACD vs Signal: {context.macd:.4f} vs {context.macd_signal:.4f}
- Market Regime: {context.market_regime}
- Confluence Score: {context.confluence_score:.2f}

Be concise and technical."""

        response = self._call_claude(
            "You are a technical analyst. Provide brief, professional explanations.",
            prompt,
            use_tools=False
        )

        return response


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_context_from_data(
    data: pd.DataFrame,
    ticker: str = "UNKNOWN",
    performance_report: Optional[Any] = None,
    mc_result: Optional[Any] = None
) -> AnalysisContext:
    """
    Create AnalysisContext from DataFrame.

    Args:
        data: DataFrame with indicators and regime data
        ticker: Stock ticker symbol
        performance_report: Optional PerformanceReport
        mc_result: Optional MonteCarloResult

    Returns:
        AnalysisContext object
    """
    latest = data.iloc[-1]

    context = AnalysisContext(
        ticker=ticker,
        current_price=float(latest['Close']),
        current_date=str(data.index[-1].date()) if hasattr(data.index[-1], 'date') else str(data.index[-1]),

        # Technical indicators
        rsi=float(latest.get('RSI', 50)),
        macd=float(latest.get('MACD', 0)),
        macd_signal=float(latest.get('MACD_Signal', 0)),
        sma_50=float(latest.get('SMA_50', latest['Close'])),
        sma_200=float(latest.get('SMA_200', latest['Close'])),
        bb_percent_b=float(latest.get('BB_Percent_B', 0.5)),
        atr=float(latest.get('ATR', 0)),
        adx=float(latest.get('ADX', 25)),

        # Regime
        market_regime=str(latest.get('Market_Regime', 'SIDEWAYS')),
        volatility_regime=str(latest.get('Volatility_Regime', 'NORMAL_VOLATILITY')),
        trend_persistence=str(latest.get('Trend_Persistence', 'RANDOM_WALK')),
        hurst_exponent=float(latest.get('Hurst_Exponent', 0.5)),
        regime_confidence=float(latest.get('Regime_Confidence', 0.5)),

        # Signal
        signal=str(latest.get('Signal', 'HOLD')),
        signal_confidence=float(latest.get('Signal_Confidence', 0.5)),
        confluence_score=float(latest.get('Confluence_Score', 0)),
        strategy=str(latest.get('Strategy', 'TREND_FOLLOWING'))
    )

    # Add comprehensive performance metrics if available
    if performance_report:
        # Basic metrics
        context.total_return = getattr(performance_report, 'total_return', None)
        context.cagr = getattr(performance_report, 'cagr', None)
        context.volatility = getattr(performance_report, 'volatility', None)
        context.sharpe_ratio = getattr(performance_report, 'sharpe_ratio', None)
        context.max_drawdown = getattr(performance_report, 'max_drawdown', None)

        # Advanced risk-adjusted metrics
        context.sortino_ratio = getattr(performance_report, 'sortino_ratio', None)
        context.calmar_ratio = getattr(performance_report, 'calmar_ratio', None)
        context.omega_ratio = getattr(performance_report, 'omega_ratio', None)
        context.profit_factor = getattr(performance_report, 'profit_factor', None)

        # Risk metrics (VaR/CVaR)
        context.var_95 = getattr(performance_report, 'var_95', None)
        context.var_99 = getattr(performance_report, 'var_99', None)
        context.cvar_95 = getattr(performance_report, 'cvar_95', None)
        context.cvar_975 = getattr(performance_report, 'cvar_975', None)

        # Probabilistic metrics
        context.probabilistic_sharpe = getattr(performance_report, 'probabilistic_sharpe', None)
        context.deflated_sharpe = getattr(performance_report, 'deflated_sharpe', None)
        sharpe_ci = getattr(performance_report, 'sharpe_confidence_interval', None)
        if sharpe_ci:
            context.sharpe_ci_lower = sharpe_ci[0]
            context.sharpe_ci_upper = sharpe_ci[1]

        # Trade statistics
        context.total_trades = getattr(performance_report, 'total_trades', None)
        context.win_rate = getattr(performance_report, 'win_rate', None)
        context.avg_win = getattr(performance_report, 'avg_win', None)
        context.avg_loss = getattr(performance_report, 'avg_loss', None)
        context.best_trade = getattr(performance_report, 'best_trade', None)
        context.worst_trade = getattr(performance_report, 'worst_trade', None)
        context.avg_trade_duration = getattr(performance_report, 'avg_trade_duration', None)

        # Statistical tests
        context.cagr_tstat = getattr(performance_report, 'cagr_tstat', None)
        context.cagr_pvalue = getattr(performance_report, 'cagr_pvalue', None)
        context.returns_skewness = getattr(performance_report, 'returns_skewness', None)
        context.returns_kurtosis = getattr(performance_report, 'returns_kurtosis', None)
        context.jarque_bera_stat = getattr(performance_report, 'jarque_bera_stat', None)
        context.jarque_bera_pvalue = getattr(performance_report, 'jarque_bera_pvalue', None)

        # Time period
        context.start_date = getattr(performance_report, 'start_date', None)
        context.end_date = getattr(performance_report, 'end_date', None)
        context.trading_days = getattr(performance_report, 'trading_days', None)
        context.years = getattr(performance_report, 'years', None)

    # Add comprehensive Monte Carlo results if available
    if mc_result:
        context.mc_cagr_percentile = getattr(mc_result, 'cagr_percentile', None)
        context.mc_is_significant = getattr(mc_result, 'is_statistically_significant', None)
        context.mc_prob_loss = getattr(mc_result, 'prob_loss_10pct', None)

        # CAGR distribution percentiles
        cagr_dist = getattr(mc_result, 'cagr_distribution', None)
        if cagr_dist is not None and hasattr(cagr_dist, '__len__') and len(cagr_dist) > 0:
            import numpy as np
            context.mc_cagr_5th = float(np.percentile(cagr_dist, 5))
            context.mc_cagr_25th = float(np.percentile(cagr_dist, 25))
            context.mc_cagr_50th = float(np.percentile(cagr_dist, 50))
            context.mc_cagr_75th = float(np.percentile(cagr_dist, 75))
            context.mc_cagr_95th = float(np.percentile(cagr_dist, 95))

        # Other MC metrics
        sharpe_dist = getattr(mc_result, 'sharpe_distribution', None)
        if sharpe_dist is not None and hasattr(sharpe_dist, '__len__') and len(sharpe_dist) > 0:
            import numpy as np
            context.mc_sharpe_median = float(np.median(sharpe_dist))

        max_dd_dist = getattr(mc_result, 'max_drawdown_distribution', None)
        if max_dd_dist is not None and hasattr(max_dd_dist, '__len__') and len(max_dd_dist) > 0:
            import numpy as np
            context.mc_max_dd_median = float(np.median(max_dd_dist))

    # Add position sizing metrics if position_sizer provided
    if hasattr(mc_result, 'kelly_fraction') if mc_result else False:
        context.kelly_fraction = getattr(mc_result, 'kelly_fraction', None)

    return context


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
    from performance_metrics import PerformanceAnalyser
    from monte_carlo import MonteCarloSimulator

    print("=" * 70)
    print("LLM AGENT TEST")
    print("=" * 70)

    # Load and process data
    print("\nLoading data...")
    collector = DataCollector()
    data = collector.get_data('AAPL', years=5)

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
    config = BacktestConfig(use_position_sizer=False)
    engine = BacktestEngine(data, config)
    results = engine.run_backtest()

    # Get performance metrics
    returns = engine.portfolio.returns()
    try:
        trades_df = engine.portfolio.trades.records_readable
    except:
        trades_df = None

    analyser = PerformanceAnalyser(returns, trades_df)
    report = analyser.generate_report()

    # Run Monte Carlo
    print("Running Monte Carlo...")
    simulator = MonteCarloSimulator(returns)
    mc_result = simulator.run_simulation(n_simulations=500, verbose=False)

    # Create context
    print("\nCreating analysis context...")
    context = create_context_from_data(
        data,
        ticker='AAPL',
        performance_report=report,
        mc_result=mc_result
    )

    print(f"\nContext created:")
    print(f"  Price: ${context.current_price:.2f}")
    print(f"  RSI: {context.rsi:.1f}")
    print(f"  Regime: {context.market_regime}")
    print(f"  Signal: {context.signal}")

    # Initialise LLM agent
    print("\n" + "=" * 70)
    print("INITIALISING LLM AGENT")
    print("=" * 70)

    agent = LLMAgent(data)
    agent.trades_df = trades_df

    # Check if API is available
    if agent.anthropic_client is None and agent.openai_client is None:
        print("\nNo API keys configured. Showing placeholder output...")
        print("\nTo enable LLM analysis, set environment variables:")
        print("  export ANTHROPIC_API_KEY=your_key")
        print("  or")
        print("  export OPENAI_API_KEY=your_key")
    else:
        # Generate recommendation
        print("\n" + "=" * 70)
        print("GENERATING TRADE RECOMMENDATION")
        print("=" * 70)

        recommendation = agent.generate_trade_recommendation(context)

        print(f"\nRecommendation: {recommendation.recommendation.value}")
        print(f"Confidence: {recommendation.confidence:.0%}")
        print(f"Entry: ${recommendation.entry_price:.2f}")
        print(f"Stop: ${recommendation.stop_loss:.2f}")
        print(f"Target: ${recommendation.take_profit:.2f}")
        print(f"R:R: 1:{recommendation.risk_reward_ratio:.1f}")
        print(f"\nRationale: {recommendation.rationale[:200]}...")

        print(f"\nRisks:")
        for risk in recommendation.risks[:3]:
            print(f"  - {risk}")

        # Generate JSON report (primary output for master agent)
        print("\n" + "=" * 70)
        print("GENERATING JSON REPORT (for master agent)")
        print("=" * 70)

        json_output = agent.generate_json_report(context)
        print("\nJSON Output Structure:")
        print(f"  - metadata: ticker, date, price, model")
        print(f"  - recommendation: {json_output['recommendation']['action']} ({json_output['recommendation']['confidence']:.0%})")
        print(f"  - trade_specifications: entry, stop, target, R:R")
        print(f"  - technical_analysis: momentum, trend, volatility")
        print(f"  - regime_analysis: {json_output['regime_analysis']['market_regime']}")
        print(f"  - signal_system: {json_output['signal_system']['current_signal']}")
        print(f"  - risk_factors: {len(json_output['risk_factors'])} items")
        print(f"  - catalysts: {len(json_output['catalysts'])} items")
        if 'performance_analysis' in json_output:
            print(f"  - performance_analysis: {json_output['performance_analysis']['overall_assessment']}")
        if 'backtest_metrics' in json_output:
            print(f"  - backtest_metrics: CAGR={json_output['backtest_metrics']['return_metrics']['cagr']:.1%}, Sharpe={json_output['backtest_metrics']['risk_adjusted_metrics']['sharpe_ratio']:.2f}")
        if 'monte_carlo' in json_output:
            print(f"  - monte_carlo: {json_output['monte_carlo']['statistical_significance']['cagr_percentile']:.0f}th percentile")

        # Show full JSON string
        print("\n" + "-" * 70)
        print("FULL JSON OUTPUT:")
        print("-" * 70)
        json_string = agent.generate_json_string(context)
        print(json_string)

        # Also show text report for comparison
        print("\n" + "=" * 70)
        print("TEXT REPORT (for human readability)")
        print("=" * 70)

        full_report = agent.generate_full_report(context)
        print(full_report)

        # Save all outputs to files
        print("\n" + "=" * 70)
        print("SAVING OUTPUTS TO FILES")
        print("=" * 70)

        output_paths = agent.save_all_outputs(
            context,
            output_dir="outputs",
            analyst_name="Fardeen Idrus",
            agent_name="Technical Analyst Agent"
        )

        print("\nFiles saved:")
        for format_type, path in output_paths.items():
            print(f"  {format_type.upper()}: {path}")

    print("\n" + "=" * 70)
    print("LLM AGENT TEST COMPLETE")
    print("=" * 70)