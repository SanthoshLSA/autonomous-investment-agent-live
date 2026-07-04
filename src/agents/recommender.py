"""
Recommender agent node running LLM synthesis for structural recommendations.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.state import InvestmentAgentState
from src.config import get_config
from src.logger import get_logger

logger = get_logger(__name__)


def create_recommender_node(llm: Any) -> Callable[[InvestmentAgentState], dict[str, Any]]:
    """Factory creating the recommender node bound to the specified Language Model.

    Args:
        llm: A LangChain-compatible LLM instance.

    Returns:
        Recommender node function.
    """

    def recommender_node(state: InvestmentAgentState) -> dict[str, Any]:
        logger.info("Recommender node started")
        composite_scores = state.get("composite_scores")
        if not composite_scores:
            err = "Recommender failed: No composite scoring data found in state."
            logger.error(err)
            return {
                "error_log": [err],
                "audit_log": ["Recommender aborted: Missing composite scores."],
            }

        config = get_config()
        risk_tolerance = config.portfolio.risk_tolerance

        # Construct payload of metrics for the LLM context
        analysis_context = {}
        for ticker, score in composite_scores.items():
            tech = state["technical_analysis"].get(ticker, {})
            risk = state["risk_analysis"].get(ticker, {})
            sent = state["sentiment_analysis"].get(ticker, {})

            analysis_context[ticker] = {
                "risk_score": score["risk_score"],
                "signal": score["signal"],
                "confidence": score["confidence"],
                "technical_overall": tech.get("overall_signal", "hold"),
                "rsi_value": tech.get("rsi_value", 50.0),
                "sharpe_ratio": risk.get("sharpe_ratio", 0.0),
                "volatility": risk.get("volatility", 0.0),
                "max_drawdown": risk.get("max_drawdown", 0.0),
                "sentiment_weighted": sent.get("weighted_sentiment", 0.0),
                "sentiment_classification": sent.get("classification", "neutral"),
            }

        system_prompt = f"""You are an elite, production-grade Investment Adviser.
Your task is to analyze the mathematical metrics of the target assets and propose a portfolio allocation.
The user has specified a {risk_tolerance.upper()} risk tolerance.

You must respond ONLY with a valid JSON document containing:
1. "allocations": A dictionary of ticker keys to allocation percentages (floats between 0.0 and 1.0). They MUST sum to 1.0 or less. Unallocated capital will remain in Cash.
2. "rationale": A dictionary mapping each ticker to a concise, structured markdown sentence explaining the decision.
3. "portfolio_summary": An executive summary describing the overall risk and investment stance of the proposed portfolio.
4. "warnings": A list of key risks or caveats identified (e.g. high volatility, bad sentiment, etc.).

Do not output any markdown code blocks, explanatory text, or conversational intros. Output ONLY raw JSON."""

        user_prompt = f"""Watchlist Analysis Context:
{json.dumps(analysis_context, indent=2)}

Risk Stance: {risk_tolerance}

Generate the JSON recommendation portfolio."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            logger.info("Invoking LLM for synthesis...")
            response = llm.invoke(messages)
            content = response.content.strip()

            # Clean markdown formatting wraps if LLM ignored instructions
            if content.startswith("```"):
                lines = content.splitlines()
                if lines[0].startswith("```json") or lines[0].startswith("```"):
                    content = "\n".join(lines[1:-1])

            recs = json.loads(content)

            # Simple fallback check to guarantee fields exist
            if "allocations" not in recs:
                raise ValueError("LLM response missing 'allocations' field.")

            logger.info("Recommender node complete", allocations=recs.get("allocations"))
            return {
                "portfolio_recommendation": recs,
                "audit_log": ["Recommender: Portfolio synthesis complete with AI rationale."],
            }

        except Exception as e:
            logger.exception(
                "Recommender node LLM failure, using deterministic conservative fallback"
            )

            # Deterministic conservative fallback allocation (Equal weight bounded by 20% max, rest Cash)
            fallback_allocations = {}
            active_tickers = list(composite_scores.keys())
            if active_tickers:
                # Divide up to 60% of portfolio equally among buy/hold assets, cash for the rest
                valid_assets = [
                    t
                    for t in active_tickers
                    if composite_scores[t]["signal"] in ["strong_buy", "buy", "hold"]
                ]
                if not valid_assets:
                    valid_assets = active_tickers

                weight = min(0.6 / len(valid_assets), config.portfolio.max_single_position)
                for t in active_tickers:
                    if t in valid_assets:
                        fallback_allocations[t] = round(weight, 4)
                    else:
                        fallback_allocations[t] = 0.0

            fallback_rec = {
                "allocations": fallback_allocations,
                "rationale": {
                    t: "Deterministic fallback due to LLM timeout/unavailability."
                    for t in active_tickers
                },
                "portfolio_summary": "Conservative equal-weight fallback allocation designed for stability.",
                "warnings": [
                    "System used rule-based fallback recommendation because LLM service was unreachable."
                ],
            }
            return {
                "portfolio_recommendation": fallback_rec,
                "error_log": [f"Recommender LLM fallback triggered: {str(e)}"],
                "audit_log": ["Recommender: Fallback portfolio generated due to LLM error."],
            }

    return recommender_node
