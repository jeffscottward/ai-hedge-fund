import json
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_company_news, get_financial_metrics, get_market_cap, search_line_items
from src.utils.api_key import get_api_key_from_state
from src.utils.llm import call_llm
from src.utils.progress import progress


class NicholasCrownSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: int = Field(description="Confidence 0-100")
    reasoning: str = Field(description="Reasoning for the decision")


def nicholas_crown_agent(state: AgentState, agent_id: str = "nicholas_crown_agent"):
    """Analyze stocks through Nicholas Crown's real-return and downside-control lens."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    analysis_data = {}
    crown_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=5, api_key=api_key)

        progress.update_status(agent_id, ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "operating_income",
                "net_income",
                "free_cash_flow",
                "capital_expenditure",
                "cash_and_equivalents",
                "total_debt",
                "shareholders_equity",
                "dividends_and_other_cash_distributions",
            ],
            end_date,
            period="ttm",
            limit=5,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching company news")
        company_news = get_company_news(ticker, end_date, limit=30, api_key=api_key)

        progress.update_status(agent_id, ticker, "Analyzing real return quality")
        real_return_analysis = analyze_real_return_quality(metrics, financial_line_items, market_cap)

        progress.update_status(agent_id, ticker, "Analyzing balance sheet risk")
        balance_sheet_analysis = analyze_balance_sheet_risk(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing volatility drag")
        volatility_analysis = analyze_volatility_drag(metrics, company_news)

        progress.update_status(agent_id, ticker, "Analyzing income durability")
        income_analysis = analyze_income_durability(metrics, financial_line_items)

        total_score = real_return_analysis["score"] + balance_sheet_analysis["score"] + volatility_analysis["score"] + income_analysis["score"]
        max_score = 40

        if total_score >= 28:
            signal = "bullish"
        elif total_score <= 18:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {
            "ticker": ticker,
            "signal_from_scores": signal,
            "score": total_score,
            "max_score": max_score,
            "real_return_analysis": real_return_analysis,
            "balance_sheet_analysis": balance_sheet_analysis,
            "volatility_analysis": volatility_analysis,
            "income_analysis": income_analysis,
            "market_cap": market_cap,
        }

        progress.update_status(agent_id, ticker, "Generating Nicholas Crown analysis")
        crown_output = generate_crown_output(
            ticker=ticker,
            analysis_data=analysis_data[ticker],
            state=state,
            agent_id=agent_id,
        )

        crown_analysis[ticker] = {
            "signal": crown_output.signal,
            "confidence": crown_output.confidence,
            "reasoning": crown_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=crown_output.reasoning)

    message = HumanMessage(content=json.dumps(crown_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(crown_analysis, "Nicholas Crown Agent")

    state["data"]["analyst_signals"][agent_id] = crown_analysis

    progress.update_status(agent_id, None, "Done")
    return {"messages": [message], "data": state["data"]}


def analyze_real_return_quality(metrics: list, financial_line_items: list, market_cap: float | None) -> dict[str, Any]:
    latest = metrics[0] if metrics else None
    score = 0
    details = []

    fcf_yield = getattr(latest, "free_cash_flow_yield", None) if latest else None
    if fcf_yield is not None:
        if fcf_yield >= 0.05:
            score += 3
            details.append(f"Free cash flow yield clears a real-return hurdle: {fcf_yield:.1%}")
        elif fcf_yield >= 0.025:
            score += 2
            details.append(f"Moderate free cash flow yield: {fcf_yield:.1%}")
        else:
            details.append(f"Thin free cash flow yield: {fcf_yield:.1%}")
    elif market_cap:
        fcf_values = [item.free_cash_flow for item in financial_line_items if getattr(item, "free_cash_flow", None) is not None]
        if fcf_values and fcf_values[0] > 0:
            implied_yield = fcf_values[0] / market_cap
            if implied_yield >= 0.05:
                score += 3
            elif implied_yield >= 0.025:
                score += 2
            details.append(f"Implied free cash flow yield: {implied_yield:.1%}")
        else:
            details.append("No positive free cash flow yield available")
    else:
        details.append("No market cap/free cash flow yield available")

    growth_fields = ["revenue_growth", "earnings_growth", "free_cash_flow_growth"]
    positive_growth = [getattr(latest, field, None) for field in growth_fields if latest and getattr(latest, field, None) is not None and getattr(latest, field, None) > 0]
    if len(positive_growth) >= 2:
        score += 3
        details.append("Multiple growth measures are positive after inflation pressure")
    elif positive_growth:
        score += 1
        details.append("Some growth measures are positive")
    else:
        details.append("Growth data is weak or unavailable")

    operating_margin = getattr(latest, "operating_margin", None) if latest else None
    if operating_margin is not None:
        if operating_margin >= 0.20:
            score += 2
            details.append(f"Strong operating margin: {operating_margin:.1%}")
        elif operating_margin >= 0.10:
            score += 1
            details.append(f"Acceptable operating margin: {operating_margin:.1%}")
        else:
            details.append(f"Low operating margin: {operating_margin:.1%}")

    return {"score": min(score, 10), "details": "; ".join(details)}


def analyze_balance_sheet_risk(metrics: list, financial_line_items: list) -> dict[str, Any]:
    latest = metrics[0] if metrics else None
    score = 0
    details = []

    debt_to_equity = getattr(latest, "debt_to_equity", None) if latest else None
    if debt_to_equity is not None:
        if debt_to_equity <= 0.5:
            score += 3
            details.append(f"Low debt-to-equity: {debt_to_equity:.2f}")
        elif debt_to_equity <= 1.5:
            score += 1
            details.append(f"Manageable debt-to-equity: {debt_to_equity:.2f}")
        else:
            details.append(f"Heavy debt-to-equity: {debt_to_equity:.2f}")
    else:
        details.append("Debt-to-equity unavailable")

    interest_coverage = getattr(latest, "interest_coverage", None) if latest else None
    if interest_coverage is not None:
        if interest_coverage >= 8:
            score += 3
            details.append(f"Strong interest coverage: {interest_coverage:.1f}x")
        elif interest_coverage >= 3:
            score += 1
            details.append(f"Adequate interest coverage: {interest_coverage:.1f}x")
        else:
            details.append(f"Weak interest coverage: {interest_coverage:.1f}x")

    current_ratio = getattr(latest, "current_ratio", None) if latest else None
    if current_ratio is not None:
        if current_ratio >= 1.5:
            score += 2
            details.append(f"Healthy liquidity: current ratio {current_ratio:.1f}")
        elif current_ratio >= 1.0:
            score += 1
            details.append(f"Adequate liquidity: current ratio {current_ratio:.1f}")
        else:
            details.append(f"Tight liquidity: current ratio {current_ratio:.1f}")

    cash_values = [item.cash_and_equivalents for item in financial_line_items if getattr(item, "cash_and_equivalents", None) is not None]
    debt_values = [item.total_debt for item in financial_line_items if getattr(item, "total_debt", None) is not None]
    if cash_values and debt_values and debt_values[0]:
        cash_to_debt = cash_values[0] / debt_values[0]
        if cash_to_debt >= 0.5:
            score += 2
            details.append(f"Cash covers a meaningful share of debt: {cash_to_debt:.1%}")
        else:
            details.append(f"Limited cash-to-debt cushion: {cash_to_debt:.1%}")

    return {"score": min(score, 10), "details": "; ".join(details)}


def analyze_volatility_drag(metrics: list, news_items: list) -> dict[str, Any]:
    latest = metrics[0] if metrics else None
    score = 10
    details = []

    leverage = getattr(latest, "debt_to_assets", None) if latest else None
    if leverage is not None and leverage > 0.6:
        score -= 3
        details.append(f"High debt-to-assets can amplify volatility: {leverage:.1%}")
    elif leverage is not None:
        details.append(f"Debt-to-assets is not extreme: {leverage:.1%}")

    negative_keywords = ["lawsuit", "fraud", "investigation", "downgrade", "bankruptcy", "default", "recall", "misses"]
    negative_count = sum(1 for item in news_items if any(word in (item.title or "").lower() for word in negative_keywords))
    if news_items and negative_count / len(news_items) > 0.25:
        score -= 3
        details.append(f"News flow has elevated negative headline risk: {negative_count}/{len(news_items)}")
    elif negative_count:
        score -= 1
        details.append(f"Some negative headlines present: {negative_count}/{len(news_items)}")
    else:
        details.append("No obvious negative headline cluster")

    pe_ratio = getattr(latest, "price_to_earnings_ratio", None) if latest else None
    if pe_ratio is not None and pe_ratio > 40:
        score -= 2
        details.append(f"High P/E leaves little room for disappointment: {pe_ratio:.1f}")

    return {"score": max(score, 0), "details": "; ".join(details)}


def analyze_income_durability(metrics: list, financial_line_items: list) -> dict[str, Any]:
    latest = metrics[0] if metrics else None
    score = 0
    details = []

    fcf_values = [item.free_cash_flow for item in financial_line_items if getattr(item, "free_cash_flow", None) is not None]
    if len(fcf_values) >= 3 and all(value > 0 for value in fcf_values[:3]):
        score += 4
        details.append("Free cash flow has been positive across recent periods")
    elif fcf_values and fcf_values[0] > 0:
        score += 2
        details.append("Latest free cash flow is positive")
    else:
        details.append("Free cash flow is weak or unavailable")

    payout_ratio = getattr(latest, "payout_ratio", None) if latest else None
    if payout_ratio is not None:
        if 0 < payout_ratio <= 0.6:
            score += 2
            details.append(f"Dividend payout looks sustainable: {payout_ratio:.1%}")
        elif payout_ratio > 0.9:
            details.append(f"Payout ratio looks stretched: {payout_ratio:.1%}")

    roic = getattr(latest, "return_on_invested_capital", None) if latest else None
    if roic is not None:
        if roic >= 0.12:
            score += 3
            details.append(f"Strong return on invested capital: {roic:.1%}")
        elif roic >= 0.06:
            score += 1
            details.append(f"Moderate return on invested capital: {roic:.1%}")
        else:
            details.append(f"Low return on invested capital: {roic:.1%}")

    capex_values = [abs(item.capital_expenditure) for item in financial_line_items if getattr(item, "capital_expenditure", None) is not None]
    if fcf_values and capex_values and fcf_values[0] > 0:
        capex_to_fcf = capex_values[0] / fcf_values[0]
        if capex_to_fcf <= 0.75:
            score += 1
            details.append(f"Capital expenditure burden is manageable: {capex_to_fcf:.1%} of FCF")
        else:
            details.append(f"Capital expenditure burden may pressure distributable cash: {capex_to_fcf:.1%} of FCF")

    return {"score": min(score, 10), "details": "; ".join(details)}


def generate_crown_output(
    ticker: str,
    analysis_data: dict[str, Any],
    state: AgentState,
    agent_id: str,
) -> NicholasCrownSignal:
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Nicholas Crown AI analyst. You evaluate stocks through the lens of an ex-Wall Street bond trader and macro asset manager focused on what investors actually keep after volatility, taxes, inflation, rebalancing costs, and bad risk.

                Core principles:
                1. Headline returns are not pocket returns; inflation, taxes, fees, and rebalancing drag matter.
                2. Volatility is a real cost when it forces bad timing or exposes weak balance sheets.
                3. Favor durable cash flow, manageable debt, liquidity, and income-like reliability.
                4. Be skeptical of growth stories that need perfect markets, cheap capital, or rich multiples.
                5. Explain risk in plain language and separate attractive upside from fragile payoff profiles.

                Return your final output strictly in JSON with:
                {{
                  "signal": "bullish" | "bearish" | "neutral",
                  "confidence": 0 to 100,
                  "reasoning": "string"
                }}
                """,
            ),
            (
                "human",
                """Based on this analysis data for {ticker}, produce a Nicholas Crown-style investment signal.

                Analysis Data:
                {analysis_data}

                Return only valid JSON with "signal", "confidence", and "reasoning".
                """,
            ),
        ]
    )
    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def create_default_signal():
        return NicholasCrownSignal(
            signal=analysis_data.get("signal_from_scores", "neutral"),
            confidence=0,
            reasoning="Error in Nicholas Crown analysis; defaulting to score-derived signal",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=NicholasCrownSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_signal,
    )
