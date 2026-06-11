"""LangGraph graph state.

All nodes read from and write to a ``BestockState`` dict.
Fields annotated with ``Annotated[list[X], add]`` use the built-in list-append
reducer so nodes can safely return partial lists without overwriting prior entries.
"""

from operator import add
from typing import Annotated, TypedDict

from bestock_agent.schemas import (
    AgentError,
    ChartArtifact,
    EmailPayload,
    IndexComparison,
    PriceBar,
    RunSummary,
    SentimentResult,
    TopGainer,
    TrendAnalysis,
)


class BestockState(TypedDict):
    # ── User inputs ───────────────────────────────────────────────────────────
    target_date: str                  # ISO-8601, e.g. "2026-06-11"
    lookback_days: int                # past trading days to analyse (3–30)
    recipient_email: str              # email address for the report
    advanced_analysis_enabled: bool   # enables sentiment / volatility / index nodes

    # ── Fetched market data ───────────────────────────────────────────────────
    top_gainer: TopGainer | None
    price_history: list[PriceBar]

    # ── Analysis results ──────────────────────────────────────────────────────
    trend_analysis: TrendAnalysis | None
    sentiment_result: SentimentResult | None
    index_comparison: IndexComparison | None

    # ── Output artefacts ─────────────────────────────────────────────────────
    # Reducers: list entries are *appended* across node calls, not replaced.
    chart_artifacts: Annotated[list[ChartArtifact], add]
    report_text: str
    email_payload: EmailPayload | None

    # ── Flow control ──────────────────────────────────────────────────────────
    errors: Annotated[list[AgentError], add]
    retry_count: int
    active_financial_provider: str    # "finnhub" | "yfinance"
    active_news_provider: str         # "brave"   | "serpapi"

    # ── Final ─────────────────────────────────────────────────────────────────
    run_summary: RunSummary | None


def initial_state(
    *,
    recipient_email: str,
    target_date: str,
    lookback_days: int = 5,
    advanced_analysis_enabled: bool = False,
) -> BestockState:
    """Return a fully-initialised state dict for graph invocation."""
    return BestockState(
        target_date=target_date,
        lookback_days=lookback_days,
        recipient_email=recipient_email,
        advanced_analysis_enabled=advanced_analysis_enabled,
        top_gainer=None,
        price_history=[],
        trend_analysis=None,
        sentiment_result=None,
        index_comparison=None,
        chart_artifacts=[],
        report_text="",
        email_payload=None,
        errors=[],
        retry_count=0,
        active_financial_provider="finnhub",
        active_news_provider="brave",
        run_summary=None,
    )
