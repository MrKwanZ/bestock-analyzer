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

STATE_SCHEMA_VERSION = 1


class BestockState(TypedDict):
    # ── User inputs ───────────────────────────────────────────────────────────
    target_date: str                  # ISO-8601, e.g. "2026-06-11"
    lookback_days: int                # past trading days to analyse (3–30)
    recipient_email: str              # email address for the report
    advanced_analysis_enabled: bool   # enables sentiment / index / volatility nodes
    enable_volatility: bool           # include volatility metrics in the email report
    skip_email: bool                  # when True, analysis completes without sending email
    state_schema_version: int         # checkpoint schema version for migration safety

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
    email_payload: EmailPayload | None

    # ── Flow control ──────────────────────────────────────────────────────────
    errors: Annotated[list[AgentError], add]
    retry_count: int
    rate_limit_backoff_used: bool      # True after retrying same provider for a rate-limit error
    active_financial_provider: str    # "alphavantage" | "yfinance"
    active_news_provider: str         # "brave"   | "serpapi"

    # ── Final ─────────────────────────────────────────────────────────────────
    run_summary: RunSummary | None


def initial_state(
    *,
    recipient_email: str,
    target_date: str,
    lookback_days: int = 5,
    advanced_analysis_enabled: bool = False,
    enable_volatility: bool = False,
    skip_email: bool = False,
) -> BestockState:
    """Return a fully-initialised state dict for graph invocation."""
    return BestockState(
        target_date=target_date,
        lookback_days=lookback_days,
        recipient_email=recipient_email,
        advanced_analysis_enabled=advanced_analysis_enabled,
        enable_volatility=enable_volatility,
        skip_email=skip_email,
        state_schema_version=STATE_SCHEMA_VERSION,
        top_gainer=None,
        price_history=[],
        trend_analysis=None,
        sentiment_result=None,
        index_comparison=None,
        chart_artifacts=[],
        email_payload=None,
        errors=[],
        retry_count=0,
        rate_limit_backoff_used=False,
        active_financial_provider="alphavantage",
        active_news_provider="brave",
        run_summary=None,
    )
