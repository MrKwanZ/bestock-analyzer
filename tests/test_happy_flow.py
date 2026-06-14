"""End-to-end happy-flow tests using fully mocked providers.

These tests verify that the complete LangGraph pipeline produces a successful
RunSummary when every node receives valid, mocked data. No real API calls are
made — all providers are patched at the node boundary.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bestock_agent.graph import app as agent_app
from bestock_agent.providers.financial_base import RateLimitError
from bestock_agent.schemas import (
    ChartArtifact,
    ChartType,
    IndexBar,
    IndexComparison,
    IndexData,
    PriceBar,
    SentimentLabel,
    SentimentResult,
    TopGainer,
)
from bestock_agent.state import initial_state


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _gainer() -> TopGainer:
    return TopGainer(
        symbol="NVDA",
        name="NVIDIA Corporation",
        price=1200.0,
        change=50.0,
        change_pct=4.3,
        volume=15_000_000,
        rvol=3.2,
    )


def _bars(n: int = 5) -> list[PriceBar]:
    base = 1100.0
    return [
        PriceBar(
            date=date(2026, 6, i + 3),
            open=base + i * 10,
            high=base + i * 10 + 5,
            low=base + i * 10 - 5,
            close=base + i * 10,
            volume=10_000_000 + i * 500_000,
        )
        for i in range(n)
    ]


def _sentiment() -> SentimentResult:
    return SentimentResult(
        symbol="NVDA",
        score=0.6,
        label=SentimentLabel.POSITIVE,
        confidence=0.8,
        summary="Strong bullish momentum driven by AI chip demand.",
    )


def _comparison() -> IndexComparison:
    return IndexComparison(
        stock_symbol="NVDA",
        sp500=IndexData(period_change_pct=1.5),
        relative_perf_vs_sp500=2.8,
        beta=1.3,
    )


def _chart_artifact() -> ChartArtifact:
    return ChartArtifact(
        chart_type=ChartType.PRICE_TREND,
        path="outputs/charts/nvda_price.png",
        title="NVDA Price Trend",
    )


# ── Happy flow: basic (no advanced analysis) ──────────────────────────────────

@pytest.mark.asyncio
async def test_happy_flow_basic_no_advanced():
    """Full graph with no advanced analysis — should reach END with success."""
    state = initial_state(
        recipient_email="test@example.com",
        target_date=str(date.today()),
        lookback_days=5,
        advanced_analysis_enabled=False,
    )
    state["active_financial_provider"] = "alphavantage"

    mock_fin = AsyncMock()
    mock_fin.get_top_nasdaq_gainer.return_value = _gainer()
    mock_fin.get_price_history.return_value = _bars()

    mock_chart = _chart_artifact()

    with (
        patch("bestock_agent.nodes.fetch_top_gainer.get_financial_provider", return_value=mock_fin),
        patch("bestock_agent.nodes.fetch_price_history.get_financial_provider", return_value=mock_fin),
        patch("bestock_agent.nodes.build_charts.generate_price_chart", return_value=mock_chart),
        patch("bestock_agent.nodes.build_charts.generate_change_chart", return_value=_chart_artifact()),
        patch("bestock_agent.nodes.build_charts.generate_index_comparison_chart", return_value=_chart_artifact()),
        patch("bestock_agent.nodes.build_charts.generate_sentiment_chart", return_value=_chart_artifact()),
        patch("bestock_agent.nodes.send_email.send_report_email", new_callable=AsyncMock),
    ):
        final = await agent_app.ainvoke(state)

    run_summary = final.get("run_summary")
    assert run_summary is not None, "run_summary should be set after successful run"
    assert run_summary.success is True
    assert final.get("top_gainer") is not None
    assert final.get("trend_analysis") is not None
    assert final.get("email_payload") is not None
    assert run_summary.stock_symbol == "NVDA"


# ── Happy flow: advanced analysis enabled ─────────────────────────────────────

@pytest.mark.asyncio
async def test_happy_flow_advanced_analysis():
    """Full graph with advanced analysis — sentiment + index comparison included."""
    state = initial_state(
        recipient_email="test@example.com",
        target_date=str(date.today()),
        lookback_days=5,
        advanced_analysis_enabled=True,
        enable_volatility=True,
    )
    state["active_financial_provider"] = "alphavantage"

    mock_fin = AsyncMock()
    mock_fin.get_top_nasdaq_gainer.return_value = _gainer()
    mock_fin.get_price_history.return_value = _bars()

    with (
        patch("bestock_agent.nodes.fetch_top_gainer.get_financial_provider", return_value=mock_fin),
        patch("bestock_agent.nodes.fetch_price_history.get_financial_provider", return_value=mock_fin),
        patch(
            "bestock_agent.nodes.fetch_news_and_sentiment.get_news_provider",
            return_value=AsyncMock(search_news=AsyncMock(return_value=[
                {"title": "NVIDIA AI chip demand surges", "snippet": "Strong Q2"}
            ])),
        ),
        patch(
            "bestock_agent.nodes.fetch_news_and_sentiment.classify_sentiment",
            return_value=_sentiment(),
        ),
        patch(
            "bestock_agent.nodes.compare_with_indices._fetch_index_bars_sync",
            return_value=[
                IndexBar(date=date(2026, 6, i + 3), close=5000.0 + i * 10, change_pct=0.2 * i)
                for i in range(6)
            ],
        ),
        patch("bestock_agent.nodes.build_charts.generate_price_chart", return_value=_chart_artifact()),
        patch("bestock_agent.nodes.build_charts.generate_change_chart", return_value=_chart_artifact()),
        patch("bestock_agent.nodes.build_charts.generate_index_comparison_chart", return_value=_chart_artifact()),
        patch("bestock_agent.nodes.build_charts.generate_sentiment_chart", return_value=_chart_artifact()),
        patch("bestock_agent.nodes.send_email.send_report_email", new_callable=AsyncMock),
    ):
        final = await agent_app.ainvoke(state)

    run_summary = final.get("run_summary")
    assert run_summary is not None
    assert run_summary.success is True
    assert final.get("sentiment_result") is not None


# ── Happy flow: email delivery confirmed ──────────────────────────────────────

@pytest.mark.asyncio
async def test_happy_flow_email_sent_flag():
    """Email send should be reflected in run_summary.email_sent."""
    state = initial_state(
        recipient_email="recipient@example.com",
        target_date=str(date.today()),
        lookback_days=5,
    )
    state["active_financial_provider"] = "alphavantage"

    mock_fin = AsyncMock()
    mock_fin.get_top_nasdaq_gainer.return_value = _gainer()
    mock_fin.get_price_history.return_value = _bars()

    with (
        patch("bestock_agent.nodes.fetch_top_gainer.get_financial_provider", return_value=mock_fin),
        patch("bestock_agent.nodes.fetch_price_history.get_financial_provider", return_value=mock_fin),
        patch("bestock_agent.nodes.build_charts.generate_price_chart", return_value=_chart_artifact()),
        patch("bestock_agent.nodes.build_charts.generate_change_chart", return_value=_chart_artifact()),
        patch("bestock_agent.nodes.build_charts.generate_index_comparison_chart", return_value=_chart_artifact()),
        patch("bestock_agent.nodes.build_charts.generate_sentiment_chart", return_value=_chart_artifact()),
        patch("bestock_agent.nodes.send_email.send_report_email", new_callable=AsyncMock),
        # Ensure SMTP credentials look real so the send path executes
        patch("bestock_agent.nodes.send_email.get_settings", return_value=MagicMock(
            smtp_user="bestockagent@gmail.com",
            smtp_password="real_password",
        )),
    ):
        final = await agent_app.ainvoke(state)

    assert final["run_summary"].email_sent is True
    assert final["run_summary"].email_recipient == "recipient@example.com"


# ── Fallback: alphavantage → yfinance ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_fallback_alphavantage_to_yfinance():
    """When alphavantage fails, the graph should retry with yfinance."""
    state = initial_state(
        recipient_email="test@example.com",
        target_date=str(date.today()),
        lookback_days=5,
    )
    state["active_financial_provider"] = "alphavantage"

    call_count = {"n": 0}

    async def _side_effect_gainer():
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("alphavantage API unavailable")
        return _gainer()

    mock_provider = AsyncMock()
    mock_provider.get_top_nasdaq_gainer.side_effect = _side_effect_gainer
    mock_provider.get_price_history.return_value = _bars()

    with (
        patch("bestock_agent.nodes.fetch_top_gainer.get_financial_provider", return_value=mock_provider),
        patch("bestock_agent.nodes.fetch_price_history.get_financial_provider", return_value=mock_provider),
        patch("bestock_agent.nodes.build_charts.generate_price_chart", return_value=_chart_artifact()),
        patch("bestock_agent.nodes.build_charts.generate_change_chart", return_value=_chart_artifact()),
        patch("bestock_agent.nodes.build_charts.generate_index_comparison_chart", return_value=_chart_artifact()),
        patch("bestock_agent.nodes.build_charts.generate_sentiment_chart", return_value=_chart_artifact()),
        patch("bestock_agent.nodes.send_email.send_report_email", new_callable=AsyncMock),
    ):
        final = await agent_app.ainvoke(state)

    # Provider should have switched and the run should ultimately succeed
    assert final.get("top_gainer") is not None
    # The retry path means fetch_top_gainer was called more than once
    assert call_count["n"] >= 2


@pytest.mark.asyncio
async def test_rate_limit_backoff_retries_same_provider_then_succeeds():
    """A first rate limit should wait, retry the same provider, and continue."""
    state = initial_state(
        recipient_email="test@example.com",
        target_date=str(date.today()),
        lookback_days=5,
    )
    state["active_financial_provider"] = "alphavantage"

    call_count = {"n": 0}

    async def _side_effect_gainer():
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RateLimitError("rate limited")
        return _gainer()

    mock_provider = AsyncMock()
    mock_provider.get_top_nasdaq_gainer.side_effect = _side_effect_gainer
    mock_provider.get_price_history.return_value = _bars()

    with (
        patch("bestock_agent.nodes.fetch_top_gainer.get_financial_provider", return_value=mock_provider),
        patch("bestock_agent.nodes.fetch_price_history.get_financial_provider", return_value=mock_provider),
        patch("bestock_agent.nodes.error_handler.wait_for_rate_limit_backoff", new_callable=AsyncMock) as wait,
        patch("bestock_agent.nodes.build_charts.generate_price_chart", return_value=_chart_artifact()),
        patch("bestock_agent.nodes.build_charts.generate_change_chart", return_value=_chart_artifact()),
        patch("bestock_agent.nodes.build_charts.generate_index_comparison_chart", return_value=_chart_artifact()),
        patch("bestock_agent.nodes.build_charts.generate_sentiment_chart", return_value=_chart_artifact()),
        patch("bestock_agent.nodes.send_email.send_report_email", new_callable=AsyncMock),
    ):
        final = await agent_app.ainvoke(state)

    wait.assert_awaited_once()
    assert final.get("top_gainer") is not None
    assert final.get("active_financial_provider") == "alphavantage"
    assert call_count["n"] == 2


# ── Error handling: unrecoverable failure → failure summary ───────────────────

@pytest.mark.asyncio
async def test_unrecoverable_error_produces_failure_summary():
    """A validation error (non-recoverable) should produce a failed RunSummary."""
    from bestock_agent.services.validation import ValidationError  # noqa: F401

    state = initial_state(
        recipient_email="test@example.com",
        target_date=str(date.today()),
        lookback_days=5,
    )

    absurd_gainer = TopGainer(
        symbol="BAD", name="Bad Corp", price=1.0, change=-10000, change_pct=-9999.0
    )
    mock_fin = AsyncMock()
    mock_fin.get_top_nasdaq_gainer.return_value = absurd_gainer
    mock_fin.get_price_history.return_value = _bars()

    with (
        patch("bestock_agent.nodes.fetch_top_gainer.get_financial_provider", return_value=mock_fin),
        patch("bestock_agent.nodes.fetch_price_history.get_financial_provider", return_value=mock_fin),
    ):
        final = await agent_app.ainvoke(state)

    run_summary = final.get("run_summary")
    assert run_summary is not None
    assert run_summary.success is False


# ── Validation: UI input helpers ──────────────────────────────────────────────

def test_validate_inputs_rejects_empty_email():
    from bestock_agent.app import _validate_inputs
    err = _validate_inputs("", False, 5)
    assert err != ""
    assert "email" in err.lower() or "recipient" in err.lower()


def test_validate_inputs_rejects_invalid_email():
    from bestock_agent.app import _validate_inputs
    err = _validate_inputs("not-an-email", False, 5)
    assert "invalid" in err.lower() or "email" in err.lower()


def test_validate_inputs_accepts_valid_email():
    from bestock_agent.app import _validate_inputs
    err = _validate_inputs("user@example.com", False, 5)
    assert err == ""


def test_validate_inputs_rejects_missing_lookback_when_advanced():
    from bestock_agent.app import _validate_inputs
    err = _validate_inputs("user@example.com", True, None)
    assert err != ""
    assert "lookback" in err.lower()


def test_validate_inputs_accepts_valid_lookback_when_advanced():
    from bestock_agent.app import _validate_inputs
    err = _validate_inputs("user@example.com", True, 10)
    assert err == ""


def test_validate_inputs_rejects_lookback_out_of_range():
    from bestock_agent.app import _validate_inputs
    err = _validate_inputs("user@example.com", True, 50)
    assert err != ""
    assert "lookback" in err.lower()
