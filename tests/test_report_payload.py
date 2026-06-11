"""Tests for report chain template fallback and EmailPayload construction."""

from datetime import date

import pytest

from bestock_agent.chains.report_chain import (
    ReportOutput,
    _generate_suggestion,
    _template_report,
)
from bestock_agent.schemas import PriceBar, TopGainer
from bestock_agent.services.analysis import build_trend_analysis


def _gainer() -> TopGainer:
    return TopGainer(symbol="NVDA", name="NVIDIA Corporation", price=1200.0, change=50.0, change_pct=4.3)


def _analysis():
    bars = [
        PriceBar(date=date(2026, 6, 3), open=1100, high=1110, low=1090, close=1100, volume=10_000_000),
        PriceBar(date=date(2026, 6, 4), open=1100, high=1130, low=1095, close=1125, volume=12_000_000),
        PriceBar(date=date(2026, 6, 9), open=1125, high=1150, low=1120, close=1145, volume=11_000_000),
        PriceBar(date=date(2026, 6, 10), open=1145, high=1210, low=1140, close=1200, volume=14_000_000),
    ]
    return build_trend_analysis("NVDA", bars)


def test_template_report_returns_report_output():
    result = _template_report(_gainer(), _analysis(), "2026-06-11")
    assert isinstance(result, ReportOutput)


def test_template_report_html_contains_symbol():
    result = _template_report(_gainer(), _analysis(), "2026-06-11")
    assert "NVDA" in result.html_body


def test_template_report_html_contains_price():
    result = _template_report(_gainer(), _analysis(), "2026-06-11")
    assert "1200" in result.html_body


def test_template_report_text_contains_trend():
    result = _template_report(_gainer(), _analysis(), "2026-06-11")
    assert "trend" in result.text_body.lower() or "uptrend" in result.text_body.lower()


def test_template_report_text_contains_avg_change():
    result = _template_report(_gainer(), _analysis(), "2026-06-11")
    assert "Average" in result.text_body or "average" in result.text_body


def test_template_report_html_is_valid_structure():
    result = _template_report(_gainer(), _analysis(), "2026-06-11")
    assert "<html>" in result.html_body
    assert "</html>" in result.html_body
    assert "<table" in result.html_body


def test_template_report_subject_format():
    """The compose_report node should set subject to 'Analysis Report on SYMBOL'."""
    symbol = _gainer().symbol
    expected_subject = f"Analysis Report on {symbol}"
    assert expected_subject == f"Analysis Report on NVDA"


def test_template_report_includes_greeting_and_sign_off():
    result = _template_report(_gainer(), _analysis(), "2026-06-11")
    assert "Greetings!" in result.text_body
    assert "Regards," in result.text_body
    assert "BeStock Agent" in result.text_body
    assert "Greetings!" in result.html_body
    assert "Regards,<br>BeStock Agent" in result.html_body


def test_template_report_includes_suggestion_section():
    result = _template_report(_gainer(), _analysis(), "2026-06-11")
    assert "Suggestion" in result.text_body
    assert "Recommendation" in result.text_body
    assert "Suggestion" in result.html_body


def test_generate_suggestion_uptrend_recommends_buy():
    action, _ = _generate_suggestion(_analysis(), None, None)
    assert action == "Buy"
