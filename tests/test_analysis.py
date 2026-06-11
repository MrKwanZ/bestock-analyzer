"""Tests for services/analysis.py — trend calculations."""

from datetime import date

import pytest

from bestock_agent.schemas import PriceBar, TrendLabel
from bestock_agent.services.analysis import (
    build_trend_analysis,
    classify_trend,
    compute_daily_changes,
    compute_volatility,
    format_price_table,
)


def _bar(close: float, d: str = "2026-06-01") -> PriceBar:
    return PriceBar(date=date.fromisoformat(d), open=close, high=close + 1, low=close - 1, close=close, volume=1_000)


# ── compute_daily_changes ─────────────────────────────────────────────────────


def test_daily_changes_single_bar():
    assert compute_daily_changes([_bar(100.0)]) == [0.0]


def test_daily_changes_empty():
    assert compute_daily_changes([]) == []


def test_daily_changes_two_bars_up():
    bars = [_bar(100.0, "2026-06-01"), _bar(110.0, "2026-06-02")]
    changes = compute_daily_changes(bars)
    assert changes[0] == 0.0
    assert abs(changes[1] - 10.0) < 1e-4


def test_daily_changes_two_bars_down():
    bars = [_bar(200.0, "2026-06-01"), _bar(190.0, "2026-06-02")]
    changes = compute_daily_changes(bars)
    assert changes[1] < 0
    assert abs(changes[1] - (-5.0)) < 1e-4


def test_daily_changes_length_matches_bars():
    bars = [_bar(100.0 + i, f"2026-06-0{i+1}") for i in range(5)]
    changes = compute_daily_changes(bars)
    assert len(changes) == len(bars)


# ── classify_trend ────────────────────────────────────────────────────────────


def test_trend_uptrend():
    # All days rising strongly
    changes = [0.0, 1.5, 2.0, 1.8, 2.2]
    assert classify_trend(changes) == TrendLabel.UPTREND


def test_trend_downtrend():
    changes = [0.0, -1.5, -2.0, -1.8, -2.2]
    assert classify_trend(changes) == TrendLabel.DOWNTREND


def test_trend_sideways():
    changes = [0.0, 0.1, -0.1, 0.2, -0.2]
    assert classify_trend(changes) == TrendLabel.SIDEWAYS


def test_trend_pullback():
    # Strong uptrend but last day reverses
    changes = [0.0, 2.0, 1.5, 1.8, -0.5]
    assert classify_trend(changes) == TrendLabel.PULLBACK


def test_trend_single_entry_is_sideways():
    assert classify_trend([0.0]) == TrendLabel.SIDEWAYS


# ── compute_volatility ────────────────────────────────────────────────────────


def test_volatility_returns_none_for_one_bar():
    assert compute_volatility([0.0]) is None


def test_volatility_returns_float_for_multiple():
    result = compute_volatility([0.0, 1.0, -1.0, 2.0, -2.0])
    assert isinstance(result, float)
    assert result > 0


def test_volatility_zero_for_flat_series():
    result = compute_volatility([0.0, 0.0, 0.0, 0.0])
    assert result == 0.0


# ── build_trend_analysis ──────────────────────────────────────────────────────


def test_build_trend_analysis_fields():
    bars = [
        _bar(100.0, "2026-06-03"),
        _bar(102.0, "2026-06-04"),
        _bar(101.0, "2026-06-05"),
        _bar(104.0, "2026-06-08"),
        _bar(105.0, "2026-06-09"),
    ]
    result = build_trend_analysis("NVDA", bars)
    assert result.symbol == "NVDA"
    assert len(result.bars) == 5
    assert len(result.daily_changes) == 5
    assert result.daily_changes[0] == 0.0
    assert isinstance(result.avg_change, float)
    assert result.trend_label in TrendLabel
    assert result.summary != ""


def test_build_trend_analysis_annotates_bars():
    bars = [_bar(100.0, "2026-06-09"), _bar(110.0, "2026-06-10")]
    result = build_trend_analysis("TEST", bars)
    assert result.bars[0].change_pct == 0.0
    assert abs(result.bars[1].change_pct - 10.0) < 1e-3


# ── format_price_table ────────────────────────────────────────────────────────


def test_format_price_table_has_header():
    bars = [_bar(100.0, "2026-06-09"), _bar(105.0, "2026-06-10")]
    changes = [0.0, 5.0]
    table = format_price_table(bars, changes)
    assert "Date" in table
    assert "Close" in table
    assert "Change" in table


def test_format_price_table_first_row_dash():
    bars = [_bar(100.0, "2026-06-09"), _bar(105.0, "2026-06-10")]
    changes = [0.0, 5.0]
    table = format_price_table(bars, changes)
    lines = table.split("\n")
    # First data row should have "—" for change
    assert "—" in lines[2]
