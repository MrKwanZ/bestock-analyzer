"""Tests for services/validation.py — all validation rules."""

from datetime import date, timedelta

import pytest

from bestock_agent.schemas import PriceBar, TopGainer
from bestock_agent.services.validation import (
    ValidationError,
    validate_email,
    validate_lookback_days,
    validate_price_bars,
    validate_target_date,
    validate_top_gainer,
)


# ── validate_lookback_days ────────────────────────────────────────────────────


def test_lookback_valid_bounds():
    validate_lookback_days(3)
    validate_lookback_days(15)
    validate_lookback_days(30)


def test_lookback_too_small():
    with pytest.raises(ValidationError) as exc:
        validate_lookback_days(2)
    assert "3" in exc.value.message and "30" in exc.value.message


def test_lookback_too_large():
    with pytest.raises(ValidationError) as exc:
        validate_lookback_days(31)
    assert "3" in exc.value.message and "30" in exc.value.message


def test_lookback_not_int():
    with pytest.raises(ValidationError) as exc:
        validate_lookback_days("five")   # type: ignore[arg-type]
    assert "integer" in exc.value.message


# ── validate_email ────────────────────────────────────────────────────────────


def test_email_valid():
    validate_email("user@example.com")
    validate_email("neil+filter@sub.domain.io")


def test_email_missing_at():
    with pytest.raises(ValidationError) as exc:
        validate_email("notanemail")
    assert "valid e-mail" in exc.value.message


def test_email_empty():
    with pytest.raises(ValidationError) as exc:
        validate_email("")
    assert "non-empty" in exc.value.message


def test_email_placeholder():
    with pytest.raises(ValidationError):
        validate_email("<YOUR-EMAIL-ADDRESS>")


# ── validate_target_date ──────────────────────────────────────────────────────


def test_date_valid_today():
    validate_target_date(str(date.today()))


def test_date_valid_past():
    validate_target_date("2026-01-15")


def test_date_future():
    future = str(date.today() + timedelta(days=1))
    with pytest.raises(ValidationError) as exc:
        validate_target_date(future)
    assert "future" in exc.value.message


def test_date_bad_format():
    with pytest.raises(ValidationError) as exc:
        validate_target_date("15/06/2026")
    assert "ISO-8601" in exc.value.message


def test_date_nonsense():
    with pytest.raises(ValidationError):
        validate_target_date("not-a-date")


# ── validate_top_gainer ───────────────────────────────────────────────────────


def _make_gainer(**kwargs) -> TopGainer:
    defaults = dict(symbol="NVDA", name="NVIDIA Corporation", price=1200.0, change=50.0, change_pct=4.3)
    return TopGainer(**{**defaults, **kwargs})


def test_top_gainer_valid():
    validate_top_gainer(_make_gainer())


def test_top_gainer_empty_symbol():
    with pytest.raises(ValidationError) as exc:
        validate_top_gainer(_make_gainer(symbol=""))
    assert "symbol" in exc.value.field


def test_top_gainer_zero_price():
    # Pydantic's Field(gt=0) rejects price=0 at construction — this is the
    # first line of defence. Both our ValidationError and pydantic's are acceptable.
    from pydantic import ValidationError as PydanticValidationError
    with pytest.raises((ValidationError, PydanticValidationError)):
        validate_top_gainer(_make_gainer(price=0.0))


def test_top_gainer_absurd_change():
    with pytest.raises(ValidationError) as exc:
        validate_top_gainer(_make_gainer(change_pct=99999.0))
    assert "change_pct" in exc.value.field


# ── validate_price_bars ───────────────────────────────────────────────────────


def _bar(d: str, close: float = 100.0, volume: int = 1_000_000) -> PriceBar:
    return PriceBar(date=date.fromisoformat(d), open=close, high=close + 1, low=close - 1, close=close, volume=volume)


def test_bars_valid():
    bars = [_bar("2026-06-09"), _bar("2026-06-10")]
    validate_price_bars(bars, "TEST")


def test_bars_empty():
    with pytest.raises(ValidationError) as exc:
        validate_price_bars([], "AAPL")
    assert "No price bars" in exc.value.message


def test_bars_insufficient():
    with pytest.raises(ValidationError) as exc:
        validate_price_bars([_bar("2026-06-10")], "AAPL", min_bars=2)
    assert "at least 2" in exc.value.message


def test_bars_zero_close():
    # Pydantic's Field(gt=0) rejects close=0 at construction — first line of
    # defence. Verify the system rejects it, regardless of which layer fires.
    from pydantic import ValidationError as PydanticValidationError
    yesterday = str(date.today() - timedelta(days=1))
    with pytest.raises((ValidationError, PydanticValidationError)):
        bars = [_bar(yesterday), PriceBar(
            date=date.today(), open=1, high=1, low=0.5, close=0, volume=100
        )]
        validate_price_bars(bars, "BAD")


def test_bars_high_less_than_low():
    yesterday = str(date.today() - timedelta(days=1))
    bars = [
        _bar(yesterday),
        PriceBar(date=date.today(), open=100, high=95, low=105, close=100, volume=500),
    ]
    with pytest.raises(ValidationError) as exc:
        validate_price_bars(bars, "BAD")
    assert "High" in exc.value.message or "Low" in exc.value.message


def test_bars_non_ascending_dates():
    bars = [_bar("2026-06-10"), _bar("2026-06-09")]
    with pytest.raises(ValidationError) as exc:
        validate_price_bars(bars, "BAD")
    assert "ascending" in exc.value.message


def test_bars_stale_data():
    old_date = str(date.today() - timedelta(days=30))
    older_date = str(date.today() - timedelta(days=31))
    bars = [_bar(older_date), _bar(old_date)]
    with pytest.raises(ValidationError) as exc:
        validate_price_bars(bars, "STALE")
    assert "stale" in exc.value.message.lower()
