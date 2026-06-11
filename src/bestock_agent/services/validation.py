"""Input and data validation helpers.

All functions raise ``ValidationError`` on failure and return cleanly on success.
They are deliberately side-effect-free so they can be called from tests without
mocking any external dependencies.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from bestock_agent.schemas import PriceBar, TopGainer


# ── Custom exception ──────────────────────────────────────────────────────────


class ValidationError(ValueError):
    """Raised when a validation check fails.

    Attributes:
        field:   The field or parameter that failed validation.
        message: Human-readable explanation of why validation failed.
    """

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        self.message = message
        super().__init__(f"[{field}] {message}")


# ── User-supplied parameter validation ────────────────────────────────────────

_MIN_LOOKBACK = 3
_MAX_LOOKBACK = 30
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_lookback_days(days: int) -> None:
    """Ensure *days* is within the permitted range [3, 30]."""
    if not isinstance(days, int):
        raise ValidationError("lookback_days", f"Must be an integer, got {type(days).__name__}")
    if days < _MIN_LOOKBACK or days > _MAX_LOOKBACK:
        raise ValidationError(
            "lookback_days",
            f"Must be between {_MIN_LOOKBACK} and {_MAX_LOOKBACK}, got {days}",
        )


def validate_email(email: str) -> None:
    """Ensure *email* looks like a valid e-mail address."""
    if not isinstance(email, str) or not email.strip():
        raise ValidationError("email", "Must be a non-empty string")
    if not _EMAIL_RE.match(email.strip()):
        raise ValidationError("email", f"Not a valid e-mail address: {email!r}")


def validate_target_date(target_date: str) -> None:
    """Ensure *target_date* is an ISO-8601 date string and not in the future."""
    try:
        d = date.fromisoformat(target_date)
    except (ValueError, TypeError):
        raise ValidationError(
            "target_date",
            f"Must be an ISO-8601 date (YYYY-MM-DD), got {target_date!r}",
        )
    if d > date.today():
        raise ValidationError(
            "target_date",
            f"Target date {target_date} is in the future — no market data available yet",
        )


# ── Fetched market-data validation ────────────────────────────────────────────


def validate_top_gainer(gainer: TopGainer) -> None:
    """Ensure the TopGainer record contains plausible values."""
    if not gainer.symbol or not gainer.symbol.strip():
        raise ValidationError("top_gainer.symbol", "Symbol must not be empty")
    if gainer.price <= 0:
        raise ValidationError(
            "top_gainer.price",
            f"Price must be positive, got {gainer.price}",
        )
    if gainer.change_pct < -99 or gainer.change_pct > 10_000:
        raise ValidationError(
            "top_gainer.change_pct",
            f"Implausible change percentage: {gainer.change_pct:.2f}%",
        )


def validate_price_bars(bars: list[PriceBar], symbol: str, min_bars: int = 2) -> None:
    """Ensure there are enough valid price bars to perform analysis."""
    if not bars:
        raise ValidationError(
            "price_history",
            f"No price bars returned for {symbol} — the symbol may be delisted or the market is closed",
        )
    if len(bars) < min_bars:
        raise ValidationError(
            "price_history",
            f"Only {len(bars)} bar(s) returned for {symbol}; need at least {min_bars} to compute changes",
        )
    for i, bar in enumerate(bars):
        if bar.close <= 0:
            raise ValidationError(
                f"price_history[{i}].close",
                f"Zero or negative close price on {bar.date}: {bar.close}",
            )
        if bar.open <= 0 or bar.high <= 0 or bar.low <= 0:
            raise ValidationError(
                f"price_history[{i}]",
                f"Zero or negative OHLC value on {bar.date}",
            )
        if bar.high < bar.low:
            raise ValidationError(
                f"price_history[{i}]",
                f"High ({bar.high}) is less than Low ({bar.low}) on {bar.date}",
            )
        if bar.volume < 0:
            raise ValidationError(
                f"price_history[{i}].volume",
                f"Negative volume on {bar.date}: {bar.volume}",
            )

    # Check that dates are strictly ascending
    for i in range(1, len(bars)):
        if bars[i].date <= bars[i - 1].date:
            raise ValidationError(
                "price_history",
                f"Bars are not in ascending date order at index {i}: {bars[i-1].date} vs {bars[i].date}",
            )

    # Warn about suspiciously stale data (last bar older than 14 calendar days)
    last_date = bars[-1].date
    if (date.today() - last_date) > timedelta(days=14):
        raise ValidationError(
            "price_history",
            f"Most recent bar ({last_date}) is more than 14 days old — data may be stale",
        )
