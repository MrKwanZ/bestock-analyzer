"""Pure-Python stock analysis calculations.

No external API calls here — takes raw PriceBar data and returns structured
TrendAnalysis. All functions are synchronous and side-effect free.
"""

import statistics

from bestock_agent.schemas import PriceBar, TrendAnalysis, TrendLabel


# ── Trend classification ──────────────────────────────────────────────────────

_UPTREND_THRESHOLD = 0.4   # avg daily change % above this → uptrend
_DOWNTREND_THRESHOLD = -0.4  # avg daily change % below this → downtrend


def compute_daily_changes(bars: list[PriceBar]) -> list[float]:
    """Return a list of daily % changes aligned to *bars*.

    The first entry is always 0.0 (no prior day to compare against).
    Subsequent entries are (close_today / close_yesterday - 1) * 100.
    """
    if not bars:
        return []
    changes: list[float] = [0.0]
    for i in range(1, len(bars)):
        prev_close = bars[i - 1].close
        curr_close = bars[i].close
        pct = ((curr_close - prev_close) / prev_close) * 100.0
        changes.append(round(pct, 4))
    return changes


def _avg_excluding_first(changes: list[float]) -> float:
    """Average of change values excluding the always-zero first entry."""
    meaningful = changes[1:] if len(changes) > 1 else changes
    return round(statistics.mean(meaningful), 4) if meaningful else 0.0


def classify_trend(daily_changes: list[float]) -> TrendLabel:
    """Classify the price trend from the list of daily change percentages."""
    if len(daily_changes) < 2:
        return TrendLabel.SIDEWAYS
    avg = _avg_excluding_first(daily_changes)
    last = daily_changes[-1]
    if avg > _UPTREND_THRESHOLD:
        # Most recent day pulled back from an otherwise bullish period
        if last < 0:
            return TrendLabel.PULLBACK
        return TrendLabel.UPTREND
    if avg < _DOWNTREND_THRESHOLD:
        return TrendLabel.DOWNTREND
    return TrendLabel.SIDEWAYS


def compute_volatility(daily_changes: list[float]) -> float | None:
    """Return annualised volatility proxy (std dev of daily % changes)."""
    meaningful = daily_changes[1:] if len(daily_changes) > 1 else daily_changes
    if len(meaningful) < 2:
        return None
    return round(statistics.stdev(meaningful), 4)


def build_trend_summary(
    avg_change: float,
    trend: TrendLabel,
    lookback: int,
    volatility: float | None = None,
) -> str:
    """Return a human-readable trend explanation, optionally including volatility."""
    direction = "gained" if avg_change >= 0 else "declined"
    vol_str = f" with a daily volatility of ±{volatility:.2f}%" if volatility else ""
    return (
        f"Over the past {lookback} trading day(s) the stock {direction} an average of "
        f"{abs(avg_change):.2f}% per day ({trend.value}){vol_str}."
    )


def build_trend_analysis(symbol: str, bars: list[PriceBar]) -> TrendAnalysis:
    """Derive all trend metrics from a list of price bars and return a TrendAnalysis."""
    daily_changes = compute_daily_changes(bars)
    avg_change = _avg_excluding_first(daily_changes)
    trend_label = classify_trend(daily_changes)
    volatility = compute_volatility(daily_changes)
    summary = build_trend_summary(avg_change, trend_label, len(bars), volatility)

    # Back-fill change_pct onto bars (mutates copies, not originals)
    annotated_bars: list[PriceBar] = []
    for i, bar in enumerate(bars):
        annotated_bars.append(bar.model_copy(update={"change_pct": daily_changes[i]}))

    return TrendAnalysis(
        symbol=symbol,
        bars=annotated_bars,
        daily_changes=daily_changes,
        avg_change=avg_change,
        trend_label=trend_label,
        volatility=volatility,
        summary=summary,
    )


def format_price_table(bars: list[PriceBar], daily_changes: list[float]) -> str:
    """Return a plain-text ASCII table of date / close / change for use in reports."""
    header = f"{'Date':<12} {'Close':>10} {'Change':>8}"
    separator = "-" * len(header)
    rows = [header, separator]
    for i, bar in enumerate(bars):
        change_str = f"{daily_changes[i]:+.2f}%" if i > 0 else "  —"
        rows.append(f"{str(bar.date):<12} ${bar.close:>9.2f} {change_str:>8}")
    return "\n".join(rows)
