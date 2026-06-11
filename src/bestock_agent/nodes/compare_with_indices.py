"""Node: compare_with_indices.

Fetches S&P 500 (^GSPC) performance over the same lookback window and computes
relative performance metrics.  Only S&P 500 is compared per Phase 5 scope.

Beta is computed as: cov(r_stock, r_sp500) / var(r_sp500)
Excess return is:    stock_period_change - sp500_period_change
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone

import numpy as np
import yfinance as yf

from bestock_agent.schemas import (
    AgentError,
    ErrorType,
    IndexBar,
    IndexComparison,
    IndexData,
)
from bestock_agent.state import BestockState

_SP500_TICKER = "^GSPC"
_SP500_NAME = "S&P 500"


def _fetch_index_bars_sync(lookback_days: int) -> list[IndexBar]:
    """Download recent ^GSPC bars; returns the last *lookback_days* trading days.

    Uses the same two-step strategy as ``yfinance_provider`` to handle
    Yahoo Finance's ~12–24 h batch-history data lag.
    """
    import pandas as pd

    ticker = yf.Ticker(_SP500_TICKER)

    end_date = date.today() + timedelta(days=1)
    start_date = end_date - timedelta(days=lookback_days * 3)
    hist = ticker.history(start=str(start_date), end=str(end_date), auto_adjust=True)
    hist = hist.dropna(subset=["Close"])

    # Supplement with the most recent completed session if lagging
    try:
        recent = ticker.history(period="1d", interval="1d", auto_adjust=True)
        recent = recent.dropna(subset=["Close"])
        if not recent.empty and not hist.empty:
            recent_date = recent.index[-1].date()
            if recent_date > hist.index[-1].date():
                hist = pd.concat([hist, recent.tail(1)])
    except Exception:
        pass

    hist = hist.tail(lookback_days + 1)  # +1 so first row gives us prev_close

    bars: list[IndexBar] = []
    closes = list(hist["Close"])
    dates = list(hist.index)

    for i in range(1, len(closes)):
        prev_close = float(closes[i - 1])
        cur_close = float(closes[i])
        change_pct = (cur_close - prev_close) / prev_close * 100.0 if prev_close else 0.0
        bars.append(
            IndexBar(
                date=dates[i].date(),
                close=round(cur_close, 4),
                change_pct=round(change_pct, 4),
            )
        )

    return bars


def _period_change(bars: list[IndexBar]) -> float:
    """Return cumulative percentage change over the full bar series."""
    if len(bars) < 2:
        return bars[0].change_pct if bars else 0.0
    first_close = bars[0].close
    last_close = bars[-1].close
    return round((last_close - first_close) / first_close * 100.0, 4)


def _compute_beta(stock_changes: list[float], index_changes: list[float]) -> float | None:
    """Compute beta = cov(stock, index) / var(index).

    Returns None when there are fewer than 2 observations.
    """
    if len(stock_changes) < 2 or len(index_changes) < 2:
        return None
    n = min(len(stock_changes), len(index_changes))
    s = np.array(stock_changes[-n:], dtype=float)
    m = np.array(index_changes[-n:], dtype=float)
    var_m = float(np.var(m, ddof=1))
    if var_m == 0.0:
        return None
    cov = float(np.cov(s, m, ddof=1)[0, 1])
    return round(cov / var_m, 4)


async def compare_with_indices(state: BestockState) -> dict:
    """Fetch ^GSPC bars, compute relative metrics, write IndexComparison to state."""
    analysis = state.get("trend_analysis")
    if analysis is None:
        return {}  # No stock data — silently skip

    lookback_days = state.get("lookback_days") or len(analysis.bars)

    try:
        sp500_bars = await asyncio.to_thread(_fetch_index_bars_sync, lookback_days)
    except Exception as exc:
        error = AgentError(
            error_type=ErrorType.NETWORK_ERROR,
            message=f"compare_with_indices: failed to fetch ^GSPC data — {exc}",
            node="compare_with_indices",
            timestamp=datetime.now(timezone.utc).isoformat(),
            recoverable=False,
        )
        return {"errors": [error]}

    if not sp500_bars:
        return {}

    sp500_period_change = _period_change(sp500_bars)
    stock_period_change = (
        analysis.bars[-1].close - analysis.bars[0].close
    ) / analysis.bars[0].close * 100.0 if len(analysis.bars) >= 2 else 0.0

    relative_perf = round(stock_period_change - sp500_period_change, 4)

    stock_changes = analysis.daily_changes[1:] if len(analysis.daily_changes) > 1 else analysis.daily_changes
    index_changes = [b.change_pct for b in sp500_bars]
    beta = _compute_beta(stock_changes, index_changes)
    excess_return = round(stock_period_change - sp500_period_change, 4)

    sp500_data = IndexData(
        symbol=_SP500_TICKER,
        name=_SP500_NAME,
        bars=sp500_bars,
        period_change_pct=sp500_period_change,
    )

    comparison = IndexComparison(
        stock_symbol=analysis.symbol,
        sp500=sp500_data,
        relative_perf_vs_sp500=relative_perf,
        beta=beta,
        excess_return=excess_return,
    )
    return {"index_comparison": comparison}
