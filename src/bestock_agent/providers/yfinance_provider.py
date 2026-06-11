"""yfinance financial data provider — used as fallback when Finnhub is unavailable.

All yfinance calls are synchronous and wrapped with asyncio.to_thread.
Top-gainer discovery uses concurrent per-ticker history calls so we avoid
MultiIndex column structure issues from the batch yf.download API.
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta

import yfinance as yf

from bestock_agent.providers.financial_base import (
    FinancialProvider,
    FinancialProviderError,
)
from bestock_agent.providers.finnhub_provider import NASDAQ_WATCHLIST
from bestock_agent.schemas import PriceBar, TopGainer

# Limit concurrent yfinance requests to avoid hitting rate limits
_SEMAPHORE_LIMIT = 8


def _latest_daily_bar(ticker: yf.Ticker) -> tuple[float, date] | None:
    """Return (close_price, bar_date) for the most recent completed session.

    Yahoo Finance's end-of-day bars can lag up to ~24 h after market close.
    We always cross-check with ``period="1d", interval="1d"`` which returns
    the most recently settled daily candle independently of the batch history.
    """
    try:
        h = ticker.history(period="1d", interval="1d", auto_adjust=True)
        h = h.dropna(subset=["Close"])
        if not h.empty:
            return float(h["Close"].iloc[-1]), h.index[-1].date()
    except Exception:
        pass
    return None


def _single_quote_sync(symbol: str) -> dict | None:
    """Return {symbol, close, prev_close, pct_change} for the latest trading day.

    Combines two yfinance calls to get the freshest possible data:
      1. ``period="1d", interval="1d"`` → most recent completed session close
      2. start/end history             → the session before that (prev_close)
    """
    try:
        ticker = yf.Ticker(symbol)

        # Step 1: most recent completed session
        latest = _latest_daily_bar(ticker)
        if latest is None:
            return None
        close_today, today_date = latest

        # Step 2: history up to (but not including) today_date for prev_close
        end_date = today_date                          # exclusive upper bound
        start_date = end_date - timedelta(days=10)    # 10 cal days → ≥ 5 trading days
        hist = ticker.history(start=str(start_date), end=str(end_date), auto_adjust=True)
        hist = hist.dropna(subset=["Close"])
        if hist.empty:
            return None

        close_prev = float(hist["Close"].iloc[-1])
        if close_prev == 0:
            return None

        pct = (close_today - close_prev) / close_prev * 100.0
        return {
            "symbol": symbol,
            "close": close_today,
            "prev_close": close_prev,
            "pct_change": pct,
        }
    except Exception:
        return None


async def _get_quote(symbol: str, sem: asyncio.Semaphore) -> dict | None:
    async with sem:
        return await asyncio.to_thread(_single_quote_sync, symbol)


async def _fetch_top_gainer_async() -> TopGainer:
    sem = asyncio.Semaphore(_SEMAPHORE_LIMIT)
    tasks = [_get_quote(sym, sem) for sym in NASDAQ_WATCHLIST]
    results = await asyncio.gather(*tasks)

    valid = [r for r in results if r is not None]
    if not valid:
        raise FinancialProviderError("yfinance returned no valid quotes for NASDAQ watchlist")

    best = max(valid, key=lambda r: r["pct_change"])
    symbol = best["symbol"]

    # Fetch the company display name via .info (fast_info does not carry names)
    name = symbol
    try:
        info = yf.Ticker(symbol).info
        name = info.get("longName") or info.get("shortName") or symbol
    except Exception:
        pass

    return TopGainer(
        symbol=symbol,
        name=name,
        price=round(best["close"], 4),
        change=round(best["close"] - best["prev_close"], 4),
        change_pct=round(best["pct_change"], 4),
    )


def _fetch_price_history_sync(symbol: str, lookback_days: int) -> list[PriceBar]:
    """Fetch the last *lookback_days* completed daily bars.

    Strategy:
      1. Query the standard end-of-day history with an explicit date range
         (``end = today + 1 day``) and drop any NaN rows — this covers the
         common case.
      2. Cross-check with ``period="1d", interval="1d"`` to detect sessions
         that Yahoo Finance has finalised but whose daily bar is not yet in
         the batch history endpoint (Yahoo's ~12–24 h data lag).  If a newer
         bar is found it is appended before the tail-slice.
    """
    import pandas as pd

    ticker = yf.Ticker(symbol)

    # ── Step 1: batch history ─────────────────────────────────────────────────
    end_date = date.today() + timedelta(days=1)
    start_date = end_date - timedelta(days=lookback_days * 3)
    hist = ticker.history(start=str(start_date), end=str(end_date), auto_adjust=True)

    if hist is None or hist.empty:
        raise FinancialProviderError(f"yfinance returned no history for {symbol}")

    hist = hist.dropna(subset=["Close"])

    # ── Step 2: supplement with most recent session if history lags ───────────
    try:
        recent = ticker.history(period="1d", interval="1d", auto_adjust=True)
        recent = recent.dropna(subset=["Close"])
        if not recent.empty and not hist.empty:
            recent_date = recent.index[-1].date()
            hist_last_date = hist.index[-1].date()
            if recent_date > hist_last_date:
                hist = pd.concat([hist, recent.tail(1)])
    except Exception:
        pass  # supplemental step is best-effort

    hist = hist.tail(lookback_days)

    bars: list[PriceBar] = []
    for idx, row in hist.iterrows():
        bar_date = idx.date() if hasattr(idx, "date") else date.fromisoformat(str(idx)[:10])
        bars.append(
            PriceBar(
                date=bar_date,
                open=round(float(row["Open"]), 4),
                high=round(float(row["High"]), 4),
                low=round(float(row["Low"]), 4),
                close=round(float(row["Close"]), 4),
                volume=int(row["Volume"]),
            )
        )
    return bars


class YfinanceProvider(FinancialProvider):
    @property
    def name(self) -> str:
        return "yfinance"

    async def get_top_nasdaq_gainer(self) -> TopGainer:
        return await _fetch_top_gainer_async()

    async def get_price_history(self, symbol: str, lookback_days: int) -> list[PriceBar]:
        return await asyncio.to_thread(_fetch_price_history_sync, symbol, lookback_days)
