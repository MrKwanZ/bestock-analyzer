"""yfinance financial data provider — used as fallback when Finnhub is unavailable.

All yfinance calls are synchronous and wrapped with asyncio.to_thread.
Top-gainer discovery uses concurrent per-ticker history calls so we avoid
MultiIndex column structure issues from the batch yf.download API.
"""

from __future__ import annotations

import asyncio
from datetime import date

import yfinance as yf

from bestock_agent.providers.financial_base import (
    FinancialProvider,
    FinancialProviderError,
)
from bestock_agent.providers.finnhub_provider import NASDAQ_WATCHLIST
from bestock_agent.schemas import PriceBar, TopGainer

# Limit concurrent yfinance requests to avoid hitting rate limits
_SEMAPHORE_LIMIT = 8


def _single_quote_sync(symbol: str) -> dict | None:
    """Return {symbol, close, prev_close, pct_change} for the latest trading day."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d", auto_adjust=True)
        if hist is None or hist.empty:
            return None
        hist = hist.dropna(subset=["Close"])
        if len(hist) < 2:
            return None
        close_today = float(hist["Close"].iloc[-1])
        close_prev = float(hist["Close"].iloc[-2])
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
    # Request 3× calendar days to cover weekends and holidays
    period = f"{lookback_days * 3}d"
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, auto_adjust=True)

    if hist is None or hist.empty:
        raise FinancialProviderError(f"yfinance returned no history for {symbol}")

    hist = hist.dropna(subset=["Close"])
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
