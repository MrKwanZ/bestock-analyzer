"""Finnhub financial data provider.

Uses the official finnhub-python SDK. All SDK calls are synchronous and are
wrapped with asyncio.to_thread so they don't block the event loop.

Free-tier limitation: Finnhub does not expose a top-gainer screener on the
free plan. We work around this by concurrently quoting a curated watchlist of
major NASDAQ stocks and ranking them by intraday % change.
"""

import asyncio
from datetime import date, timedelta

import finnhub

from bestock_agent.providers.financial_base import (
    FinancialProviderError,
    RateLimitError,
    FinancialProvider,
)
from bestock_agent.schemas import PriceBar, TopGainer

# Top-50 NASDAQ stocks by market cap — used for the free-tier top-gainer search
NASDAQ_WATCHLIST: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "AMD",
    "INTC", "COST", "NFLX", "TMUS", "CSCO", "QCOM", "INTU", "CMCSA", "AMGN",
    "AMAT", "ISRG", "VRTX", "ADI", "MU", "PANW", "LRCX", "KLAC", "SNPS",
    "CDNS", "REGN", "MELI", "MDLZ", "FTNT", "MAR", "ADBE", "CRWD", "DXCM",
    "MNST", "WDAY", "ORLY", "PCAR", "NXPI", "MRVL", "SMCI", "APP", "CEG",
    "ON", "IDXX", "BIIB", "ILMN", "TTWO",
]

# Concurrent quote requests — stay inside Finnhub's 60 req/min free limit
_CONCURRENCY = 10


class FinnhubProvider(FinancialProvider):
    def __init__(self, api_key: str) -> None:
        self._client = finnhub.Client(api_key=api_key)

    @property
    def name(self) -> str:
        return "finnhub"

    async def _quote(self, symbol: str, semaphore: asyncio.Semaphore) -> dict | None:
        async with semaphore:
            try:
                data = await asyncio.to_thread(self._client.quote, symbol)
                if data and data.get("c", 0) > 0:
                    return {"symbol": symbol, **data}
            except Exception:
                pass
        return None

    async def get_top_nasdaq_gainer(self) -> TopGainer:
        sem = asyncio.Semaphore(_CONCURRENCY)
        tasks = [self._quote(sym, sem) for sym in NASDAQ_WATCHLIST]
        results = await asyncio.gather(*tasks)

        valid = [r for r in results if r is not None and r.get("dp") is not None]
        if not valid:
            raise FinancialProviderError("Finnhub returned no valid quotes")

        best = max(valid, key=lambda r: r["dp"])
        symbol = best["symbol"]

        # Fetch company profile for the full name
        try:
            profile = await asyncio.to_thread(self._client.company_profile2, symbol=symbol)
            name = profile.get("name", symbol)
        except Exception:
            name = symbol

        return TopGainer(
            symbol=symbol,
            name=name,
            price=round(float(best["c"]), 4),
            change=round(float(best["d"]), 4),
            change_pct=round(float(best["dp"]), 4),
        )

    async def get_price_history(self, symbol: str, lookback_days: int) -> list[PriceBar]:
        # Use tomorrow as the upper bound so the most recently completed NYSE
        # session is included when the caller is in a timezone ahead of UTC
        # (e.g. UTC+8 midnight = 16:00 UTC, before NYSE's 20:00 UTC close).
        to_dt = date.today() + timedelta(days=1)
        from_dt = to_dt - timedelta(days=lookback_days * 3)
        to_ts = int(to_dt.strftime("%s") if hasattr(to_dt, "strftime") else
                    (to_dt - date(1970, 1, 1)).total_seconds())
        from_ts = int((from_dt - date(1970, 1, 1)).total_seconds())

        try:
            candles = await asyncio.to_thread(
                self._client.stock_candles, symbol, "D", from_ts, to_ts
            )
        except Exception as exc:
            raise FinancialProviderError(f"Finnhub candle request failed: {exc}") from exc

        if not candles or candles.get("s") != "ok":
            raise FinancialProviderError(
                f"Finnhub returned no candle data for {symbol}: {candles}"
            )

        closes = candles["c"]
        highs = candles["h"]
        lows = candles["l"]
        opens = candles["o"]
        volumes = candles["v"]
        timestamps = candles["t"]

        bars: list[PriceBar] = []
        for i, ts in enumerate(timestamps):
            bar_date = date.fromtimestamp(ts)
            bars.append(
                PriceBar(
                    date=bar_date,
                    open=round(float(opens[i]), 4),
                    high=round(float(highs[i]), 4),
                    low=round(float(lows[i]), 4),
                    close=round(float(closes[i]), 4),
                    volume=int(volumes[i]),
                )
            )

        # Return only the last N trading days
        return bars[-lookback_days:] if len(bars) >= lookback_days else bars
