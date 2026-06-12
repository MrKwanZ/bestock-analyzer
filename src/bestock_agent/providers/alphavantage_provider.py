"""Alpha Vantage financial data provider — primary source.

Uses the Alpha Vantage REST API via httpx (async).

Free-tier limits: 25 requests / day, ~5 requests / minute.
Typical usage per agent run (≤5 requests):
  1. TOP_GAINERS_LOSERS  → broad US market; filters applied locally
  2. TIME_SERIES_DAILY   → 20-day volume history for RVOL check (per candidate, max 3)
  3. OVERVIEW            → company display name
  4. TIME_SERIES_DAILY   → OHLCV price history (for the chosen stock)

Filter criteria applied to select the top gainer:
  - Symbol length ≤ 4 characters
  - Close price ≥ $10
  - Volume ≥ 500,000
  - Percentage gain ≥ 5%
  - RVOL (today's volume / 20-day avg volume) ≥ 2
"""

from __future__ import annotations

from datetime import date

import httpx

from bestock_agent.providers.financial_base import (
    FinancialProvider,
    FinancialProviderError,
    RateLimitError,
)
from bestock_agent.schemas import PriceBar, TopGainer

_BASE_URL = "https://www.alphavantage.co/query"
_TIMEOUT = 20.0

# Filter thresholds
_MIN_PRICE   = 10.0
_MIN_VOLUME  = 500_000
_MIN_PCT     = 5.0
_MIN_RVOL    = 2.0
_MAX_SYM_LEN = 4

# Max number of candidates to attempt RVOL checks on before giving up
_MAX_RVOL_ATTEMPTS = 3


class AlphaVantageProvider(FinancialProvider):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client = httpx.AsyncClient(timeout=_TIMEOUT)

    @property
    def name(self) -> str:
        return "alphavantage"

    async def _get(self, params: dict) -> dict:
        """Make a GET request and return parsed JSON.

        Raises:
            RateLimitError: on HTTP 429 or an Alpha Vantage rate-limit body.
            FinancialProviderError: on any other HTTP / network failure.
        """
        params = {**params, "apikey": self._api_key}
        try:
            resp = await self._client.get(_BASE_URL, params=params)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                raise RateLimitError("Alpha Vantage rate limit (HTTP 429)") from exc
            raise FinancialProviderError(f"Alpha Vantage HTTP error: {exc}") from exc
        except httpx.RequestError as exc:
            raise FinancialProviderError(f"Alpha Vantage network error: {exc}") from exc

        data: dict = resp.json()

        # Alpha Vantage signals rate limits / plan violations in the response body
        if "Note" in data:
            raise RateLimitError(f"Alpha Vantage rate limit: {data['Note']}")
        if "Information" in data:
            raise RateLimitError(f"Alpha Vantage plan limit: {data['Information']}")

        return data

    async def _compute_rvol(self, symbol: str, today_volume: int) -> float | None:
        """Fetch TIME_SERIES_DAILY and return rvol = today_vol / 20-day avg vol.

        Returns None if the history is insufficient or the request fails.
        """
        try:
            data = await self._get({
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": "compact",   # last 100 trading days
            })
        except Exception:
            return None

        time_series: dict = data.get("Time Series (Daily)", {})
        if not time_series:
            return None

        # Sort dates descending; skip the most recent (may be today, partial data)
        # Use the 20 completed sessions before today for the average
        sorted_dates = sorted(time_series.keys(), reverse=True)
        prev_volumes: list[float] = []
        for d in sorted_dates[:21]:   # up to 21 bars; skip index 0 (today) below
            row = time_series[d]
            try:
                prev_volumes.append(float(row["5. volume"]))
            except (KeyError, ValueError):
                continue

        # prev_volumes[0] is the most recent settled bar (yesterday or today's EOD).
        # We want the 20 bars *before* today's trade as the historical baseline.
        history_vols = prev_volumes[1:21] if len(prev_volumes) > 1 else prev_volumes
        if not history_vols:
            return None

        avg_vol_20d = sum(history_vols) / len(history_vols)
        if avg_vol_20d <= 0:
            return None

        return round(today_volume / avg_vol_20d, 2)

    async def get_top_nasdaq_gainer(self) -> TopGainer:
        """Return today's top NASDAQ-market gainer passing all quality filters.

        Filters applied (in order):
          1. Symbol ≤ 4 characters
          2. Close ≥ $10
          3. Volume ≥ 500,000
          4. Percentage gain ≥ 5%
          5. RVOL (today / 20-day avg) ≥ 2  [checked via TIME_SERIES_DAILY]

        The TOP_GAINERS_LOSERS list is already sorted descending by % change, so
        the first candidate that passes all five filters is chosen.
        """
        data = await self._get({"function": "TOP_GAINERS_LOSERS"})

        gainers: list[dict] = data.get("top_gainers", [])
        if not gainers:
            raise FinancialProviderError("Alpha Vantage returned no top gainers")

        # ── Apply quick filters (no extra API calls) ───────────────────────────
        def _parse(g: dict) -> dict | None:
            sym = g.get("ticker", "").strip()
            if not sym or len(sym) > _MAX_SYM_LEN:
                return None
            try:
                price      = float(g.get("price", "0"))
                volume     = int(float(g.get("volume", "0")))
                change_pct = float(
                    g.get("change_percentage", "0%").replace("%", "").replace("+", "")
                )
                change     = float(g.get("change_amount", "0").replace("+", ""))
            except (ValueError, TypeError):
                return None

            if price < _MIN_PRICE or volume < _MIN_VOLUME or change_pct < _MIN_PCT:
                return None

            return {
                "symbol":     sym,
                "price":      price,
                "volume":     volume,
                "change_pct": change_pct,
                "change":     change,
            }

        candidates = [c for g in gainers if (c := _parse(g)) is not None]
        if not candidates:
            raise FinancialProviderError(
                f"No stock passed filters (symbol ≤{_MAX_SYM_LEN} chars, "
                f"close ≥${_MIN_PRICE}, vol ≥{_MIN_VOLUME:,}, "
                f"change ≥{_MIN_PCT}%) in today's top-gainers list "
                f"(scanned {len(gainers)} entries)"
            )

        # ── RVOL check (fetches TIME_SERIES_DAILY per candidate) ───────────────
        chosen: dict | None = None
        chosen_rvol: float | None = None

        for c in candidates[:_MAX_RVOL_ATTEMPTS]:
            rvol = await self._compute_rvol(c["symbol"], c["volume"])
            c["rvol"] = rvol
            if rvol is not None and rvol >= _MIN_RVOL and chosen is None:
                chosen = c
                chosen_rvol = rvol

        # If no candidate cleared the RVOL bar, fall back to the best % gainer
        if chosen is None:
            chosen = candidates[0]
            chosen_rvol = chosen.get("rvol")  # may be None if RVOL fetch failed

        symbol: str = chosen["symbol"]

        # ── Company display name ───────────────────────────────────────────────
        name: str = symbol
        try:
            overview = await self._get({"function": "OVERVIEW", "symbol": symbol})
            name = overview.get("Name") or symbol
        except Exception:
            pass

        return TopGainer(
            symbol=symbol,
            name=name,
            price=round(chosen["price"], 4),
            change=round(chosen["change"], 4),
            change_pct=round(chosen["change_pct"], 4),
            volume=chosen["volume"],
            rvol=chosen_rvol,
        )

    async def get_price_history(self, symbol: str, lookback_days: int) -> list[PriceBar]:
        """Return the last *lookback_days* completed daily bars via TIME_SERIES_DAILY."""
        data = await self._get({
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "compact",   # last 100 trading days — more than enough
        })

        time_series: dict = data.get("Time Series (Daily)", {})
        if not time_series:
            raise FinancialProviderError(
                f"Alpha Vantage returned no daily data for {symbol}"
            )

        # Dates are YYYY-MM-DD strings; sort ascending and keep the last N
        sorted_dates = sorted(time_series.keys())[-lookback_days:]

        bars: list[PriceBar] = []
        for date_str in sorted_dates:
            row = time_series[date_str]
            try:
                bars.append(PriceBar(
                    date=date.fromisoformat(date_str),
                    open=round(float(row["1. open"]), 4),
                    high=round(float(row["2. high"]), 4),
                    low=round(float(row["3. low"]), 4),
                    close=round(float(row["4. close"]), 4),
                    volume=int(float(row["5. volume"])),
                ))
            except (KeyError, ValueError) as exc:
                raise FinancialProviderError(
                    f"Alpha Vantage malformed bar for {symbol} on {date_str}: {exc}"
                ) from exc

        if not bars:
            raise FinancialProviderError(
                f"Alpha Vantage returned empty history for {symbol}"
            )

        return bars
