"""yfinance financial data provider — fallback when Alpha Vantage is unavailable.

All yfinance calls are synchronous and wrapped with asyncio.to_thread.
Top-gainer discovery uses yfinance's built-in Yahoo Finance screener
(yf.screen) to fetch the day's biggest gainers across the US market,
then applies the same quality filters as the primary provider.

Filter criteria:
  - Symbol length ≤ 4 characters
  - Close price ≥ $10
  - Volume ≥ 500,000
  - Percentage gain ≥ 5%
  - RVOL (today's volume / 20-day avg volume) ≥ 2
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta

import yfinance as yf

from bestock_agent.providers.financial_base import (
    FinancialProvider,
    FinancialProviderError,
)
from bestock_agent.schemas import PriceBar, TopGainer

# Filter thresholds (mirrors alphavantage_provider for consistency)
_MIN_PRICE   = 10.0
_MIN_VOLUME  = 500_000
_MIN_PCT     = 5.0       # percentage points (5.0 = 5%)
_MIN_RVOL    = 2.0
_MAX_SYM_LEN = 4

# Limit concurrent yfinance RVOL-history requests
_SEMAPHORE_LIMIT = 5
# How many top candidates to compute RVOL for
_MAX_RVOL_CANDIDATES = 10


def _screen_day_gainers_sync() -> list[dict]:
    """Call yf.screen('day_gainers') and return raw quote dicts."""
    result = yf.screen("day_gainers", count=100)
    return result.get("quotes", [])


def _compute_rvol_sync(symbol: str, today_volume: int) -> float | None:
    """Return rvol = today_vol / 20-day avg volume, or None on failure."""
    try:
        ticker = yf.Ticker(symbol)
        # Fetch ~25 trading days so we have at least 20 complete sessions
        hist = ticker.history(period="25d", interval="1d", auto_adjust=True)
        hist = hist.dropna(subset=["Volume"])
        if len(hist) < 2:
            return None
        # Exclude the last bar (today) — use the 20 sessions before it
        prev_vols = hist["Volume"].iloc[-21:-1]
        if prev_vols.empty:
            return None
        avg_vol_20d = float(prev_vols.mean())
        if avg_vol_20d <= 0:
            return None
        return round(today_volume / avg_vol_20d, 2)
    except Exception:
        return None


async def _compute_rvol(symbol: str, today_volume: int, sem: asyncio.Semaphore) -> float | None:
    async with sem:
        return await asyncio.to_thread(_compute_rvol_sync, symbol, today_volume)


async def _fetch_top_gainer_async() -> TopGainer:
    """Discover today's top qualifying gainer using the Yahoo Finance screener."""

    # ── Step 1: get day gainers from Yahoo Finance ─────────────────────────────
    try:
        quotes: list[dict] = await asyncio.to_thread(_screen_day_gainers_sync)
    except Exception as exc:
        raise FinancialProviderError(f"yfinance screener failed: {exc}") from exc

    if not quotes:
        raise FinancialProviderError("yfinance screener returned no quotes")

    # ── Step 2: quick filters (no extra requests needed) ───────────────────────
    # regularMarketChangePercent is in percentage points (5.5 → 5.5%)
    candidates: list[dict] = []
    for q in quotes:
        sym    = q.get("symbol", "")
        price  = float(q.get("regularMarketPrice", 0) or 0)
        volume = int(q.get("regularMarketVolume", 0) or 0)
        pct    = float(q.get("regularMarketChangePercent", 0) or 0)
        change = float(q.get("regularMarketChange", 0) or 0)

        if (
            sym
            and len(sym) <= _MAX_SYM_LEN
            and price  >= _MIN_PRICE
            and volume >= _MIN_VOLUME
            and pct    >= _MIN_PCT
        ):
            candidates.append({
                "symbol":     sym,
                "close":      price,
                "volume":     volume,
                "pct_change": pct,
                "change":     change,
                "name":       q.get("longName") or q.get("shortName") or sym,
            })

    if not candidates:
        raise FinancialProviderError(
            f"No stock passed filters (symbol ≤{_MAX_SYM_LEN} chars, "
            f"close ≥${_MIN_PRICE}, vol ≥{_MIN_VOLUME:,}, "
            f"change ≥{_MIN_PCT}%) from today's Yahoo Finance screener"
        )

    # Sort by % change descending; evaluate top N for RVOL
    candidates.sort(key=lambda c: c["pct_change"], reverse=True)
    top = candidates[:_MAX_RVOL_CANDIDATES]

    # ── Step 3: compute RVOL concurrently for top candidates ───────────────────
    sem = asyncio.Semaphore(_SEMAPHORE_LIMIT)
    rvol_values: list[float | None] = await asyncio.gather(
        *[_compute_rvol(c["symbol"], c["volume"], sem) for c in top]
    )

    for c, rvol in zip(top, rvol_values):
        c["rvol"] = rvol

    # ── Step 4: pick best — prefer RVOL ≥ 2, fall back to highest % change ────
    rvol_qualified = [c for c in top if c.get("rvol") is not None and c["rvol"] >= _MIN_RVOL]
    chosen = rvol_qualified[0] if rvol_qualified else top[0]

    return TopGainer(
        symbol=chosen["symbol"],
        name=chosen["name"],
        price=round(chosen["close"], 4),
        change=round(chosen["change"], 4),
        change_pct=round(chosen["pct_change"], 4),
        volume=chosen["volume"],
        rvol=chosen.get("rvol"),
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

    # ── Sanity check + scale correction ──────────────────────────────────────
    # Yahoo Finance's batch history occasionally returns stale/incorrect prices
    # (e.g. un-finalised split/dividend adjustments). Cross-check the
    # second-to-last bar (previous session) against fast_info.previous_close
    # which comes from the real-time quote feed and is reliable.
    #
    # When a >50% discrepancy is detected, apply a uniform scale-correction
    # factor (fi_prev / history_prev) to all bars *except* the last one —
    # the most-recent bar is trusted because it was either already correct in
    # the batch history or was supplemented by period="1d" above.
    if len(bars) >= 2:
        try:
            fi_prev = ticker.fast_info.previous_close
            if fi_prev and fi_prev > 0:
                history_prev = bars[-2].close
                ratio = max(history_prev, fi_prev) / max(min(history_prev, fi_prev), 0.01)
                if ratio > 1.5:
                    factor = fi_prev / history_prev
                    corrected: list[PriceBar] = []
                    for i, bar in enumerate(bars):
                        if i < len(bars) - 1:
                            corrected.append(PriceBar(
                                date=bar.date,
                                open=round(bar.open * factor, 4),
                                high=round(bar.high * factor, 4),
                                low=round(bar.low * factor, 4),
                                close=round(bar.close * factor, 4),
                                volume=bar.volume,
                            ))
                        else:
                            corrected.append(bar)
                    bars = corrected
        except Exception:
            pass  # correction is best-effort; proceed with uncorrected bars

    return bars


class YfinanceProvider(FinancialProvider):
    @property
    def name(self) -> str:
        return "yfinance"

    async def get_top_nasdaq_gainer(self) -> TopGainer:
        return await _fetch_top_gainer_async()

    async def get_price_history(self, symbol: str, lookback_days: int) -> list[PriceBar]:
        return await asyncio.to_thread(_fetch_price_history_sync, symbol, lookback_days)
