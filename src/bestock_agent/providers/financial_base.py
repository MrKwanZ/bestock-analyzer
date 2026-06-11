"""Abstract base class for financial data providers."""

from abc import ABC, abstractmethod

from bestock_agent.schemas import PriceBar, TopGainer


class FinancialProviderError(Exception):
    """Raised when a financial provider call fails unrecoverably."""


class RateLimitError(FinancialProviderError):
    """Raised when the provider returns a rate-limit response."""


class FinancialProvider(ABC):
    """Common interface for financial data providers (Finnhub, yfinance, …)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider identifier, e.g. ``"finnhub"``."""

    @abstractmethod
    async def get_top_nasdaq_gainer(self) -> TopGainer:
        """Return the NASDAQ stock with the highest intraday percentage gain."""

    @abstractmethod
    async def get_price_history(self, symbol: str, lookback_days: int) -> list[PriceBar]:
        """Return OHLCV bars for *symbol* covering *lookback_days* trading days."""
