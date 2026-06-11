"""Provider factory — returns the correct FinancialProvider for a given name."""

from bestock_agent.providers.financial_base import FinancialProvider
from bestock_agent.providers.finnhub_provider import FinnhubProvider
from bestock_agent.providers.yfinance_provider import YfinanceProvider


def get_financial_provider(name: str) -> FinancialProvider:
    """Instantiate and return the named financial provider.

    Args:
        name: ``"finnhub"`` or ``"yfinance"``

    Raises:
        ValueError: if *name* is not recognised.
    """
    if name == "finnhub":
        from bestock_agent.config import get_settings
        return FinnhubProvider(api_key=get_settings().finnhub_api_key)
    if name == "yfinance":
        return YfinanceProvider()
    raise ValueError(f"Unknown financial provider: {name!r}")
