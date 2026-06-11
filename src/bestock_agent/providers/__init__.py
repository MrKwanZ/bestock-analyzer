"""Provider factories — return the correct provider instance for a given name."""

from bestock_agent.providers.financial_base import FinancialProvider
from bestock_agent.providers.finnhub_provider import FinnhubProvider
from bestock_agent.providers.news_base import NewsProvider
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


def get_news_provider(name: str) -> NewsProvider:
    """Instantiate and return the named news provider.

    Args:
        name: ``"brave"`` or ``"serpapi"``

    Raises:
        ValueError: if *name* is not recognised.
    """
    from bestock_agent.config import get_settings
    settings = get_settings()

    if name == "brave":
        from bestock_agent.providers.brave_provider import BraveProvider
        return BraveProvider(api_key=settings.brave_api_key)
    if name == "serpapi":
        from bestock_agent.providers.serpapi_provider import SerpAPIProvider
        return SerpAPIProvider(api_key=settings.serpapi_api_key)
    raise ValueError(f"Unknown news provider: {name!r}")
