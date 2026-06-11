"""Node: fetch_news_and_sentiment.

Searches for recent news about the selected stock using the active news
provider (Brave → SerpAPI fallback), then runs the sentiment chain to produce
a structured SentimentResult.

Errors in this node are non-fatal — the graph continues without sentiment data
so that the rest of the report can still be generated.
"""

from __future__ import annotations

from datetime import datetime, timezone

from bestock_agent.chains.sentiment_chain import classify_sentiment
from bestock_agent.config import get_settings
from bestock_agent.providers import get_news_provider
from bestock_agent.providers.news_base import (
    NewsProviderError,
    NewsRateLimitError,
)
from bestock_agent.schemas import AgentError, ErrorType
from bestock_agent.state import BestockState

_MAX_ARTICLES = 10
_NEWS_PROVIDERS = ["brave", "serpapi"]


async def fetch_news_and_sentiment(state: BestockState) -> dict:
    """Fetch news headlines and classify sentiment; write SentimentResult to state."""
    gainer = state.get("top_gainer")
    if gainer is None:
        return {}

    settings = get_settings()
    symbol = gainer.symbol
    company_name = gainer.name
    query = f"{company_name} ({symbol}) stock news"

    articles = []
    provider_used = None
    errors = []

    # Try each news provider in order until one succeeds
    for provider_name in _NEWS_PROVIDERS:
        api_key = (
            settings.brave_api_key if provider_name == "brave" else settings.serpapi_api_key
        )
        if not api_key or api_key.startswith("<"):
            continue  # Skip providers with placeholder keys

        try:
            provider = get_news_provider(provider_name)
            articles = await provider.search_news(query, max_results=_MAX_ARTICLES)
            provider_used = provider_name
            break
        except NewsRateLimitError as exc:
            errors.append(
                AgentError(
                    error_type=ErrorType.RATE_LIMIT,
                    message=f"News provider {provider_name!r} rate limited: {exc}",
                    node="fetch_news_and_sentiment",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    recoverable=True,
                    fallback_used=True,
                )
            )
        except NewsProviderError as exc:
            errors.append(
                AgentError(
                    error_type=ErrorType.NETWORK_ERROR,
                    message=f"News provider {provider_name!r} failed: {exc}",
                    node="fetch_news_and_sentiment",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    recoverable=True,
                    fallback_used=True,
                )
            )

    if not articles:
        # All providers failed or had placeholder keys — return a neutral placeholder
        from bestock_agent.schemas import SentimentLabel, SentimentResult
        sentiment = SentimentResult(
            symbol=symbol,
            score=0.0,
            label=SentimentLabel.NEUTRAL,
            confidence=0.0,
            summary="News data unavailable — no valid news API key configured.",
        )
        result: dict = {"sentiment_result": sentiment}
        if errors:
            result["errors"] = errors
        return result

    try:
        sentiment = classify_sentiment(
            articles=articles,
            symbol=symbol,
            openai_model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
        )
    except Exception as exc:
        errors.append(
            AgentError(
                error_type=ErrorType.TOOL_ERROR,
                message=f"Sentiment chain failed: {exc}",
                node="fetch_news_and_sentiment",
                timestamp=datetime.now(timezone.utc).isoformat(),
                recoverable=False,
            )
        )
        from bestock_agent.schemas import SentimentLabel, SentimentResult
        sentiment = SentimentResult(
            symbol=symbol,
            score=0.0,
            label=SentimentLabel.NEUTRAL,
            confidence=0.0,
            summary=f"Sentiment analysis failed ({provider_used} provider, {len(articles)} articles fetched).",
        )

    result = {"sentiment_result": sentiment}
    if errors:
        result["errors"] = errors
    return result
