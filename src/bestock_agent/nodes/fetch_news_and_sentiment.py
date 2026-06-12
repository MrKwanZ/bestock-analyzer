"""Node: fetch_news_and_sentiment.

Searches for recent news about the selected stock using the active news
provider (Brave → SerpAPI fallback), then runs the sentiment chain to produce
a structured SentimentResult.

If the first search returns too few articles, the query is automatically
refined (via query_refinement_chain) and a second attempt is made before
falling back to a different provider.

Errors in this node are non-fatal — the graph continues without sentiment data
so that the rest of the report can still be generated.
"""

from __future__ import annotations

from datetime import datetime, timezone

from bestock_agent.chains.query_refinement_chain import refine_query
from bestock_agent.chains.sentiment_chain import classify_sentiment
from bestock_agent.config import get_settings
from bestock_agent.logging import get_logger
from bestock_agent.providers import get_news_provider
from bestock_agent.providers.news_base import (
    NewsProviderError,
    NewsRateLimitError,
)
from bestock_agent.schemas import AgentError, ErrorType, SentimentLabel, SentimentResult
from bestock_agent.services.backoff import wait_for_rate_limit_backoff
from bestock_agent.state import BestockState

log = get_logger("fetch_news_and_sentiment")

_MAX_ARTICLES = 10
_MIN_ARTICLES_FOR_GOOD_SENTIMENT = 3   # fewer than this → try a refined query
_NEWS_PROVIDERS = ["brave", "serpapi"]


def _neutral_sentiment(symbol: str, summary: str) -> SentimentResult:
    return SentimentResult(
        symbol=symbol,
        score=0.0,
        label=SentimentLabel.NEUTRAL,
        confidence=0.0,
        summary=summary,
    )


async def fetch_news_and_sentiment(state: BestockState) -> dict:
    """Fetch news headlines and classify sentiment; write SentimentResult to state."""
    gainer = state.get("top_gainer")
    if gainer is None:
        return {}

    settings = get_settings()
    symbol = gainer.symbol
    company_name = gainer.name
    original_query = f"{company_name} ({symbol}) stock news"

    articles: list = []
    provider_used: str | None = None
    errors: list[AgentError] = []
    backoff_attempted: set[str] = set()

    # ── Try each news provider in order ───────────────────────────────────────
    for provider_name in _NEWS_PROVIDERS:
        api_key = (
            settings.brave_api_key if provider_name == "brave"
            else settings.serpapi_api_key
        )
        if not api_key or api_key.startswith("<"):
            continue

        # First attempt with the original query
        query = original_query
        try:
            provider = get_news_provider(provider_name)
            articles = await provider.search_news(query, max_results=_MAX_ARTICLES)
            provider_used = provider_name
        except NewsRateLimitError as exc:
            errors.append(AgentError(
                error_type=ErrorType.RATE_LIMIT,
                message=f"News provider {provider_name!r} rate limited: {exc}",
                node="fetch_news_and_sentiment",
                timestamp=datetime.now(timezone.utc).isoformat(),
                recoverable=True,
                fallback_used=True,
            ))
            log.node_error("fetch_news_and_sentiment", "RATE_LIMIT", str(exc))
            if provider_name not in backoff_attempted:
                backoff_attempted.add(provider_name)
                await wait_for_rate_limit_backoff(
                    settings.rate_limit_backoff_seconds,
                    provider=provider_name,
                    node="fetch_news_and_sentiment",
                )
                try:
                    articles = await provider.search_news(query, max_results=_MAX_ARTICLES)
                    provider_used = provider_name
                except NewsRateLimitError as retry_exc:
                    errors.append(AgentError(
                        error_type=ErrorType.RATE_LIMIT,
                        message=f"News provider {provider_name!r} rate limited after backoff: {retry_exc}",
                        node="fetch_news_and_sentiment",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        recoverable=True,
                        fallback_used=True,
                    ))
                    log.node_error("fetch_news_and_sentiment", "RATE_LIMIT", str(retry_exc))
                    continue
                except NewsProviderError as retry_exc:
                    errors.append(AgentError(
                        error_type=ErrorType.NETWORK_ERROR,
                        message=f"News provider {provider_name!r} failed after backoff: {retry_exc}",
                        node="fetch_news_and_sentiment",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        recoverable=True,
                        fallback_used=True,
                    ))
                    log.node_error("fetch_news_and_sentiment", "NETWORK_ERROR", str(retry_exc))
                    continue
            else:
                continue
        except NewsProviderError as exc:
            errors.append(AgentError(
                error_type=ErrorType.NETWORK_ERROR,
                message=f"News provider {provider_name!r} failed: {exc}",
                node="fetch_news_and_sentiment",
                timestamp=datetime.now(timezone.utc).isoformat(),
                recoverable=True,
                fallback_used=True,
            ))
            log.node_error("fetch_news_and_sentiment", "NETWORK_ERROR", str(exc))
            continue

        # ── Query refinement if too few results ────────────────────────────────
        if len(articles) < _MIN_ARTICLES_FOR_GOOD_SENTIMENT:
            feedback = (
                f"Only {len(articles)} article(s) returned; need more specific results."
            )
            refined = refine_query(
                symbol=symbol,
                company_name=company_name,
                original_query=original_query,
                feedback=feedback,
                openai_model=settings.openai_model,
                openai_api_key=settings.openai_api_key,
            )
            if refined != original_query:
                log.query_refined(original_query, refined)
                try:
                    more = await provider.search_news(refined, max_results=_MAX_ARTICLES)
                    if len(more) > len(articles):
                        articles = more
                        query = refined
                except Exception:
                    pass  # keep original articles; refinement is best-effort

        if articles:
            log.info(
                "news_fetched",
                provider=provider_name,
                query=query[:80],
                article_count=len(articles),
            )
            break  # success — stop trying providers

    # ── No articles from any provider ─────────────────────────────────────────
    if not articles:
        sentiment = _neutral_sentiment(
            symbol,
            "News data unavailable — no valid news API key configured or all providers failed.",
        )
        result: dict = {"sentiment_result": sentiment}
        if errors:
            result["errors"] = errors
        return result

    # ── Sentiment classification ───────────────────────────────────────────────
    try:
        sentiment = classify_sentiment(
            articles=articles,
            symbol=symbol,
            openai_model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
        )
        log.info(
            "sentiment_classified",
            symbol=symbol,
            label=sentiment.label.value,
            score=sentiment.score,
            confidence=sentiment.confidence,
        )
    except Exception as exc:
        errors.append(AgentError(
            error_type=ErrorType.TOOL_ERROR,
            message=f"Sentiment chain failed: {exc}",
            node="fetch_news_and_sentiment",
            timestamp=datetime.now(timezone.utc).isoformat(),
            recoverable=False,
        ))
        log.node_error("fetch_news_and_sentiment", "TOOL_ERROR", str(exc))
        sentiment = _neutral_sentiment(
            symbol,
            f"Sentiment analysis failed ({provider_used} provider, {len(articles)} articles fetched).",
        )

    result = {"sentiment_result": sentiment}
    if errors:
        result["errors"] = errors
    return result
