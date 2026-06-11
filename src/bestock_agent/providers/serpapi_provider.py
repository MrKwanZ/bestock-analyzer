"""SerpAPI news provider — fallback when Brave Search is unavailable.

Uses the ``google-search-results`` package (already a project dependency).
All SerpAPI calls are synchronous and are wrapped with asyncio.to_thread.
"""

from __future__ import annotations

import asyncio

from bestock_agent.providers.news_base import (
    NewsArticle,
    NewsProvider,
    NewsProviderError,
    NewsRateLimitError,
)


def _sync_search(query: str, api_key: str, max_results: int) -> list[NewsArticle]:
    try:
        from serpapi import GoogleSearch  # google-search-results package
    except ImportError as exc:
        raise NewsProviderError("google-search-results package not installed") from exc

    params = {
        "engine": "google_news",
        "q": query,
        "api_key": api_key,
        "num": max_results,
        "hl": "en",
        "gl": "us",
    }
    try:
        search = GoogleSearch(params)
        data = search.get_dict()
    except Exception as exc:
        if "rate" in str(exc).lower():
            raise NewsRateLimitError(f"SerpAPI rate limit: {exc}") from exc
        raise NewsProviderError(f"SerpAPI request failed: {exc}") from exc

    articles: list[NewsArticle] = []
    for item in data.get("news_results", []):
        source = item.get("source", {})
        articles.append(
            NewsArticle(
                title=item.get("title", ""),
                source=source.get("name", "") if isinstance(source, dict) else str(source),
                published_at=item.get("date", ""),
                snippet=item.get("snippet", ""),
                url=item.get("link", ""),
            )
        )
    return articles


class SerpAPIProvider(NewsProvider):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "serpapi"

    async def search_news(self, query: str, max_results: int = 10) -> list[NewsArticle]:
        return await asyncio.to_thread(_sync_search, query, self._api_key, max_results)
