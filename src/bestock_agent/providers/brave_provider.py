"""Brave Search news provider.

Uses the Brave News Search REST endpoint.  All requests are made with httpx
(already a project dependency) so no extra packages are required.
"""

from __future__ import annotations

import httpx

from bestock_agent.providers.news_base import (
    NewsArticle,
    NewsProvider,
    NewsProviderError,
    NewsRateLimitError,
)

_BRAVE_NEWS_URL = "https://api.search.brave.com/res/v1/news/search"
_TIMEOUT = 12.0


class BraveProvider(NewsProvider):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "brave"

    async def search_news(self, query: str, max_results: int = 10) -> list[NewsArticle]:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._api_key,
        }
        params = {
            "q": query,
            "count": min(max_results, 20),
            "country": "US",
            "search_lang": "en",
            "freshness": "pw",  # past week
        }
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(_BRAVE_NEWS_URL, headers=headers, params=params)
                if resp.status_code == 429:
                    raise NewsRateLimitError(f"Brave Search rate limit: {resp.text[:200]}")
                if resp.status_code == 401:
                    raise NewsProviderError("Brave Search auth failed: check BRAVE_API_KEY")
                resp.raise_for_status()
                data = resp.json()
        except NewsProviderError:
            raise
        except httpx.TimeoutException as exc:
            raise NewsProviderError(f"Brave Search timed out: {exc}") from exc
        except Exception as exc:
            raise NewsProviderError(f"Brave Search request failed: {exc}") from exc

        articles: list[NewsArticle] = []
        for item in data.get("results", []):
            articles.append(
                NewsArticle(
                    title=item.get("title", ""),
                    source=item.get("meta_url", {}).get("hostname", ""),
                    published_at=item.get("page_age", ""),
                    snippet=item.get("description", ""),
                    url=item.get("url", ""),
                )
            )
        return articles
