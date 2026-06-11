"""Abstract base class for news / search providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class NewsProviderError(Exception):
    """Raised when a news provider call fails unrecoverably."""


class NewsRateLimitError(NewsProviderError):
    """Raised when the provider returns a rate-limit response."""


@dataclass
class NewsArticle:
    title: str
    source: str
    published_at: str
    snippet: str
    url: str = ""


class NewsProvider(ABC):
    """Common interface for news / search providers (Brave, SerpAPI, …)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider identifier, e.g. ``"brave"``."""

    @abstractmethod
    async def search_news(self, query: str, max_results: int = 10) -> list[NewsArticle]:
        """Return news articles matching *query*, newest first."""
