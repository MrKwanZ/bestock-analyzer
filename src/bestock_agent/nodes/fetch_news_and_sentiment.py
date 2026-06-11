"""Node: fetch_news_and_sentiment.

Searches for recent news about the selected stock using the active news provider,
then runs the sentiment chain to produce a structured SentimentResult.
Implemented in Phase 5.
"""

from bestock_agent.state import BestockState


async def fetch_news_and_sentiment(state: BestockState) -> dict:
    """Fetch news headlines and classify sentiment; write SentimentResult to state."""
    raise NotImplementedError("Implemented in Phase 5")
