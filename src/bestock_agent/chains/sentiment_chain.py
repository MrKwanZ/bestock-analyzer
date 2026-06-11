"""LCEL chain for classifying sentiment from news articles.

Primary path : ChatOpenAI with structured output → SentimentResult.
Token guard  : snippets are truncated to 400 chars each; max 8 articles
               per batch.  If > 8 articles arrive they are chunked, each
               chunk summarised, and the summaries merged.
Fallback     : keyword-based scoring used when LLM is unavailable or the
               API key is a placeholder.
"""

from __future__ import annotations

import re
from typing import Sequence

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from bestock_agent.providers.news_base import NewsArticle
from bestock_agent.schemas import (
    SentimentDriver,
    SentimentLabel,
    SentimentResult,
)

# ── Token-limit constants ─────────────────────────────────────────────────────

_MAX_SNIPPET_CHARS = 400
_BATCH_SIZE = 8


# ── Structured LLM output ─────────────────────────────────────────────────────


class _SentimentOutput(BaseModel):
    score: float = Field(
        ge=-1.0, le=1.0,
        description="Sentiment score from -1.0 (very negative) to +1.0 (very positive)",
    )
    label: str = Field(description="One of: positive, neutral, negative")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the classification")
    summary: str = Field(description="One-sentence summary of the overall sentiment")
    positive_drivers: list[str] = Field(
        default_factory=list,
        description="Up to 3 bullish points found in the articles",
    )
    negative_drivers: list[str] = Field(
        default_factory=list,
        description="Up to 3 bearish points found in the articles",
    )


_SYSTEM = """\
You are a financial sentiment analyst. Analyse the provided news headlines and snippets
for a specific stock and return a structured sentiment assessment. Be objective and concise.
"""

_HUMAN = """\
Stock: {symbol}

News articles:
{articles_text}

Classify the overall sentiment and identify the key bullish and bearish factors.
"""


def _build_chain(model: str, api_key: str):
    llm = ChatOpenAI(model=model, api_key=api_key, temperature=0.0)
    structured = llm.with_structured_output(_SentimentOutput)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM), ("human", _HUMAN)]
    )
    return prompt | structured


def _format_articles(articles: Sequence[NewsArticle]) -> str:
    lines = []
    for i, a in enumerate(articles, 1):
        snippet = (a.snippet or "")[:_MAX_SNIPPET_CHARS]
        lines.append(f"{i}. [{a.source}] {a.title}\n   {snippet}")
    return "\n\n".join(lines)


# ── Keyword fallback ──────────────────────────────────────────────────────────

_POSITIVE = {
    "surge", "rally", "gain", "gains", "rise", "rises", "beat", "beats", "record",
    "strong", "upgrade", "buy", "outperform", "breakthrough", "bullish", "profit",
    "growth", "revenue", "exceed", "exceeds", "positive", "optimistic", "demand",
}
_NEGATIVE = {
    "drop", "drops", "fall", "falls", "decline", "declines", "down", "miss", "misses",
    "weak", "downgrade", "sell", "underperform", "loss", "losses", "risk", "concern",
    "lawsuit", "investigation", "recall", "warning", "bearish", "disappoint",
    "disappoints", "layoff", "layoffs",
}


def _keyword_sentiment(articles: Sequence[NewsArticle], symbol: str) -> SentimentResult:
    pos_hits: list[str] = []
    neg_hits: list[str] = []
    sources: list[str] = []

    for a in articles:
        text = f"{a.title} {a.snippet}".lower()
        words = set(re.findall(r"\b\w+\b", text))
        p = words & _POSITIVE
        n = words & _NEGATIVE
        if p:
            pos_hits.append(f"{a.title[:80]} ({a.source})")
        if n:
            neg_hits.append(f"{a.title[:80]} ({a.source})")
        if a.source:
            sources.append(a.source)

    total = len(pos_hits) + len(neg_hits) or 1
    raw_score = (len(pos_hits) - len(neg_hits)) / total
    score = round(max(-1.0, min(1.0, raw_score)), 3)

    if score > 0.1:
        label = SentimentLabel.POSITIVE
    elif score < -0.1:
        label = SentimentLabel.NEGATIVE
    else:
        label = SentimentLabel.NEUTRAL

    drivers = [
        SentimentDriver(text=t, sentiment=SentimentLabel.POSITIVE, source="keyword")
        for t in pos_hits[:3]
    ] + [
        SentimentDriver(text=t, sentiment=SentimentLabel.NEGATIVE, source="keyword")
        for t in neg_hits[:3]
    ]

    return SentimentResult(
        symbol=symbol,
        score=score,
        label=label,
        confidence=0.5,
        drivers=drivers,
        sources=list(set(sources))[:5],
        summary=(
            f"Keyword analysis of {len(articles)} article(s): "
            f"{len(pos_hits)} bullish signal(s), {len(neg_hits)} bearish signal(s)."
        ),
    )


def _output_to_result(out: _SentimentOutput, articles: Sequence[NewsArticle], symbol: str) -> SentimentResult:
    try:
        label = SentimentLabel(out.label.lower())
    except ValueError:
        label = SentimentLabel.NEUTRAL

    drivers = [
        SentimentDriver(text=t, sentiment=SentimentLabel.POSITIVE, source="llm")
        for t in out.positive_drivers
    ] + [
        SentimentDriver(text=t, sentiment=SentimentLabel.NEGATIVE, source="llm")
        for t in out.negative_drivers
    ]
    sources = list({a.source for a in articles if a.source})[:5]

    return SentimentResult(
        symbol=symbol,
        score=round(out.score, 3),
        label=label,
        confidence=round(out.confidence, 3),
        drivers=drivers,
        sources=sources,
        summary=out.summary,
    )


# ── Public interface ──────────────────────────────────────────────────────────


def classify_sentiment(
    articles: Sequence[NewsArticle],
    symbol: str,
    openai_model: str,
    openai_api_key: str,
) -> SentimentResult:
    """Classify sentiment from *articles* about *symbol*.

    Uses ChatOpenAI when a valid key is present; falls back to keyword scoring
    when the key is missing or a placeholder.  Articles are chunked in batches
    of ``_BATCH_SIZE`` to stay within context limits.
    """
    if not articles:
        return SentimentResult(
            symbol=symbol,
            score=0.0,
            label=SentimentLabel.NEUTRAL,
            confidence=0.0,
            summary="No news articles found for sentiment analysis.",
        )

    key_is_placeholder = not openai_api_key or openai_api_key.startswith("<")
    if key_is_placeholder:
        return _keyword_sentiment(articles, symbol)

    chain = _build_chain(openai_model, openai_api_key)

    # ── Batch processing for token safety ─────────────────────────────────────
    batches = [articles[i : i + _BATCH_SIZE] for i in range(0, len(articles), _BATCH_SIZE)]

    try:
        if len(batches) == 1:
            articles_text = _format_articles(batches[0])
            out: _SentimentOutput = chain.invoke({"symbol": symbol, "articles_text": articles_text})
            return _output_to_result(out, articles, symbol)

        # Multiple batches: summarise each then merge
        batch_results: list[_SentimentOutput] = []
        for batch in batches:
            articles_text = _format_articles(batch)
            result = chain.invoke({"symbol": symbol, "articles_text": articles_text})
            batch_results.append(result)

        avg_score = sum(r.score for r in batch_results) / len(batch_results)
        avg_confidence = sum(r.confidence for r in batch_results) / len(batch_results)
        merged_label = (
            SentimentLabel.POSITIVE if avg_score > 0.1
            else SentimentLabel.NEGATIVE if avg_score < -0.1
            else SentimentLabel.NEUTRAL
        )
        merged_pos = [d for r in batch_results for d in r.positive_drivers][:3]
        merged_neg = [d for r in batch_results for d in r.negative_drivers][:3]

        drivers = [
            SentimentDriver(text=t, sentiment=SentimentLabel.POSITIVE, source="llm")
            for t in merged_pos
        ] + [
            SentimentDriver(text=t, sentiment=SentimentLabel.NEGATIVE, source="llm")
            for t in merged_neg
        ]
        sources = list({a.source for a in articles if a.source})[:5]

        return SentimentResult(
            symbol=symbol,
            score=round(avg_score, 3),
            label=merged_label,
            confidence=round(avg_confidence, 3),
            drivers=drivers,
            sources=sources,
            summary=batch_results[0].summary,
        )

    except Exception:
        return _keyword_sentiment(articles, symbol)
