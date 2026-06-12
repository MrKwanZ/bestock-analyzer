"""LCEL chain: search-query refinement for news lookups.

When the initial news search returns too few articles or low-confidence
sentiment, this chain generates a more targeted query so the news provider
can return higher-quality results before the agent falls back to a different
provider.

Primary path: ChatOpenAI rewrites the query with stock-specific context.
Fallback: pure-Python rule-based refinement (no API call required).
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

_SYSTEM = """\
You are a financial news search assistant. Given an original search query that
returned insufficient or irrelevant results, produce a single, improved search
query optimised for finding recent, specific news about the given stock.

Rules:
- Keep the refined query under 12 words.
- Include the stock ticker symbol AND one or two relevant financial terms.
- Do NOT include explanations — output only the refined query string.
"""

_HUMAN = """\
Stock: {symbol} ({company_name})
Original query: {original_query}
Issue: {feedback}

Refined query:"""


def _build_chain(model: str, api_key: str):
    llm = ChatOpenAI(model=model, api_key=api_key, temperature=0.0)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM), ("human", _HUMAN)]
    )
    return prompt | llm | StrOutputParser()


def _fallback_refinement(
    symbol: str,
    company_name: str,
    original_query: str,
) -> str:
    """Rule-based query refinement — no LLM required."""
    # If the original query is very short or generic, add financial context
    financial_terms = ["earnings", "stock price", "market performance", "analyst"]
    has_financial_term = any(t in original_query.lower() for t in financial_terms)

    if not has_financial_term:
        return f"{symbol} {company_name} stock earnings market news"

    # Otherwise just ensure the ticker appears prominently
    if symbol not in original_query:
        return f"{symbol} {original_query}"

    return f"{symbol} {company_name} latest financial news"


def refine_query(
    symbol: str,
    company_name: str,
    original_query: str,
    feedback: str,
    openai_model: str,
    openai_api_key: str,
) -> str:
    """Return a refined news search query.

    Tries the LLM chain first; falls back to rule-based refinement if the
    API key is a placeholder or the chain raises an exception.
    """
    key_is_placeholder = not openai_api_key or openai_api_key.startswith("<")
    if not key_is_placeholder:
        try:
            chain = _build_chain(openai_model, openai_api_key)
            result: str = chain.invoke({
                "symbol": symbol,
                "company_name": company_name,
                "original_query": original_query,
                "feedback": feedback,
            })
            refined = result.strip().strip('"').strip("'")
            if refined:
                return refined
        except Exception:
            pass  # fall through to rule-based

    return _fallback_refinement(symbol, company_name, original_query)
