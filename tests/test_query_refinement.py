"""Tests for chains/query_refinement_chain.py."""


from bestock_agent.chains.query_refinement_chain import _fallback_refinement, refine_query


# ── _fallback_refinement (no LLM required) ────────────────────────────────────

def test_fallback_adds_financial_terms_when_missing():
    result = _fallback_refinement("NVDA", "NVIDIA Corporation", "NVIDIA news")
    assert "NVDA" in result
    # Should append financial-context words
    assert any(w in result.lower() for w in ["earnings", "stock", "market", "financial", "analyst"])


def test_fallback_ensures_symbol_in_query():
    result = _fallback_refinement("AAPL", "Apple Inc", "quarterly results announcement")
    assert "AAPL" in result


def test_fallback_returns_string():
    result = _fallback_refinement("TSLA", "Tesla Inc", "TSLA stock earnings")
    assert isinstance(result, str)
    assert len(result) > 0


# ── refine_query — fallback path (placeholder API key) ────────────────────────

def test_refine_query_fallback_with_placeholder_key():
    result = refine_query(
        symbol="NVDA",
        company_name="NVIDIA Corporation",
        original_query="NVIDIA news",
        feedback="Only 1 article returned",
        openai_model="gpt-4o-mini",
        openai_api_key="<YOUR-API-KEY>",
    )
    assert isinstance(result, str)
    assert len(result) > 0
    assert "NVDA" in result


def test_refine_query_fallback_with_empty_key():
    result = refine_query(
        symbol="AAPL",
        company_name="Apple Inc",
        original_query="Apple news today",
        feedback="Too few results",
        openai_model="gpt-4o-mini",
        openai_api_key="",
    )
    assert isinstance(result, str)
    assert "AAPL" in result


def test_refine_query_different_for_generic_input():
    """A very generic query should produce a more specific result."""
    refined = refine_query(
        symbol="META",
        company_name="Meta Platforms",
        original_query="news",
        feedback="Query too vague",
        openai_model="gpt-4o-mini",
        openai_api_key="",
    )
    # Fallback should at least include the symbol or company
    assert "META" in refined or "Meta" in refined
