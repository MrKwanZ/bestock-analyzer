"""LangGraph StateGraph assembly for the NASDAQ Best Stock Analyzer.

Graph topology
──────────────

  START
    │
    ▼
  fetch_top_gainer ──(error)──────────────────────────┐
    │                                                   │
    ▼                                                   │
  fetch_price_history ──(error)───────────────────────►│
    │                                                   │
    ▼                                                   ▼
  analyze_trend ──(error)───────────────────────► error_handler ──► END
    │                                                   ▲
    ├──(advanced=True)──► fetch_news_and_sentiment      │
    │                           │ (error: degrade)      │
    │                           ▼                       │
    │                    compare_with_indices            │
    │                           │                       │
    └──(advanced=False)─────────┤                       │
                                ▼                       │
                          build_charts                  │
                                │                       │
                                ▼                       │
                          compose_report                │
                                │                       │
                                ▼                       │
                          send_email ──(error)──────────┘
                                │
                                ▼
                               END
"""

from langgraph.graph import END, START, StateGraph

from bestock_agent.nodes.analyze_trend import analyze_trend
from bestock_agent.nodes.build_charts import build_charts
from bestock_agent.nodes.compare_with_indices import compare_with_indices
from bestock_agent.nodes.compose_report import compose_report
from bestock_agent.nodes.error_handler import error_handler
from bestock_agent.nodes.fetch_news_and_sentiment import fetch_news_and_sentiment
from bestock_agent.nodes.fetch_price_history import fetch_price_history
from bestock_agent.nodes.fetch_top_gainer import fetch_top_gainer
from bestock_agent.nodes.send_email import send_email
from bestock_agent.state import BestockState

# ── Routing constants ─────────────────────────────────────────────────────────

_NODE_FETCH_TOP_GAINER = "fetch_top_gainer"
_NODE_FETCH_PRICE_HISTORY = "fetch_price_history"
_NODE_ANALYZE_TREND = "analyze_trend"
_NODE_FETCH_NEWS = "fetch_news_and_sentiment"
_NODE_COMPARE_INDICES = "compare_with_indices"
_NODE_BUILD_CHARTS = "build_charts"
_NODE_COMPOSE_REPORT = "compose_report"
_NODE_SEND_EMAIL = "send_email"
_NODE_ERROR_HANDLER = "error_handler"

_MAX_RETRIES = 2


# ── Routing helpers ───────────────────────────────────────────────────────────


def _latest_error_node(state: BestockState) -> str | None:
    """Return the ``node`` field of the most recent AgentError, or None."""
    if state["errors"]:
        return state["errors"][-1].node
    return None


def _has_unrecoverable_error_from(state: BestockState, node: str) -> bool:
    """True if the latest error originated from *node* and is not recoverable."""
    errors = state["errors"]
    if not errors:
        return False
    latest = errors[-1]
    return latest.node == node and not latest.recoverable


def _has_error_from(state: BestockState, node: str) -> bool:
    """True if *any* error in state originated from *node*."""
    return any(e.node == node for e in state["errors"])


# ── Routing functions ─────────────────────────────────────────────────────────


def _route_fetch_top_gainer(state: BestockState) -> str:
    """Route after fetch_top_gainer: proceed or hand off to error_handler."""
    if state["top_gainer"] is None or _has_error_from(state, _NODE_FETCH_TOP_GAINER):
        return _NODE_ERROR_HANDLER
    return _NODE_FETCH_PRICE_HISTORY


def _route_fetch_price_history(state: BestockState) -> str:
    """Route after fetch_price_history: proceed or hand off to error_handler."""
    if not state["price_history"] or _has_error_from(state, _NODE_FETCH_PRICE_HISTORY):
        return _NODE_ERROR_HANDLER
    return _NODE_ANALYZE_TREND


def _route_analyze_trend(state: BestockState) -> str:
    """Route after analyze_trend: branch on advanced-analysis flag or error."""
    if _has_unrecoverable_error_from(state, _NODE_ANALYZE_TREND):
        return _NODE_ERROR_HANDLER
    if state["advanced_analysis_enabled"]:
        return _NODE_FETCH_NEWS
    return _NODE_BUILD_CHARTS


def _route_fetch_news(state: BestockState) -> str:
    """Route after fetch_news_and_sentiment.

    News / sentiment failure is *non-fatal*: we degrade gracefully to
    build_charts rather than routing to error_handler.
    """
    return _NODE_COMPARE_INDICES


def _route_send_email(state: BestockState) -> str:
    """Route after send_email: surface error or finish."""
    if _has_error_from(state, _NODE_SEND_EMAIL):
        return _NODE_ERROR_HANDLER
    return END


def _route_error_handler(state: BestockState) -> str:
    """Route after error_handler: retry the failing node if budget remains.

    After error_handler runs it either:
    - Bumped retry_count and possibly switched provider → re-route to the node
      that originated the error so it re-runs with the new provider.
    - Saturated retry_count → route to END with failure RunSummary.
    """
    from bestock_agent.services.fallback import failing_node

    if state["retry_count"] >= _MAX_RETRIES:
        return END

    target = failing_node(state)
    if target and target in (
        _NODE_FETCH_TOP_GAINER,
        _NODE_FETCH_PRICE_HISTORY,
        _NODE_ANALYZE_TREND,
        _NODE_SEND_EMAIL,
    ):
        return target
    return END


# ── Graph assembly ────────────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """Construct and return the compiled LangGraph StateGraph."""
    graph = StateGraph(BestockState)

    # Register nodes
    graph.add_node(_NODE_FETCH_TOP_GAINER, fetch_top_gainer)
    graph.add_node(_NODE_FETCH_PRICE_HISTORY, fetch_price_history)
    graph.add_node(_NODE_ANALYZE_TREND, analyze_trend)
    graph.add_node(_NODE_FETCH_NEWS, fetch_news_and_sentiment)
    graph.add_node(_NODE_COMPARE_INDICES, compare_with_indices)
    graph.add_node(_NODE_BUILD_CHARTS, build_charts)
    graph.add_node(_NODE_COMPOSE_REPORT, compose_report)
    graph.add_node(_NODE_SEND_EMAIL, send_email)
    graph.add_node(_NODE_ERROR_HANDLER, error_handler)

    # Entry point
    graph.add_edge(START, _NODE_FETCH_TOP_GAINER)

    # Conditional edges
    graph.add_conditional_edges(
        _NODE_FETCH_TOP_GAINER,
        _route_fetch_top_gainer,
        {
            _NODE_FETCH_PRICE_HISTORY: _NODE_FETCH_PRICE_HISTORY,
            _NODE_ERROR_HANDLER: _NODE_ERROR_HANDLER,
        },
    )
    graph.add_conditional_edges(
        _NODE_FETCH_PRICE_HISTORY,
        _route_fetch_price_history,
        {
            _NODE_ANALYZE_TREND: _NODE_ANALYZE_TREND,
            _NODE_ERROR_HANDLER: _NODE_ERROR_HANDLER,
        },
    )
    graph.add_conditional_edges(
        _NODE_ANALYZE_TREND,
        _route_analyze_trend,
        {
            _NODE_FETCH_NEWS: _NODE_FETCH_NEWS,
            _NODE_BUILD_CHARTS: _NODE_BUILD_CHARTS,
            _NODE_ERROR_HANDLER: _NODE_ERROR_HANDLER,
        },
    )

    # fetch_news → compare_with_indices always (graceful degradation on error)
    graph.add_conditional_edges(
        _NODE_FETCH_NEWS,
        _route_fetch_news,
        {_NODE_COMPARE_INDICES: _NODE_COMPARE_INDICES},
    )

    # compare_with_indices → build_charts always (errors are non-fatal)
    graph.add_edge(_NODE_COMPARE_INDICES, _NODE_BUILD_CHARTS)

    # Linear pipeline tail
    graph.add_edge(_NODE_BUILD_CHARTS, _NODE_COMPOSE_REPORT)
    graph.add_edge(_NODE_COMPOSE_REPORT, _NODE_SEND_EMAIL)

    graph.add_conditional_edges(
        _NODE_SEND_EMAIL,
        _route_send_email,
        {
            END: END,
            _NODE_ERROR_HANDLER: _NODE_ERROR_HANDLER,
        },
    )

    # error_handler can retry or terminate
    graph.add_conditional_edges(
        _NODE_ERROR_HANDLER,
        _route_error_handler,
        {
            _NODE_FETCH_TOP_GAINER: _NODE_FETCH_TOP_GAINER,
            _NODE_FETCH_PRICE_HISTORY: _NODE_FETCH_PRICE_HISTORY,
            _NODE_ANALYZE_TREND: _NODE_ANALYZE_TREND,
            _NODE_SEND_EMAIL: _NODE_SEND_EMAIL,
            END: END,
        },
    )

    return graph


# Pre-compiled graph instance for use by CLI, Gradio UI, and tests.
app = build_graph().compile()
