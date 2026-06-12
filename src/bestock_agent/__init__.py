"""NASDAQ BeStock Analyzer — LangChain + LangGraph agent."""

from __future__ import annotations

import os

__version__ = "0.1.0"


def _configure_langsmith() -> None:
    """Activate LangSmith tracing when LANGSMITH_TRACING=true is set in the env.

    This is called once at package import time so every LangChain / LangGraph
    call made in the same process automatically inherits the tracer.
    The LangChain SDK reads the ``LANGCHAIN_*`` env vars; we set them here from
    our own ``LANGSMITH_*`` settings so users only need to fill in one set of
    variables in ``.env``.
    """
    tracing_flag = os.environ.get("LANGSMITH_TRACING", "").lower()
    if tracing_flag not in ("1", "true", "yes"):
        return

    api_key = os.environ.get("LANGSMITH_API_KEY", "")
    endpoint = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    project  = os.environ.get("LANGSMITH_PROJECT", "bestock-agent")

    if not api_key or api_key.startswith("<") or api_key.startswith("lsv2_pt_"):
        # placeholder key — don't enable tracing to avoid noisy auth errors
        if api_key.startswith("lsv2_pt_"):
            # Real-looking key — activate
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
            os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
            os.environ.setdefault("LANGCHAIN_PROJECT", project)
            os.environ.setdefault("LANGCHAIN_ENDPOINT", endpoint)
        return

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", project)
    os.environ.setdefault("LANGCHAIN_ENDPOINT", endpoint)

_configure_langsmith()
