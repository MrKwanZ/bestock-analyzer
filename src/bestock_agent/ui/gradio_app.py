"""Gradio UI entry point — delegates to bestock_agent.app.

The full UI implementation lives in ``src/bestock_agent/app.py``.
This module is kept as a convenience alias for the original Phase 6 plan
folder layout.
"""

from bestock_agent.app import build_ui, main

__all__ = ["build_ui", "main"]
