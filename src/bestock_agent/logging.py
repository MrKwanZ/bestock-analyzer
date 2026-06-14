"""Structured application logging for the NASDAQ BeStock Analyzer.

Wraps Python's standard ``logging`` module with helper methods for the key
pipeline events that matter most for observability:

  - Provider selection and request latency
  - Provider fallback switches
  - Rate-limit backoff waits
  - Validation failures
  - LLM token usage
  - Email delivery status
  - Node-level errors and retries

Usage::

    from bestock_agent.logging import get_logger
    log = get_logger("fetch_top_gainer")

    log.provider_call("alphavantage", "get_top_nasdaq_gainer", latency_ms=312)
    log.fallback_switch("alphavantage", "yfinance", reason="rate limit hit")
    log.backoff_waiting(provider="alphavantage", node="fetch_top_gainer", seconds=20)
    log.validation_failure("change_pct", "Value 99999 exceeds plausible range")
    log.token_usage(model="gpt-5.4-mini", prompt=420, completion=180, total=600)
    log.email_status(sent=True, recipient="user@example.com")
    log.node_error("fetch_top_gainer", "TOOL_ERROR", "API returned 503", retry=1)
"""

from __future__ import annotations

import logging
import os
import time


# ── Root logger configuration ─────────────────────────────────────────────────

_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
_LOG_FORMAT = os.environ.get("LOG_FORMAT", "text")   # "text" | "json"

_root = logging.getLogger("bestock_agent")
if not _root.handlers:
    _handler = logging.StreamHandler()
    if _LOG_FORMAT == "json":
        _fmt = '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":%(message)s}'
    else:
        _fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    _handler.setFormatter(logging.Formatter(_fmt, datefmt="%Y-%m-%dT%H:%M:%S"))
    _root.addHandler(_handler)
    _root.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
    _root.propagate = False


# ── Timer context manager ─────────────────────────────────────────────────────


class _Timer:
    """Simple wall-clock timer."""

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_ms = round((time.perf_counter() - self._start) * 1000)


def timer() -> _Timer:
    """Return a context-manager timer; access ``t.elapsed_ms`` after exit."""
    return _Timer()


# ── Structured logger wrapper ─────────────────────────────────────────────────


class BestockLogger:
    """Logger with structured event helpers."""

    def __init__(self, name: str) -> None:
        self._log = logging.getLogger(f"bestock_agent.{name}")

    # ── Raw helpers ───────────────────────────────────────────────────────────

    def debug(self, msg: str, **fields: object) -> None:
        self._log.debug(self._fmt(msg, fields))

    def info(self, msg: str, **fields: object) -> None:
        self._log.info(self._fmt(msg, fields))

    def warning(self, msg: str, **fields: object) -> None:
        self._log.warning(self._fmt(msg, fields))

    def error(self, msg: str, **fields: object) -> None:
        self._log.error(self._fmt(msg, fields))

    @staticmethod
    def _fmt(msg: str, fields: dict) -> str:
        if not fields:
            return msg
        kv = "  ".join(f"{k}={v!r}" for k, v in fields.items())
        return f"{msg}  [{kv}]"

    # ── Domain-specific helpers ───────────────────────────────────────────────

    def provider_call(
        self,
        provider: str,
        operation: str,
        *,
        latency_ms: int | None = None,
        symbol: str | None = None,
    ) -> None:
        """Log a successful provider API call."""
        fields: dict = {"provider": provider, "op": operation}
        if symbol:
            fields["symbol"] = symbol
        if latency_ms is not None:
            fields["latency_ms"] = latency_ms
        self.info("provider_call", **fields)

    def fallback_switch(
        self,
        from_provider: str,
        to_provider: str,
        *,
        reason: str = "",
    ) -> None:
        """Log a provider fallback switch."""
        self.warning(
            "provider_fallback",
            from_provider=from_provider,
            to_provider=to_provider,
            reason=reason,
        )

    def backoff_waiting(self, *, provider: str, node: str, seconds: int) -> None:
        """Log a rate-limit backoff wait before retrying the same provider."""
        self.warning(
            "rate_limit_backoff_wait",
            provider=provider,
            node=node,
            seconds=seconds,
        )

    def validation_failure(self, field: str, message: str) -> None:
        """Log an input validation failure."""
        self.warning("validation_failure", field=field, detail=message)

    def token_usage(
        self,
        model: str,
        *,
        prompt: int = 0,
        completion: int = 0,
        total: int = 0,
    ) -> None:
        """Log LLM token consumption."""
        self.info(
            "token_usage",
            model=model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )

    def email_status(self, *, sent: bool, recipient: str, error: str = "") -> None:
        """Log email delivery outcome."""
        if sent:
            self.info("email_sent", recipient=recipient)
        else:
            self.warning("email_skipped_or_failed", recipient=recipient, error=error)

    def node_error(
        self,
        node: str,
        error_type: str,
        message: str,
        *,
        retry: int = 0,
        recoverable: bool = True,
    ) -> None:
        """Log a pipeline node error."""
        self.error(
            "node_error",
            node=node,
            error_type=error_type,
            recoverable=recoverable,
            retry=retry,
            detail=message[:200],
        )

    def query_refined(self, original: str, refined: str, *, via: str = "llm") -> None:
        """Log a news query refinement."""
        self.info("query_refined", via=via, original=original[:80], refined=refined[:80])


# ── Factory ───────────────────────────────────────────────────────────────────


def get_logger(name: str) -> BestockLogger:
    """Return a BestockLogger scoped to *name* under the bestock_agent namespace."""
    return BestockLogger(name)
