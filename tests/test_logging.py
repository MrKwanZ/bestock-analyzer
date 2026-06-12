"""Tests for bestock_agent.logging — structured logger helpers."""

import logging

import pytest

from bestock_agent.logging import BestockLogger, get_logger, timer


def test_get_logger_returns_bestock_logger():
    log = get_logger("test_node")
    assert isinstance(log, BestockLogger)


def test_logger_name_is_scoped():
    log = get_logger("my_node")
    assert log._log.name == "bestock_agent.my_node"


def test_provider_call_does_not_raise():
    log = get_logger("fetch_top_gainer")
    log.provider_call("alphavantage", "get_top_nasdaq_gainer", latency_ms=150, symbol="NVDA")


def test_fallback_switch_does_not_raise():
    log = get_logger("error_handler")
    log.fallback_switch("alphavantage", "yfinance", reason="rate limit")


def test_validation_failure_does_not_raise():
    log = get_logger("fetch_top_gainer")
    log.validation_failure("change_pct", "Value 99999 is out of range")


def test_token_usage_does_not_raise():
    log = get_logger("report_chain")
    log.token_usage("gpt-4o-mini", prompt=300, completion=150, total=450)


def test_email_status_sent_does_not_raise():
    log = get_logger("send_email")
    log.email_status(sent=True, recipient="user@example.com")


def test_email_status_failed_does_not_raise():
    log = get_logger("send_email")
    log.email_status(sent=False, recipient="user@example.com", error="SMTP timeout")


def test_node_error_does_not_raise():
    log = get_logger("fetch_top_gainer")
    log.node_error("fetch_top_gainer", "TOOL_ERROR", "API returned 503", retry=1)


def test_query_refined_does_not_raise():
    log = get_logger("fetch_news")
    log.query_refined("NVDA news", "NVDA NVIDIA stock earnings news")


def test_timer_measures_elapsed():
    with timer() as t:
        pass
    assert isinstance(t.elapsed_ms, int)
    assert t.elapsed_ms >= 0


def test_fmt_includes_key_value():
    """BestockLogger._fmt should embed key=value pairs in the message string."""
    log = get_logger("test_fmt")
    formatted = log._fmt("test_event", {"provider": "alphavantage", "latency_ms": 99})
    assert "provider" in formatted
    assert "alphavantage" in formatted
