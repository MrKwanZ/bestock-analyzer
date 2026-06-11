"""Shared fixtures for the test suite."""

import json
from pathlib import Path

import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def top_gainers_fixture() -> dict:
    return json.loads((FIXTURES_DIR / "top_gainers.json").read_text())


@pytest.fixture()
def price_history_fixture() -> dict:
    return json.loads((FIXTURES_DIR / "price_history.json").read_text())


@pytest.fixture()
def news_results_fixture() -> dict:
    return json.loads((FIXTURES_DIR / "news_results.json").read_text())
