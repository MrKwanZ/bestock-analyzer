"""Shared pytest fixtures for checkpoint-aware graph tests."""

from __future__ import annotations

import pytest

from bestock_agent.checkpoint import create_checkpointer, make_thread_id, run_config
from bestock_agent.graph import compile_app


@pytest.fixture
async def checkpointer(tmp_path):
    """Isolated SQLite checkpointer backed by a temp file."""
    db_path = tmp_path / "test.db"
    saver = await create_checkpointer(db_path)
    yield saver
    await saver.conn.close()


@pytest.fixture
async def agent_app(checkpointer):
    """Compiled graph with checkpointing enabled and no email interrupt."""
    return compile_app(checkpointer=checkpointer, interrupt_before_send=False)


@pytest.fixture
async def agent_app_with_interrupt(checkpointer):
    """Compiled graph that pauses before send_email."""
    return compile_app(checkpointer=checkpointer, interrupt_before_send=True)


@pytest.fixture
def thread_config():
    """Fresh thread config for an isolated graph run."""
    thread_id = make_thread_id("test")
    return run_config(thread_id), thread_id
