"""LangGraph checkpoint helpers — SQLite persistence and schema-safe resume."""

from __future__ import annotations

import asyncio
import inspect
from enum import Enum
from pathlib import Path
from uuid import uuid4

import aiosqlite
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

import bestock_agent.schemas as schema_module
from bestock_agent.config import get_settings
from bestock_agent.state import STATE_SCHEMA_VERSION, BestockState
from bestock_agent.state_migration import StaleCheckpointError, hydrate_state

_checkpointer: BaseCheckpointSaver | None = None


def _schema_msgpack_allowlist() -> list[tuple[str, str]]:
    """Register every Pydantic model / enum written into graph checkpoints."""
    module_name = schema_module.__name__
    allowed: list[tuple[str, str]] = []
    for name, obj in vars(schema_module).items():
        if not inspect.isclass(obj) or obj.__module__ != module_name:
            continue
        if issubclass(obj, BaseModel):
            allowed.append((module_name, name))
        elif issubclass(obj, Enum) and obj is not Enum:
            allowed.append((module_name, name))
    return sorted(allowed)


def get_checkpoint_serde() -> JsonPlusSerializer:
    """Return a checkpoint serializer that knows about ``bestock_agent.schemas`` types."""
    return JsonPlusSerializer(allowed_msgpack_modules=_schema_msgpack_allowlist())


async def create_checkpointer(db_path: Path) -> AsyncSqliteSaver:
    """Open (or create) a persistent AsyncSqliteSaver at *db_path*."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(db_path))
    saver = AsyncSqliteSaver(conn, serde=get_checkpoint_serde())
    await saver.setup()
    return saver


def _init_checkpointer_sync() -> BaseCheckpointSaver:
    settings = get_settings()
    if not settings.checkpoint_enabled:
        return MemorySaver(serde=get_checkpoint_serde())

    async def _setup() -> AsyncSqliteSaver:
        return await create_checkpointer(settings.checkpoint_db_path)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_setup())

    raise RuntimeError(
        "Checkpointer cannot be initialised inside a running event loop. "
        "Import bestock_agent.graph before starting async code."
    )


def get_shared_checkpointer() -> BaseCheckpointSaver:
    """Return the process-wide checkpointer singleton."""
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = _init_checkpointer_sync()
    return _checkpointer


def make_thread_id(prefix: str = "run") -> str:
    """Return a version-scoped thread ID for a new graph run."""
    return f"v{STATE_SCHEMA_VERSION}-{prefix}-{uuid4().hex[:12]}"


def run_config(thread_id: str) -> dict:
    """Build the LangGraph ``config`` dict for a checkpoint thread."""
    return {"configurable": {"thread_id": thread_id}}


async def _snapshot_values(app, config: dict) -> dict | None:
    snapshot = await app.aget_state(config)
    values = snapshot.values
    if not values:
        return None
    return dict(values)


async def prepare_invoke_state(app, config: dict, fresh_state: BestockState) -> BestockState:
    """Validate and hydrate checkpoint state before ``ainvoke``."""
    stored = await _snapshot_values(app, config)
    if stored is None:
        return fresh_state

    stored_version = stored.get("state_schema_version", 0)
    if stored_version != STATE_SCHEMA_VERSION:
        raise StaleCheckpointError(stored_version, STATE_SCHEMA_VERSION)

    return hydrate_state(stored, fresh_state)


async def assert_resumable(app, config: dict) -> None:
    """Ensure an existing checkpoint can be resumed with the current schema."""
    stored = await _snapshot_values(app, config)
    if stored is None:
        raise StaleCheckpointError(0, STATE_SCHEMA_VERSION)

    stored_version = stored.get("state_schema_version", 0)
    if stored_version != STATE_SCHEMA_VERSION:
        raise StaleCheckpointError(stored_version, STATE_SCHEMA_VERSION)
