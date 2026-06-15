"""Tests for LangGraph checkpoint persistence and schema safety."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.types import Command

from bestock_agent.checkpoint import (
    assert_resumable,
    create_checkpointer,
    make_thread_id,
    prepare_invoke_state,
    run_config,
)
from bestock_agent.graph import compile_app
from bestock_agent.schemas import ChartArtifact, ChartType, PriceBar, TopGainer
from bestock_agent.state import STATE_SCHEMA_VERSION, initial_state
from bestock_agent.state_migration import StaleCheckpointError, hydrate_state


def _gainer() -> TopGainer:
    return TopGainer(
        symbol="NVDA",
        name="NVIDIA Corporation",
        price=1200.0,
        change=50.0,
        change_pct=4.3,
    )


def _bars() -> list[PriceBar]:
    return [
        PriceBar(
            date=date(2026, 6, i + 3),
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.0 + i,
            volume=1000,
        )
        for i in range(5)
    ]


def _chart() -> ChartArtifact:
    return ChartArtifact(
        chart_type=ChartType.PRICE_TREND,
        path="outputs/charts/nvda_price.png",
        title="NVDA Price Trend",
    )


def _mock_patches():
    mock_fin = AsyncMock()
    mock_fin.get_top_nasdaq_gainer.return_value = _gainer()
    mock_fin.get_price_history.return_value = _bars()
    return (
        patch("bestock_agent.nodes.fetch_top_gainer.get_financial_provider", return_value=mock_fin),
        patch("bestock_agent.nodes.fetch_price_history.get_financial_provider", return_value=mock_fin),
        patch("bestock_agent.nodes.build_charts.generate_price_chart", return_value=_chart()),
        patch("bestock_agent.nodes.build_charts.generate_change_chart", return_value=_chart()),
        patch("bestock_agent.nodes.build_charts.generate_index_comparison_chart", return_value=_chart()),
        patch("bestock_agent.nodes.build_charts.generate_sentiment_chart", return_value=_chart()),
        patch("bestock_agent.nodes.send_email.send_report_email", new_callable=AsyncMock),
    )


@pytest.mark.asyncio
async def test_thread_id_embeds_schema_version():
    thread_id = make_thread_id("cli")
    assert thread_id.startswith(f"v{STATE_SCHEMA_VERSION}-cli-")


@pytest.mark.asyncio
async def test_hydrate_fills_new_field_from_defaults():
    stored = initial_state(
        recipient_email="test@example.com",
        target_date="2026-06-11",
        lookback_days=5,
    )
    del stored["state_schema_version"]
    overrides = initial_state(
        recipient_email="test@example.com",
        target_date="2026-06-11",
        lookback_days=5,
    )
    hydrated = hydrate_state(stored, overrides)
    assert hydrated["state_schema_version"] == STATE_SCHEMA_VERSION
    assert hydrated["retry_count"] == 0


@pytest.mark.asyncio
async def test_version_mismatch_raises_stale_checkpoint():
    from types import SimpleNamespace

    app = SimpleNamespace()

    async def _aget_state(_config):
        return SimpleNamespace(values={"state_schema_version": 0})

    app.aget_state = _aget_state
    fresh = initial_state(
        recipient_email="test@example.com",
        target_date="2026-06-11",
        lookback_days=5,
    )

    with pytest.raises(StaleCheckpointError):
        await prepare_invoke_state(app, run_config("stale-thread"), fresh)

    with pytest.raises(StaleCheckpointError):
        await assert_resumable(app, run_config("stale-thread"))


@pytest.mark.asyncio
async def test_checkpoint_saved_after_each_node(agent_app, thread_config):
    config, _ = thread_config
    state = initial_state(
        recipient_email="test@example.com",
        target_date=str(date.today()),
        lookback_days=5,
    )
    state["active_financial_provider"] = "alphavantage"
    state = await prepare_invoke_state(agent_app, config, state)

    with _mock_patches()[0], _mock_patches()[1], _mock_patches()[2], _mock_patches()[3], _mock_patches()[4], _mock_patches()[5], _mock_patches()[6]:
        await agent_app.ainvoke(state, config=config)

    checkpoints = [item async for item in agent_app.checkpointer.alist(config)]
    assert len(checkpoints) >= 5


@pytest.mark.asyncio
async def test_state_roundtrip_via_get_state(agent_app, thread_config):
    config, _ = thread_config
    state = initial_state(
        recipient_email="test@example.com",
        target_date=str(date.today()),
        lookback_days=5,
    )
    state["active_financial_provider"] = "alphavantage"
    state = await prepare_invoke_state(agent_app, config, state)

    with _mock_patches()[0], _mock_patches()[1], _mock_patches()[2], _mock_patches()[3], _mock_patches()[4], _mock_patches()[5], _mock_patches()[6]:
        await agent_app.ainvoke(state, config=config)

    snapshot = await agent_app.aget_state(config)
    values = snapshot.values
    assert values.get("top_gainer") is not None
    assert values.get("email_payload") is not None
    assert values.get("state_schema_version") == STATE_SCHEMA_VERSION


@pytest.mark.asyncio
async def test_thread_isolation(agent_app):
    config_a = run_config(make_thread_id("a"))
    config_b = run_config(make_thread_id("b"))

    state_a = initial_state(
        recipient_email="a@example.com",
        target_date=str(date.today()),
        lookback_days=5,
    )
    state_b = initial_state(
        recipient_email="b@example.com",
        target_date=str(date.today()),
        lookback_days=5,
    )
    for state in (state_a, state_b):
        state["active_financial_provider"] = "alphavantage"

    with _mock_patches()[0], _mock_patches()[1], _mock_patches()[2], _mock_patches()[3], _mock_patches()[4], _mock_patches()[5], _mock_patches()[6]:
        await agent_app.ainvoke(state_a, config=config_a)
        await agent_app.ainvoke(state_b, config=config_b)

    snap_a = await agent_app.aget_state(config_a)
    snap_b = await agent_app.aget_state(config_b)
    assert snap_a.values["recipient_email"] == "a@example.com"
    assert snap_b.values["recipient_email"] == "b@example.com"


@pytest.mark.asyncio
async def test_resume_after_interrupt(agent_app_with_interrupt, thread_config):
    config, _ = thread_config
    state = initial_state(
        recipient_email="recipient@example.com",
        target_date=str(date.today()),
        lookback_days=5,
    )
    state["active_financial_provider"] = "alphavantage"
    state = await prepare_invoke_state(agent_app_with_interrupt, config, state)

    mock_fin = AsyncMock()
    mock_fin.get_top_nasdaq_gainer.return_value = _gainer()
    mock_fin.get_price_history.return_value = _bars()

    with (
        patch("bestock_agent.nodes.fetch_top_gainer.get_financial_provider", return_value=mock_fin),
        patch("bestock_agent.nodes.fetch_price_history.get_financial_provider", return_value=mock_fin),
        patch("bestock_agent.nodes.build_charts.generate_price_chart", return_value=_chart()),
        patch("bestock_agent.nodes.build_charts.generate_change_chart", return_value=_chart()),
        patch("bestock_agent.nodes.build_charts.generate_index_comparison_chart", return_value=_chart()),
        patch("bestock_agent.nodes.build_charts.generate_sentiment_chart", return_value=_chart()),
        patch("bestock_agent.nodes.send_email.send_report_email", new_callable=AsyncMock),
        patch("bestock_agent.nodes.send_email.get_settings", return_value=MagicMock(
            smtp_user="bestockagent@gmail.com",
            smtp_password="real_password",
        )),
    ):
        await agent_app_with_interrupt.ainvoke(state, config=config)
        paused = await agent_app_with_interrupt.aget_state(config)
        assert "send_email" in paused.next

        await agent_app_with_interrupt.ainvoke(Command(resume=True), config=config)
        final = await agent_app_with_interrupt.aget_state(config)

    assert final.values["run_summary"] is not None
    assert final.values["run_summary"].email_sent is True


@pytest.mark.asyncio
async def test_sqlite_persists_across_instances(tmp_path):
    db_path = tmp_path / "persist.db"
    config = run_config(make_thread_id("persist"))
    state = initial_state(
        recipient_email="test@example.com",
        target_date=str(date.today()),
        lookback_days=5,
    )
    state["active_financial_provider"] = "alphavantage"

    cp1 = await create_checkpointer(db_path)
    app1 = compile_app(checkpointer=cp1, interrupt_before_send=False)
    with _mock_patches()[0], _mock_patches()[1], _mock_patches()[2], _mock_patches()[3], _mock_patches()[4], _mock_patches()[5], _mock_patches()[6]:
        await app1.ainvoke(state, config=config)
    await cp1.conn.close()

    cp2 = await create_checkpointer(db_path)
    app2 = compile_app(checkpointer=cp2, interrupt_before_send=False)
    snapshot = await app2.aget_state(config)
    assert snapshot.values.get("top_gainer") is not None
    await cp2.conn.close()


@pytest.mark.asyncio
async def test_nested_pydantic_new_optional_field():
    raw = {
        "symbol": "NVDA",
        "name": "NVIDIA",
        "price": 1200.0,
        "change": 50.0,
        "change_pct": 4.3,
    }
    gainer = TopGainer.model_validate(raw)
    assert gainer.market_cap is None


def test_checkpoint_serde_registers_schema_types():
    from bestock_agent.checkpoint import _schema_msgpack_allowlist, get_checkpoint_serde

    allowlist = _schema_msgpack_allowlist()
    assert ("bestock_agent.schemas", "TopGainer") in allowlist
    assert ("bestock_agent.schemas", "EmailPayload") in allowlist
    assert ("bestock_agent.schemas", "TrendLabel") in allowlist

    serde = get_checkpoint_serde()
    gainer = TopGainer(
        symbol="NVDA",
        name="NVIDIA Corporation",
        price=1200.0,
        change=50.0,
        change_pct=4.3,
    )
    payload = serde.dumps_typed(gainer)
    restored = serde.loads_typed(payload)
    assert isinstance(restored, TopGainer)
    assert restored.symbol == "NVDA"
