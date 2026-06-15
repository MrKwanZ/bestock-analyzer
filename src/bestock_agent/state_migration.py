"""State hydration and version migration for LangGraph checkpoints."""

from __future__ import annotations

from typing import Any

from bestock_agent.schemas import (
    AgentError,
    ChartArtifact,
    EmailPayload,
    IndexComparison,
    PriceBar,
    RunSummary,
    SentimentResult,
    TopGainer,
    TrendAnalysis,
)
from bestock_agent.state import STATE_SCHEMA_VERSION, BestockState, initial_state

# Chained upgraders keyed by the source version. Extend when bumping
# STATE_SCHEMA_VERSION and backward resume is required.
_MIGRATIONS: dict[int, Any] = {}


class StaleCheckpointError(Exception):
    """Raised when a checkpoint was written with an incompatible schema version."""

    def __init__(self, stored_version: int, current_version: int) -> None:
        self.stored_version = stored_version
        self.current_version = current_version
        super().__init__(
            f"Checkpoint schema version {stored_version} does not match "
            f"current version {current_version}."
        )


def _user_inputs_from_state(state: BestockState) -> dict[str, Any]:
    return {
        "recipient_email": state["recipient_email"],
        "target_date": state["target_date"],
        "lookback_days": state["lookback_days"],
        "advanced_analysis_enabled": state["advanced_analysis_enabled"],
        "enable_volatility": state["enable_volatility"],
        "skip_email": state.get("skip_email", False),
    }


def _coerce_model(value: Any, model_cls: type) -> Any:
    if value is None:
        return None
    if isinstance(value, model_cls):
        return value
    return model_cls.model_validate(value)


def _coerce_model_list(values: Any, model_cls: type) -> list[Any]:
    if not values:
        return []
    return [_coerce_model(item, model_cls) for item in values]


def migrate_state(stored: dict[str, Any], from_version: int) -> dict[str, Any]:
    """Upgrade a stored checkpoint dict through the migration chain."""
    current = dict(stored)
    version = from_version
    while version < STATE_SCHEMA_VERSION:
        migrator = _MIGRATIONS.get(version)
        if migrator is None:
            raise StaleCheckpointError(version, STATE_SCHEMA_VERSION)
        current = migrator(current)
        version += 1
    return current


def hydrate_state(stored: dict[str, Any], overrides: BestockState) -> BestockState:
    """Merge checkpoint values over fresh defaults from ``initial_state``."""
    base = initial_state(**_user_inputs_from_state(overrides))
    merged: dict[str, Any] = dict(base)

    for key, value in stored.items():
        if key in ("chart_artifacts", "errors"):
            merged[key] = _coerce_model_list(value, ChartArtifact if key == "chart_artifacts" else AgentError)
        elif key == "price_history":
            merged[key] = _coerce_model_list(value, PriceBar)
        elif key == "top_gainer":
            merged[key] = _coerce_model(value, TopGainer)
        elif key == "trend_analysis":
            merged[key] = _coerce_model(value, TrendAnalysis)
        elif key == "sentiment_result":
            merged[key] = _coerce_model(value, SentimentResult)
        elif key == "index_comparison":
            merged[key] = _coerce_model(value, IndexComparison)
        elif key == "email_payload":
            merged[key] = _coerce_model(value, EmailPayload)
        elif key == "run_summary":
            merged[key] = _coerce_model(value, RunSummary)
        else:
            merged[key] = value

    merged["state_schema_version"] = STATE_SCHEMA_VERSION
    return merged  # type: ignore[return-value]
