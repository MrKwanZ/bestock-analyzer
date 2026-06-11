"""Plotly chart generation and export.

Phase 3: basic price-trend and daily-change charts.
Phase 5 will add index-comparison and sentiment charts.
"""

from __future__ import annotations

import os
from pathlib import Path

import plotly.graph_objects as go

from bestock_agent.schemas import ChartArtifact, ChartType, PriceBar

_CHART_BG = "#0d1117"
_GRID_COLOR = "#21262d"
_TEXT_COLOR = "#c9d1d9"
_ACCENT_BLUE = "#58a6ff"
_GREEN = "#3fb950"
_RED = "#f85149"


def _ensure_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_price_chart(
    symbol: str,
    bars: list[PriceBar],
    output_dir: str = "outputs/charts",
) -> ChartArtifact:
    """Generate a closing-price trend line chart and save it as a PNG."""
    dates = [str(b.date) for b in bars]
    closes = [b.close for b in bars]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=closes,
            mode="lines+markers",
            name="Close",
            line=dict(color=_ACCENT_BLUE, width=2),
            marker=dict(size=6, color=_ACCENT_BLUE),
            hovertemplate="<b>%{x}</b><br>Close: $%{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"{symbol} — Closing Price",
            font=dict(color=_TEXT_COLOR, size=16),
        ),
        paper_bgcolor=_CHART_BG,
        plot_bgcolor=_CHART_BG,
        font=dict(color=_TEXT_COLOR),
        xaxis=dict(
            gridcolor=_GRID_COLOR,
            title="Date",
            tickfont=dict(color=_TEXT_COLOR),
        ),
        yaxis=dict(
            gridcolor=_GRID_COLOR,
            title="Price (USD)",
            tickprefix="$",
            tickfont=dict(color=_TEXT_COLOR),
        ),
        margin=dict(l=60, r=30, t=60, b=60),
        width=800,
        height=450,
    )

    out_dir = _ensure_output_dir(output_dir)
    filename = out_dir / f"{symbol}_price_trend.png"
    fig.write_image(str(filename))

    return ChartArtifact(
        path=str(filename),
        chart_type=ChartType.PRICE_TREND,
        title=f"{symbol} Closing Price Trend",
    )


def generate_change_chart(
    symbol: str,
    daily_changes: list[float],
    dates: list[str],
    output_dir: str = "outputs/charts",
) -> ChartArtifact:
    """Generate a daily % change bar chart and save it as a PNG."""
    # Exclude the first zero-change entry
    plot_dates = dates[1:] if len(dates) > 1 else dates
    plot_changes = daily_changes[1:] if len(daily_changes) > 1 else daily_changes
    bar_colors = [_GREEN if c >= 0 else _RED for c in plot_changes]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_dates,
            y=plot_changes,
            marker_color=bar_colors,
            hovertemplate="<b>%{x}</b><br>Change: %{y:+.2f}%<extra></extra>",
            name="Daily Change %",
        )
    )
    fig.add_hline(y=0, line_color=_TEXT_COLOR, line_width=0.8)

    fig.update_layout(
        title=dict(
            text=f"{symbol} — Daily Change (%)",
            font=dict(color=_TEXT_COLOR, size=16),
        ),
        paper_bgcolor=_CHART_BG,
        plot_bgcolor=_CHART_BG,
        font=dict(color=_TEXT_COLOR),
        xaxis=dict(gridcolor=_GRID_COLOR, title="Date", tickfont=dict(color=_TEXT_COLOR)),
        yaxis=dict(
            gridcolor=_GRID_COLOR,
            title="Change (%)",
            ticksuffix="%",
            tickfont=dict(color=_TEXT_COLOR),
        ),
        margin=dict(l=60, r=30, t=60, b=60),
        width=800,
        height=400,
    )

    out_dir = _ensure_output_dir(output_dir)
    filename = out_dir / f"{symbol}_daily_change.png"
    fig.write_image(str(filename))

    return ChartArtifact(
        path=str(filename),
        chart_type=ChartType.DAILY_CHANGE,
        title=f"{symbol} Daily Change (%)",
    )
