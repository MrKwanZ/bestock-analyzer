"""Plotly chart generation and export.

Phase 3: basic price-trend and daily-change charts.
Phase 5: index-comparison (S&P 500 overlay) and sentiment gauge charts.
"""

from __future__ import annotations

import os
from pathlib import Path

import plotly.graph_objects as go

from bestock_agent.schemas import (
    ChartArtifact,
    ChartType,
    IndexComparison,
    PriceBar,
    SentimentResult,
)

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


# ── Phase 5: advanced charts ──────────────────────────────────────────────────

_ORANGE = "#d29922"


def generate_index_comparison_chart(
    symbol: str,
    stock_bars: list[PriceBar],
    comparison: IndexComparison,
    output_dir: str = "outputs/charts",
) -> ChartArtifact:
    """Normalised performance overlay: stock vs S&P 500.

    Both series are rebased to 100 at the first available date so they can be
    compared on the same axis regardless of absolute price level.
    """
    if not stock_bars or not comparison.sp500.bars:
        raise ValueError("Not enough data to generate index comparison chart")

    # Align on common dates (index bars may differ in count)
    n = min(len(stock_bars), len(comparison.sp500.bars))
    sb = stock_bars[-n:]
    ib = comparison.sp500.bars[-n:]

    stock_closes = [b.close for b in sb]
    index_closes = [b.close for b in ib]
    dates = [str(b.date) for b in sb]

    base_stock = stock_closes[0]
    base_index = index_closes[0]
    norm_stock = [round(c / base_stock * 100, 4) for c in stock_closes]
    norm_index = [round(c / base_index * 100, 4) for c in index_closes]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=norm_stock,
            mode="lines+markers",
            name=symbol,
            line=dict(color=_ACCENT_BLUE, width=2),
            marker=dict(size=6),
            hovertemplate="<b>%{x}</b><br>" + symbol + ": %{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=norm_index,
            mode="lines+markers",
            name="S&P 500",
            line=dict(color=_ORANGE, width=2, dash="dot"),
            marker=dict(size=5),
            hovertemplate="<b>%{x}</b><br>S&P 500: %{y:.1f}<extra></extra>",
        )
    )
    fig.add_hline(y=100, line_color=_TEXT_COLOR, line_width=0.6, line_dash="dash")

    fig.update_layout(
        title=dict(
            text=f"{symbol} vs S&P 500 — Normalised Performance (base = 100)",
            font=dict(color=_TEXT_COLOR, size=15),
        ),
        paper_bgcolor=_CHART_BG,
        plot_bgcolor=_CHART_BG,
        font=dict(color=_TEXT_COLOR),
        xaxis=dict(gridcolor=_GRID_COLOR, title="Date", tickfont=dict(color=_TEXT_COLOR)),
        yaxis=dict(
            gridcolor=_GRID_COLOR,
            title="Indexed Performance",
            tickfont=dict(color=_TEXT_COLOR),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=_TEXT_COLOR),
            x=0.01, y=0.99,
        ),
        margin=dict(l=60, r=30, t=70, b=60),
        width=800,
        height=450,
    )

    out_dir = _ensure_output_dir(output_dir)
    filename = out_dir / f"{symbol}_vs_sp500.png"
    fig.write_image(str(filename))

    return ChartArtifact(
        path=str(filename),
        chart_type=ChartType.INDEX_COMPARISON,
        title=f"{symbol} vs S&P 500 Normalised Performance",
    )


def generate_sentiment_chart(
    sentiment: SentimentResult,
    output_dir: str = "outputs/charts",
) -> ChartArtifact:
    """Horizontal gauge chart showing the sentiment score (-1 → +1).

    The gauge uses a red-yellow-green colour scale and annotates the
    score, label, and confidence inline.
    """
    symbol = sentiment.symbol
    score = sentiment.score
    confidence = sentiment.confidence
    label = sentiment.label.value.capitalize()

    gauge_color = _GREEN if score > 0.1 else _RED if score < -0.1 else _ORANGE

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            number={"suffix": "", "font": {"color": gauge_color, "size": 36}},
            delta={"reference": 0, "valueformat": "+.3f"},
            gauge={
                "axis": {"range": [-1, 1], "tickcolor": _TEXT_COLOR, "tickfont": {"color": _TEXT_COLOR}},
                "bar": {"color": gauge_color, "thickness": 0.6},
                "bgcolor": _CHART_BG,
                "borderwidth": 0,
                "steps": [
                    {"range": [-1, -0.1], "color": "#3d1a1a"},
                    {"range": [-0.1, 0.1], "color": "#1e2a1e"},
                    {"range": [0.1, 1], "color": "#1a2e1a"},
                ],
                "threshold": {
                    "line": {"color": _TEXT_COLOR, "width": 2},
                    "thickness": 0.75,
                    "value": score,
                },
            },
            title={
                "text": (
                    f"<b>{symbol} Sentiment</b><br>"
                    f"<span style='font-size:14px;color:{_TEXT_COLOR}'>"
                    f"{label} &nbsp;|&nbsp; Confidence {confidence:.0%}</span>"
                ),
                "font": {"color": _TEXT_COLOR, "size": 18},
            },
        )
    )

    fig.update_layout(
        paper_bgcolor=_CHART_BG,
        font=dict(color=_TEXT_COLOR),
        margin=dict(l=40, r=40, t=80, b=40),
        width=600,
        height=350,
    )

    out_dir = _ensure_output_dir(output_dir)
    filename = out_dir / f"{symbol}_sentiment.png"
    fig.write_image(str(filename))

    return ChartArtifact(
        path=str(filename),
        chart_type=ChartType.SENTIMENT,
        title=f"{symbol} Sentiment Score ({label})",
    )
