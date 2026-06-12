"""Matplotlib chart generation — no browser dependency required.

Replaces the previous Plotly + kaleido implementation. All charts are
rendered entirely in Python and saved as PNGs using matplotlib's Agg backend.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.use("Agg")   # headless, no display needed

from bestock_agent.schemas import (
    ChartArtifact,
    ChartType,
    IndexComparison,
    PriceBar,
    SentimentResult,
)

# ── Shared theme ──────────────────────────────────────────────────────────────

_BG      = "#0d1117"
_AXES_BG = "#161b22"
_GRID    = "#21262d"
_TEXT    = "#c9d1d9"
_BLUE    = "#58a6ff"
_GREEN   = "#3fb950"
_RED     = "#f85149"
_ORANGE  = "#d29922"

_FONT = {"fontsize": 10, "color": _TEXT, "fontfamily": "sans-serif"}


def _apply_dark_theme(fig: plt.Figure, ax: plt.Axes) -> None:
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_AXES_BG)
    ax.tick_params(colors=_TEXT, labelsize=9)
    ax.spines["bottom"].set_color(_GRID)
    ax.spines["left"].set_color(_GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.grid(True, color=_GRID, linewidth=0.5, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)


def _ensure_dir(output_dir: str) -> Path:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(str(path), dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Chart 1: Closing price trend ──────────────────────────────────────────────

def generate_price_chart(
    symbol: str,
    bars: list[PriceBar],
    output_dir: str = "outputs/charts",
) -> ChartArtifact:
    dates  = [str(b.date) for b in bars]
    closes = [b.close for b in bars]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    _apply_dark_theme(fig, ax)

    ax.plot(dates, closes, color=_BLUE, linewidth=2, marker="o", markersize=5,
            markerfacecolor=_BLUE, zorder=3)
    ax.fill_between(range(len(closes)), closes,
                    min(closes) * 0.995,
                    color=_BLUE, alpha=0.12)

    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=30, ha="right", **_FONT)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.2f}"))
    ax.set_xlabel("Date", **_FONT)
    ax.set_ylabel("Close Price (USD)", **_FONT)
    ax.set_title(f"{symbol} — Closing Price", color=_TEXT, fontsize=14, pad=12)

    # Annotate last point
    ax.annotate(
        f"${closes[-1]:,.2f}",
        xy=(len(closes) - 1, closes[-1]),
        xytext=(8, 6), textcoords="offset points",
        color=_BLUE, fontsize=9,
    )

    out = _ensure_dir(output_dir)
    path = out / f"{symbol}_price_trend.png"
    _save(fig, path)

    return ChartArtifact(
        path=str(path),
        chart_type=ChartType.PRICE_TREND,
        title=f"{symbol} Closing Price Trend",
    )


# ── Chart 2: Daily % change bars ──────────────────────────────────────────────

def generate_change_chart(
    symbol: str,
    daily_changes: list[float],
    dates: list[str],
    output_dir: str = "outputs/charts",
) -> ChartArtifact:
    plot_dates   = dates[1:] if len(dates) > 1 else dates
    plot_changes = daily_changes[1:] if len(daily_changes) > 1 else daily_changes
    colors = [_GREEN if c >= 0 else _RED for c in plot_changes]

    fig, ax = plt.subplots(figsize=(9, 4))
    _apply_dark_theme(fig, ax)

    x = range(len(plot_dates))
    bars_obj = ax.bar(x, plot_changes, color=colors, width=0.6, zorder=3)

    ax.axhline(0, color=_TEXT, linewidth=0.8, alpha=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_dates, rotation=30, ha="right", **_FONT)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.1f}%"))
    ax.set_xlabel("Date", **_FONT)
    ax.set_ylabel("Daily Change (%)", **_FONT)
    ax.set_title(f"{symbol} — Daily Change (%)", color=_TEXT, fontsize=14, pad=12)

    # Value labels on bars
    for bar, val in zip(bars_obj, plot_changes):
        offset = 0.3 if val >= 0 else -0.8
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + offset,
            f"{val:+.1f}%",
            ha="center", va="bottom", fontsize=8, color=_TEXT,
        )

    out = _ensure_dir(output_dir)
    path = out / f"{symbol}_daily_change.png"
    _save(fig, path)

    return ChartArtifact(
        path=str(path),
        chart_type=ChartType.DAILY_CHANGE,
        title=f"{symbol} Daily Change (%)",
    )


# ── Chart 3: Stock vs S&P 500 normalised overlay ──────────────────────────────

def generate_index_comparison_chart(
    symbol: str,
    stock_bars: list[PriceBar],
    comparison: IndexComparison,
    output_dir: str = "outputs/charts",
) -> ChartArtifact:
    if not stock_bars or not comparison.sp500.bars:
        raise ValueError("Not enough data for index comparison chart")

    n  = min(len(stock_bars), len(comparison.sp500.bars))
    sb = stock_bars[-n:]
    ib = comparison.sp500.bars[-n:]

    dates        = [str(b.date) for b in sb]
    stock_closes = [b.close for b in sb]
    index_closes = [b.close for b in ib]

    norm_stock = [c / stock_closes[0] * 100 for c in stock_closes]
    norm_index = [c / index_closes[0] * 100 for c in index_closes]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    _apply_dark_theme(fig, ax)

    ax.plot(dates, norm_stock, color=_BLUE, linewidth=2, marker="o", markersize=5,
            label=symbol, zorder=3)
    ax.plot(dates, norm_index, color=_ORANGE, linewidth=2, marker="s", markersize=4,
            linestyle="--", label="S&P 500", zorder=3)
    ax.axhline(100, color=_TEXT, linewidth=0.6, linestyle=":", alpha=0.5)

    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=30, ha="right", **_FONT)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.set_xlabel("Date", **_FONT)
    ax.set_ylabel("Indexed Performance (base = 100)", **_FONT)
    ax.set_title(
        f"{symbol} vs S&P 500 — Normalised Performance (base = 100)",
        color=_TEXT, fontsize=13, pad=12,
    )
    legend = ax.legend(facecolor=_AXES_BG, edgecolor=_GRID, labelcolor=_TEXT, fontsize=9)

    out = _ensure_dir(output_dir)
    path = out / f"{symbol}_vs_sp500.png"
    _save(fig, path)

    return ChartArtifact(
        path=str(path),
        chart_type=ChartType.INDEX_COMPARISON,
        title=f"{symbol} vs S&P 500 Normalised Performance",
    )


# ── Chart 4: Sentiment gauge ──────────────────────────────────────────────────

def generate_sentiment_chart(
    sentiment: SentimentResult,
    output_dir: str = "outputs/charts",
) -> ChartArtifact:
    symbol     = sentiment.symbol
    score      = sentiment.score          # -1 … +1
    confidence = sentiment.confidence
    label      = sentiment.label.value.capitalize()

    gauge_color = _GREEN if score > 0.1 else _RED if score < -0.1 else _ORANGE

    fig, ax = plt.subplots(figsize=(7, 3.5))
    _apply_dark_theme(fig, ax)
    ax.grid(False)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.15, 1.1)
    ax.axis("off")
    fig.patch.set_facecolor(_BG)

    # Background track
    track = mpatches.FancyBboxPatch(
        (-1, 0.3), 2, 0.28,
        boxstyle="round,pad=0.02",
        linewidth=0, facecolor=_GRID,
    )
    ax.add_patch(track)

    # Filled portion (score mapped from [-1,1] to [0,2])
    fill_width = max(0.01, score + 1)   # 0 → 2
    fill = mpatches.FancyBboxPatch(
        (-1, 0.3), fill_width, 0.28,
        boxstyle="round,pad=0.02",
        linewidth=0, facecolor=gauge_color, alpha=0.85,
    )
    ax.add_patch(fill)

    # Needle line at score position
    ax.plot([score, score], [0.26, 0.62], color=_TEXT, linewidth=2, zorder=5)

    # Labels
    ax.text(0, 0.9, f"{symbol} Sentiment", ha="center", va="center",
            color=_TEXT, fontsize=15, fontweight="bold")
    ax.text(0, 0.72, f"{label}  |  Score {score:+.2f}  |  Confidence {confidence:.0%}",
            ha="center", va="center", color=gauge_color, fontsize=11)

    # Scale labels
    for xv, lbl in [(-1, "−1\nBearish"), (0, "0\nNeutral"), (1, "+1\nBullish")]:
        ax.text(xv, 0.1, lbl, ha="center", va="top",
                color=_TEXT, fontsize=8, alpha=0.8)

    out = _ensure_dir(output_dir)
    path = out / f"{symbol}_sentiment.png"
    _save(fig, path)

    return ChartArtifact(
        path=str(path),
        chart_type=ChartType.SENTIMENT,
        title=f"{symbol} Sentiment Score ({label})",
    )
