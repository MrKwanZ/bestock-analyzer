"""Command-line interface for the NASDAQ BeStock Analyzer."""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date

from bestock_agent.config import get_settings
from bestock_agent.graph import app
from bestock_agent.services.analysis import format_price_table
from bestock_agent.state import initial_state


# ── Formatting helpers ────────────────────────────────────────────────────────


def _hr(char: str = "─", width: int = 52) -> str:
    return char * width


def _print_header(title: str) -> None:
    print(f"\n{_hr()}")
    print(f"  {title}")
    print(_hr())


def _pct_color(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"


# ── Main graph runner ─────────────────────────────────────────────────────────


async def run_agent(
    *,
    recipient: str,
    target_date: str,
    lookback_days: int,
    advanced: bool,
    provider: str,
) -> None:
    print("\n🚀  NASDAQ BeStock Analyzer — starting run")
    print(f"    Date     : {target_date}")
    print(f"    Lookback : {lookback_days} trading days")
    print(f"    Provider : {provider}")
    print(f"    Recipient: {recipient}")
    print(f"    Advanced : {'yes' if advanced else 'no'}")
    print()

    state = initial_state(
        recipient_email=recipient,
        target_date=target_date,
        lookback_days=lookback_days,
        advanced_analysis_enabled=advanced,
    )
    state["active_financial_provider"] = provider

    print("🔍  Fetching top NASDAQ gainer …")
    final = await app.ainvoke(state)

    run_summary = final.get("run_summary")

    # ── Top gainer ────────────────────────────────────────────────────────────
    gainer = final.get("top_gainer")
    if gainer:
        _print_header("Top NASDAQ Gainer")
        print(f"  Name   : {gainer.name}")
        print(f"  Symbol : {gainer.symbol}")
        print(f"  Close  : ${gainer.price:,.2f}")
        print(f"  Change : {_pct_color(gainer.change_pct)}")

    # ── 5-day performance ─────────────────────────────────────────────────────
    analysis = final.get("trend_analysis")
    if analysis:
        _print_header(f"{len(analysis.bars)}-Day Performance")
        print(format_price_table(analysis.bars, analysis.daily_changes))
        print()
        print(f"  Average Daily Change : {_pct_color(analysis.avg_change)}")
        print(f"  Trend                : {analysis.trend_label.value.title()}")
        if analysis.volatility is not None:
            print(f"  Volatility (Std Dev) : ±{analysis.volatility:.2f}%")
        print(f"\n  {analysis.summary}")

    # ── Sentiment (advanced only) ─────────────────────────────────────────────
    sentiment = final.get("sentiment_result")
    if sentiment and advanced:
        _print_header("Market Sentiment")
        score_sign = "+" if sentiment.score >= 0 else ""
        print(f"  Label      : {sentiment.label.value.capitalize()}")
        print(f"  Score      : {score_sign}{sentiment.score:.2f}  (confidence {sentiment.confidence:.0%})")
        print(f"  Summary    : {sentiment.summary}")
        drivers = [d for d in sentiment.drivers]
        if drivers:
            print("  Drivers    :")
            for d in drivers[:4]:
                bullet = "  ↑" if d.sentiment.value == "positive" else "  ↓"
                print(f"    {bullet} {d.text}")

    # ── Index comparison (advanced only) ─────────────────────────────────────
    comparison = final.get("index_comparison")
    if comparison and advanced:
        _print_header("S&P 500 Comparison")
        stock_period = comparison.sp500.period_change_pct + comparison.relative_perf_vs_sp500
        beta_str = f"{comparison.beta:.2f}" if comparison.beta is not None else "N/A"
        print(f"  {comparison.stock_symbol:<8} period return : {_pct_color(stock_period)}")
        print(f"  S&P 500  period return : {_pct_color(comparison.sp500.period_change_pct)}")
        print(f"  Relative performance  : {_pct_color(comparison.relative_perf_vs_sp500)}")
        print(f"  Beta                  : {beta_str}")

    # ── Charts ────────────────────────────────────────────────────────────────
    charts = final.get("chart_artifacts", [])
    if charts:
        _print_header("Charts Generated")
        for c in charts:
            print(f"  [{c.chart_type.value}] {c.path}")

    # ── Email status ──────────────────────────────────────────────────────────
    _print_header("Email")
    payload = final.get("email_payload")
    if payload:
        print(f"  Subject : {payload.subject}")
        print(f"  To      : {payload.recipient}")
    if run_summary:
        if run_summary.email_sent:
            status = "✅  Sent"
        elif not run_summary.success and run_summary.errors:
            email_err = next((e for e in run_summary.errors if e.node == "send_email"), None)
            status = f"❌  Failed — {email_err.message}" if email_err else "❌  Failed"
        else:
            status = "⚠️   Skipped (SMTP credentials not configured)"
        print(f"  Status  : {status}")

    # ── Fatal errors (no analysis produced) ──────────────────────────────────
    if run_summary and not run_summary.success and not final.get("top_gainer"):
        print(f"\n❌  Run failed before fetching data: {run_summary.message}")
        for err in (run_summary.errors or []):
            print(f"    [{err.error_type.value}] {err.node}: {err.message}")
        print(f"\n{_hr()}\n")
        sys.exit(1)

    print(f"\n{_hr()}\n")


# ── Argument parsing ──────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="bestock",
        description="NASDAQ BeStock Analyzer — find and report the top NASDAQ gainer",
    )
    parser.add_argument(
        "--date",
        default=str(date.today()),
        help="Target date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=settings.default_lookback_days,
        help=f"Lookback trading days (default: {settings.default_lookback_days})",
    )
    parser.add_argument(
        "--recipient",
        default=str(settings.default_email_recipient),
        help="Email recipient address",
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        default=settings.advanced_analysis_enabled,
        help="Enable sentiment, volatility, and index-comparison analysis",
    )
    parser.add_argument(
        "--provider",
        choices=["finnhub", "yfinance"],
        default="finnhub",
        help="Primary financial data provider (default: finnhub)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    asyncio.run(
        run_agent(
            recipient=args.recipient,
            target_date=args.date,
            lookback_days=args.days,
            advanced=args.advanced,
            provider=args.provider,
        )
    )


if __name__ == "__main__":
    main()
