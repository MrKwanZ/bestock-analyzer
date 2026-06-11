"""LCEL chain for composing the stock analysis email report.

Primary path: ChatOpenAI with structured output → HTML + plain-text email body.
Fallback: pure-Python template used when the LLM call fails (e.g. invalid key).
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from bestock_agent.schemas import TrendAnalysis, TopGainer
from bestock_agent.services.analysis import format_price_table


class ReportOutput(BaseModel):
    """Structured output produced by the report chain."""

    html_body: str = Field(description="Professional HTML email body with inline styles, suitable for email clients")
    text_body: str = Field(description="Plain-text fallback of the same report for non-HTML email clients")


_SYSTEM = """\
You are a professional financial analyst assistant. Generate a concise, well-formatted stock analysis email report.
Keep the language clear, factual, and professional. Do not provide explicit buy/sell financial advice.
"""

_HUMAN = """\
Prepare an email body for the following NASDAQ stock analysis.

## Stock
- Name: {name}
- Symbol: {symbol}
- Today's Close: ${price:.2f}
- Today's Change: {change_pct:+.2f}%
- Analysis Date: {analysis_date}

## {lookback_days}-Day Performance
{price_table}

## Quantitative Summary
- Average Daily Change: {avg_change:+.2f}%
- Trend Classification: {trend_label}
- Daily Volatility (Std Dev): {volatility}

## Trend Explanation
{trend_summary}

---
Return both an HTML version (with subtle inline styling, a readable table, and section headers)
and a plain-text version. The plain-text version should be fully readable without HTML rendering.
"""


def _build_chain(model: str, api_key: str):
    llm = ChatOpenAI(model=model, api_key=api_key, temperature=0.3)
    structured_llm = llm.with_structured_output(ReportOutput)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM), ("human", _HUMAN)]
    )
    return prompt | structured_llm


# ── Template fallback ─────────────────────────────────────────────────────────


def _template_report(gainer: TopGainer, analysis: TrendAnalysis, analysis_date: str) -> ReportOutput:
    """Pure-Python fallback used when the LLM chain is unavailable."""
    table = format_price_table(analysis.bars, analysis.daily_changes)
    vol_str = f"±{analysis.volatility:.2f}%" if analysis.volatility else "N/A"

    trend_emoji = {"uptrend": "📈", "downtrend": "📉", "sideways": "➡️", "pullback": "↩️"}.get(
        analysis.trend_label.value, ""
    )

    text_body = f"""\
NASDAQ Stock Analysis Report
{'=' * 40}
Company : {gainer.name} ({gainer.symbol})
Date    : {analysis_date}
Price   : ${gainer.price:.2f}
Change  : {gainer.change_pct:+.2f}%

{len(analysis.bars)}-Day Performance
{'-' * 40}
{table}

Summary
{'-' * 40}
Average Daily Change : {analysis.avg_change:+.2f}%
Trend                : {trend_emoji} {analysis.trend_label.value.title()}
Daily Volatility     : {vol_str}

{analysis.summary}

---
This report was generated automatically by the NASDAQ BeStock Analyzer.
"""

    html_rows = ""
    for i, bar in enumerate(analysis.bars):
        chg = analysis.daily_changes[i]
        chg_str = f"{chg:+.2f}%" if i > 0 else "—"
        color = "#3fb950" if chg >= 0 else "#f85149"
        html_rows += (
            f"<tr>"
            f"<td style='padding:6px 12px;border-bottom:1px solid #30363d'>{bar.date}</td>"
            f"<td style='padding:6px 12px;border-bottom:1px solid #30363d;text-align:right'>${bar.close:.2f}</td>"
            f"<td style='padding:6px 12px;border-bottom:1px solid #30363d;text-align:right;color:{color}'>{chg_str}</td>"
            f"</tr>"
        )

    html_body = f"""\
<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;background:#0d1117;color:#c9d1d9;padding:24px;max-width:680px;margin:auto">
  <h1 style="color:#58a6ff;font-size:20px">NASDAQ Stock Analysis Report</h1>
  <table style="border-collapse:collapse;margin-bottom:16px">
    <tr><td style="padding:4px 12px 4px 0;color:#8b949e">Company</td><td><strong>{gainer.name}</strong> ({gainer.symbol})</td></tr>
    <tr><td style="padding:4px 12px 4px 0;color:#8b949e">Date</td><td>{analysis_date}</td></tr>
    <tr><td style="padding:4px 12px 4px 0;color:#8b949e">Close Price</td><td>${gainer.price:.2f}</td></tr>
    <tr><td style="padding:4px 12px 4px 0;color:#8b949e">Today's Change</td>
        <td style="color:{'#3fb950' if gainer.change_pct >= 0 else '#f85149'}">{gainer.change_pct:+.2f}%</td></tr>
  </table>

  <h2 style="color:#58a6ff;font-size:16px">{len(analysis.bars)}-Day Performance</h2>
  <table style="border-collapse:collapse;width:100%;margin-bottom:16px">
    <thead>
      <tr style="background:#161b22">
        <th style="padding:8px 12px;text-align:left;color:#8b949e;border-bottom:1px solid #30363d">Date</th>
        <th style="padding:8px 12px;text-align:right;color:#8b949e;border-bottom:1px solid #30363d">Close</th>
        <th style="padding:8px 12px;text-align:right;color:#8b949e;border-bottom:1px solid #30363d">Change</th>
      </tr>
    </thead>
    <tbody>{html_rows}</tbody>
  </table>

  <h2 style="color:#58a6ff;font-size:16px">Trend Analysis</h2>
  <table style="border-collapse:collapse;margin-bottom:16px">
    <tr><td style="padding:4px 12px 4px 0;color:#8b949e">Avg Daily Change</td><td>{analysis.avg_change:+.2f}%</td></tr>
    <tr><td style="padding:4px 12px 4px 0;color:#8b949e">Trend</td><td>{trend_emoji} {analysis.trend_label.value.title()}</td></tr>
    <tr><td style="padding:4px 12px 4px 0;color:#8b949e">Volatility</td><td>{vol_str}</td></tr>
  </table>
  <p style="color:#8b949e;font-size:13px">{analysis.summary}</p>
  <hr style="border-color:#30363d"/>
  <p style="color:#6e7681;font-size:11px">Generated by NASDAQ BeStock Analyzer</p>
</body>
</html>"""

    return ReportOutput(html_body=html_body, text_body=text_body)


# ── Public interface ──────────────────────────────────────────────────────────


def compose_report(
    gainer: TopGainer,
    analysis: TrendAnalysis,
    analysis_date: str,
    openai_model: str,
    openai_api_key: str,
) -> ReportOutput:
    """Compose the report using the LLM chain with a template fallback."""
    # Skip LLM if the API key is the placeholder from .env
    key_is_placeholder = not openai_api_key or openai_api_key.startswith("<")
    if not key_is_placeholder:
        try:
            chain = _build_chain(openai_model, openai_api_key)
            vol_str = f"±{analysis.volatility:.2f}%" if analysis.volatility else "N/A"
            result = chain.invoke({
                "name": gainer.name,
                "symbol": gainer.symbol,
                "price": gainer.price,
                "change_pct": gainer.change_pct,
                "analysis_date": analysis_date,
                "lookback_days": len(analysis.bars),
                "price_table": format_price_table(analysis.bars, analysis.daily_changes),
                "avg_change": analysis.avg_change,
                "trend_label": analysis.trend_label.value.title(),
                "volatility": vol_str,
                "trend_summary": analysis.summary,
            })
            return result
        except Exception:
            pass  # Fall through to template

    return _template_report(gainer, analysis, analysis_date)
