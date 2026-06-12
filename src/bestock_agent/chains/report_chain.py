"""LCEL chain for composing the stock analysis email report.

Primary path: ChatOpenAI with structured output → HTML + plain-text email body.
Fallback: pure-Python template used when the LLM call fails (e.g. invalid key).
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from bestock_agent.schemas import (
    IndexComparison,
    SentimentLabel,
    SentimentResult,
    TrendAnalysis,
    TrendLabel,
    TopGainer,
)
from bestock_agent.services.analysis import build_trend_summary, format_price_table

# ── Constants ──────────────────────────────────────────────────────────────────

_GREETING = "Greetings!"
_SIGN_OFF_TEXT = "Regards,\nBeStock Agent"
_SIGN_OFF_HTML = "Regards,<br>BeStock Agent"

_DISCLAIMER_RAW  = """<ol>
<li>This is informational guidance only and not personal financial advice.</li>
<li>A 15-20 minute stock data latency is expected.</li>
</ol>"""
_DISCLAIMER_TEXT = f"Disclaimer: {_DISCLAIMER_RAW}"
_DISCLAIMER_HTML = f"<strong>Disclaimer: {_DISCLAIMER_RAW}</strong>"


class ReportOutput(BaseModel):
    """Structured output produced by the report chain."""

    html_body: str = Field(description="Professional HTML email body with inline styles, suitable for email clients")
    text_body: str = Field(description="Plain-text fallback of the same report for non-HTML email clients")


_SYSTEM = """\
You are a professional financial analyst assistant. Generate a concise, well-formatted stock analysis email report.
Keep the language clear, factual, and professional.
"""

_HUMAN = """\
Prepare an email body for the following NASDAQ stock analysis.

Begin the email with "Greetings!" and end with:
Regards,
BeStock Agent
(put a line break between "Regards," and "BeStock Agent")

Immediately after the greeting, open the report by stating the stock name ({name}),
its ticker symbol ({symbol}), and today's percentage change ({change_pct:+.2f}%).

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
{volatility_line}
## Trend Explanation
{trend_summary}
{sentiment_section}{index_section}{suggestion_content}
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


# ── Section helpers ────────────────────────────────────────────────────────────


def _opening_line_text(gainer: TopGainer) -> str:
    """Opening sentence for plain-text emails — name, symbol, and today's change."""
    return (
        f"Today's Top Stock in NASDAQ: {gainer.name} ({gainer.symbol}), "
        f"with a {gainer.change_pct:+.2f}% change today."
    )


def _opening_line_html(gainer: TopGainer) -> str:
    """Opening paragraph for HTML emails — name, symbol, and today's change."""
    color = "#3fb950" if gainer.change_pct >= 0 else "#f85149"
    return (
        f'<p style="color:#c9d1d9;font-size:14px;margin-bottom:12px">'
        f"Today's top NASDAQ performer is <strong>{gainer.name}</strong> "
        f"({gainer.symbol}), with a "
        f'<span style="color:{color}">{gainer.change_pct:+.2f}%</span> change today.</p>'
    )


def _sentiment_section_text(sentiment: SentimentResult | None) -> str:
    if sentiment is None or not sentiment.summary:
        return ""
    label_upper = sentiment.label.value.upper()
    drivers = [d.text for d in sentiment.drivers]
    driver_str = ""
    if drivers:
        bullet_items = "\n".join(f"  • {d}" for d in drivers[:4])
        driver_str = f"\nKey Drivers:\n{bullet_items}"
    return (
        f"\n## Market Sentiment\n"
        f"- Sentiment: {label_upper}  (score {sentiment.score:+.2f}, confidence {sentiment.confidence:.0%})\n"
        f"- Summary: {sentiment.summary}{driver_str}\n"
    )


def _index_section_text(comparison: IndexComparison | None) -> str:
    if comparison is None:
        return ""
    sp = comparison.sp500
    beta_str = f"{comparison.beta:.2f}" if comparison.beta is not None else "N/A"
    stock_period = sp.period_change_pct + comparison.relative_perf_vs_sp500
    return (
        f"\n## S&P 500 Comparison\n"
        f"- {comparison.stock_symbol} period change: {stock_period:+.2f}%\n"
        f"- S&P 500 period change: {sp.period_change_pct:+.2f}%\n"
        f"- Relative performance vs S&P 500: {comparison.relative_perf_vs_sp500:+.2f}%\n"
        f"- Beta: {beta_str}\n"
    )


def _trend_summary_for_report(analysis: TrendAnalysis, include_volatility: bool) -> str:
    """Return trend explanation text, omitting volatility unless explicitly enabled."""
    volatility = analysis.volatility if include_volatility else None
    return build_trend_summary(
        analysis.avg_change,
        analysis.trend_label,
        len(analysis.bars),
        volatility,
    )


def _generate_suggestion(
    analysis: TrendAnalysis,
    sentiment: SentimentResult | None,
    comparison: IndexComparison | None,
    *,
    include_volatility: bool = True,
) -> tuple[str, str]:
    """Return (action, rationale) where action is Buy, Hold, or Sell."""
    score = 0.0
    reasons: list[str] = []

    trend = analysis.trend_label
    if trend == TrendLabel.UPTREND:
        score += 2
        reasons.append("the stock is in an uptrend")
    elif trend == TrendLabel.DOWNTREND:
        score -= 2
        reasons.append("the stock is in a downtrend")
    elif trend == TrendLabel.PULLBACK:
        score -= 0.5
        reasons.append("the stock is showing a pullback")
    else:
        reasons.append("price action is relatively sideways")

    if analysis.avg_change > 0.5:
        score += 1
        reasons.append(f"average daily change is positive ({analysis.avg_change:+.2f}%)")
    elif analysis.avg_change < -0.5:
        score -= 1
        reasons.append(f"average daily change is negative ({analysis.avg_change:+.2f}%)")

    if sentiment is not None and sentiment.confidence > 0:
        if sentiment.label == SentimentLabel.POSITIVE:
            score += 1.5
            reasons.append("market sentiment is positive")
        elif sentiment.label == SentimentLabel.NEGATIVE:
            score -= 1.5
            reasons.append("market sentiment is negative")
        else:
            reasons.append("market sentiment is neutral")

    if comparison is not None:
        rel = comparison.relative_perf_vs_sp500
        if rel > 1.0:
            score += 1
            reasons.append(f"it is outperforming the S&P 500 ({rel:+.2f}%)")
        elif rel < -1.0:
            score -= 1
            reasons.append(f"it is underperforming the S&P 500 ({rel:+.2f}%)")

    if (
        include_volatility
        and analysis.volatility is not None
        and analysis.volatility > 3.0
    ):
        score -= 0.5
        reasons.append(f"elevated volatility (±{analysis.volatility:.2f}%) suggests added caution")

    if score >= 2:
        action = "Buy"
    elif score <= -2:
        action = "Sell"
    else:
        action = "Hold"

    joined = ", ".join(reasons[:4]) if reasons else "the available signals are mixed"
    rationale = f"Based on {joined}, a **{action}** stance is suggested for consideration."
    return action, rationale


def _suggestion_section_text(action: str, rationale: str) -> str:
    return f"\n## Suggestion\n- Recommendation: {action}\n- Rationale: {rationale}\n"


def _suggestion_section_html(action: str, rationale: str) -> str:
    action_color = (
        "#3fb950" if action == "Buy"
        else "#f85149" if action == "Sell"
        else "#d29922"
    )
    plain_rationale = rationale.replace("**", "")
    return (
        f"  <h2 style='color:#58a6ff;font-size:16px'>Suggestion</h2>\n"
        f"  <table style='border-collapse:collapse;margin-bottom:16px'>\n"
        f"    <tr><td style='padding:4px 12px 4px 0;color:#8b949e'>Recommendation</td>"
        f"<td style='color:{action_color}'><strong>{action}</strong></td></tr>\n"
        f"  </table>\n"
        f"  <p style='color:#8b949e;font-size:13px'>{plain_rationale}</p>\n"
    )


def _finalize_email_bodies(html_body: str, text_body: str) -> ReportOutput:
    """Inject the disclaimer at the top and ensure greeting + sign-off are present."""
    # ── Plain text ─────────────────────────────────────────────────────────────
    if _GREETING not in text_body:
        text_body = f"{_GREETING}\n\n{text_body.lstrip()}"
    # Insert disclaimer right after the greeting line
    if _DISCLAIMER_TEXT not in text_body:
        text_body = text_body.replace(
            _GREETING,
            f"{_GREETING}\n{_DISCLAIMER_TEXT}",
            1,
        )
    if _SIGN_OFF_TEXT not in text_body:
        text_body = f"{text_body.rstrip()}\n\n{_SIGN_OFF_TEXT}\n"

    # ── HTML ───────────────────────────────────────────────────────────────────
    if _GREETING not in html_body:
        # Insert greeting right after the opening <body ...> tag
        body_end = html_body.index(">", html_body.index("<body")) + 1
        html_body = (
            html_body[:body_end]
            + f"\n  <p style='color:#c9d1d9;font-size:14px;margin-bottom:8px'>{_GREETING}</p>"
            + html_body[body_end:]
        )
    # Insert disclaimer right after the greeting paragraph
    if _DISCLAIMER_HTML not in html_body:
        greeting_tag = f">{_GREETING}</p>"
        disclaimer_para = (
            f"  <p style='font-size:20px;margin-bottom:16px'>{_DISCLAIMER_HTML}</p>"
            f"<p>{_GREETING}</p>\n"
        )
        html_body = html_body.replace(greeting_tag, disclaimer_para, 1)

    if "BeStock Agent" not in html_body:
        html_body = html_body.replace(
            "</body>",
            f"  <p style='color:#c9d1d9;font-size:14px;margin-top:24px'>{_SIGN_OFF_HTML}</p>\n</body>",
        )

    return ReportOutput(html_body=html_body, text_body=text_body)


# ── Template fallback ─────────────────────────────────────────────────────────


def _template_report(
    gainer: TopGainer,
    analysis: TrendAnalysis,
    analysis_date: str,
    advanced: bool = False,
    enable_volatility: bool = False,
    sentiment: SentimentResult | None = None,
    comparison: IndexComparison | None = None,
) -> ReportOutput:
    """Pure-Python fallback used when the LLM chain is unavailable."""
    table = format_price_table(analysis.bars, analysis.daily_changes)
    vol_str = f"±{analysis.volatility:.2f}%" if analysis.volatility else "N/A"

    trend_emoji = {"uptrend": "📈", "downtrend": "📉", "sideways": "➡️", "pullback": "↩️"}.get(
        analysis.trend_label.value, ""
    )

    sentiment_block = ""
    if sentiment is not None and sentiment.summary:
        label_cap = sentiment.label.value.capitalize()
        sentiment_block = (
            f"\nSentiment\n{'-' * 40}\n"
            f"Label      : {label_cap}\n"
            f"Score      : {sentiment.score:+.2f}  (confidence {sentiment.confidence:.0%})\n"
            f"Summary    : {sentiment.summary}\n"
        )

    index_block = ""
    if comparison is not None:
        beta_str = f"{comparison.beta:.2f}" if comparison.beta is not None else "N/A"
        index_block = (
            f"\nS&P 500 Comparison\n{'-' * 40}\n"
            f"{analysis.symbol} period return : {comparison.sp500.period_change_pct + comparison.relative_perf_vs_sp500:+.2f}%\n"
            f"S&P 500 period return         : {comparison.sp500.period_change_pct:+.2f}%\n"
            f"Relative performance          : {comparison.relative_perf_vs_sp500:+.2f}%\n"
            f"Beta                          : {beta_str}\n"
        )

    trend_summary = _trend_summary_for_report(analysis, enable_volatility)

    # Volatility is shown only when the Volatility advanced setting is enabled.
    vol_line = f"Daily Volatility     : {vol_str}\n" if enable_volatility else ""
    suggestion_block = ""
    if advanced:
        action, rationale = _generate_suggestion(
            analysis,
            sentiment,
            comparison,
            include_volatility=enable_volatility,
        )
        plain_rationale = rationale.replace("**", "")
        suggestion_block = (
            f"\nSuggestion\n{'-' * 40}\n"
            f"Recommendation : {action}\n"
            f"Rationale      : {plain_rationale}\n"
        )

    text_body = f"""\
{_GREETING}
{_DISCLAIMER_TEXT}

{_opening_line_text(gainer)}
Please find below the Analysis Report:
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
{vol_line}
{trend_summary}
{sentiment_block}{index_block}{suggestion_block}
{_SIGN_OFF_TEXT}
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

    # Optional advanced HTML sections
    html_sentiment_section = ""
    if sentiment is not None and sentiment.summary:
        sent_color = (
            "#3fb950" if sentiment.score > 0.1
            else "#f85149" if sentiment.score < -0.1
            else "#d29922"
        )
        sent_label = sentiment.label.value.capitalize()
        driver_rows = ""
        for d in sentiment.drivers[:4]:
            dir_color = "#3fb950" if d.sentiment.value == "positive" else "#f85149"
            driver_rows += f"<li style='margin:3px 0;color:{dir_color}'>{d.text}</li>"
        driver_html = f"<ul style='margin:6px 0 0 16px;padding:0'>{driver_rows}</ul>" if driver_rows else ""
        html_sentiment_section = (
            f"  <h2 style='color:#58a6ff;font-size:16px'>Market Sentiment</h2>\n"
            f"  <table style='border-collapse:collapse;margin-bottom:8px'>\n"
            f"    <tr><td style='padding:4px 12px 4px 0;color:#8b949e'>Label</td>"
            f"<td style='color:{sent_color}'><strong>{sent_label}</strong></td></tr>\n"
            f"    <tr><td style='padding:4px 12px 4px 0;color:#8b949e'>Score</td>"
            f"<td>{sentiment.score:+.2f} (confidence {sentiment.confidence:.0%})</td></tr>\n"
            f"  </table>\n"
            f"  <p style='color:#8b949e;font-size:13px'>{sentiment.summary}</p>\n"
            f"  {driver_html}\n"
        )

    html_index_section = ""
    if comparison is not None:
        beta_str = f"{comparison.beta:.2f}" if comparison.beta is not None else "N/A"
        rel_color = "#3fb950" if comparison.relative_perf_vs_sp500 >= 0 else "#f85149"
        stock_period = comparison.sp500.period_change_pct + comparison.relative_perf_vs_sp500
        html_index_section = (
            f"  <h2 style='color:#58a6ff;font-size:16px'>S&amp;P 500 Comparison</h2>\n"
            f"  <table style='border-collapse:collapse;margin-bottom:16px'>\n"
            f"    <tr><td style='padding:4px 12px 4px 0;color:#8b949e'>{comparison.stock_symbol} Period Return</td>"
            f"<td>{stock_period:+.2f}%</td></tr>\n"
            f"    <tr><td style='padding:4px 12px 4px 0;color:#8b949e'>S&amp;P 500 Period Return</td>"
            f"<td>{comparison.sp500.period_change_pct:+.2f}%</td></tr>\n"
            f"    <tr><td style='padding:4px 12px 4px 0;color:#8b949e'>Relative Performance</td>"
            f"<td style='color:{rel_color}'>{comparison.relative_perf_vs_sp500:+.2f}%</td></tr>\n"
            f"    <tr><td style='padding:4px 12px 4px 0;color:#8b949e'>Beta</td><td>{beta_str}</td></tr>\n"
            f"  </table>\n"
        )

    html_vol_row = (
        f"    <tr><td style='padding:4px 12px 4px 0;color:#8b949e'>Volatility</td><td>{vol_str}</td></tr>\n"
        if enable_volatility else ""
    )
    html_suggestion_section = ""
    if advanced:
        action, rationale = _generate_suggestion(
            analysis,
            sentiment,
            comparison,
            include_volatility=enable_volatility,
        )
        html_suggestion_section = _suggestion_section_html(action, rationale)

    html_body = f"""\
<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;background:#0d1117;color:#c9d1d9;padding:24px;max-width:680px;margin:auto">
  <p style="color:#c9d1d9;font-size:14px;margin-bottom:8px">{_GREETING}</p>
  <p style="color:#c9d1d9;font-size:13px;margin-bottom:16px">{_DISCLAIMER_HTML}</p>
  {_opening_line_html(gainer)}
  <h1 style="color:#58a6ff;font-size:20px">NASDAQ Stock Analysis Report on {gainer.name} ({gainer.symbol})</h1>
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
{html_vol_row}  </table>
  <p style="color:#8b949e;font-size:13px">{trend_summary}</p>
{html_sentiment_section}{html_index_section}{html_suggestion_section}  <p style="color:#c9d1d9;font-size:14px;margin-top:24px">{_SIGN_OFF_HTML}</p>
</body>
</html>"""

    return ReportOutput(html_body=html_body, text_body=text_body)


# ── Public interface ───────────────────────────────────────────────────────────


def compose_report(
    gainer: TopGainer,
    analysis: TrendAnalysis,
    analysis_date: str,
    openai_model: str,
    openai_api_key: str,
    advanced: bool = False,
    enable_volatility: bool = False,
    sentiment: SentimentResult | None = None,
    comparison: IndexComparison | None = None,
) -> ReportOutput:
    """Compose the report using the LLM chain with a template fallback.

    Volatility is included only when *enable_volatility* is True.
    The suggestion section is included only when *advanced* is True.
    """
    vol_str = f"±{analysis.volatility:.2f}%" if analysis.volatility else "N/A"
    trend_summary = _trend_summary_for_report(analysis, enable_volatility)

    volatility_line = (
        f"- Daily Volatility (Std Dev): {vol_str}\n" if enable_volatility else ""
    )
    suggestion_content = ""
    if advanced:
        action, rationale = _generate_suggestion(
            analysis,
            sentiment,
            comparison,
            include_volatility=enable_volatility,
        )
        suggestion_content = _suggestion_section_text(action, rationale)

    key_is_placeholder = not openai_api_key or openai_api_key.startswith("<")
    if not key_is_placeholder:
        try:
            chain = _build_chain(openai_model, openai_api_key)
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
                "volatility_line": volatility_line,
                "trend_summary": trend_summary,
                "sentiment_section": _sentiment_section_text(sentiment),
                "index_section": _index_section_text(comparison),
                "suggestion_content": suggestion_content,
            })
            return _finalize_email_bodies(result.html_body, result.text_body)
        except Exception:
            pass  # Fall through to template

    return _template_report(
        gainer,
        analysis,
        analysis_date,
        advanced=advanced,
        enable_volatility=enable_volatility,
        sentiment=sentiment,
        comparison=comparison,
    )
