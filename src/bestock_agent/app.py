"""Gradio web UI for the NASDAQ BeStock Analyzer — Phase 6."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import date

import gradio as gr

from bestock_agent.config import get_settings
from bestock_agent.graph import app as agent_app
from bestock_agent.services.validation import ValidationError, validate_email
from bestock_agent.state import initial_state

# ── Constants ─────────────────────────────────────────────────────────────────

_TREND_EMOJI = {
    "uptrend": "📈",
    "downtrend": "📉",
    "sideways": "➡️",
    "pullback": "↩️",
}

_LOOKBACK_MIN = 3
_LOOKBACK_MAX = 30

_PROGRESS_HTML = """
<div class="bs-progress-wrap">
  <p class="bs-progress-label">Analysis in progress…</p>
</div>
"""

# ── Minimal CSS — only things Gradio's default theme cannot provide ───────────

_CSS = """
/* Unified font across the whole app */
*, *::before, *::after,
body, .gradio-container, button, input, textarea, select,
label, p, span, h1, h2, h3, h4, h5, h6, li {
    font-family: Arial, sans-serif !important;
}

/* Progress status */
.bs-progress-wrap { width:100%; padding:16px 0; }
.bs-progress-label { font-size:14px; margin:0; text-align:center; }

/* Hide Gradio's built-in loading overlay on the status panel */
#bs-status-panel-wrapper .generating,
#bs-status-panel-wrapper .progress-level,
#bs-status-panel-wrapper .eta-bar,
#bs-status-panel-wrapper .loader {
    display:none !important;
}

/* Result output cards */
.bs-results { font-family:Arial,sans-serif; width:100%; }
.bs-result-card {
    border:1px solid #e5e7eb; border-radius:10px;
    padding:18px 22px; margin-bottom:12px;
    width:100%; box-sizing:border-box;
}
.dark .bs-result-card { border-color:#374151; }

.bs-section-label {
    font-size:11px; text-transform:uppercase;
    letter-spacing:.06em; margin:0 0 8px;
    color:#6b7280;
}
.dark .bs-section-label { color:#9ca3af; }

.bs-stock-header { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; margin-bottom:12px; }
.bs-stock-name  { font-size:24px; font-weight:700; }
.bs-stock-badge {
    font-size:13px; font-weight:600;
    background:#f3f4f6; padding:2px 8px; border-radius:4px;
}
.dark .bs-stock-badge { background:#374151; }

.bs-metrics { display:flex; gap:24px; flex-wrap:wrap; }
.bs-metric-label { font-size:11px; color:#6b7280; margin:0; }
.dark .bs-metric-label { color:#9ca3af; }
.bs-metric-value { font-size:20px; font-weight:600; margin:2px 0; }

.bs-pos { color:#16a34a; } .dark .bs-pos { color:#4ade80; }
.bs-neg { color:#dc2626; } .dark .bs-neg { color:#f87171; }
.bs-neu { color:#b45309; } .dark .bs-neu { color:#fbbf24; }

.bs-summary { font-size:13px; color:#6b7280; margin:10px 0 0; }
.dark .bs-summary { color:#9ca3af; }

.bs-email-ok {
    border:1px solid #16a34a; border-radius:8px;
    padding:12px 16px; display:flex; align-items:center; gap:10px;
    color:#15803d; font-size:14px; background:#f0fdf4;
}
.dark .bs-email-ok { background:#052e16; color:#4ade80; border-color:#166534; }
.bs-email-warn {
    border:1px solid #b45309; border-radius:8px;
    padding:12px 16px; display:flex; align-items:center; gap:10px;
    color:#92400e; font-size:14px; background:#fffbeb;
}
.dark .bs-email-warn { background:#1c1003; color:#fbbf24; border-color:#92400e; }
.bs-run-error {
    border:1px solid #dc2626; border-radius:8px; padding:16px;
    color:#dc2626; font-size:14px; background:#fef2f2;
}
.dark .bs-run-error { background:#1a0000; color:#f87171; border-color:#991b1b; }

.bs-email-pending {
    border:1px solid #2563eb; border-radius:8px;
    padding:12px 16px; display:flex; align-items:center; gap:10px;
    color:#1d4ed8; font-size:14px; background:#eff6ff;
}
.dark .bs-email-pending { background:#0c1929; color:#60a5fa; border-color:#1d4ed8; }

.bs-email-cancelled {
    border:1px solid #6b7280; border-radius:8px;
    padding:12px 16px; display:flex; align-items:center; gap:10px;
    color:#4b5563; font-size:14px; background:#f9fafb;
}
.dark .bs-email-cancelled { background:#1f2937; color:#9ca3af; border-color:#4b5563; }

/* Email confirmation panel */
#bs-confirm-panel {
    border:1px solid #e5e7eb; border-radius:10px;
    padding:16px 20px; margin-top:8px; width:100%; box-sizing:border-box;
}
.dark #bs-confirm-panel { border-color:#374151; }

/* "How to use" section padding */
.bs-instructions { padding:5px; }
"""


# ── Result HTML helpers ───────────────────────────────────────────────────────

def _card(content: str) -> str:
    return f"<div class='bs-result-card'>{content}</div>"


def _section_label(title: str) -> str:
    return f"<p class='bs-section-label'>{title}</p>"


def _email_status_html(
    recipient: str,
    *,
    sent: bool = False,
    pending: bool = False,
    cancelled: bool = False,
) -> str:
    if cancelled:
        return (
            "<div class='bs-email-cancelled'><span>🚫</span>"
            "<span>Email sending was cancelled.</span></div>"
        )
    if pending:
        return (
            f"<div class='bs-email-pending'><span>📧</span>"
            f"<span>Report ready — confirm below to send to <strong>{recipient}</strong>.</span></div>"
        )
    if sent:
        return (
            f"<div class='bs-email-ok'><span>✅</span>"
            f"<span>Analysis report sent to <strong>{recipient}</strong></span></div>"
        )
    return (
        "<div class='bs-email-warn'><span>⚠️</span>"
        "<span>Email could not be sent — check SMTP credentials in .env</span></div>"
    )


def _build_result_html(
    final: dict,
    show_advanced: bool,
    show_sentiment: bool,
    show_index: bool,
    show_volatility: bool,
    *,
    email_cancelled: bool = False,
) -> str:
    gainer    = final.get("top_gainer")
    analysis  = final.get("trend_analysis")
    sentiment = final.get("sentiment_result")
    comparison = final.get("index_comparison")
    run_summary = final.get("run_summary")

    # ── Stock card ─────────────────────────────────────────────────────────────
    pct_cls    = "bs-pos" if gainer.change_pct >= 0 else "bs-neg"
    change_sign = "+" if gainer.change_pct >= 0 else ""
    stock_html = (
        f"<div class='bs-stock-header'>"
        f"<span class='bs-stock-name'>{gainer.name}</span>"
        f"<span class='bs-stock-badge'>{gainer.symbol}</span>"
        f"</div>"
        f"<div class='bs-metrics'>"
        f"<div><p class='bs-metric-label'>CLOSE PRICE</p>"
        f"<p class='bs-metric-value'>${gainer.price:,.2f}</p></div>"
        f"<div><p class='bs-metric-label'>TODAY'S CHANGE</p>"
        f"<p class='bs-metric-value {pct_cls}'>{change_sign}{gainer.change_pct:.2f}%</p></div>"
        f"</div>"
    )

    sections = ""

    # ── Performance card ───────────────────────────────────────────────────────
    if analysis:
        emoji   = _TREND_EMOJI.get(analysis.trend_label.value, "")
        vol_str = f"±{analysis.volatility:.2f}%" if analysis.volatility else "—"
        metrics = (
            f"<div class='bs-metrics'>"
            f"<div><p class='bs-metric-label'>TREND</p>"
            f"<p class='bs-metric-value'>{emoji} {analysis.trend_label.value.title()}</p></div>"
            f"<div><p class='bs-metric-label'>AVG DAILY CHANGE</p>"
            f"<p class='bs-metric-value'>{'+' if analysis.avg_change >= 0 else ''}{analysis.avg_change:.2f}%</p></div>"
        )
        # Volatility only shown when its toggle is enabled
        if show_advanced and show_volatility:
            metrics += (
                f"<div><p class='bs-metric-label'>VOLATILITY</p>"
                f"<p class='bs-metric-value'>{vol_str}</p></div>"
            )
        metrics += f"</div><p class='bs-summary'>{analysis.summary}</p>"
        sections += _card(_section_label(f"{len(analysis.bars)}-Day Performance Summary") + metrics)

    # ── Sentiment card ─────────────────────────────────────────────────────────
    if show_advanced and show_sentiment and sentiment:
        s_cls = (
            "bs-pos" if sentiment.score > 0.1
            else "bs-neg" if sentiment.score < -0.1
            else "bs-neu"
        )
        sent_html = (
            f"<div class='bs-metrics'>"
            f"<div><p class='bs-metric-label'>LABEL</p>"
            f"<p class='bs-metric-value {s_cls}'>{sentiment.label.value.capitalize()}</p></div>"
            f"<div><p class='bs-metric-label'>SCORE</p>"
            f"<p class='bs-metric-value {s_cls}'>{'+' if sentiment.score >= 0 else ''}{sentiment.score:.2f}</p></div>"
            f"<div><p class='bs-metric-label'>CONFIDENCE</p>"
            f"<p class='bs-metric-value'>{sentiment.confidence:.0%}</p></div>"
            f"</div>"
            f"<p class='bs-summary'>{sentiment.summary}</p>"
        )
        sections += _card(_section_label("Market Sentiment") + sent_html)

    # ── S&P 500 comparison card ────────────────────────────────────────────────
    if show_advanced and show_index and comparison:
        stock_period = comparison.sp500.period_change_pct + comparison.relative_perf_vs_sp500
        rel          = comparison.relative_perf_vs_sp500
        rel_cls      = "bs-pos" if rel >= 0 else "bs-neg"
        beta_str     = f"{comparison.beta:.2f}" if comparison.beta is not None else "N/A"
        idx_html = (
            f"<div class='bs-metrics'>"
            f"<div><p class='bs-metric-label'>{gainer.symbol} PERIOD RETURN</p>"
            f"<p class='bs-metric-value'>{'+' if stock_period >= 0 else ''}{stock_period:.2f}%</p></div>"
            f"<div><p class='bs-metric-label'>S&amp;P 500 PERIOD RETURN</p>"
            f"<p class='bs-metric-value'>{'+' if comparison.sp500.period_change_pct >= 0 else ''}{comparison.sp500.period_change_pct:.2f}%</p></div>"
            f"<div><p class='bs-metric-label'>RELATIVE PERFORMANCE</p>"
            f"<p class='bs-metric-value {rel_cls}'>{'+' if rel >= 0 else ''}{rel:.2f}%</p></div>"
            f"<div><p class='bs-metric-label'>BETA</p>"
            f"<p class='bs-metric-value'>{beta_str}</p></div>"
            f"</div>"
        )
        sections += _card(_section_label("S&P 500 Comparison") + idx_html)

    # ── Email notification ─────────────────────────────────────────────────────
    email_html = ""
    if run_summary:
        payload = final.get("email_payload")
        addr = (
            str(payload.recipient) if payload
            else run_summary.email_recipient or final.get("recipient_email", "—")
        )
        if email_cancelled:
            email_html = _email_status_html(addr, cancelled=True)
        elif run_summary.email_sent:
            email_html = _email_status_html(addr, sent=True)
        elif "awaiting email confirmation" in run_summary.message.lower():
            email_html = _email_status_html(addr, pending=True)
        else:
            email_html = _email_status_html(addr, sent=False)

    return (
        "<div class='bs-results'>"
        + _card(stock_html)
        + sections
        + email_html
        + "</div>"
    )


def _error_html(message: str) -> str:
    return (
        f"<div class='bs-run-error'>"
        f"<strong>❌ Run failed</strong><br><span>{message}</span></div>"
    )


# ── Validation helpers ────────────────────────────────────────────────────────

def _validate_lookback(value, *, required: bool = False) -> tuple[int | None, str]:
    """Return (int_value, error_message). error_message is '' if valid."""
    if value is None or value == "":
        if required:
            return None, "Lookback days is required when Advanced Settings is enabled."
        return None, "Lookback days must be a whole number."
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None, "Lookback days must be a whole number."
    if not (_LOOKBACK_MIN <= v <= _LOOKBACK_MAX):
        return None, f"Lookback days must be between {_LOOKBACK_MIN} and {_LOOKBACK_MAX}."
    return v, ""


def _validate_inputs(
    recipient_input: str,
    show_advanced: bool,
    lookback_raw,
) -> str:
    """Return a user-facing error message, or '' if all inputs are valid."""
    recipient = recipient_input.strip()
    if not recipient:
        return "Email Recipient is required."

    try:
        validate_email(recipient)
    except ValidationError as exc:
        return f"Invalid Email Recipient — {exc.message}"

    if show_advanced:
        _, err = _validate_lookback(lookback_raw, required=True)
        if err:
            return f"Invalid Lookback Days — {err}"

    return ""


# ── UI context helpers ────────────────────────────────────────────────────────

def _ui_context(
    show_advanced: bool,
    enable_sentiment: bool,
    enable_index: bool,
    enable_volatility: bool,
) -> dict:
    return {
        "show_advanced": show_advanced,
        "enable_sentiment": enable_sentiment,
        "enable_index": enable_index,
        "enable_volatility": enable_volatility,
    }


def _result_from_state(final: dict, ctx: dict, *, email_cancelled: bool = False) -> str:
    return _build_result_html(
        final,
        ctx["show_advanced"],
        ctx["enable_sentiment"],
        ctx["enable_index"],
        ctx["enable_volatility"],
        email_cancelled=email_cancelled,
    )


def _hide_confirm() -> dict:
    return gr.update(visible=False)


def _show_confirm(recipient: str) -> tuple[dict, str]:
    return (
        gr.update(visible=True),
        f"Send the analysis report to **{recipient}**?",
    )


# ── Agent runner ──────────────────────────────────────────────────────────────

async def _run_analysis(
    recipient_input: str,
    show_advanced: bool,
    lookback_raw,
    enable_sentiment: bool,
    enable_index: bool,
    enable_volatility: bool,
) -> AsyncGenerator[tuple, None]:
    """Validate, run analysis (without sending email), then prompt for confirmation."""
    err = _validate_inputs(recipient_input, show_advanced, lookback_raw)
    if err:
        yield _error_html(err), _hide_confirm(), "", None, {}
        return

    yield _PROGRESS_HTML, _hide_confirm(), "", None, {}

    settings  = get_settings()
    recipient = recipient_input.strip()

    if show_advanced:
        days, err = _validate_lookback(lookback_raw, required=True)
        if err:
            yield _error_html(f"Invalid Lookback Days — {err}"), _hide_confirm(), "", None, {}
            return
    else:
        days = settings.default_lookback_days

    advanced = show_advanced and (enable_sentiment or enable_index or enable_volatility)
    ctx = _ui_context(show_advanced, enable_sentiment, enable_index, enable_volatility)

    state = initial_state(
        recipient_email=recipient,
        target_date=str(date.today()),
        lookback_days=days,
        advanced_analysis_enabled=advanced,
        enable_volatility=show_advanced and enable_volatility,
        skip_email=True,
    )
    state["active_financial_provider"] = "alphavantage"

    final       = await agent_app.ainvoke(state)
    run_summary = final.get("run_summary")
    gainer      = final.get("top_gainer")

    if not gainer or not run_summary or not run_summary.success:
        errors = final.get("errors", [])
        msg = errors[-1].message if errors else (
            run_summary.message if run_summary else "Unknown error"
        )
        yield _error_html(msg), _hide_confirm(), "", None, {}
        return

    confirm_panel, confirm_text = _show_confirm(recipient)
    yield (
        _result_from_state(final, ctx),
        confirm_panel,
        confirm_text,
        final,
        ctx,
    )


async def _confirm_send_email(pending_state: dict | None, ctx: dict):
    """User confirmed — send the prepared email."""
    if not pending_state:
        return _error_html("No analysis to send."), _hide_confirm(), "", None, {}

    from bestock_agent.nodes.send_email import send_email

    state = dict(pending_state)
    state["skip_email"] = False
    send_result = await send_email(state)
    merged = {**pending_state, **send_result}

    return (
        _result_from_state(merged, ctx),
        _hide_confirm(),
        "",
        None,
        ctx,
    )


async def _cancel_send_email(pending_state: dict | None, ctx: dict):
    """User declined — keep results but do not send email."""
    if not pending_state:
        return "", _hide_confirm(), "", None, {}

    return (
        _result_from_state(pending_state, ctx, email_cancelled=True),
        _hide_confirm(),
        "",
        None,
        ctx,
    )


def _toggle_advanced(checked: bool):
    return gr.update(visible=checked)


def _reset(default_days: int):
    return (
        "",       # recipient_input
        False,    # adv_toggle
        default_days,   # lookback_days
        False,    # enable_sentiment
        False,    # enable_index
        False,    # enable_volatility
        gr.update(visible=False),   # adv_panel
        "",       # status_panel
        _hide_confirm(),  # confirm_panel
        "",       # confirm_text
        None,     # pending_state
        {},       # ui_context
    )


# ── Instructions ──────────────────────────────────────────────────────────────

def _instructions_md() -> str:
    return f"""\
### How to use

**Email Recipient** — enter a valid email address to receive analysis report.

**Start** — Start the agent to find today's top NASDAQ gainer and analyse its performance. When analysis completes, confirm with **Yes** or **No** whether to send the report to the email recipient.

**Advanced Settings** — tick the checkbox to reveal extra options:
- **Lookback Days** — required number of past trading days included in the price history ({_LOOKBACK_MIN} – {_LOOKBACK_MAX}).
- **Sentiment Analysis** — Classify market sentiment from recent news headlines.
- **S&P 500 Comparison** — compares the stock's performance against the S&P 500 index.
- **Volatility** — show daily price volatility (standard deviation) in the performance summary.

**Reset** — clears all inputs and the results panel.
"""


# ── UI layout ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    settings = get_settings()
    default_days = settings.default_lookback_days

    with gr.Blocks(title="NASDAQ BeStock Analyzer") as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.Markdown("# 📈 NASDAQ BeStock Analyzer")
        gr.Markdown(
            "Finds you the best NASDAQ stock of the day, "
            "analyses its performance, and delivers a report to your inbox."
        )

        # ── Inputs ────────────────────────────────────────────────────────────
        recipient_input = gr.Textbox(
            label="Email Recipient",
            lines=1,
        )

        adv_toggle = gr.Checkbox(label="Advanced Settings", value=False)

        with gr.Column(visible=False) as adv_panel:
            with gr.Row():
                lookback_days = gr.Number(
                    label="Lookback Days",
                    value=default_days,
                    minimum=_LOOKBACK_MIN,
                    maximum=_LOOKBACK_MAX,
                    precision=0,
                    info=f"Required. Number of past trading days ({_LOOKBACK_MIN}–{_LOOKBACK_MAX}).",
                )
                enable_sentiment = gr.Checkbox(
                    label="Sentiment Analysis",
                    value=False,
                    info="Classify market sentiment from recent news headlines.",
                )
                enable_index = gr.Checkbox(
                    label="S&P 500 Comparison",
                    value=False,
                    info="Compare performance against the S&P 500 index.",
                )
                enable_volatility = gr.Checkbox(
                    label="Volatility",
                    value=False,
                    info="Show daily price volatility (std dev) in results.",
                )

        # ── Buttons ───────────────────────────────────────────────────────────
        with gr.Row():
            start_btn = gr.Button("▶  Start", variant="primary")
            reset_btn = gr.Button("↺  Reset", variant="secondary")

        # ── Status / result panel ──────────────────────────────────────────────
        with gr.Column(elem_id="bs-status-panel-wrapper"):
            status_panel = gr.HTML(value="", elem_id="bs-status-panel")

        # ── Email confirmation (shown after analysis completes) ───────────────
        pending_state = gr.State(None)
        ui_context = gr.State({})

        with gr.Column(visible=False, elem_id="bs-confirm-panel") as confirm_panel:
            confirm_text = gr.Markdown("")
            with gr.Row():
                confirm_yes = gr.Button("Yes", variant="primary")
                confirm_no = gr.Button("No", variant="secondary")

        # ── Instructions ──────────────────────────────────────────────────────
        with gr.Group(elem_classes="bs-instructions"):
            gr.Markdown(_instructions_md())

        # ── Wiring ────────────────────────────────────────────────────────────

        adv_toggle.change(
            fn=_toggle_advanced,
            inputs=[adv_toggle],
            outputs=[adv_panel],
        )

        start_btn.click(
            fn=_run_analysis,
            inputs=[
                recipient_input, adv_toggle, lookback_days,
                enable_sentiment, enable_index, enable_volatility,
            ],
            outputs=[
                status_panel,
                confirm_panel,
                confirm_text,
                pending_state,
                ui_context,
            ],
            show_progress="hidden",
        )

        confirm_yes.click(
            fn=_confirm_send_email,
            inputs=[pending_state, ui_context],
            outputs=[
                status_panel,
                confirm_panel,
                confirm_text,
                pending_state,
                ui_context,
            ],
            show_progress="hidden",
        )

        confirm_no.click(
            fn=_cancel_send_email,
            inputs=[pending_state, ui_context],
            outputs=[
                status_panel,
                confirm_panel,
                confirm_text,
                pending_state,
                ui_context,
            ],
        )

        reset_btn.click(
            fn=lambda: _reset(default_days),
            inputs=[],
            outputs=[
                recipient_input,
                adv_toggle,
                lookback_days,
                enable_sentiment,
                enable_index,
                enable_volatility,
                adv_panel,
                status_panel,
                confirm_panel,
                confirm_text,
                pending_state,
                ui_context,
            ],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False, css=_CSS)


if __name__ == "__main__":
    main()
