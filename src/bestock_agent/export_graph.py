"""Export the compiled LangGraph topology to the outputs/ folder."""

from __future__ import annotations

import argparse
import base64
import re
import time
from pathlib import Path

import httpx

from bestock_agent.graph import build_graph

_DEFAULT_OUTPUT_DIR = Path("outputs")

# Visual hint in the diagram: Gradio pauses here for email confirmation.
_HITL_NODE = "send_email"
_HITL_LABEL = "send_email<br/>(Gradio interrupt)"


def _annotate_mermaid(mermaid: str) -> str:
    """Highlight the human-in-the-loop gate and document checkpointing."""
    annotated = mermaid.replace(
        f"{_HITL_NODE}({_HITL_NODE})",
        f'{_HITL_NODE}("{_HITL_LABEL}")',
    )
    if "classDef hitl" not in annotated:
        annotated = annotated.replace(
            "classDef last fill:#bfb6fc",
            "classDef hitl fill:#dbeafe,stroke:#2563eb,stroke-width:2px\n\tclassDef last fill:#bfb6fc",
        )
        annotated = re.sub(
            rf"(\t{_HITL_NODE}\(.+\))",
            rf"\1:::hitl",
            annotated,
            count=1,
        )
    note = (
        "\n\t%% Checkpointing: state persisted to SQLite after every node "
        "(.checkpoints/bestock.db). Gradio uses interrupt_before send_email.\n"
    )
    if note not in annotated:
        annotated = annotated.replace("graph TD;", f"graph TD;{note}")
    return annotated


def _render_mermaid_png(mermaid: str, *, max_retries: int = 3, retry_delay: float = 1.0) -> bytes:
    """Render Mermaid source to PNG via mermaid.ink."""
    encoded = base64.urlsafe_b64encode(mermaid.encode("utf-8")).decode().rstrip("=")
    url = f"https://mermaid.ink/img/{encoded}?type=png"
    last_err: Exception | None = None
    for _ in range(max_retries):
        try:
            response = httpx.get(url, timeout=30.0, follow_redirects=True)
            response.raise_for_status()
            return response.content
        except Exception as exc:  # noqa: BLE001 — retry on any render failure
            last_err = exc
            time.sleep(retry_delay)
    raise RuntimeError(f"Failed to render Mermaid PNG: {last_err}") from last_err


def export_graph(output_dir: Path | str = _DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    """Write Mermaid source and PNG diagram for the compiled agent graph.

    Returns a dict with ``mermaid`` and ``png`` paths. PNG rendering uses the
    mermaid.ink API (requires network access).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    graph = build_graph().compile().get_graph()
    mermaid_path = out / "bestock_graph.mmd"
    png_path = out / "bestock_graph.png"

    mermaid = _annotate_mermaid(graph.draw_mermaid())
    mermaid_path.write_text(mermaid, encoding="utf-8")
    png_path.write_bytes(_render_mermaid_png(mermaid))

    return {"mermaid": mermaid_path, "png": png_path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the BeStock LangGraph diagram (Mermaid + PNG)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory for exported files (default: outputs)",
    )
    args = parser.parse_args()
    paths = export_graph(args.output_dir)
    print(f"Wrote {paths['mermaid']}")
    print(f"Wrote {paths['png']}")


if __name__ == "__main__":
    main()
