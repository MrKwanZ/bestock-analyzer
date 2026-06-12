"""Export the compiled LangGraph topology to the outputs/ folder."""

from __future__ import annotations

import argparse
from pathlib import Path

from bestock_agent.graph import app

_DEFAULT_OUTPUT_DIR = Path("outputs")


def export_graph(output_dir: Path | str = _DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    """Write Mermaid source and PNG diagram for the compiled agent graph.

    Returns a dict with ``mermaid`` and ``png`` paths. PNG rendering uses the
    mermaid.ink API (requires network access).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    graph = app.get_graph()
    mermaid_path = out / "bestock_graph.mmd"
    png_path = out / "bestock_graph.png"

    mermaid_path.write_text(graph.draw_mermaid(), encoding="utf-8")
    png_path.write_bytes(graph.draw_mermaid_png(max_retries=3, retry_delay=1.0))

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
