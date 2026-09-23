#!/usr/bin/env python
"""
Render the support graph to ``docs/images/support_graph.png``.

The assignment requires the graph image produced by ``draw_mermaid_png()``.
That call renders through the public mermaid.ink service, so it needs outbound
internet access; run this on a machine that has it.

    python scripts/render_graph.py

Three outputs are written, best first:
  * support_graph.png  -- the deliverable (needs network, or a local pyppeteer)
  * support_graph.mmd  -- the Mermaid source (always written, no network)
  * support_graph.txt  -- an ASCII rendering (needs `pip install grandalf`)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from support_system.graph import build_application  # noqa: E402

OUT = ROOT / "docs" / "images"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # force_offline: drawing the graph must never depend on credentials.
    graph = build_application(force_offline=True).graph.get_graph()

    (OUT / "support_graph.mmd").write_text(graph.draw_mermaid(), encoding="utf-8")
    print(f"wrote {OUT / 'support_graph.mmd'}")

    try:
        from langchain_core.runnables.graph import MermaidDrawMethod

        png = graph.draw_mermaid_png(draw_method=MermaidDrawMethod.API)
        (OUT / "support_graph.png").write_bytes(png)
        print(f"wrote {OUT / 'support_graph.png'} ({len(png)} bytes)")
    except Exception as exc:  # noqa: BLE001
        print(f"PNG not rendered: {type(exc).__name__}: {str(exc)[:160]}")
        print("  -> mermaid.ink needs outbound internet access.")
        print("  -> offline alternative: pip install pyppeteer, then re-run.")

    try:
        (OUT / "support_graph.txt").write_text(graph.draw_ascii(), encoding="utf-8")
        print(f"wrote {OUT / 'support_graph.txt'}")
    except Exception as exc:  # noqa: BLE001
        print(f"ASCII not rendered ({type(exc).__name__}); pip install grandalf")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
