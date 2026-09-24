#!/usr/bin/env python
"""
Re-run the deliverable notebook and save it with fresh outputs.

    python scripts/build_notebook.py              # execute in place
    python scripts/build_notebook.py --html       # also export an .html copy
    python scripts/build_notebook.py --no-execute # export only, keep outputs as they are

Why a script rather than "just run the cells": the deliverable must be an
``.ipynb`` whose outputs are *stored in the file*, so a grader sees the results
without running anything. Executing it from the command line guarantees that
every cell ran, in order, in one clean kernel -- which clicking through a UI
does not.

Exit code is non-zero if any cell raised, so this can be used as a check.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "customer_support_langgraph.ipynb"


def execute(path: Path, timeout: int) -> None:
    print(f"executing {path.name} ...")
    subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            f"--ExecutePreprocessor.timeout={timeout}",
            str(path),
        ],
        check=True,
        cwd=path.parent,
    )


def export(path: Path, fmt: str) -> None:
    print(f"exporting {fmt} ...")
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", fmt, str(path)],
        check=True,
        cwd=path.parent,
    )


def verify(path: Path) -> int:
    """Confirm the stored notebook really carries outputs and no errors."""
    import nbformat

    nb = nbformat.read(path, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    with_output = [c for c in code_cells if c.get("outputs")]
    errors = [
        o for c in code_cells for o in c.get("outputs", [])
        if o.get("output_type") == "error"
    ]

    print(f"\n  cells          : {len(nb.cells)} ({len(code_cells)} code)")
    print(f"  with outputs   : {len(with_output)}/{len(code_cells)}")
    print(f"  execution errors: {len(errors)}")
    for error in errors:
        print(f"    !! {error['ename']}: {error['evalue'][:120]}")

    if errors:
        print("\nFAILED: the notebook contains execution errors.")
        return 1
    if not with_output:
        print("\nFAILED: no outputs were stored -- the notebook did not run.")
        return 1
    print(f"\nOK: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    parser.add_argument("--no-execute", action="store_true",
                        help="skip running the cells; only verify and export")
    parser.add_argument("--html", action="store_true", help="also write an .html copy")
    parser.add_argument("--pdf", action="store_true",
                        help="also write a .pdf copy (needs a LaTeX install)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="per-cell timeout in seconds (default 600)")
    args = parser.parse_args()

    path = args.notebook.resolve()
    if not path.is_file():
        print(f"No such notebook: {path}")
        return 1

    if not args.no_execute:
        try:
            execute(path, args.timeout)
        except subprocess.CalledProcessError as exc:
            print(f"\nnbconvert failed with exit code {exc.returncode}.")
            print("The notebook was left as it was; fix the failing cell and re-run.")
            return exc.returncode

    status = verify(path)
    if status:
        return status

    for wanted, fmt in ((args.html, "html"), (args.pdf, "pdf")):
        if not wanted:
            continue
        try:
            export(path, fmt)
            print(f"  wrote {path.with_suffix('.' + fmt)}")
        except subprocess.CalledProcessError as exc:
            print(f"  {fmt} export failed (exit {exc.returncode}).")
            if fmt == "pdf":
                print("  PDF export needs LaTeX. Easier: export HTML and print it to PDF.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
