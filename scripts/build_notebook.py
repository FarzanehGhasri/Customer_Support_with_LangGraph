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
import json
import re
import subprocess
import sys
from pathlib import Path

#: Real git conflict markers sit at the very start of a line and have an exact
#: shape. Matching loosely would fire on ordinary output -- this notebook prints
#: banners made of "=" characters, which a naive "=======" search flags as a
#: conflict. Anchoring to the line start and pinning the length avoids that:
#: inside a .ipynb every output line is quoted and indented, so only markers
#: git itself inserted begin at column 0.
_CONFLICT_RE = re.compile(r"^(<{7} |={7}$|>{7} |\|{7} )", re.MULTILINE)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "customer_support_langgraph.ipynb"


def preflight(path: Path) -> int:
    """Fail early, and usefully, if the file is not a readable notebook.

    nbconvert's own failure for a damaged file is a forty-line traceback ending
    in a JSONDecodeError, which says nothing about the cause or the cure. The
    overwhelmingly common cause is a git conflict: the notebook is committed
    with its outputs, so running it locally always dirties the file and the next
    pull collides. Diagnose that here and say exactly what to type.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        print(f"FAILED: {path.name} is not UTF-8 text ({exc}).")
        print("  Restore it with:  git checkout -- " + _repo_relative(path))
        return 1

    conflicts = [m.start() for m in _CONFLICT_RE.finditer(raw)]
    if conflicts:
        line = raw.count("\n", 0, conflicts[0]) + 1
        print(f"FAILED: {path.name} contains git conflict markers (first at line {line}).")
        print()
        print("  This happens because the notebook is committed WITH its outputs, so")
        print("  running it locally changes the file and the next `git pull` collides.")
        print()
        print("  Take the committed version and start again:")
        print(f"      git checkout -- {_repo_relative(path)}")
        print("  If a merge is still in progress, abort it first:")
        print("      git merge --abort")
        print("  Or force the remote copy regardless of merge state:")
        print("      git fetch origin")
        print(f"      git checkout origin/<branch> -- {_repo_relative(path)}")
        return 1

    try:
        json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"FAILED: {path.name} is not valid JSON -- {exc.msg} at line {exc.lineno}.")
        print()
        print("  The offending line:")
        lines = raw.splitlines()
        for number in range(max(1, exc.lineno - 2), min(len(lines), exc.lineno + 1) + 1):
            marker = ">>" if number == exc.lineno else "  "
            print(f"    {marker} {number:5} {lines[number - 1][:100]}")
        print()
        print("  A notebook is a JSON file; something has damaged it. Restore it with:")
        print(f"      git checkout -- {_repo_relative(path)}")
        return 1

    return 0


def _repo_relative(path: Path) -> str:
    """Path as you would type it from the repo root, with forward slashes."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


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

    # Check the file is readable before handing it to nbconvert, whose failure
    # for a damaged notebook is an unhelpful traceback.
    status = preflight(path)
    if status:
        return status

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
