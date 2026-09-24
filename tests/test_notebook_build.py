"""
Tests for the notebook build script's preflight check.

The check exists because nbconvert's failure on a damaged notebook is a long
traceback that names neither the cause nor the cure. These tests pin the two
things that matter: real damage is caught with an actionable message, and the
healthy notebook is not flagged by mistake.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from support_system.config import PROJECT_ROOT

NOTEBOOK = PROJECT_ROOT / "notebooks" / "customer_support_langgraph.ipynb"


def _load_script():
    """Import scripts/build_notebook.py, which is not on the import path."""
    path = PROJECT_ROOT / "scripts" / "build_notebook.py"
    spec = importlib.util.spec_from_file_location("build_notebook", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


@pytest.fixture(scope="module")
def good_source() -> str:
    return NOTEBOOK.read_text(encoding="utf-8")


def test_the_committed_notebook_passes_preflight(script):
    assert script.preflight(NOTEBOOK) == 0


def test_banner_output_is_not_mistaken_for_a_conflict(script, good_source):
    """Regression: the notebook prints '=' banners, and a naive search for
    '=======' flagged the healthy file as conflicted."""
    assert "=" * 20 in good_source          # the banners really are in there
    assert script._CONFLICT_RE.search(good_source) is None


@pytest.mark.parametrize(
    "marker",
    ["<<<<<<< HEAD", "=======", ">>>>>>> origin/main", "||||||| merged common ancestors"],
)
def test_every_conflict_marker_shape_is_detected(script, good_source, tmp_path, capsys, marker):
    lines = good_source.splitlines(keepends=True)
    lines.insert(39, marker + "\n")
    damaged = tmp_path / "conflicted.ipynb"
    damaged.write_text("".join(lines), encoding="utf-8")

    assert script.preflight(damaged) == 1
    out = capsys.readouterr().out
    assert "conflict markers" in out
    assert "line 40" in out
    assert "git checkout --" in out          # tells the reader what to type


def test_invalid_json_is_reported_with_the_offending_line(script, good_source, tmp_path, capsys):
    lines = good_source.splitlines(keepends=True)
    lines[39] = "  metadata: {\n"            # unquoted key
    damaged = tmp_path / "broken.ipynb"
    damaged.write_text("".join(lines), encoding="utf-8")

    assert script.preflight(damaged) == 1
    out = capsys.readouterr().out
    assert "not valid JSON" in out
    assert "line 40" in out
    assert "metadata: {" in out              # shows the actual bad line


def test_a_non_utf8_file_is_reported(script, tmp_path, capsys):
    damaged = tmp_path / "binary.ipynb"
    damaged.write_bytes(b"\xff\xfe\x00not a notebook")
    assert script.preflight(damaged) == 1
    assert "not UTF-8" in capsys.readouterr().out


def test_verify_requires_stored_outputs(script, tmp_path, capsys):
    """A notebook with no outputs is not a deliverable."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
    stripped = tmp_path / "stripped.ipynb"
    stripped.write_text(json.dumps(nb), encoding="utf-8")

    assert script.verify(stripped) == 1
    assert "no outputs were stored" in capsys.readouterr().out


def test_verify_rejects_a_notebook_containing_an_error(script, tmp_path, capsys):
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = [{
                "output_type": "error", "ename": "ValueError",
                "evalue": "boom", "traceback": [],
            }]
            break
    failed = tmp_path / "errored.ipynb"
    failed.write_text(json.dumps(nb), encoding="utf-8")

    assert script.verify(failed) == 1
    assert "execution errors" in capsys.readouterr().out


def test_verify_accepts_the_committed_notebook(script):
    assert script.verify(NOTEBOOK) == 0
