"""
Minimal ``.env`` loader.

Why not ``python-dotenv``?  One fewer dependency to install behind a proxy, and
the file format we need is three lines of parsing.  It lives in the config
package because configuration loading is that package's single responsibility.

Rules (deliberately the same as python-dotenv's common subset):
* ``KEY=value`` per line;
* ``#`` starts a comment, blank lines are ignored;
* surrounding single/double quotes are stripped;
* **existing environment variables win** -- a real exported variable is never
  silently overwritten by the file.

The file is read as ``utf-8-sig``, not ``utf-8``. Windows PowerShell 5.1 writes
a byte-order mark when told ``-Encoding utf8``, and that BOM would otherwise
become part of the first key: the variable would be named ``\ufeffOPENAI_API_KEY``
and the real one would look unset, for no visible reason.
"""

from __future__ import annotations

import os
from pathlib import Path

from .settings import PROJECT_ROOT

#: Guard so repeated ``Settings.from_env()`` calls don't re-read the file.
_LOADED: set[Path] = set()


def load_dotenv(path: Path | str | None = None, *, override: bool = False) -> dict[str, str]:
    """Load ``path`` (default: ``<project root>/.env``) into ``os.environ``.

    Args:
        path: File to read. Missing files are ignored -- CI and graders may not
            have one, and that must not be an error.
        override: When True, values in the file replace already-exported ones.

    Returns:
        The key/value pairs that were applied to the environment.
    """
    env_path = Path(path) if path is not None else PROJECT_ROOT / ".env"
    env_path = env_path.expanduser()

    if not override and env_path in _LOADED:
        return {}
    if not env_path.is_file():
        return {}

    applied: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if key in os.environ and not override:
            continue  # an explicitly exported variable outranks the file
        os.environ[key] = value
        applied[key] = value

    _LOADED.add(env_path)
    return applied
