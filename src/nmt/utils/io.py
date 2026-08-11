"""Small, dependency-free I/O helpers.

Everything the pipeline produces that a human or the report generator needs to
read back is written through this module, in UTF-8, with a trailing newline.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Absolute path of the repository root.

    Resolved from this file's location (``<root>/src/nmt/utils/io.py``) so that
    scripts, notebooks and the Gradio app all agree on where ``data/`` and
    ``artifacts/`` live regardless of the working directory they are launched
    from.
    """
    return Path(__file__).resolve().parents[3]


def ensure_dir(path: Path | str) -> Path:
    """Create ``path`` as a directory (including parents) and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path | str) -> Path:
    """Create the parent directory of ``path`` and return ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# --- JSON -------------------------------------------------------------------


def write_json(path: Path | str, payload: Any, *, indent: int = 2) -> Path:
    """Write ``payload`` as pretty-printed UTF-8 JSON."""
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, ensure_ascii=False, default=str)
        handle.write("\n")
    return path


def read_json(path: Path | str) -> Any:
    """Read a UTF-8 JSON document."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


# --- JSON Lines -------------------------------------------------------------


def append_jsonl(path: Path | str, record: dict[str, Any]) -> None:
    """Append one record to a JSON Lines file.

    Used for the per-step training log: appending is crash-safe, so a run that
    dies mid-epoch still leaves a readable history behind.
    """
    path = ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Read a JSON Lines file into a list of dictionaries."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# --- Plain text -------------------------------------------------------------


def write_lines(path: Path | str, lines: Iterable[str]) -> Path:
    """Write an iterable of strings, one per line."""
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line.rstrip("\n") + "\n")
    return path


def read_lines(path: Path | str) -> list[str]:
    """Read a text file into a list of newline-stripped strings."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def iter_lines(path: Path | str) -> Iterator[str]:
    """Stream a text file line by line without loading it into memory."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            yield line.rstrip("\n")


def human_bytes(size: float) -> str:
    """Format a byte count for logs and tables (e.g. ``"149.1 MB"``)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"
