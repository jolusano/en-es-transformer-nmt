"""Console + file logging with a consistent format across every entry point."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_CONFIGURED = False


def setup_logging(
    level: int | str = logging.INFO,
    *,
    log_file: Path | str | None = None,
) -> logging.Logger:
    """Configure the root logger once per process.

    Parameters
    ----------
    level
        Threshold for the console handler.
    log_file
        Optional path; when given, a second handler mirrors everything at
        ``DEBUG`` level into that file so a training run leaves a full trace on
        disk even though the console stays readable.
    """
    global _CONFIGURED

    root = logging.getLogger()
    if _CONFIGURED:
        root.setLevel(level)
        return root

    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # These libraries are chatty at INFO and drown out our own messages.
    for noisy in (
        "matplotlib",
        "fontTools",  # logs one line per font table on every PDF/SVG export
        "PIL",
        "httpx",
        "urllib3",
        "asyncio",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, configuring the root one if needed."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
