"""Fetch the raw Tatoeba exports.

The instructor's brief points at https://tatoeba.org/en/downloads.  That page
publishes *per-language sentence dumps* and a single global *links* file; it
does not publish a ready-made English-Spanish parallel file, so the aligned
corpus is reconstructed locally in :mod:`nmt.data.pairing`.

Doing the join ourselves rather than downloading a pre-packaged bitext has two
advantages worth stating in the report:

1. the corpus is pinned to a dated snapshot we control, and
2. we keep the Tatoeba sentence ids, which makes it possible to trace any
   individual training or test example back to the public database.

Three files are needed (~172 MB compressed):

===========================  ========  =============================
file                         size      contents
===========================  ========  =============================
``eng_sentences.tsv.bz2``    ~24 MB    ``id \\t lang \\t text``
``spa_sentences.tsv.bz2``    ~6 MB     ``id \\t lang \\t text``
``links.tar.bz2``            ~142 MB   ``links.csv``: ``id \\t id``
===========================  ========  =============================
"""

from __future__ import annotations

import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from nmt.utils.io import human_bytes, project_root
from nmt.utils.logging_utils import get_logger

logger = get_logger(__name__)

_BASE = "https://downloads.tatoeba.org/exports"


@dataclass(frozen=True)
class RemoteFile:
    """One downloadable artefact."""

    name: str
    url: str
    description: str


DEFAULT_SOURCES: tuple[RemoteFile, ...] = (
    RemoteFile(
        name="eng_sentences.tsv.bz2",
        url=f"{_BASE}/per_language/eng/eng_sentences.tsv.bz2",
        description="All English sentences with their Tatoeba ids.",
    ),
    RemoteFile(
        name="spa_sentences.tsv.bz2",
        url=f"{_BASE}/per_language/spa/spa_sentences.tsv.bz2",
        description="All Spanish sentences with their Tatoeba ids.",
    ),
    RemoteFile(
        name="links.tar.bz2",
        url=f"{_BASE}/links.tar.bz2",
        description="Global translation graph: pairs of linked sentence ids.",
    ),
)

#: Fallback: the OPUS redistribution of Tatoeba, already sentence-aligned.
#: Much smaller and faster, used when ``downloads.tatoeba.org`` is unreachable.
OPUS_FALLBACK = RemoteFile(
    name="opus-tatoeba-en-es.zip",
    url="https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-es.txt.zip",
    description="Pre-aligned English-Spanish bitext (OPUS mirror of Tatoeba).",
)


def _download(url: str, destination: Path, *, timeout: int = 120) -> Path:
    """Stream ``url`` to ``destination`` via a temporary file.

    Writing to ``.part`` first means an interrupted download can never leave a
    truncated file that a later run would mistake for a complete one.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    logger.info("Downloading %s", url)
    request = urllib.request.Request(
        url, headers={"User-Agent": "AIG230-NMT-project/1.0"}
    )
    with (
        urllib.request.urlopen(request, timeout=timeout) as response,  # noqa: S310
        partial.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle, length=1 << 20)

    partial.replace(destination)
    logger.info(
        "  -> %s (%s)", destination.name, human_bytes(destination.stat().st_size)
    )
    return destination


def download_tatoeba_exports(
    raw_dir: Path | str | None = None,
    *,
    force: bool = False,
) -> dict[str, Path]:
    """Download every raw export, skipping files that are already present.

    Parameters
    ----------
    raw_dir
        Target directory; defaults to ``<project root>/data/raw``.
    force
        Re-download even when the file already exists.

    Returns
    -------
    dict
        Maps each file name to its path on disk.
    """
    raw_dir = Path(raw_dir) if raw_dir else project_root() / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for source in DEFAULT_SOURCES:
        destination = raw_dir / source.name
        if destination.exists() and not force:
            logger.info(
                "Present, skipping: %s (%s)",
                source.name,
                human_bytes(destination.stat().st_size),
            )
        else:
            _download(source.url, destination)
        paths[source.name] = destination

    return paths


def download_opus_fallback(raw_dir: Path | str | None = None) -> Path:
    """Download the pre-aligned OPUS mirror (used only if Tatoeba is down)."""
    raw_dir = Path(raw_dir) if raw_dir else project_root() / "data" / "raw"
    destination = raw_dir / OPUS_FALLBACK.name
    if destination.exists():
        return destination
    return _download(OPUS_FALLBACK.url, destination)


def main() -> None:
    """``python -m nmt.data.download`` entry point."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument(
        "--force", action="store_true", help="re-download existing files"
    )
    args = parser.parse_args()

    download_tatoeba_exports(args.raw_dir, force=args.force)


if __name__ == "__main__":
    main()
