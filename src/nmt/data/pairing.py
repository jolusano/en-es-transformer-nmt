"""Reconstruct the English-Spanish bitext from the raw Tatoeba exports.

Tatoeba stores sentences and *translation links* separately.  ``links.csv``
contains roughly 3.4 x 10^7 undirected id pairs spanning every language in the
database; only a small fraction connect an English sentence to a Spanish one.

The join is done in three passes so that peak memory stays proportional to the
two languages we care about rather than to the size of the link table:

1. load ``eng_sentences.tsv`` into ``{id: text}``  (~2 x 10^6 rows),
2. load ``spa_sentences.tsv`` into ``{id: text}``  (~4 x 10^5 rows),
3. stream ``links.csv`` line by line, emitting a pair whenever the left id is
   English **and** the right id is Spanish.

Because Tatoeba stores each link in both orientations, the asymmetric test in
step 3 yields every English-Spanish pair exactly once with no need to
deduplicate ids afterwards.

The output keeps the sentence ids.  They cost little and they make the corpus
auditable: any example quoted in the report or flagged during error analysis
can be looked up at ``https://tatoeba.org/en/sentences/show/<id>``.
"""

from __future__ import annotations

import bz2
import csv
import io
import re
import tarfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from nmt.utils.io import project_root, write_json
from nmt.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Tatoeba text can contain quote characters that Python's csv module would
# otherwise treat as field delimiters, so quoting is disabled throughout.
_CSV_KWARGS = {"delimiter": "\t", "quoting": csv.QUOTE_NONE}

_INTERNAL_WHITESPACE = re.compile(r"\s+")


def _sanitise(text: str) -> str:
    """Collapse internal whitespace so a field can never break the TSV.

    A handful of Tatoeba sentences contain literal tab or newline characters.
    Since :func:`nmt.data.cleaning.normalise` collapses all whitespace runs to
    a single space anyway, doing it here loses nothing and keeps the interim
    file parseable.
    """
    return _INTERNAL_WHITESPACE.sub(" ", text).strip()


@dataclass
class PairingReport:
    """Counters describing how the bitext was assembled."""

    english_sentences: int = 0
    spanish_sentences: int = 0
    links_scanned: int = 0
    pairs_found: int = 0
    unique_english_used: int = 0
    unique_spanish_used: int = 0
    malformed_rows: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _read_sentences(path: Path, expected_lang: str) -> dict[int, str]:
    """Load one ``id \\t lang \\t text`` export into a dictionary.

    Rows whose language code does not match ``expected_lang`` are skipped;
    the per-language dumps are already filtered but the check is cheap and
    guards against a mis-specified path.
    """
    sentences: dict[int, str] = {}
    malformed = 0

    with bz2.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, **_CSV_KWARGS):
            if len(row) < 3:
                malformed += 1
                continue
            sentence_id, lang, text = row[0], row[1], row[2]
            if lang != expected_lang:
                continue
            try:
                sentences[int(sentence_id)] = text
            except ValueError:
                malformed += 1

    logger.info(
        "Loaded %s %s sentences from %s (%d malformed rows skipped)",
        f"{len(sentences):,}",
        expected_lang,
        path.name,
        malformed,
    )
    return sentences


def _open_links(path: Path) -> io.TextIOWrapper:
    """Return a text stream over ``links.csv``.

    Accepts either the ``.tar.bz2`` archive published by Tatoeba or an already
    extracted ``.csv`` / ``.csv.bz2``.
    """
    if path.suffixes[-2:] == [".tar", ".bz2"] or path.name.endswith(".tar.bz2"):
        archive = tarfile.open(path, "r:bz2")  # noqa: SIM115 (closed by the caller's `with`)
        member = next(
            (m for m in archive.getmembers() if m.name.endswith("links.csv")), None
        )
        if member is None:
            raise FileNotFoundError(f"links.csv not found inside {path}")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise OSError(f"could not read {member.name} from {path}")
        return io.TextIOWrapper(extracted, encoding="utf-8", newline="")

    if path.suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", newline="")

    return path.open("r", encoding="utf-8", newline="")


def build_sentence_pairs(
    raw_dir: Path | str | None = None,
    *,
    output_path: Path | str | None = None,
    report_path: Path | str | None = None,
) -> tuple[Path, PairingReport]:
    """Join the exports into a single TSV of aligned sentence pairs.

    The output has four columns and one header row::

        en_id \\t es_id \\t en \\t es

    Returns
    -------
    (Path, PairingReport)
        Location of the written file and the counters gathered on the way.
    """
    raw_dir = Path(raw_dir) if raw_dir else project_root() / "data" / "raw"
    output_path = Path(
        output_path or project_root() / "data" / "interim" / "pairs_raw.tsv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = PairingReport()

    english = _read_sentences(raw_dir / "eng_sentences.tsv.bz2", "eng")
    spanish = _read_sentences(raw_dir / "spa_sentences.tsv.bz2", "spa")
    report.english_sentences = len(english)
    report.spanish_sentences = len(spanish)

    used_english: set[int] = set()
    used_spanish: set[int] = set()

    logger.info("Streaming links.csv (this takes about a minute)...")
    # Written by hand rather than through csv.writer: sentences legitimately
    # contain double quotes, and csv's QUOTE_NONE dialect refuses to emit a
    # field containing the quote character. _sanitise() has already guaranteed
    # no field can contain a tab or a newline, so a plain join is both correct
    # and considerably faster over 3.4e7 rows.
    with (
        _open_links(raw_dir / "links.tar.bz2") as links,
        output_path.open("w", encoding="utf-8", newline="\n") as out_handle,
    ):
        out_handle.write("en_id\tes_id\ten\tes\n")

        for row in csv.reader(links, **_CSV_KWARGS):
            report.links_scanned += 1
            if len(row) < 2:
                report.malformed_rows += 1
                continue
            try:
                left, right = int(row[0]), int(row[1])
            except ValueError:
                report.malformed_rows += 1
                continue

            # Asymmetric test: links are stored both ways round, so checking
            # only (English, Spanish) yields each pair exactly once.
            en_text = english.get(left)
            if en_text is None:
                continue
            es_text = spanish.get(right)
            if es_text is None:
                continue

            out_handle.write(
                f"{left}\t{right}\t{_sanitise(en_text)}\t{_sanitise(es_text)}\n"
            )
            used_english.add(left)
            used_spanish.add(right)
            report.pairs_found += 1

            if report.links_scanned % 5_000_000 == 0:
                logger.info(
                    "  scanned %s links, %s pairs so far",
                    f"{report.links_scanned:,}",
                    f"{report.pairs_found:,}",
                )

    report.unique_english_used = len(used_english)
    report.unique_spanish_used = len(used_spanish)
    report.notes.append(
        "Tatoeba is a many-to-many translation graph: one English sentence may "
        "link to several Spanish sentences and vice versa. Those alternatives "
        "are genuine paraphrases and are kept, but they are also the reason "
        "the train/val/test split groups sentences into connected components "
        "(see nmt.data.splitting)."
    )

    logger.info(
        "Wrote %s pairs to %s (%s unique EN, %s unique ES)",
        f"{report.pairs_found:,}",
        output_path,
        f"{report.unique_english_used:,}",
        f"{report.unique_spanish_used:,}",
    )

    if report_path is not None:
        write_json(report_path, report.to_dict())

    return output_path, report


def main() -> None:
    """``python -m nmt.data.pairing`` entry point."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    build_sentence_pairs(
        args.raw_dir, output_path=args.output, report_path=args.report
    )


if __name__ == "__main__":
    main()
