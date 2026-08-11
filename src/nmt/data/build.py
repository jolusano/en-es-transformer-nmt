"""End-to-end dataset construction.

Runs the whole preprocessing pipeline and writes everything the rest of the
project (and the report) depends on::

    python -m nmt.data.build

Stages
------
1. download the raw Tatoeba exports (skipped if present)
2. join them into a bitext                      -> data/interim/pairs_raw.tsv
3. normalise and filter                         -> cleaning report
4. split by connected component                 -> data/processed/*.tsv
5. fit the joint subword and word vocabularies  -> artifacts/tokenizers/
6. compute corpus statistics                    -> artifacts/results/dataset_stats.json

Everything numeric that appears in the report's data section is read back out
of ``dataset_stats.json``, so re-running this script is what refreshes those
tables and figures.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from nmt.data.cleaning import CleaningRules, clean_pairs
from nmt.data.download import download_tatoeba_exports
from nmt.data.pairing import build_sentence_pairs
from nmt.data.splitting import SplitSizes, split_pairs
from nmt.data.tokenizer import SubwordTokenizer, WordTokenizer
from nmt.utils.io import ensure_dir, project_root, read_json, write_json
from nmt.utils.logging_utils import get_logger, setup_logging
from nmt.utils.seed import seed_everything

logger = get_logger(__name__)




def load_raw_pairs(path: Path) -> list[tuple[int, int, str, str]]:
    """Read ``pairs_raw.tsv`` produced by :mod:`nmt.data.pairing`."""
    rows: list[tuple[int, int, str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader, None)  # header
        for row in reader:
            if len(row) >= 4:
                rows.append((int(row[0]), int(row[1]), row[2], row[3]))
    logger.info("Loaded %s raw pairs from %s", f"{len(rows):,}", path.name)
    return rows


def write_split(path: Path, pairs: list[tuple[str, str]]) -> Path:
    """Write one split as a two-column ``en \\t es`` TSV with a header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sentences contain double quotes, which csv.writer refuses to emit under
    # QUOTE_NONE. Normalisation has already collapsed every whitespace run to a
    # single space, so no field can contain a tab or newline and a plain join
    # is unambiguous.
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("en\tes\n")
        for english, spanish in pairs:
            handle.write(f"{english}\t{spanish}\n")
    return path


def read_split(path: Path) -> list[tuple[str, str]]:
    """Read a split written by :func:`write_split`."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader, None)
        return [(row[0], row[1]) for row in reader if len(row) >= 2]


# ---------------------------------------------------------------------------
# Corpus statistics
# ---------------------------------------------------------------------------


def _length_summary(values: list[int]) -> dict[str, float]:
    """Descriptive statistics for a length distribution."""
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)

    def percentile(p: float) -> float:
        position = min(n - 1, max(0, int(round(p * (n - 1)))))
        return float(ordered[position])

    # The mean is computed once and reused: putting `sum(ordered)` inside the
    # variance generator would re-scan the list for every element, which is
    # O(n^2) and takes minutes on the training split.
    mean = sum(ordered) / n
    variance = sum((value - mean) ** 2 for value in ordered) / n

    return {
        "count": n,
        "mean": mean,
        "std": variance**0.5,
        "min": float(ordered[0]),
        "p25": percentile(0.25),
        "median": percentile(0.50),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": float(ordered[-1]),
    }


def _vocabulary_summary(sentences: list[str], tokenizer: WordTokenizer) -> dict:
    """Type/token statistics for one language."""
    counter: Counter[str] = Counter()
    for sentence in sentences:
        counter.update(tokenizer.tokenize(sentence))

    tokens = sum(counter.values())
    types = len(counter)
    hapax = sum(1 for count in counter.values() if count == 1)

    #: Fraction of running text covered by the N most frequent types --
    #: the number that justifies truncating the word vocabulary.
    ordered = [count for _, count in counter.most_common()]
    cumulative: dict[str, float] = {}
    running = 0
    for cutoff in (1_000, 5_000, 10_000, 20_000, 32_000):
        running = sum(ordered[:cutoff])
        cumulative[f"coverage_top_{cutoff}"] = running / tokens if tokens else 0.0

    return {
        "tokens": tokens,
        "types": types,
        "type_token_ratio": types / tokens if tokens else 0.0,
        "hapax_legomena": hapax,
        "hapax_fraction": hapax / types if types else 0.0,
        "most_common": counter.most_common(25),
        **cumulative,
    }


def compute_statistics(
    splits: dict[str, list[tuple[str, str]]],
    subword: SubwordTokenizer,
    word: WordTokenizer,
) -> dict:
    """Everything the exploration section of the report needs."""
    stats: dict[str, object] = {"splits": {}}

    for name, pairs in splits.items():
        english = [pair[0] for pair in pairs]
        spanish = [pair[1] for pair in pairs]

        entry: dict[str, object] = {
            "pairs": len(pairs),
            "unique_en": len(set(english)),
            "unique_es": len(set(spanish)),
            "length_words": {
                "en": _length_summary([len(s.split()) for s in english]),
                "es": _length_summary([len(s.split()) for s in spanish]),
            },
            "length_chars": {
                "en": _length_summary([len(s) for s in english]),
                "es": _length_summary([len(s) for s in spanish]),
            },
        }

        # Subword statistics are expensive; compute them on a sample for the
        # large training split and in full for the small evaluation splits.
        sample_en = english[:50_000]
        sample_es = spanish[:50_000]
        entry["length_subwords"] = {
            "en": _length_summary([len(subword.encode(s)) for s in sample_en]),
            "es": _length_summary([len(subword.encode(s)) for s in sample_es]),
        }
        entry["subword_fertility"] = {
            "en": (
                sum(len(subword.encode(s)) for s in sample_en)
                / max(1, sum(len(s.split()) for s in sample_en))
            ),
            "es": (
                sum(len(subword.encode(s)) for s in sample_es)
                / max(1, sum(len(s.split()) for s in sample_es))
            ),
        }

        if name == "train":
            entry["vocabulary"] = {
                "en": _vocabulary_summary(english, word),
                "es": _vocabulary_summary(spanish, word),
            }
        else:
            entry["word_oov"] = {
                "en": word.coverage(english),
                "es": word.coverage(spanish),
            }

        stats["splits"][name] = entry  # type: ignore[index]

    stats["tokenizers"] = {
        "subword": {
            "kind": "sentencepiece-bpe",
            "vocab_size": subword.vocab_size,
        },
        "word": {
            "kind": "frequency-truncated word level",
            "vocab_size": word.vocab_size,
        },
    }
    return stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None, help="project root")
    parser.add_argument("--subword-vocab", type=int, default=16_000)
    parser.add_argument("--word-vocab", type=int, default=32_000)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--test-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-download", action="store_true", help="assume raw files are present"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="use only the first N raw pairs (for a fast smoke test)",
    )
    args = parser.parse_args()

    root = args.root or project_root()
    setup_logging(log_file=root / "artifacts" / "logs" / "build_dataset.log")
    seed_everything(args.seed)

    started = time.time()
    raw_dir = root / "data" / "raw"
    interim = root / "data" / "interim"
    processed = ensure_dir(root / "data" / "processed")
    tokenizer_dir = ensure_dir(root / "artifacts" / "tokenizers")
    results_dir = ensure_dir(root / "artifacts" / "results")

    # --- 1-2. acquire and join -------------------------------------------
    if not args.skip_download:
        download_tatoeba_exports(raw_dir)

    pairs_path = interim / "pairs_raw.tsv"
    # A file left behind by an interrupted run is worse than no file at all:
    # it would be silently reused and every downstream stage would work on an
    # empty corpus. Anything implausibly small is treated as absent.
    if pairs_path.exists() and pairs_path.stat().st_size < 1_000_000:
        logger.warning(
            "%s is only %d bytes -- treating it as an incomplete run and rebuilding.",
            pairs_path.name,
            pairs_path.stat().st_size,
        )
        pairs_path.unlink()

    pairing_report_path = results_dir / "pairing_report.json"

    if not pairs_path.exists():
        pairs_path, pairing_report = build_sentence_pairs(
            raw_dir,
            output_path=pairs_path,
            report_path=pairing_report_path,
        )
        pairing_summary = pairing_report.to_dict()
    elif pairing_report_path.exists():
        # Reuse the counters from the run that produced this file. Recording
        # only {"reused": True} here would silently drop the corpus-provenance
        # numbers from dataset_stats.json, and the report reads them from
        # there -- so a second `build` invocation would blank out a table.
        logger.info("Reusing existing %s and its pairing report", pairs_path.name)
        pairing_summary = read_json(pairing_report_path)
    else:
        logger.warning(
            "Reusing %s but %s is missing; corpus-provenance statistics will be "
            "unavailable. Delete the TSV and re-run to regenerate them.",
            pairs_path.name,
            pairing_report_path.name,
        )
        pairing_summary = {"reused": True}

    raw_pairs = load_raw_pairs(pairs_path)
    if args.limit:
        raw_pairs = raw_pairs[: args.limit]
        logger.warning("SMOKE TEST: limited to %s pairs", f"{len(raw_pairs):,}")

    # --- 3. clean ---------------------------------------------------------
    rules = CleaningRules(max_tokens=args.max_tokens)
    cleaned, cleaning_report = clean_pairs(raw_pairs, rules)
    logger.info(
        "Cleaning kept %s / %s pairs (%.1f%%)",
        f"{len(cleaned):,}",
        f"{len(raw_pairs):,}",
        100 * len(cleaned) / max(1, len(raw_pairs)),
    )
    write_json(results_dir / "cleaning_report.json", cleaning_report.to_dict())

    # --- 4. split ---------------------------------------------------------
    splits, split_report = split_pairs(
        cleaned,
        SplitSizes(validation=args.validation_fraction, test=args.test_fraction),
        seed=args.seed,
    )
    write_json(results_dir / "split_report.json", split_report.to_dict())
    for name, pairs in splits.items():
        write_split(processed / f"{name}.tsv", pairs)
        logger.info("  %-10s %s pairs", name, f"{len(pairs):,}")

    # --- 5. fit tokenisers on the TRAINING split only ---------------------
    # Fitting on the full corpus would leak test-set surface forms into the
    # vocabulary, which flatters the out-of-vocabulary rate we report.
    training_text = [sentence for pair in splits["train"] for sentence in pair]

    logger.info("Training joint subword tokeniser...")
    subword = SubwordTokenizer.train(
        training_text,
        tokenizer_dir,
        vocab_size=args.subword_vocab,
        prefix="joint_bpe",
    )

    logger.info("Building word-level vocabulary...")
    word = WordTokenizer.train(training_text, vocab_size=args.word_vocab)
    word.save(tokenizer_dir, prefix="word")

    # --- 6. statistics ----------------------------------------------------
    logger.info("Computing corpus statistics...")
    statistics = compute_statistics(splits, subword, word)
    statistics["pairing"] = pairing_summary
    statistics["cleaning"] = cleaning_report.to_dict()
    statistics["split"] = split_report.to_dict()
    statistics["config"] = vars(args) | {"rules": asdict(rules)}
    statistics["elapsed_seconds"] = round(time.time() - started, 1)

    write_json(results_dir / "dataset_stats.json", statistics)
    logger.info(
        "Done in %.1fs. Statistics -> %s",
        time.time() - started,
        results_dir / "dataset_stats.json",
    )


if __name__ == "__main__":
    main()
