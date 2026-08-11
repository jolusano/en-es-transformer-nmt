"""Recompute the error analysis from stored hypotheses, without a GPU.

    python reports/reanalyse_errors.py

``nmt.evaluation.evaluate`` saves every source/reference/hypothesis triple into
``evaluation_<split>.json`` precisely so that the analysis layer can be revised
without re-decoding the test set. This script re-runs
:func:`nmt.evaluation.error_analysis.analyse` over those stored outputs and
rewrites the ``error_analysis`` block in place.

Written when a bug was found in sentence-level BLEU: it was computed at fixed
4-gram order, so any sentence shorter than four tokens scored zero however good
it was. Since a quarter of this test set is under four tokens, the "worst
translations" ranking had filled up with sentences the model had translated
*perfectly*. Re-decoding four systems to fix a scoring function would have cost
an hour of GPU time for no new model output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nmt.constants import DIRECTIONS
from nmt.evaluation.error_analysis import analyse, to_markdown
from nmt.utils.io import project_root, read_json, write_json
from nmt.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def reanalyse(path: Path) -> dict[str, float]:
    """Rewrite one evaluation report's error analysis. Returns before/after means."""
    report = read_json(path)
    changes: dict[str, float] = {}

    for direction in DIRECTIONS:
        entry = report["directions"][direction]
        outputs = entry["outputs"]

        before = entry["error_analysis"]["mean_sentence_bleu"]
        entry["error_analysis"] = analyse(
            [o["source"] for o in outputs],
            [o["reference"] for o in outputs],
            [o["hypothesis"] for o in outputs],
            direction,
        )
        after = entry["error_analysis"]["mean_sentence_bleu"]
        changes[direction] = after - before
        logger.info(
            "  %-6s mean sentence BLEU %.2f -> %.2f (%+.2f)",
            direction, before, after, after - before,
        )

    write_json(path, report)

    markdown = "\n\n".join(
        to_markdown(report["directions"][d]["error_analysis"]) for d in DIRECTIONS
    )
    (path.parent / f"error_analysis_{report['split']}.md").write_text(
        markdown, encoding="utf-8"
    )
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    setup_logging()
    results_dir = args.results or project_root() / "artifacts" / "results"

    paths = sorted(results_dir.glob(f"*/evaluation_{args.split}.json"))
    if not paths:
        logger.warning("No evaluation reports found under %s", results_dir)
        return

    for path in paths:
        logger.info("%s", path.parent.name)
        reanalyse(path)

    logger.info(
        "Rewrote %d report(s). Re-run nmt.viz.make_figures and "
        "reports/build_report_data.py to propagate.",
        len(paths),
    )


if __name__ == "__main__":
    main()
