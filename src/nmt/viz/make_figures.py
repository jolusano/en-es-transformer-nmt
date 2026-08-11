"""Regenerate every figure.

    python -m nmt.viz.make_figures            # everything available
    python -m nmt.viz.make_figures --only architecture

Figures whose input data does not exist yet are skipped with a note rather than
raising, so this is safe to run before any model has been trained -- the
architecture and corpus figures are produced regardless.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nmt.utils.io import project_root
from nmt.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

GROUPS = ("architecture", "data", "training", "results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument(
        "--only", nargs="*", choices=GROUPS, default=list(GROUPS),
        help="restrict to these figure groups",
    )
    args = parser.parse_args()

    setup_logging()
    root = project_root()
    output_dir = args.output or root / "reports" / "figures"
    results_dir = args.results or root / "artifacts" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0

    if "architecture" in args.only:
        from nmt.viz import architecture

        produced = architecture.draw_all(output_dir)
        total += len(produced)
        logger.info("Architecture figures: %d", len(produced))

    if "data" in args.only:
        from nmt.viz import data_plots

        stats = results_dir / "dataset_stats.json"
        if stats.exists():
            produced = data_plots.draw_all(output_dir, stats_path=stats)
            total += len(produced)
            logger.info("Corpus figures: %d", len(produced))
        else:
            logger.warning(
                "Skipping corpus figures: %s not found. Run `python -m nmt.data.build`.",
                stats,
            )

    if "training" in args.only:
        from nmt.viz import training_plots

        if results_dir.exists():
            produced = training_plots.draw_all(results_dir, output_dir)
            total += len(produced)
            logger.info("Training figures: %d", len(produced))
            if not produced:
                logger.warning("No training runs found under %s", results_dir)

    if "results" in args.only:
        from nmt.viz import results_plots

        if results_dir.exists():
            produced = results_plots.draw_all(results_dir, output_dir)
            total += len(produced)
            logger.info("Evaluation figures: %d", len(produced))
            if not produced:
                logger.warning(
                    "No evaluation reports found. Run `python -m nmt.evaluation.evaluate`."
                )

    logger.info("Wrote %d figures to %s", total, output_dir)


if __name__ == "__main__":
    main()
