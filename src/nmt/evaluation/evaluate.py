"""Evaluate a trained checkpoint on the test set.

    python -m nmt.evaluation.evaluate --checkpoint artifacts/checkpoints/bpe_scratch/best_bleu.pt

Produces, per direction:

* sacreBLEU and chrF2, plus the from-scratch BLEU as a cross-check,
* greedy vs beam comparison,
* BLEU stratified by source length,
* the failure-mode census and the worst examples,
* every hypothesis, so the numbers can be recomputed without a GPU.

Everything lands in ``artifacts/results/<run>/`` as JSON, which is what the
report reads.  No number in the report is typed in by hand.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from nmt.constants import DIRECTION_NAMES, DIRECTIONS
from nmt.data.build import read_split
from nmt.evaluation.bleu import compare_implementations, corpus_bleu
from nmt.evaluation.error_analysis import analyse, to_markdown
from nmt.inference.search import DecodeConfig
from nmt.inference.translator import Translator
from nmt.utils.io import ensure_dir, project_root, write_json
from nmt.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def evaluate_direction(
    translator: Translator,
    pairs: list[tuple[str, str]],
    direction: str,
    *,
    batch_size: int = 32,
    progress: bool = True,
) -> dict[str, object]:
    """Decode one direction and score it."""
    started = time.time()
    sources, hypotheses, references = translator.translate_pairs(
        pairs, direction, batch_size=batch_size, progress=progress
    )
    elapsed = time.time() - started

    comparison = compare_implementations(hypotheses, references)
    analysis = analyse(sources, references, hypotheses, direction)

    logger.info(
        "%s | BLEU %.2f | chrF2 %.2f | %.1fs (%.1f sentences/s)",
        DIRECTION_NAMES[direction],
        comparison["sacrebleu"]["bleu"],
        comparison["sacrebleu"]["chrf2"],
        elapsed,
        len(pairs) / elapsed if elapsed else 0.0,
    )

    return {
        "direction": direction,
        "direction_name": DIRECTION_NAMES[direction],
        "sentences": len(pairs),
        "seconds": round(elapsed, 2),
        "sentences_per_second": round(len(pairs) / elapsed, 2) if elapsed else 0.0,
        "bleu": comparison["sacrebleu"]["bleu"],
        "chrf2": comparison["sacrebleu"]["chrf2"],
        "metrics": comparison,
        "error_analysis": analysis,
        "outputs": [
            {"source": s, "reference": r, "hypothesis": h}
            for s, r, h in zip(sources, references, hypotheses)
        ],
    }


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    split: str = "test",
    processed_dir: Path | None = None,
    results_dir: Path | None = None,
    decode_config: DecodeConfig | None = None,
    limit: int | None = None,
    batch_size: int = 32,
    device: str = "auto",
    compare_greedy: bool = True,
) -> dict[str, object]:
    """Full evaluation of one checkpoint, written to ``results_dir``."""
    root = project_root()
    processed_dir = processed_dir or root / "data" / "processed"
    run_name = checkpoint.parent.name
    results_dir = ensure_dir(results_dir or root / "artifacts" / "results" / run_name)

    decode_config = decode_config or DecodeConfig()
    translator = Translator.from_checkpoint(
        checkpoint, device=device, decode_config=decode_config
    )

    pairs = read_split(processed_dir / f"{split}.tsv")
    if limit:
        pairs = pairs[:limit]
    logger.info("Evaluating %s on %s pairs from %s", checkpoint.name, len(pairs), split)

    report: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "run": run_name,
        "split": split,
        "pairs": len(pairs),
        "decoding": vars(decode_config),
        "directions": {},
    }

    for direction in DIRECTIONS:
        report["directions"][direction] = evaluate_direction(  # type: ignore[index]
            translator, pairs, direction, batch_size=batch_size
        )

    # --- greedy vs beam ---------------------------------------------------
    # Reported because it isolates how much of the score comes from the model
    # and how much from the search; a large gap means the model's single-best
    # token is often wrong even when its distribution is right.
    if compare_greedy and decode_config.strategy == "beam":
        greedy_config = DecodeConfig(strategy="greedy", beam_size=1)
        greedy_translator = Translator.from_checkpoint(
            checkpoint, device=device, decode_config=greedy_config
        )
        greedy_scores: dict[str, float] = {}
        for direction in DIRECTIONS:
            _, hypotheses, references = greedy_translator.translate_pairs(
                pairs, direction, batch_size=batch_size, progress=False
            )
            greedy_scores[direction] = corpus_bleu(hypotheses, references).score
        report["greedy_bleu"] = greedy_scores
        report["beam_gain"] = {
            direction: report["directions"][direction]["bleu"] - greedy_scores[direction]  # type: ignore[index]
            for direction in DIRECTIONS
        }

    # --- headline numbers -------------------------------------------------
    report["summary"] = {
        "bleu": {
            direction: report["directions"][direction]["bleu"]  # type: ignore[index]
            for direction in DIRECTIONS
        },
        "chrf2": {
            direction: report["directions"][direction]["chrf2"]  # type: ignore[index]
            for direction in DIRECTIONS
        },
        "mean_bleu": sum(
            report["directions"][direction]["bleu"] for direction in DIRECTIONS  # type: ignore[index]
        )
        / len(DIRECTIONS),
    }

    write_json(results_dir / f"evaluation_{split}.json", report)

    markdown = "\n\n".join(
        to_markdown(report["directions"][direction]["error_analysis"])  # type: ignore[index]
        for direction in DIRECTIONS
    )
    (results_dir / f"error_analysis_{split}.md").write_text(markdown, encoding="utf-8")

    logger.info(
        "Wrote %s and %s",
        results_dir / f"evaluation_{split}.json",
        results_dir / f"error_analysis_{split}.md",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument(
        "--strategy", default="beam", choices=["beam", "greedy"]
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-greedy-comparison", action="store_true")
    args = parser.parse_args()

    setup_logging()
    evaluate_checkpoint(
        args.checkpoint,
        split=args.split,
        decode_config=DecodeConfig(
            strategy=args.strategy,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
        ),
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
        compare_greedy=not args.no_greedy_comparison,
    )


if __name__ == "__main__":
    main()
