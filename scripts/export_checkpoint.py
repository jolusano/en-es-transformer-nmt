"""Strip a training checkpoint down to what inference needs.

    python scripts/export_checkpoint.py --all
    python scripts/export_checkpoint.py artifacts/checkpoints/bpe_scratch/best_bleu.pt

A checkpoint written during training carries far more than the model. Adam keeps
two moment tensors per parameter, so the optimiser state alone is twice the size
of the weights, and with the scheduler state and metadata a 37.6M-parameter
model lands around 450 MB. All of that exists so training can be *resumed*; none
of it is read at inference.

This produces a ``*_release.pt`` containing only the weights, the model config
and the tokeniser path -- roughly a quarter the size, and small enough to attach
to a GitHub Release or fit comfortably in a Drive folder.

Keep the full checkpoints if you may want to resume training. Ship the slim ones.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nmt.utils.io import human_bytes, project_root  # noqa: E402
from nmt.utils.logging_utils import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

#: Everything the inference path reads. `Translator.from_checkpoint` needs
#: model_state and model_config; tokenizer_path lets it find the vocabulary
#: without being told, and the rest is provenance worth a few hundred bytes.
KEEP = ("model_state", "model_config", "tokenizer_path", "epoch", "global_step")


def export(source: Path, destination: Path | None = None) -> Path:
    """Write a slim copy of ``source`` and report the saving."""
    destination = destination or source.with_name(f"{source.stem}_release.pt")

    payload = torch.load(source, map_location="cpu", weights_only=False)
    slim = {key: payload[key] for key in KEEP if key in payload}

    # The evaluation numbers are small and make the artefact self-describing:
    # anyone downloading it can see what it scored without the repository.
    metrics = payload.get("metrics", {})
    if metrics:
        slim["metrics"] = {
            "epoch": metrics.get("epoch"),
            "validation": metrics.get("validation"),
            "bleu_detail": metrics.get("bleu_detail"),
        }

    torch.save(slim, destination)

    before, after = source.stat().st_size, destination.stat().st_size
    logger.info(
        "%-22s %10s -> %10s  (%.0f%% smaller)",
        source.parent.name + "/" + source.name,
        human_bytes(before),
        human_bytes(after),
        100 * (1 - after / before),
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path, nargs="?", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--all", action="store_true",
        help="export best_bleu.pt for every run under artifacts/checkpoints/",
    )
    args = parser.parse_args()
    setup_logging()

    if args.all:
        root = project_root() / "artifacts" / "checkpoints"
        paths = sorted(root.glob("*/best_bleu.pt"))
        if not paths:
            logger.error("No checkpoints found under %s", root)
            return
        for path in paths:
            export(path)
        logger.info("Wrote %d release checkpoint(s).", len(paths))
    elif args.checkpoint:
        export(args.checkpoint, args.output)
    else:
        parser.error("give a checkpoint path or --all")


if __name__ == "__main__":
    main()
