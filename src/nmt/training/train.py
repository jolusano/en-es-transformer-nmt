"""Train one experiment from a YAML configuration.

    python -m nmt.training.train --config configs/bpe_scratch.yaml

Wires together the data pipeline, the model, and :class:`nmt.training.Trainer`,
and writes everything needed to reproduce or evaluate the run into
``artifacts/results/<name>/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from nmt.config import ExperimentConfig
from nmt.constants import DIRECTIONS
from nmt.data.build import read_split
from nmt.data.dataset import (
    TranslationDataset,
    build_dataloader,
    estimate_steps_per_epoch,
    padding_waste,
)
from nmt.data.tokenizer import load_tokenizer
from nmt.evaluation.bleu import corpus_bleu
from nmt.inference.search import DecodeConfig
from nmt.inference.translator import Translator, build_model_from_config
from nmt.training.trainer import Trainer
from nmt.utils.devices import resolve_device
from nmt.utils.io import ensure_dir, write_json
from nmt.utils.logging_utils import get_logger, setup_logging
from nmt.utils.seed import seed_everything

logger = get_logger(__name__)


def make_validation_bleu_callback(
    pairs: list[tuple[str, str]],
    tokenizer,
    config: ExperimentConfig,
    device: torch.device,
    *,
    sample_size: int,
):
    """Build a callback that decodes a fixed sample and returns BLEU.

    Greedy decoding is used here rather than beam search: this runs after every
    epoch, and the purpose is a comparable *trend*, not the best achievable
    score. The final numbers in the report come from the separate beam-search
    evaluation pass.

    The sample is fixed across epochs so the curve is not confounded by which
    sentences happened to be drawn.
    """
    sample = pairs[:sample_size]
    if not sample:
        return None

    def callback(model: torch.nn.Module) -> dict[str, float]:
        translator = Translator(
            model,
            tokenizer,
            device=device,
            decode_config=DecodeConfig(strategy="greedy", beam_size=1),
            max_length=config.data.max_length,
        )
        was_training = model.training
        model.eval()

        scores: dict[str, float] = {}
        for direction in DIRECTIONS:
            _, hypotheses, references = translator.translate_pairs(
                sample,
                direction,
                batch_size=config.data.eval_batch_size,
                progress=False,
            )
            scores[direction] = corpus_bleu(hypotheses, references).score

        if was_training:
            model.train()

        scores["bleu"] = sum(scores[d] for d in DIRECTIONS) / len(DIRECTIONS)
        return scores

    return callback


def run(config: ExperimentConfig, *, resume: Path | None = None) -> dict:
    """Execute one experiment end to end."""
    seed_everything(config.train.seed)
    device = resolve_device(config.train.device)

    # --- data -------------------------------------------------------------
    tokenizer = load_tokenizer(config.data.tokenizer)
    if tokenizer.vocab_size != config.model.vocab_size:
        logger.warning(
            "Config says vocab_size=%d but the tokeniser has %d entries; "
            "using the tokeniser's value.",
            config.model.vocab_size,
            tokenizer.vocab_size,
        )
        config.model.vocab_size = tokenizer.vocab_size

    processed = Path(config.data.processed_dir)
    train_pairs = read_split(processed / "train.tsv")
    validation_pairs = read_split(processed / "validation.tsv")

    if config.data.limit_train_pairs:
        train_pairs = train_pairs[: config.data.limit_train_pairs]
        logger.warning("Limited training data to %s pairs", f"{len(train_pairs):,}")

    train_dataset = TranslationDataset(
        train_pairs, tokenizer, max_length=config.data.max_length
    )
    validation_dataset = TranslationDataset(
        validation_pairs, tokenizer, max_length=config.data.max_length
    )
    logger.info(
        "Datasets: %s training examples, %s validation (both directions)",
        f"{len(train_dataset):,}",
        f"{len(validation_dataset):,}",
    )

    train_loader = build_dataloader(
        train_dataset,
        max_tokens=config.data.max_tokens_per_batch,
        shuffle=True,
        num_workers=config.data.num_workers,
        seed=config.train.seed,
        pin_memory=device.type == "cuda",
    )
    validation_loader = build_dataloader(
        validation_dataset,
        batch_size=config.data.eval_batch_size,
        shuffle=False,
        num_workers=0,
    )

    # --- model ------------------------------------------------------------
    model = build_model_from_config(config.model)

    if config.model.pretrained_embeddings:
        matrix = np.load(config.model.pretrained_embeddings)
        model.load_pretrained_embeddings(
            torch.from_numpy(matrix),
            freeze=config.model.freeze_embeddings_epochs > 0,
        )

    # --- train ------------------------------------------------------------
    bleu_callback = make_validation_bleu_callback(
        validation_pairs,
        tokenizer,
        config,
        device,
        sample_size=config.train.validation_bleu_samples,
    )

    trainer = Trainer(
        model,
        config,
        train_loader,
        validation_loader,
        device=device,
        bleu_callback=bleu_callback,
    )

    if resume is not None:
        trainer.load(resume, resume=True)

    # Record batching diagnostics before training so the report can justify
    # the token-bucketing sampler with real numbers.
    results_dir = ensure_dir(config.train.results_dir)
    write_json(
        results_dir / "data_summary.json",
        {
            "train_pairs": len(train_pairs),
            "validation_pairs": len(validation_pairs),
            "train_examples": len(train_dataset),
            "validation_examples": len(validation_dataset),
            "tokens": train_dataset.token_count(),
            "batches_per_epoch": len(train_loader),
            "estimated_steps_per_epoch": estimate_steps_per_epoch(
                train_dataset, max_tokens=config.data.max_tokens_per_batch
            ),
            "padding": padding_waste(
                validation_dataset, max_tokens=config.data.max_tokens_per_batch
            ),
        },
    )
    config.to_yaml(results_dir / "config.yaml")

    return trainer.fit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="override config")
    parser.add_argument("--device", default=None, help="override config")
    parser.add_argument(
        "--limit-train-pairs",
        type=int,
        default=None,
        help="train on a subset, for a quick end-to-end smoke test",
    )
    parser.add_argument("--name", default=None, help="override the run name")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    if args.name:
        config.name = args.name
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.device is not None:
        config.train.device = args.device
    if args.limit_train_pairs is not None:
        config.data.limit_train_pairs = args.limit_train_pairs
    config = config.resolve()

    setup_logging(log_file=Path(config.train.results_dir) / "train.log")
    logger.info("Experiment '%s': %s", config.name, config.description)

    run(config, resume=args.resume)


if __name__ == "__main__":
    main()
