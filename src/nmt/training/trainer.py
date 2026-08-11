"""The training loop.

Written by hand rather than delegated to a high-level trainer, as the brief
requires.  Everything the model does per step is visible here: forward, loss,
backward, clip, step, schedule, log.

Features that earn their complexity
-----------------------------------
*Gradient accumulation* lets a large effective batch be simulated on a small
GPU.  Transformers are known to prefer large batches, and the 8,192-token
batch used here is already near the limit of a free Colab T4; accumulating over
two or four micro-batches reaches the 16k-32k tokens that stabilise training
without needing more memory.

*Gradient clipping* by global norm bounds the damage a single pathological
batch can do.  Transformer gradients are usually well-behaved but occasionally
spike, and one unclipped spike can undo an epoch of progress.

*Mixed precision* (CUDA only) roughly halves memory and speeds up matrix
multiplication.  It is deliberately disabled on Apple's MPS backend, where it
is currently slower than fp32 for models of this size and occasionally produces
NaNs in the attention softmax.

*Validation BLEU during training* is computed on a fixed sample every
evaluation.  Validation loss and BLEU do not peak at the same epoch -- loss
usually starts rising while BLEU is still improving -- so selecting the
checkpoint on loss alone leaves quality on the table.  Both curves are logged
and the report shows the divergence.

Everything numeric is appended to ``train_log.jsonl`` as it happens, so an
interrupted Colab session still leaves a complete, plottable history behind.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from nmt.config import ExperimentConfig
from nmt.training.loss import LabelSmoothedCrossEntropy, perplexity
from nmt.training.scheduler import build_scheduler
from nmt.utils.devices import count_parameters, describe_device, supports_amp
from nmt.utils.io import append_jsonl, ensure_dir, write_json
from nmt.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class EpochMetrics:
    """Aggregated numbers for one pass over a dataset."""

    loss: float = 0.0
    nll: float = 0.0
    accuracy: float = 0.0
    perplexity: float = 0.0
    tokens: int = 0
    seconds: float = 0.0
    tokens_per_second: float = 0.0
    bleu: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MetricAccumulator:
    """Token-weighted running averages.

    Averaging per *batch* would over-weight the short-sentence batches, which
    the token-bucketing sampler makes numerous. Weighting by token count gives
    the true corpus-level mean.
    """

    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.nll_sum = 0.0
        self.correct = 0.0
        self.tokens = 0

    def update(self, stats: dict[str, float]) -> None:
        tokens = stats["tokens"]
        self.loss_sum += stats["loss"] * tokens
        self.nll_sum += stats["nll"] * tokens
        self.correct += stats["accuracy"] * tokens
        self.tokens += tokens

    def result(self, seconds: float = 0.0) -> EpochMetrics:
        tokens = max(1, self.tokens)
        nll = self.nll_sum / tokens
        return EpochMetrics(
            loss=self.loss_sum / tokens,
            nll=nll,
            accuracy=self.correct / tokens,
            perplexity=perplexity(nll),
            tokens=self.tokens,
            seconds=seconds,
            tokens_per_second=self.tokens / seconds if seconds > 0 else 0.0,
        )


class Trainer:
    """Owns the optimisation loop for one experiment.

    Parameters
    ----------
    model
        Any module whose ``forward(source, target_input)`` returns logits --
        the transformer and the LSTM baseline both satisfy this, which is what
        lets them share this loop.
    config
        The resolved experiment configuration.
    train_loader, validation_loader
        Batched data.
    device
        Where to run.
    bleu_callback
        Optional ``callable(model) -> dict`` invoked at each evaluation to
        compute BLEU. Injected rather than imported so that
        :mod:`nmt.training` does not depend on :mod:`nmt.evaluation`.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        *,
        device: torch.device,
        bleu_callback: Callable[[nn.Module], dict[str, float]] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.device = device
        self.bleu_callback = bleu_callback

        self.checkpoint_dir = ensure_dir(config.train.checkpoint_dir)
        self.results_dir = ensure_dir(config.train.results_dir)
        self.log_path = self.results_dir / "train_log.jsonl"

        self.criterion = LabelSmoothedCrossEntropy(
            smoothing=config.optim.label_smoothing
        )
        self.optimizer = self._build_optimizer()

        steps_per_epoch = max(1, len(train_loader))
        self.total_steps = (
            steps_per_epoch
            * config.train.epochs
            // max(1, config.optim.gradient_accumulation_steps)
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            config.optim.scheduler,
            d_model=config.model.d_model,
            warmup_steps=config.optim.warmup_steps,
            total_steps=self.total_steps,
        )

        self.use_amp = config.train.amp and supports_amp(device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.global_step = 0
        self.best_validation_loss = math.inf
        self.best_bleu = -math.inf
        self.epochs_without_improvement = 0
        self.history: list[dict[str, Any]] = []

    # --- setup --------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build the optimiser with decay applied only where it belongs.

        Weight decay on LayerNorm gains and on bias terms is a mistake that is
        easy to make and quietly harmful: shrinking a normalisation gain
        towards zero suppresses the signal the layer is meant to pass through.
        Only genuine weight matrices are decayed.
        """
        decay, no_decay = [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(parameter)
            else:
                decay.append(parameter)

        groups = [
            {"params": decay, "weight_decay": self.config.optim.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        logger.info(
            "Optimiser groups: %s decayed tensors, %s undecayed",
            len(decay),
            len(no_decay),
        )

        kind = self.config.optim.optimizer.lower()
        common = {
            "lr": self.config.optim.learning_rate,
            "betas": tuple(self.config.optim.betas),
            "eps": self.config.optim.eps,
        }
        if kind == "adamw":
            return torch.optim.AdamW(groups, **common)
        if kind == "adam":
            return torch.optim.Adam(groups, **common)
        raise ValueError(f"unknown optimizer {kind!r}")

    # --- one epoch ----------------------------------------------------------

    def train_epoch(self, epoch: int) -> EpochMetrics:
        """One pass over the training data."""
        self.model.train()
        accumulator = MetricAccumulator()
        accumulation = max(1, self.config.optim.gradient_accumulation_steps)
        started = time.time()

        sampler = getattr(self.train_loader, "batch_sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        self.optimizer.zero_grad(set_to_none=True)

        for index, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)

            with torch.autocast(
                device_type=self.device.type, enabled=self.use_amp, dtype=torch.float16
            ):
                logits = self.model(batch.source, batch.decoder_input)
                loss, stats = self.criterion(logits, batch.labels)

            # Scale so that the accumulated gradient equals the gradient of the
            # mean loss over the whole effective batch, not their sum.
            self.scaler.scale(loss / accumulation).backward()
            accumulator.update(stats)

            if (index + 1) % accumulation == 0:
                # Unscale before clipping: clipping a scaled gradient would
                # apply a threshold that depends on the loss scaler's current
                # (and constantly changing) value.
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.optim.max_grad_norm
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1

                if self.global_step % self.config.train.log_every == 0:
                    self._log_step(epoch, stats, float(grad_norm))

        return accumulator.result(time.time() - started)

    def _log_step(self, epoch: int, stats: dict[str, float], grad_norm: float) -> None:
        record = {
            "event": "step",
            "epoch": epoch,
            "step": self.global_step,
            "loss": round(stats["loss"], 4),
            "nll": round(stats["nll"], 4),
            "accuracy": round(stats["accuracy"], 4),
            "learning_rate": self.scheduler.get_last_lr()[0],
            "grad_norm": round(grad_norm, 4),
        }
        append_jsonl(self.log_path, record)
        logger.info(
            "epoch %2d | step %6d | loss %.3f | ppl %6.1f | acc %.3f | lr %.2e",
            epoch,
            self.global_step,
            stats["loss"],
            perplexity(stats["nll"]),
            stats["accuracy"],
            record["learning_rate"],
        )

    @torch.no_grad()
    def evaluate(self, loader: DataLoader | None = None) -> EpochMetrics:
        """Teacher-forced loss over a dataset (no decoding)."""
        loader = loader or self.validation_loader
        self.model.eval()
        accumulator = MetricAccumulator()
        started = time.time()

        for batch in loader:
            batch = batch.to(self.device)
            with torch.autocast(
                device_type=self.device.type, enabled=self.use_amp, dtype=torch.float16
            ):
                logits = self.model(batch.source, batch.decoder_input)
                _, stats = self.criterion(logits, batch.labels)
            accumulator.update(stats)

        return accumulator.result(time.time() - started)

    # --- full run -----------------------------------------------------------

    def fit(self) -> dict[str, Any]:
        """Run training to completion or to early stopping.

        Returns the run manifest, which is also written to
        ``<results_dir>/training_summary.json`` and is what the report reads.
        """
        parameters = count_parameters(self.model)
        manifest: dict[str, Any] = {
            "name": self.config.name,
            "description": self.config.description,
            "config": self.config.to_dict(),
            "device": describe_device(self.device),
            "parameters": parameters,
            "amp": self.use_amp,
            "steps_per_epoch": len(self.train_loader),
            "planned_total_steps": self.total_steps,
            "history": self.history,
        }
        write_json(self.results_dir / "training_summary.json", manifest)

        logger.info(
            "Starting '%s': %s parameters, %s batches/epoch, %d epochs on %s",
            self.config.name,
            f"{parameters['total']:,}",
            f"{len(self.train_loader):,}",
            self.config.train.epochs,
            self.device,
        )

        freeze_epochs = self.config.model.freeze_embeddings_epochs
        run_started = time.time()

        for epoch in range(1, self.config.train.epochs + 1):
            # Unfreeze pre-trained embeddings once the rest of the model has
            # adapted to their geometry.
            if (
                freeze_epochs
                and epoch == freeze_epochs + 1
                and hasattr(self.model, "freeze_embeddings")
            ):
                self.model.freeze_embeddings(False)
                logger.info("Unfroze embeddings at epoch %d", epoch)
                # Rebuild the optimiser so the newly-trainable embedding
                # parameters are actually placed in a parameter group.
                self.optimizer = self._build_optimizer()

            train_metrics = self.train_epoch(epoch)
            validation_metrics = self.evaluate()

            if self.bleu_callback is not None:
                scores = self.bleu_callback(self.model)
                validation_metrics.bleu = scores.get("bleu")
            else:
                scores = {}

            record = {
                "event": "epoch",
                "epoch": epoch,
                "step": self.global_step,
                "train": train_metrics.to_dict(),
                "validation": validation_metrics.to_dict(),
                "bleu_detail": scores,
                "learning_rate": self.scheduler.get_last_lr()[0],
                "elapsed_seconds": round(time.time() - run_started, 1),
            }
            self.history.append(record)
            append_jsonl(self.log_path, record)

            logger.info(
                "epoch %2d | train loss %.3f ppl %6.1f | val loss %.3f ppl %6.1f%s | %.0f tok/s",
                epoch,
                train_metrics.loss,
                train_metrics.perplexity,
                validation_metrics.loss,
                validation_metrics.perplexity,
                f" | val BLEU {validation_metrics.bleu:.2f}"
                if validation_metrics.bleu is not None
                else "",
                train_metrics.tokens_per_second,
            )

            improved = validation_metrics.loss < self.best_validation_loss
            if improved:
                self.best_validation_loss = validation_metrics.loss
                self.epochs_without_improvement = 0
                self._save("best_loss.pt", epoch, record)
            else:
                self.epochs_without_improvement += 1

            if (
                validation_metrics.bleu is not None
                and validation_metrics.bleu > self.best_bleu
            ):
                self.best_bleu = validation_metrics.bleu
                self._save("best_bleu.pt", epoch, record)

            self._save("last.pt", epoch, record)
            manifest["history"] = self.history
            write_json(self.results_dir / "training_summary.json", manifest)

            if self.epochs_without_improvement >= self.config.train.early_stopping_patience:
                logger.info(
                    "Early stopping: validation loss has not improved for %d epochs.",
                    self.epochs_without_improvement,
                )
                break

        manifest["history"] = self.history
        manifest["best_validation_loss"] = self.best_validation_loss
        manifest["best_validation_bleu"] = (
            self.best_bleu if self.best_bleu > -math.inf else None
        )
        manifest["total_seconds"] = round(time.time() - run_started, 1)
        manifest["epochs_completed"] = len(self.history)
        write_json(self.results_dir / "training_summary.json", manifest)

        logger.info(
            "Finished '%s' in %.1f min. Best validation loss %.4f%s",
            self.config.name,
            manifest["total_seconds"] / 60,
            self.best_validation_loss,
            f", best BLEU {self.best_bleu:.2f}" if self.best_bleu > -math.inf else "",
        )
        return manifest

    # --- checkpointing ------------------------------------------------------

    def _save(self, filename: str, epoch: int, record: dict[str, Any]) -> Path:
        """Write a checkpoint carrying everything needed to resume or serve it."""
        path = self.checkpoint_dir / filename
        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "model_config": self.config.model.to_dict(),
            "experiment_config": self.config.to_dict(),
            "tokenizer_path": self.config.data.tokenizer,
            "epoch": epoch,
            "global_step": self.global_step,
            "metrics": record,
        }
        torch.save(payload, path)
        return path

    def load(self, path: Path | str, *, resume: bool = True) -> dict[str, Any]:
        """Restore a checkpoint, optionally including optimiser state.

        Resuming with the optimiser and scheduler state is what makes an
        interrupted Colab session recoverable: without the Adam moments and the
        step counter, restarting would re-enter the warmup phase and undo
        progress.
        """
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state"])

        if resume:
            self.optimizer.load_state_dict(payload["optimizer_state"])
            self.scheduler.load_state_dict(payload["scheduler_state"])
            self.global_step = payload.get("global_step", 0)
            logger.info(
                "Resumed from %s at epoch %s, step %s",
                Path(path).name,
                payload.get("epoch"),
                self.global_step,
            )
        return payload
