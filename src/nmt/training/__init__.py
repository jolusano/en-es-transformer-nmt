"""Optimisation: loss, schedules and the training loop."""

from nmt.training.loss import LabelSmoothedCrossEntropy, perplexity
from nmt.training.scheduler import (
    build_scheduler,
    cosine_schedule,
    inverse_sqrt_schedule,
    schedule_preview,
)
from nmt.training.trainer import EpochMetrics, MetricAccumulator, Trainer

__all__ = [
    "EpochMetrics",
    "LabelSmoothedCrossEntropy",
    "MetricAccumulator",
    "Trainer",
    "build_scheduler",
    "cosine_schedule",
    "inverse_sqrt_schedule",
    "perplexity",
    "schedule_preview",
]
