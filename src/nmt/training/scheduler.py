"""Learning-rate schedules.

Transformers are unusually sensitive to the learning rate early in training,
and the reason is structural rather than incidental.  At initialisation the
attention weights are near-uniform, so every position's output is roughly the
average of all positions and carries almost no information about *which*
position mattered.  The gradients that result are dominated by noise.  Adam
normalises by a running estimate of the gradient's second moment, and that
estimate is itself unreliable in the first few dozen steps -- so a large step
taken then is large in an essentially arbitrary direction.  The classic symptom
is a loss that falls for 200 steps and then diverges to NaN.

Warmup fixes this by starting the learning rate near zero and ramping it up
linearly, giving the second-moment estimate time to stabilise before the model
takes meaningful steps.

Two schedules are provided.

``inverse_sqrt`` (the default, from Vaswani et al., 2017)
    .. math::
        lr(t) = d_{model}^{-0.5} \\cdot
                \\min\\!\\left(t^{-0.5},\\; t \\cdot t_{warmup}^{-1.5}\\right)

    Linear ramp to a peak at ``t = warmup_steps``, then decay proportional to
    :math:`1/\\sqrt{t}`.  The :math:`d_{model}^{-0.5}` factor makes the peak
    scale sensibly with model width.  Its practical virtue is that it is not
    parameterised by the *total* number of steps, so a run can be stopped early
    or extended without invalidating the schedule -- which matters when
    training in Colab sessions that may be interrupted.

``cosine``
    Linear warmup then a cosine decay to a small floor.  Usually a little
    better *if* the total step budget is known in advance and actually
    completed.  Offered for the ablation.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def inverse_sqrt_schedule(
    optimizer: Optimizer, *, d_model: int, warmup_steps: int
) -> LambdaLR:
    """The original transformer schedule.

    Notes
    -----
    ``LambdaLR`` multiplies the optimiser's base learning rate by the returned
    factor, so the base ``lr`` in the config acts as an overall gain on the
    curve rather than as the peak value.  The peak reached at
    ``t = warmup_steps`` is ``base_lr * (d_model * warmup_steps)^-0.5``.
    """
    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be positive")

    scale = d_model**-0.5

    def factor(step: int) -> float:
        # LambdaLR counts from 0; step 0 would divide by zero.
        step = max(1, step)
        return scale * min(step**-0.5, step * warmup_steps**-1.5)

    return LambdaLR(optimizer, lr_lambda=factor)


def cosine_schedule(
    optimizer: Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_factor: float = 0.01,
) -> LambdaLR:
    """Linear warmup followed by cosine decay to ``min_factor``."""
    if total_steps <= warmup_steps:
        raise ValueError("total_steps must exceed warmup_steps")

    def factor(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_factor + (1.0 - min_factor) * cosine

    return LambdaLR(optimizer, lr_lambda=factor)


def constant_schedule(optimizer: Optimizer, *, warmup_steps: int = 0) -> LambdaLR:
    """Constant learning rate after an optional linear warmup (ablation)."""

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0

    return LambdaLR(optimizer, lr_lambda=factor)


def build_scheduler(
    optimizer: Optimizer,
    kind: str,
    *,
    d_model: int,
    warmup_steps: int,
    total_steps: int | None = None,
) -> LambdaLR:
    """Factory dispatching on the config string."""
    kind = kind.lower()
    if kind in ("inverse_sqrt", "noam"):
        return inverse_sqrt_schedule(
            optimizer, d_model=d_model, warmup_steps=warmup_steps
        )
    if kind == "cosine":
        if total_steps is None:
            raise ValueError("the cosine schedule needs total_steps")
        return cosine_schedule(
            optimizer, warmup_steps=warmup_steps, total_steps=total_steps
        )
    if kind == "constant":
        return constant_schedule(optimizer, warmup_steps=warmup_steps)
    raise ValueError(f"unknown scheduler {kind!r}")


def schedule_preview(
    kind: str,
    *,
    d_model: int,
    warmup_steps: int,
    total_steps: int,
    base_lr: float = 1.0,
) -> list[float]:
    """Compute the schedule without an optimiser, for plotting.

    Used by the figure in the report that shows the warmup/decay curve.
    """
    scale = d_model**-0.5
    values: list[float] = []

    for step in range(1, total_steps + 1):
        if kind in ("inverse_sqrt", "noam"):
            factor = scale * min(step**-0.5, step * warmup_steps**-1.5)
        elif kind == "cosine":
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                factor = 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        else:
            factor = 1.0 if step >= warmup_steps else step / max(1, warmup_steps)
        values.append(base_lr * factor)

    return values
