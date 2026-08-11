"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, *, deterministic: bool = False) -> int:
    """Seed every random number generator the project touches.

    Parameters
    ----------
    seed
        The seed applied to :mod:`random`, :mod:`numpy` and :mod:`torch`.
    deterministic
        When ``True``, ask cuDNN for deterministic kernels.  This makes runs
        bit-reproducible on CUDA at a noticeable throughput cost, so it is off
        by default and enabled only for the unit tests.

    Returns
    -------
    int
        The seed that was applied, so it can be recorded in the run manifest.

    Notes
    -----
    Seeding does not make a *multi-worker* :class:`~torch.utils.data.DataLoader`
    deterministic on its own; :func:`worker_init_fn` below handles that, and it
    is wired up in :mod:`nmt.data.dataset`.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # `warn_only` keeps the run alive when an op has no deterministic
        # kernel (several do not on MPS) instead of raising.
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

    return seed


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker a distinct but reproducible seed.

    Without this, forked workers inherit the parent's numpy state and every
    worker draws the *same* "random" numbers.
    """
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)
