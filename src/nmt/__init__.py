"""Bidirectional English/Spanish neural machine translation.

AIG230 — Natural Language Processing, Seneca Polytechnic.
Final Project, Option 1: Transformer-Based Neural Machine Translation.

Group 7
    Jose Luis Sanchez Noriega
    Bikash Subedi

The package is organised so that each stage of the pipeline lives in its own
sub-package and can be exercised independently:

    nmt.data        corpus acquisition, cleaning, splitting, tokenisation
    nmt.model       transformer components written from primitives
    nmt.training    the optimisation loop and its schedules
    nmt.inference   autoregressive decoding (greedy and beam search)
    nmt.evaluation  BLEU / chrF scoring and error analysis
    nmt.viz         figure generation
    nmt.utils       seeding, logging and I/O helpers
"""

__version__ = "1.0.0"
__authors__ = ("Jose Luis Sanchez Noriega", "Bikash Subedi")

import importlib
from typing import TYPE_CHECKING

_SUBPACKAGES = (
    "data",
    "evaluation",
    "inference",
    "model",
    "training",
    "utils",
    "viz",
)

__all__ = [*_SUBPACKAGES, "__version__"]


def __getattr__(name: str):
    """Import sub-packages lazily.

    ``nmt.viz`` pulls in matplotlib and ``nmt.model`` pulls in torch; importing
    them eagerly would make ``python -m nmt.data.build`` pay for both. Lazy
    access keeps ``import nmt`` cheap while ``nmt.model`` still works.
    """
    if name in _SUBPACKAGES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover - helps editors resolve the lazy imports
    from nmt import (  # noqa: F401  (re-exported lazily via __getattr__)
        data,
        evaluation,
        inference,
        model,
        training,
        utils,
        viz,
    )
