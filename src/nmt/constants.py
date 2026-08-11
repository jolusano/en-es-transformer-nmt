"""Vocabulary constants shared by every component of the pipeline.

Both translation directions are handled by a *single* model, so the source and
target share one joint vocabulary.  Two reserved *direction tags* tell the
encoder which way to translate; they are prepended to the source sentence:

    "<2en> Buenos dias."   ->   "Good morning."
    "<2es> Good morning."  ->   "Buenos dias."

This is the tagging trick introduced by Johnson et al. (2017) for Google's
multilingual NMT system.  It costs two vocabulary entries and lets one set of
weights serve both directions, which roughly doubles the amount of supervision
each parameter receives.

The four core special tokens occupy fixed low indices.  Index 0 is ``<pad>`` so
that a zero-filled tensor is a valid, fully-padded batch, and so that
``padding_idx=0`` can be passed straight to :class:`torch.nn.Embedding`.
"""

from __future__ import annotations

from typing import Final

# --- Core special tokens (fixed indices) -----------------------------------
PAD_TOKEN: Final[str] = "<pad>"
UNK_TOKEN: Final[str] = "<unk>"
BOS_TOKEN: Final[str] = "<s>"
EOS_TOKEN: Final[str] = "</s>"

PAD_ID: Final[int] = 0
UNK_ID: Final[int] = 1
BOS_ID: Final[int] = 2
EOS_ID: Final[int] = 3

SPECIAL_TOKENS: Final[tuple[str, ...]] = (
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
)

# --- Direction tags ---------------------------------------------------------
TAG_EN: Final[str] = "<2en>"  # "translate into English"
TAG_ES: Final[str] = "<2es>"  # "translate into Spanish"

DIRECTION_TAGS: Final[tuple[str, ...]] = (TAG_EN, TAG_ES)

#: Maps a direction identifier to the tag that must prefix the source sentence.
#: ``"es-en"`` means "source is Spanish, target is English", so it takes
#: ``<2en>``.
DIRECTION_TO_TAG: Final[dict[str, str]] = {
    "es-en": TAG_EN,
    "en-es": TAG_ES,
}

#: The two supported directions, in a stable order for reporting.
DIRECTIONS: Final[tuple[str, ...]] = ("en-es", "es-en")

#: Human-readable names used in figures and tables.
DIRECTION_NAMES: Final[dict[str, str]] = {
    "en-es": "English -> Spanish",
    "es-en": "Spanish -> English",
}

#: Language code of the *source* side of each direction.
SOURCE_LANG: Final[dict[str, str]] = {"en-es": "en", "es-en": "es"}

#: Language code of the *target* side of each direction.
TARGET_LANG: Final[dict[str, str]] = {"en-es": "es", "es-en": "en"}

#: Every reserved symbol, in index order.  ``SPECIAL_TOKENS`` first so their
#: ids stay pinned to 0-3 regardless of how many tags are added later.
RESERVED_TOKENS: Final[tuple[str, ...]] = SPECIAL_TOKENS + DIRECTION_TAGS


def opposite(direction: str) -> str:
    """Return the reverse of ``direction`` (``"en-es"`` <-> ``"es-en"``)."""
    if direction not in DIRECTION_TO_TAG:
        raise ValueError(
            f"unknown direction {direction!r}; expected one of {DIRECTIONS}"
        )
    source, target = direction.split("-")
    return f"{target}-{source}"
