"""Attention masks.

A transformer needs two logically different masks, and conflating them is one
of the most common sources of silent bugs in seq2seq code, so they are built by
two separate functions here:

**Padding mask** -- "this position is filler, ignore it".  Batching sentences of
different lengths requires padding to a rectangle; without a mask the model
would attend to ``<pad>`` and, worse, the encoder's representation of a short
sentence would depend on how long the *other* sentences in its batch happened
to be.

**Causal (look-ahead) mask** -- "this position may not see the future".  During
training the decoder is given the entire target sentence at once so all
positions can be computed in parallel, but position *t* must only attend to
positions ``<= t``.  Without this the model trivially learns to copy the next
token from its input and scores near-perfect training loss while being useless
at inference, where the future does not exist.

Convention used throughout the project: **``True`` means "attend here"**.
The alternative convention (``True`` means "block") is equally common and
mixing the two silently inverts the model, so every mask function in this
module returns keep-masks and every consumer treats them that way.

Shapes are broadcast-ready for attention scores of shape
``(batch, heads, query_len, key_len)``.
"""

from __future__ import annotations

import torch

from nmt.constants import PAD_ID


def padding_mask(tokens: torch.Tensor, pad_id: int = PAD_ID) -> torch.Tensor:
    """Mark real (non-pad) positions in a batch of token ids.

    Parameters
    ----------
    tokens
        ``(batch, seq_len)`` integer ids.

    Returns
    -------
    torch.Tensor
        Boolean ``(batch, 1, 1, seq_len)``.  The two singleton axes broadcast
        over heads and over query positions: the mask says which *keys* are
        usable, and that is the same for every head and every query.
    """
    return tokens.ne(pad_id).unsqueeze(1).unsqueeze(2)


def causal_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    """Lower-triangular mask allowing each position to see itself and the past.

    Returns
    -------
    torch.Tensor
        Boolean ``(1, 1, size, size)``.  Entry ``[.., i, j]`` is ``True`` when
        ``j <= i``.

    Examples
    --------
    For ``size=4`` the mask is::

        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]

    Row *i* is the set of keys query *i* may look at.  The diagonal is included
    because a position must be able to attend to itself -- the decoder input at
    step *i* is the token *before* the one being predicted, so seeing it is not
    cheating.
    """
    mask = torch.ones(size, size, dtype=torch.bool, device=device).tril()
    return mask.unsqueeze(0).unsqueeze(0)


def decoder_mask(
    tokens: torch.Tensor, pad_id: int = PAD_ID
) -> torch.Tensor:
    """Combine padding and causal constraints for decoder self-attention.

    A key is attendable only if it is **both** a real token **and** not in the
    future, hence the logical AND.

    Returns
    -------
    torch.Tensor
        Boolean ``(batch, 1, seq_len, seq_len)``.
    """
    length = tokens.size(1)
    return padding_mask(tokens, pad_id) & causal_mask(length, tokens.device)


def cross_attention_mask(source_tokens: torch.Tensor, pad_id: int = PAD_ID) -> torch.Tensor:
    """Mask for decoder-to-encoder attention.

    Only the *source* padding matters here.  The decoder is allowed to look at
    every real source position from any target position -- translation is not
    monotonic, and forbidding a target word from attending to a later source
    word would make it impossible to handle the adjective-noun reordering that
    English and Spanish require ("red house" / "casa roja").

    Returns
    -------
    torch.Tensor
        Boolean ``(batch, 1, 1, source_len)``.
    """
    return padding_mask(source_tokens, pad_id)


def apply_mask(scores: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Set masked attention scores to the lowest finite value of their dtype.

    ``-inf`` would be the mathematically natural choice, but a query whose keys
    are *all* masked -- which happens for a fully-padded row in the last batch
    of an epoch -- then produces ``softmax([-inf, ...]) = NaN`` and poisons
    every gradient in the step.  ``torch.finfo(dtype).min`` gives a uniform
    distribution over the masked row instead, which is harmless because that
    row's output is discarded by the loss mask anyway.
    """
    if mask is None:
        return scores
    return scores.masked_fill(~mask, torch.finfo(scores.dtype).min)


def lengths_to_mask(lengths: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    """Convert a vector of sequence lengths into a boolean keep-mask.

    Used by the LSTM baseline, which tracks lengths rather than pad ids.
    """
    max_length = max_length or int(lengths.max().item())
    positions = torch.arange(max_length, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)
