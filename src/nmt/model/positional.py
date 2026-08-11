"""Positional encoding.

Self-attention is **permutation-equivariant**: it computes a weighted sum over
the value vectors, and a sum does not care about order.  Shuffle the words of
the input and every attention output is shuffled identically -- the model
literally cannot tell "the dog bit the man" from "the man bit the dog".  Since
word order is most of what distinguishes a translation from a bag of words,
position has to be injected explicitly.

Two schemes are implemented so the report can justify the choice empirically.

Sinusoidal (Vaswani et al., 2017; the default here)
    Fixed, parameter-free encodings built from sine and cosine waves at
    geometrically-spaced frequencies:

    .. math::
        PE_{(pos, 2i)}   &= \\sin\\!\\left(pos / 10000^{2i/d}\\right) \\\\
        PE_{(pos, 2i+1)} &= \\cos\\!\\left(pos / 10000^{2i/d}\\right)

    Each dimension is a sinusoid whose wavelength runs from :math:`2\\pi` up to
    :math:`10000 \\cdot 2\\pi`, so the vector for a position is something like a
    binary counter written in continuous values: fast dimensions distinguish
    neighbours, slow dimensions distinguish distant regions.

    The property that motivates the design is that **relative** position is a
    linear function of absolute position.  For any fixed offset *k* there is a
    matrix :math:`M_k` (a 2x2 rotation applied per frequency pair) with
    :math:`PE_{pos+k} = M_k \\, PE_{pos}`.  Attention can therefore learn
    "attend three tokens back" as a single linear map that works at every
    position, and -- because the formula is a function, not a lookup table --
    it extends to sequence lengths never seen in training.

Learned
    An ordinary embedding table indexed by position.  Slightly more flexible,
    marginally better on in-distribution lengths, and strictly unable to
    handle a sentence longer than the table.  Included as an ablation.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sine/cosine positional encodings added to the token embeddings.

    Parameters
    ----------
    d_model
        Model width; must be even so that sine and cosine pair up.
    max_length
        Size of the pre-computed table.  Longer inputs are handled by
        extending the table on the fly, so this is a cache size rather than a
        hard limit.
    dropout
        Applied *after* adding the positional signal, as in the original paper.
    """

    def __init__(
        self, d_model: int, *, max_length: int = 5_000, dropout: float = 0.1
    ) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")

        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # `register_buffer` puts the table in `state_dict` and moves it with
        # `.to(device)`, but keeps it out of `parameters()` so the optimiser
        # never touches it.
        self.register_buffer(
            "encoding", self._build_table(d_model, max_length), persistent=False
        )

    @staticmethod
    def _build_table(d_model: int, max_length: int) -> torch.Tensor:
        """Compute the ``(1, max_length, d_model)`` encoding table."""
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)

        # The exponent is evaluated in log-space: computing 10000^(-2i/d)
        # directly underflows to zero in float32 for large i.
        index = torch.arange(0, d_model, 2, dtype=torch.float32)
        frequency = torch.exp(-math.log(10_000.0) * index / d_model)

        table = torch.zeros(max_length, d_model)
        table[:, 0::2] = torch.sin(position * frequency)
        table[:, 1::2] = torch.cos(position * frequency)
        return table.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to ``x`` of shape ``(batch, seq, d_model)``."""
        length = x.size(1)

        if length > self.encoding.size(1):
            # Grow the cache rather than failing: the formula is defined for
            # every position, which is the point of using it.
            self.encoding = self._build_table(self.d_model, length).to(
                device=x.device, dtype=self.encoding.dtype
            )

        return self.dropout(x + self.encoding[:, :length].to(x.dtype))

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, max_length={self.encoding.size(1)}"


class LearnedPositionalEncoding(nn.Module):
    """Trainable absolute position embeddings (ablation baseline)."""

    def __init__(
        self, d_model: int, *, max_length: int = 512, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(max_length, d_model)
        self.dropout = nn.Dropout(dropout)
        # Match the scale of the sinusoidal table so the two variants start
        # from a comparable signal-to-noise ratio against the token embeddings.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        if length > self.max_length:
            raise ValueError(
                f"sequence length {length} exceeds the learned table "
                f"({self.max_length}); this is the failure mode sinusoidal "
                "encodings do not have"
            )
        positions = torch.arange(length, device=x.device)
        return self.dropout(x + self.embedding(positions).unsqueeze(0))

    def extra_repr(self) -> str:
        return f"max_length={self.max_length}"


def build_positional_encoding(
    kind: str, d_model: int, *, max_length: int = 5_000, dropout: float = 0.1
) -> nn.Module:
    """Factory dispatching on the config string."""
    kind = kind.lower()
    if kind == "sinusoidal":
        return SinusoidalPositionalEncoding(
            d_model, max_length=max_length, dropout=dropout
        )
    if kind == "learned":
        return LearnedPositionalEncoding(
            d_model, max_length=min(max_length, 512), dropout=dropout
        )
    raise ValueError(f"unknown positional encoding {kind!r}")
