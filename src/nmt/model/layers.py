"""Encoder and decoder blocks.

Each block is a stack of **sublayers**, and every sublayer is wrapped in the
same pattern: a residual connection around a normalised transformation.

Residual connections
    ``x + f(x)`` gives the gradient a path back to the input that does not pass
    through ``f``: differentiating gives ``1 + f'(x)``, so even if ``f'``
    vanishes the ``1`` survives. Without them a 6-layer transformer barely
    trains. They also change what each block has to learn: the block only needs
    to compute a *correction* to the running representation rather than
    reproduce it, which is why the residual stream is often described as a
    channel that each layer reads from and writes back into.

Layer normalisation
    Normalises each token's feature vector to zero mean and unit variance
    across the feature axis, then rescales with learned gain and bias. Note it
    is *per token*, unlike batch norm: this matters because sequences have
    different lengths and padded positions would otherwise corrupt batch
    statistics, and because it makes the model behave identically whether it
    sees a batch of 256 or the single sentence a user types into the app.

Pre-LN vs Post-LN
-----------------
The original paper puts normalisation *after* the residual add::

    x = LayerNorm(x + Sublayer(x))            # post-LN

The now-standard variant normalises the sublayer's *input* instead::

    x = x + Sublayer(LayerNorm(x))            # pre-LN

Pre-LN leaves the residual path completely un-normalised from input to output,
so gradients reach the early layers undamped. Post-LN needs a carefully tuned
learning-rate warmup to avoid diverging in the first few hundred steps; pre-LN
trains stably at higher learning rates with a much shorter warmup. This project
defaults to pre-LN for exactly that reason -- a Colab session that diverges at
step 300 is an hour wasted -- and the report includes the comparison. Pre-LN
requires one extra ``LayerNorm`` at the very end of each stack, since the last
sublayer's output would otherwise never be normalised.
"""

from __future__ import annotations

import torch
from torch import nn

from nmt.model.attention import MultiHeadAttention


def _activation(name: str) -> nn.Module:
    """Look up an activation by name."""
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        # SwiGLU-style gating is out of scope; these two cover the ablation.
    }
    key = name.lower()
    if key not in activations:
        raise ValueError(f"unknown activation {name!r}; expected one of {list(activations)}")
    return activations[key]()


class PositionwiseFeedForward(nn.Module):
    """The per-position MLP applied after attention.

    ``Linear(d_model -> d_ff) -> activation -> dropout -> Linear(d_ff -> d_model)``

    "Position-wise" means the *same* MLP is applied independently to every
    position -- it moves no information between tokens. That division of labour
    is the point: attention is the only operation that mixes positions, and the
    feed-forward network is the only operation with nonlinear depth. Attention
    decides *what to look at*; the feed-forward network decides *what to make
    of it*.

    ``d_ff`` is conventionally ``4 * d_model``. The expansion matters because
    a nonlinearity applied in the same dimensionality as its input can express
    much less than one applied in a wider space and projected back; this block
    holds roughly two thirds of the parameters in a transformer layer.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.linear_in = nn.Linear(d_model, d_ff)
        self.activation = _activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(d_ff, d_model)

        nn.init.xavier_uniform_(self.linear_in.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)
        nn.init.zeros_(self.linear_in.bias)
        nn.init.zeros_(self.linear_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_out(self.dropout(self.activation(self.linear_in(x))))


class EncoderLayer(nn.Module):
    """One encoder block: self-attention, then feed-forward.

    The encoder is **bidirectional** -- every position may attend to every
    other, in both directions. It is not predicting anything, it is building a
    representation, so there is no reason to hide the future from it. Only
    padding is masked.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        *,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "relu",
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first

        self.self_attention = MultiHeadAttention(
            d_model, num_heads, dropout=attention_dropout
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model, d_ff, dropout=dropout, activation=activation
        )

        self.norm_attention = nn.LayerNorm(d_model)
        self.norm_feed_forward = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        store_attention: bool = False,
    ) -> torch.Tensor:
        if self.norm_first:
            normed = self.norm_attention(x)
            x = x + self.dropout(
                self.self_attention(
                    normed, normed, normed, mask, store_attention=store_attention
                )
            )
            x = x + self.dropout(self.feed_forward(self.norm_feed_forward(x)))
        else:
            x = self.norm_attention(
                x
                + self.dropout(
                    self.self_attention(x, x, x, mask, store_attention=store_attention)
                )
            )
            x = self.norm_feed_forward(x + self.dropout(self.feed_forward(x)))
        return x


class DecoderLayer(nn.Module):
    """One decoder block: masked self-attention, cross-attention, feed-forward.

    The three sublayers do three different jobs:

    1. **Masked self-attention** over what has been generated so far -- this is
       where the decoder maintains target-language fluency and agreement
       ("casa" ... "roja").
    2. **Cross-attention** into the encoder output -- this is where the source
       sentence actually enters, and its weights are the closest thing the
       model has to a word alignment. The visualisations in the report plot
       exactly this tensor.
    3. **Feed-forward** -- per-position processing, as in the encoder.

    Note the asymmetry in cross-attention: queries come from the decoder, keys
    and values from the encoder. The decoder asks "what am I looking for?" and
    the encoder's memory answers.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        *,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "relu",
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first

        self.self_attention = MultiHeadAttention(
            d_model, num_heads, dropout=attention_dropout
        )
        self.cross_attention = MultiHeadAttention(
            d_model, num_heads, dropout=attention_dropout
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model, d_ff, dropout=dropout, activation=activation
        )

        self.norm_self = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_feed_forward = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        *,
        store_attention: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            ``(batch, target_len, d_model)`` decoder state.
        memory
            ``(batch, source_len, d_model)`` encoder output.
        self_mask
            Combined causal + padding mask for the target side.
        memory_mask
            Source padding mask for cross-attention.
        """
        if self.norm_first:
            normed = self.norm_self(x)
            x = x + self.dropout(
                self.self_attention(
                    normed, normed, normed, self_mask, store_attention=store_attention
                )
            )
            normed = self.norm_cross(x)
            x = x + self.dropout(
                self.cross_attention(
                    normed, memory, memory, memory_mask, store_attention=store_attention
                )
            )
            x = x + self.dropout(self.feed_forward(self.norm_feed_forward(x)))
        else:
            x = self.norm_self(
                x
                + self.dropout(
                    self.self_attention(
                        x, x, x, self_mask, store_attention=store_attention
                    )
                )
            )
            x = self.norm_cross(
                x
                + self.dropout(
                    self.cross_attention(
                        x, memory, memory, memory_mask, store_attention=store_attention
                    )
                )
            )
            x = self.norm_feed_forward(x + self.dropout(self.feed_forward(x)))
        return x


class Encoder(nn.Module):
    """A stack of :class:`EncoderLayer` blocks."""

    def __init__(self, layers: list[EncoderLayer], *, norm_first: bool = True) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        d_model = layers[0].norm_attention.normalized_shape[0]
        # Pre-LN leaves the final sublayer output un-normalised; this closes it.
        self.norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        store_attention: bool = False,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask, store_attention=store_attention)
        return self.norm(x)

    @property
    def attention_weights(self) -> list[torch.Tensor]:
        """Per-layer self-attention maps from the last forward pass."""
        return [
            layer.self_attention.last_attention_weights
            for layer in self.layers
            if layer.self_attention.last_attention_weights is not None
        ]


class Decoder(nn.Module):
    """A stack of :class:`DecoderLayer` blocks."""

    def __init__(self, layers: list[DecoderLayer], *, norm_first: bool = True) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        d_model = layers[0].norm_self.normalized_shape[0]
        self.norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        *,
        store_attention: bool = False,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x, memory, self_mask, memory_mask, store_attention=store_attention
            )
        return self.norm(x)

    @property
    def cross_attention_weights(self) -> list[torch.Tensor]:
        """Per-layer cross-attention maps -- the source/target alignments."""
        return [
            layer.cross_attention.last_attention_weights
            for layer in self.layers
            if layer.cross_attention.last_attention_weights is not None
        ]

    @property
    def self_attention_weights(self) -> list[torch.Tensor]:
        """Per-layer masked self-attention maps."""
        return [
            layer.self_attention.last_attention_weights
            for layer in self.layers
            if layer.self_attention.last_attention_weights is not None
        ]
