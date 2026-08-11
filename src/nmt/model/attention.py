"""Multi-head attention, built from primitives.

Implemented with explicit tensor operations rather than
:class:`torch.nn.MultiheadAttention` so that every step of the computation is
visible and the attention weights can be pulled out for the visualisations in
the report.

The computation, in one line:

.. math::
    \\mathrm{Attention}(Q, K, V) =
        \\mathrm{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right) V

Read it as a **differentiable dictionary lookup**.  Each query vector is
compared against every key by dot product; the scores are turned into a
probability distribution by softmax; the output is that distribution's weighted
average of the value vectors.  A hard lookup would take the single best-matching
key -- attention takes a soft blend, which is what makes it differentiable and
therefore trainable.

Why divide by :math:`\\sqrt{d_k}`
---------------------------------
If the components of *q* and *k* are independent with mean 0 and variance 1,
then :math:`q \\cdot k = \\sum_{i=1}^{d_k} q_i k_i` has mean 0 and variance
:math:`d_k`, so its standard deviation grows like :math:`\\sqrt{d_k}`.  With
:math:`d_k = 64` the raw scores would routinely reach +-8 or more.  Softmax over
values that spread out saturates: it becomes nearly one-hot, and the gradient
it passes back is proportional to :math:`p(1-p) \\approx 0`.  Dividing by
:math:`\\sqrt{d_k}` restores unit variance, keeps softmax in its responsive
range, and is the difference between a model that trains and one that stalls.

Why *multiple* heads
--------------------
A single softmax produces one distribution per query, so one head can only
attend to essentially one place at a time.  Translating "the red house" into
"la casa roja" requires simultaneously tracking the noun (for gender agreement
on both the article and the adjective) and the adjective (to move it after the
noun).  Splitting :math:`d_{model}` into *h* independent subspaces of size
:math:`d_{model}/h` lets *h* such relations be attended to in parallel, at
identical total cost -- the split is a reshape, not extra computation.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from nmt.model.masking import apply_mask


class ScaledDotProductAttention(nn.Module):
    """The attention kernel, isolated so it can be unit-tested on its own."""

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        query
            ``(batch, heads, query_len, d_k)``
        key, value
            ``(batch, heads, key_len, d_k)``
        mask
            Boolean keep-mask broadcastable to
            ``(batch, heads, query_len, key_len)``; ``True`` means attendable.

        Returns
        -------
        (context, weights)
            ``context`` is ``(batch, heads, query_len, d_k)``;
            ``weights`` is ``(batch, heads, query_len, key_len)`` and is
            returned rather than discarded so the report can plot it.
        """
        d_k = query.size(-1)

        # (B, H, Lq, d_k) @ (B, H, d_k, Lk) -> (B, H, Lq, Lk)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = apply_mask(scores, mask)

        weights = torch.softmax(scores, dim=-1)
        # Dropout on the *weights* (not the output) is what the original paper
        # specifies: it randomly forces the model to route information through
        # alternative positions, which stops any single alignment from becoming
        # load-bearing.
        weights = self.dropout(weights)

        # (B, H, Lq, Lk) @ (B, H, Lk, d_k) -> (B, H, Lq, d_k)
        context = torch.matmul(weights, value)
        return context, weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention with separate Q/K/V/O projections.

    Parameters
    ----------
    d_model
        Model width.  Must be divisible by ``num_heads``.
    num_heads
        Number of parallel attention subspaces.  Each head works in
        ``d_k = d_model // num_heads`` dimensions, so total compute is
        independent of the head count.
    dropout
        Dropout on the attention weights.
    bias
        Whether the four projections carry bias terms.  The original paper
        omits them; they are kept optional and default to ``True`` because
        they cost 4 * d_model parameters and marginally help on small data.

    Notes
    -----
    The four projections are kept as separate :class:`~torch.nn.Linear` modules
    rather than one fused matrix.  A fused ``qkv`` projection is faster, but it
    only works for *self*-attention (where Q, K and V share an input) and this
    same class also serves cross-attention, where Q comes from the decoder and
    K/V come from the encoder.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_query = nn.Linear(d_model, d_model, bias=bias)
        self.w_key = nn.Linear(d_model, d_model, bias=bias)
        self.w_value = nn.Linear(d_model, d_model, bias=bias)
        self.w_out = nn.Linear(d_model, d_model, bias=bias)

        self.attention = ScaledDotProductAttention(dropout=dropout)

        #: Attention weights from the most recent forward pass, kept for
        #: visualisation.  Detached, so holding on to it cannot leak the graph.
        self.last_attention_weights: torch.Tensor | None = None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Xavier-uniform initialisation for the projections.

        Xavier keeps the variance of activations roughly constant through the
        layer, which matters here because the residual stream accumulates the
        output of every block: a projection that amplified its input would
        compound across depth.
        """
        for module in (self.w_query, self.w_key, self.w_value, self.w_out):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, L, d_model)`` -> ``(B, H, L, d_k)``.

        The transpose is what puts the head axis next to the batch axis, so the
        subsequent ``matmul`` treats ``(B, H)`` as independent problems and runs
        all heads in one batched operation.
        """
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, H, L, d_k)`` -> ``(B, L, d_model)``: the inverse of the split."""
        batch, _, length, _ = x.shape
        # `contiguous()` is required because `transpose` only changes strides,
        # and `view` needs a contiguous buffer.
        return x.transpose(1, 2).contiguous().view(batch, length, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        store_attention: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        query
            ``(batch, query_len, d_model)``.
        key, value
            ``(batch, key_len, d_model)``.  For self-attention all three are
            the same tensor; for cross-attention ``query`` is the decoder
            state and ``key``/``value`` are the encoder output.
        mask
            Boolean keep-mask broadcastable to the score shape.
        store_attention
            Retain the weights on ``self.last_attention_weights``.  Off during
            training so the graph is not pinned in memory.
        """
        q = self._split_heads(self.w_query(query))
        k = self._split_heads(self.w_key(key))
        v = self._split_heads(self.w_value(value))

        context, weights = self.attention(q, k, v, mask)

        if store_attention:
            self.last_attention_weights = weights.detach()

        # The output projection is not cosmetic: without it the heads' outputs
        # would simply be concatenated, and there would be no way for
        # information discovered by one head to be mixed into the subspace
        # another head reads from.
        return self.w_out(self._merge_heads(context))

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, num_heads={self.num_heads}, d_k={self.d_k}"
