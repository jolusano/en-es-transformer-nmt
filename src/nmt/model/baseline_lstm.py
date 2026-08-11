"""Recurrent baseline: bidirectional LSTM encoder + LSTM decoder with attention.

This is the architecture the transformer replaced -- Bahdanau et al. (2015)
with an LSTM instead of a GRU -- and it is included so the report's evaluation
section compares against something meaningful rather than only reporting an
absolute BLEU number that no reader can calibrate.

It is a *fair* baseline, deliberately:

* the same joint vocabulary, the same direction tags, the same data,
* the same embedding width and a hidden size chosen to land within a few
  percent of the transformer's parameter count,
* the same training loop, optimiser, label smoothing and decoding code.

so the comparison isolates the architecture.  What differs is how information
crosses the sentence:

**The recurrent bottleneck.**  The decoder state at step *t* is computed from
the state at *t-1*.  Information from source word 1 reaches target word 20 only
by being carried through 20 sequential updates, and each update multiplies the
signal by a Jacobian -- the path length between two positions is O(n), and
gradients along it decay geometrically.  Attention gives every pair of positions
a path of length O(1).  This is the single most important difference, and it is
what the length-bucketed BLEU breakdown in the report is designed to expose:
the recurrent model should degrade faster as sentences get longer.

**Sequential computation.**  The recurrence cannot be parallelised across time,
so the LSTM's training throughput is bound by sentence length in a way the
transformer's is not, even though the transformer does more arithmetic.

The attention here is single-head additive (Bahdanau) attention rather than
multi-head dot-product attention, which is the historically accurate choice and
also makes the point that "attention" alone was not the transformer's
contribution -- doing away with the recurrence was.
"""

from __future__ import annotations

import torch
from torch import nn

from nmt.constants import PAD_ID


class BahdanauAttention(nn.Module):
    """Additive attention: score(h, s) = v^T tanh(W_h h + W_s s).

    Unlike dot-product attention this has learned parameters in the scoring
    function itself, which is why it does not need the ``1/sqrt(d_k)`` scaling:
    ``v`` can absorb the scale during training.  It is also slower, since the
    scores cannot be computed with a single matrix multiplication.
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.encoder_projection = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.decoder_projection = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        decoder_state: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        decoder_state
            ``(batch, decoder_dim)`` -- the current decoder hidden state.
        encoder_outputs
            ``(batch, source_len, encoder_dim)``.
        mask
            ``(batch, source_len)`` boolean keep-mask.

        Returns
        -------
        (context, weights)
            ``(batch, encoder_dim)`` and ``(batch, source_len)``.
        """
        projected = self.encoder_projection(encoder_outputs)
        query = self.decoder_projection(decoder_state).unsqueeze(1)
        scores = self.score(torch.tanh(projected + query)).squeeze(-1)

        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class AttentionSeq2Seq(nn.Module):
    """Bi-LSTM encoder, single-layer LSTM decoder, additive attention.

    Parameters are sized from the same ``ModelConfig`` the transformer uses so
    that the two models can be compared at matched capacity; ``hidden_size``
    defaults to ``d_model`` and the encoder is bidirectional, so the encoder
    output width is ``2 * hidden_size``.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        embedding_dim: int = 512,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        nn.init.normal_(self.embedding.weight, std=embedding_dim**-0.5)
        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # The bidirectional encoder produces 2*hidden features; the decoder is
        # unidirectional (it cannot see its own future), so its initial state
        # is built by projecting the concatenated final states down.
        self.bridge_hidden = nn.Linear(2 * hidden_size, hidden_size)
        self.bridge_cell = nn.Linear(2 * hidden_size, hidden_size)

        self.attention = BahdanauAttention(2 * hidden_size, hidden_size, hidden_size)

        # The decoder input at each step is [embedding ; context], the
        # input-feeding scheme of Luong et al. (2015): the previous step's
        # context is fed back so the decoder can track what it has already
        # translated.
        self.decoder = nn.LSTM(
            embedding_dim + 2 * hidden_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.output_projection = nn.Linear(hidden_size + 2 * hidden_size, embedding_dim)

        if tie_embeddings:
            self.generator = None
            self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        else:
            self.generator = nn.Linear(embedding_dim, vocab_size)
            self.output_bias = None

        self.last_attention_weights: torch.Tensor | None = None

    def _project(self, features: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.output_projection(features))
        if self.generator is not None:
            return self.generator(hidden)
        return torch.nn.functional.linear(
            hidden, self.embedding.weight, self.output_bias
        )

    def encode(self, source: torch.Tensor) -> tuple[torch.Tensor, tuple, torch.Tensor]:
        """Encode the source sentence.

        Returns the per-position outputs, the initial decoder state, and the
        source keep-mask.
        """
        mask = source.ne(PAD_ID)
        embedded = self.dropout(self.embedding(source))
        outputs, (hidden, cell) = self.encoder(embedded)

        # Concatenate the final forward and backward states of the top layer.
        forward, backward = hidden[-2], hidden[-1]
        initial_hidden = torch.tanh(
            self.bridge_hidden(torch.cat([forward, backward], dim=-1))
        ).unsqueeze(0)
        forward_cell, backward_cell = cell[-2], cell[-1]
        initial_cell = torch.tanh(
            self.bridge_cell(torch.cat([forward_cell, backward_cell], dim=-1))
        ).unsqueeze(0)

        return outputs, (initial_hidden, initial_cell), mask

    def forward(
        self,
        source: torch.Tensor,
        target_input: torch.Tensor,
        *,
        store_attention: bool = False,
    ) -> torch.Tensor:
        """Teacher-forced pass, matching the transformer's signature.

        The time loop is explicit because attention needs the *current*
        decoder state to compute the context, so the steps cannot be run in
        one call to ``nn.LSTM`` -- which is precisely the sequential dependency
        the transformer removes.
        """
        encoder_outputs, state, source_mask = self.encode(source)
        batch, target_length = target_input.shape

        context = encoder_outputs.new_zeros(batch, 2 * self.hidden_size)
        embedded = self.dropout(self.embedding(target_input))

        logits = []
        weights_per_step = []

        for step in range(target_length):
            step_input = torch.cat([embedded[:, step], context], dim=-1).unsqueeze(1)
            output, state = self.decoder(step_input, state)
            decoder_state = output.squeeze(1)

            context, weights = self.attention(
                decoder_state, encoder_outputs, source_mask
            )
            logits.append(self._project(torch.cat([decoder_state, context], dim=-1)))
            if store_attention:
                weights_per_step.append(weights)

        if store_attention and weights_per_step:
            self.last_attention_weights = torch.stack(weights_per_step, dim=1).detach()

        return torch.stack(logits, dim=1)

    def parameter_breakdown(self) -> dict[str, int]:
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        return {
            "embedding": count(self.embedding),
            "encoder": count(self.encoder),
            "decoder": count(self.decoder),
            "attention": count(self.attention),
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
