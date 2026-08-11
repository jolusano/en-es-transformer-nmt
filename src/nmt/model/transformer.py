"""The complete bidirectional translation transformer.

Assembles the components in :mod:`nmt.model.attention`, :mod:`nmt.model.layers`
and :mod:`nmt.model.positional` into one encoder-decoder model that translates
**both** English->Spanish and Spanish->English with a single set of weights.

Data flow for one training step::

    source ids  ->  embed x sqrt(d_model)  ->  + positional  ->  encoder  ->  memory
                                                                              |
    target ids (shifted right) -> embed -> + positional -> decoder <----------+
                                                              |
                                                          generator -> logits

Three design choices distinguish this from a textbook single-direction model.

**One shared vocabulary and one shared embedding table.**  The encoder
embedding, the decoder embedding and the output projection are the *same*
matrix.  Three-way tying is possible only because source and target draw from
the same joint vocabulary, and it removes about 17M parameters at ``d_model=512,
V=16k`` -- roughly a third of the model.  On a corpus this size that is
regularisation as much as it is economy: the alternative is three separate
tables, each seeing a third of the gradient signal.

**A direction tag on the source.**  ``<2es>`` or ``<2en>`` is prepended to
every source sentence.  The encoder sees it at position 0 where every other
position can attend to it, so the entire encoding is conditioned on the
requested output language.

**Optional decoupling of embedding width from model width.**  When
``embedding_dim`` differs from ``d_model`` (the pre-trained-vector experiments
use 300-dimensional MUSE vectors against a 512-wide residual stream) the model
inserts a learned projection on the way in and its transpose-shaped counterpart
on the way out.  Everything between those two projections is identical across
experiments, so a difference in results is attributable to the embeddings
rather than to a change of architecture.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nmt.config import ModelConfig
from nmt.constants import PAD_ID
from nmt.model.layers import Decoder, DecoderLayer, Encoder, EncoderLayer
from nmt.model.masking import cross_attention_mask, decoder_mask, padding_mask
from nmt.model.positional import build_positional_encoding
from nmt.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TranslationTransformer(nn.Module):
    """Encoder-decoder transformer for bidirectional EN<->ES translation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        d_model = config.d_model
        embedding_dim = config.effective_embedding_dim

        # --- embeddings -----------------------------------------------------
        # `padding_idx=0` pins the <pad> row to zero and, more importantly,
        # excludes it from gradient updates: without it, <pad> would slowly
        # drift and stop being the neutral element the masks assume.
        self.embedding = nn.Embedding(
            config.vocab_size, embedding_dim, padding_idx=PAD_ID
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=embedding_dim**-0.5)
        with torch.no_grad():
            self.embedding.weight[PAD_ID].zero_()

        # Bridges between embedding space and the residual stream. Identity
        # when the two widths agree, which is the default configuration.
        self.input_projection: nn.Module = (
            nn.Linear(embedding_dim, d_model, bias=False)
            if embedding_dim != d_model
            else nn.Identity()
        )
        self.output_projection: nn.Module = (
            nn.Linear(d_model, embedding_dim, bias=False)
            if embedding_dim != d_model
            else nn.Identity()
        )

        # Embeddings are multiplied by sqrt(d_model) before the positional
        # signal is added. With std = 1/sqrt(d) initialisation each embedding
        # has norm ~1, while the sinusoidal table has entries in [-1, 1] and
        # norm ~sqrt(d/2). Without the rescale the positional signal would
        # swamp the token identity at the start of training.
        self.embedding_scale = math.sqrt(d_model)

        self.positional_encoding = build_positional_encoding(
            config.positional_encoding,
            d_model,
            max_length=config.max_position,
            dropout=config.dropout,
        )

        # --- stacks ---------------------------------------------------------
        layer_kwargs = {
            "dropout": config.dropout,
            "attention_dropout": config.attention_dropout,
            "activation": config.activation,
            "norm_first": config.norm_first,
        }
        self.encoder = Encoder(
            [
                EncoderLayer(d_model, config.num_heads, config.d_ff, **layer_kwargs)
                for _ in range(config.num_encoder_layers)
            ],
            norm_first=config.norm_first,
        )
        self.decoder = Decoder(
            [
                DecoderLayer(d_model, config.num_heads, config.d_ff, **layer_kwargs)
                for _ in range(config.num_decoder_layers)
            ],
            norm_first=config.norm_first,
        )

        # --- output layer ---------------------------------------------------
        if config.tie_embeddings:
            # No separate parameter: logits are computed against the embedding
            # matrix itself in `project_to_vocab`. The bias is kept untied
            # because it encodes per-token frequency, which is a property of
            # the output distribution rather than of the embedding geometry.
            self.generator = None
            self.output_bias = nn.Parameter(torch.zeros(config.vocab_size))
        else:
            self.generator = nn.Linear(embedding_dim, config.vocab_size)
            nn.init.xavier_uniform_(self.generator.weight)
            nn.init.zeros_(self.generator.bias)
            self.output_bias = None

        self._embeddings_frozen = False

    # --- embedding helpers --------------------------------------------------

    def load_pretrained_embeddings(
        self, matrix: torch.Tensor | Any, *, freeze: bool = False
    ) -> None:
        """Overwrite the embedding table with pre-trained vectors."""
        matrix = torch.as_tensor(matrix, dtype=self.embedding.weight.dtype)
        if matrix.shape != self.embedding.weight.shape:
            raise ValueError(
                f"embedding matrix has shape {tuple(matrix.shape)}, expected "
                f"{tuple(self.embedding.weight.shape)}"
            )
        with torch.no_grad():
            self.embedding.weight.copy_(matrix)
            self.embedding.weight[PAD_ID].zero_()
        if freeze:
            self.freeze_embeddings(True)
        logger.info(
            "Loaded pre-trained embeddings %s (frozen=%s)",
            tuple(matrix.shape),
            freeze,
        )

    def freeze_embeddings(self, frozen: bool = True) -> None:
        """Toggle gradient flow through the embedding table.

        Used by the pre-trained experiment: holding the vectors fixed for the
        first epochs lets the randomly-initialised encoder and decoder adapt to
        the given geometry instead of destroying it with large early gradients,
        after which unfreezing allows task-specific fine-tuning.
        """
        self.embedding.weight.requires_grad_(not frozen)
        self._embeddings_frozen = frozen

    @property
    def embeddings_frozen(self) -> bool:
        return self._embeddings_frozen

    # --- forward components -------------------------------------------------

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Ids -> scaled, position-aware ``(batch, seq, d_model)`` vectors."""
        x = self.embedding(tokens)
        x = self.input_projection(x)
        return self.positional_encoding(x * self.embedding_scale)

    def encode(
        self,
        source: torch.Tensor,
        source_mask: torch.Tensor | None = None,
        *,
        store_attention: bool = False,
    ) -> torch.Tensor:
        """Run the encoder stack.

        Returns ``(batch, source_len, d_model)`` -- the *memory* that every
        decoding step attends to.  Computed once per sentence, then reused for
        every generated token, which is why encoding cost is negligible next to
        decoding cost.
        """
        if source_mask is None:
            source_mask = padding_mask(source)
        return self.encoder(
            self.embed(source), source_mask, store_attention=store_attention
        )

    def decode(
        self,
        memory: torch.Tensor,
        target_input: torch.Tensor,
        *,
        source: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        store_attention: bool = False,
    ) -> torch.Tensor:
        """Run the decoder stack, returning ``(batch, target_len, d_model)``."""
        if target_mask is None:
            target_mask = decoder_mask(target_input)
        if memory_mask is None:
            if source is None:
                raise ValueError("decode() needs either memory_mask or source")
            memory_mask = cross_attention_mask(source)

        return self.decoder(
            self.embed(target_input),
            memory,
            target_mask,
            memory_mask,
            store_attention=store_attention,
        )

    def project_to_vocab(self, hidden: torch.Tensor) -> torch.Tensor:
        """Decoder states -> vocabulary logits.

        With tied embeddings the logit for token *v* is the dot product of the
        decoder state with *v*'s embedding: the same vector that encodes "what
        this token means" on the way in scores "how well does this token fit"
        on the way out.
        """
        hidden = self.output_projection(hidden)
        if self.generator is not None:
            return self.generator(hidden)
        return torch.nn.functional.linear(
            hidden, self.embedding.weight, self.output_bias
        )

    def forward(
        self,
        source: torch.Tensor,
        target_input: torch.Tensor,
        *,
        store_attention: bool = False,
    ) -> torch.Tensor:
        """Teacher-forced forward pass.

        Parameters
        ----------
        source
            ``(batch, source_len)`` ids, already carrying the direction tag.
        target_input
            ``(batch, target_len)`` ids, already shifted right by the collate
            function (begins with ``<s>``).

        Returns
        -------
        torch.Tensor
            ``(batch, target_len, vocab_size)`` logits.
        """
        source_mask = padding_mask(source)
        memory = self.encoder(
            self.embed(source), source_mask, store_attention=store_attention
        )
        hidden = self.decoder(
            self.embed(target_input),
            memory,
            decoder_mask(target_input),
            source_mask,
            store_attention=store_attention,
        )
        return self.project_to_vocab(hidden)

    # --- introspection ------------------------------------------------------

    def attention_maps(self) -> dict[str, list[torch.Tensor]]:
        """Attention tensors from the most recent ``store_attention`` pass."""
        return {
            "encoder_self": self.encoder.attention_weights,
            "decoder_self": self.decoder.self_attention_weights,
            "cross": self.decoder.cross_attention_weights,
        }

    def parameter_breakdown(self) -> dict[str, int]:
        """Parameter counts by component, for the report's model table."""
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        breakdown = {
            "embedding": count(self.embedding),
            "encoder": count(self.encoder),
            "decoder": count(self.decoder),
            "projections": count(self.input_projection) + count(self.output_projection),
            "generator": count(self.generator) if self.generator is not None else 0,
            "output_bias": self.output_bias.numel() if self.output_bias is not None else 0,
        }
        # `parameters()` yields tied tensors once, so this is the true total.
        breakdown["total"] = sum(p.numel() for p in self.parameters())
        breakdown["trainable"] = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return breakdown

    # --- persistence --------------------------------------------------------

    def save(self, path: Path | str, **extra: Any) -> Path:
        """Write weights plus the config needed to rebuild the model.

        Saving the config alongside the weights is what lets the Gradio app
        load a checkpoint without being told which experiment produced it.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.state_dict(),
                "model_config": self.config.to_dict(),
                **extra,
            },
            path,
        )
        return path

    @classmethod
    def from_checkpoint(
        cls,
        path: Path | str,
        *,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> tuple[TranslationTransformer, dict[str, Any]]:
        """Rebuild a model from a checkpoint written by :meth:`save`.

        Returns
        -------
        (model, payload)
            The model in ``eval`` mode, and the full checkpoint dictionary so
            callers can read metadata such as the tokeniser path or the epoch.
        """
        payload = torch.load(path, map_location=map_location, weights_only=False)
        config = ModelConfig(**payload["model_config"])
        model = cls(config)
        model.load_state_dict(payload["model_state"], strict=strict)
        model.eval()
        return model, payload


def build_model(config: ModelConfig) -> TranslationTransformer:
    """Construct a model and log its size."""
    model = TranslationTransformer(config)
    breakdown = model.parameter_breakdown()
    logger.info(
        "Model: %s total parameters (%.1fM), %s trainable",
        f"{breakdown['total']:,}",
        breakdown["total"] / 1e6,
        f"{breakdown['trainable']:,}",
    )
    return model
