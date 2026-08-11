"""Transformer components, written from primitives."""

from nmt.model.attention import MultiHeadAttention, ScaledDotProductAttention
from nmt.model.baseline_lstm import AttentionSeq2Seq
from nmt.model.layers import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    PositionwiseFeedForward,
)
from nmt.model.masking import (
    causal_mask,
    cross_attention_mask,
    decoder_mask,
    padding_mask,
)
from nmt.model.positional import (
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding,
)
from nmt.model.transformer import TranslationTransformer, build_model

__all__ = [
    "AttentionSeq2Seq",
    "Decoder",
    "DecoderLayer",
    "Encoder",
    "EncoderLayer",
    "LearnedPositionalEncoding",
    "MultiHeadAttention",
    "PositionwiseFeedForward",
    "ScaledDotProductAttention",
    "SinusoidalPositionalEncoding",
    "TranslationTransformer",
    "build_model",
    "causal_mask",
    "cross_attention_mask",
    "decoder_mask",
    "padding_mask",
]
