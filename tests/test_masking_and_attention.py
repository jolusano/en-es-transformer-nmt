"""Masks, attention and end-to-end causality.

These are the properties whose violation is *invisible* in a loss curve --- a
leaking causal mask makes training loss look excellent --- so they are asserted
directly rather than trusted.
"""

from __future__ import annotations

import math

import pytest
import torch

from nmt.config import ModelConfig
from nmt.constants import PAD_ID
from nmt.model.attention import MultiHeadAttention, ScaledDotProductAttention
from nmt.model.masking import (
    causal_mask,
    cross_attention_mask,
    decoder_mask,
    padding_mask,
)
from nmt.model.transformer import TranslationTransformer
from nmt.utils.seed import seed_everything


@pytest.fixture(autouse=True)
def _seed():
    seed_everything(0, deterministic=True)


# --- masks ------------------------------------------------------------------


def test_padding_mask_marks_real_positions():
    tokens = torch.tensor([[5, 6, 7, PAD_ID, PAD_ID], [5, PAD_ID, PAD_ID, PAD_ID, PAD_ID]])
    mask = padding_mask(tokens)

    assert mask.shape == (2, 1, 1, 5)
    assert mask[0, 0, 0].tolist() == [True, True, True, False, False]
    assert mask[1, 0, 0].tolist() == [True, False, False, False, False]


def test_causal_mask_is_lower_triangular_including_diagonal():
    mask = causal_mask(4).squeeze()

    assert mask.shape == (4, 4)
    assert mask.diagonal().all(), "a position must be able to attend to itself"
    assert not mask[0, 1:].any(), "position 0 must not see the future"
    assert mask[3].all(), "the last position sees everything"
    assert mask.tril().equal(mask)


def test_decoder_mask_is_the_conjunction_of_both():
    tokens = torch.tensor([[5, 6, 7, PAD_ID]])
    mask = decoder_mask(tokens)

    assert mask.shape == (1, 1, 4, 4)
    # Row 2 may see keys 0..2, but key 3 is padding.
    assert mask[0, 0, 2].tolist() == [True, True, True, False]
    # Row 0 sees only itself.
    assert mask[0, 0, 0].tolist() == [True, False, False, False]


def test_cross_attention_mask_hides_only_source_padding():
    source = torch.tensor([[5, 6, PAD_ID]])
    mask = cross_attention_mask(source)

    assert mask.shape == (1, 1, 1, 3)
    # No causal constraint: any target position may see any real source token.
    assert mask[0, 0, 0].tolist() == [True, True, False]


# --- attention --------------------------------------------------------------


def test_attention_weights_form_a_distribution():
    attention = ScaledDotProductAttention()
    query = torch.randn(2, 4, 6, 16)
    key = torch.randn(2, 4, 6, 16)
    value = torch.randn(2, 4, 6, 16)

    context, weights = attention(query, key, value)

    assert context.shape == (2, 4, 6, 16)
    assert weights.shape == (2, 4, 6, 6)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 4, 6), atol=1e-5)
    assert (weights >= 0).all()


def test_masked_positions_receive_no_weight():
    attention = ScaledDotProductAttention()
    query = key = value = torch.randn(1, 1, 5, 8)
    mask = causal_mask(5)

    _, weights = attention(query, key, value, mask)

    upper = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    assert weights[0, 0][upper].abs().max() == 0.0


def test_fully_masked_row_does_not_produce_nan():
    """A fully-padded row must degrade to a uniform distribution, not NaN.

    This is why the implementation fills with `finfo.min` rather than `-inf`:
    softmax of an all -inf row is NaN and would poison every gradient in the
    step.
    """
    attention = ScaledDotProductAttention()
    query = key = value = torch.randn(1, 1, 3, 4)
    mask = torch.zeros(1, 1, 3, 3, dtype=torch.bool)

    context, weights = attention(query, key, value, mask)

    assert torch.isfinite(context).all()
    assert torch.isfinite(weights).all()


def test_scaling_keeps_scores_in_the_responsive_range():
    """Without the 1/sqrt(d_k) factor, softmax saturates and gradients vanish."""
    d_k = 64
    query = torch.randn(1, 1, 1, d_k)
    key = torch.randn(1, 1, 200, d_k)

    unscaled = (query @ key.transpose(-2, -1)).std()
    scaled = ((query @ key.transpose(-2, -1)) / math.sqrt(d_k)).std()

    assert unscaled > 4 * scaled
    assert 0.5 < scaled < 2.0


def test_multi_head_shapes_and_head_split_roundtrip():
    attention = MultiHeadAttention(64, 8)
    x = torch.randn(3, 7, 64)

    assert attention(x, x, x).shape == (3, 7, 64)
    assert torch.allclose(attention._merge_heads(attention._split_heads(x)), x)


def test_multi_head_rejects_indivisible_width():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(64, 7)


# --- end-to-end -------------------------------------------------------------


def _small_model() -> TranslationTransformer:
    return TranslationTransformer(
        ModelConfig(
            vocab_size=200, d_model=32, num_heads=4, d_ff=64,
            num_encoder_layers=2, num_decoder_layers=2,
            dropout=0.0, attention_dropout=0.0,
        )
    ).eval()


def test_model_output_shape():
    model = _small_model()
    source = torch.randint(4, 200, (3, 9))
    target = torch.randint(4, 200, (3, 7))

    assert model(source, target).shape == (3, 7, 200)


def test_decoder_cannot_see_the_future():
    """Perturbing a future target token must not change earlier outputs.

    If this fails, the model is being handed the answer and its training loss
    is meaningless.
    """
    model = _small_model()
    source = torch.randint(4, 200, (2, 9))
    target = torch.randint(4, 200, (2, 8))

    with torch.no_grad():
        before = model(source, target)
        perturbed = target.clone()
        perturbed[:, 5] = 199
        after = model(source, perturbed)

    assert torch.equal(before[:, :5], after[:, :5]), "information leaked backwards"
    assert not torch.allclose(before[:, 5:], after[:, 5:]), "the change had no effect"


def test_extra_source_padding_does_not_change_outputs():
    """A sentence's translation must not depend on its batch-mates' lengths."""
    model = _small_model()
    source = torch.randint(4, 200, (2, 6))
    target = torch.randint(4, 200, (2, 5))

    with torch.no_grad():
        base = model(source, target)
        padded = torch.cat([source, torch.zeros(2, 4, dtype=torch.long)], dim=1)
        extended = model(padded, target)

    assert torch.allclose(base, extended, atol=1e-5)


def test_embeddings_are_tied_three_ways():
    model = _small_model()
    breakdown = model.parameter_breakdown()

    assert model.generator is None, "tied models have no separate output matrix"
    # The embedding table is counted once even though three roles use it.
    assert breakdown["total"] < (
        breakdown["embedding"] * 2 + breakdown["encoder"] + breakdown["decoder"]
    )


def test_pad_embedding_stays_zero():
    model = _small_model()
    assert model.embedding.weight[PAD_ID].abs().max() == 0.0
