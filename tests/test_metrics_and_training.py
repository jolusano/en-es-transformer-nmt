"""BLEU, the loss function, the schedule and decoding."""

from __future__ import annotations

import math

import pytest
import torch

from nmt.config import ModelConfig
from nmt.constants import BOS_ID, PAD_ID
from nmt.evaluation.bleu import corpus_bleu, sacrebleu_score, sentence_bleu, tokenize_13a
from nmt.evaluation.error_analysis import classify
from nmt.inference.search import DecodeConfig, greedy_decode
from nmt.model.transformer import TranslationTransformer
from nmt.training.loss import LabelSmoothedCrossEntropy, perplexity
from nmt.training.scheduler import schedule_preview
from nmt.utils.seed import seed_everything


@pytest.fixture(autouse=True)
def _seed():
    seed_everything(0, deterministic=True)


# --- BLEU -------------------------------------------------------------------


def test_identical_output_scores_one_hundred():
    references = ["the cat sat on the mat", "she went to the market yesterday"]
    assert corpus_bleu(references, references).score == pytest.approx(100.0, abs=1e-6)


def test_clipping_caps_repeated_ngrams_at_their_reference_count():
    """Without clipping, "the the the the" would score 100% unigram precision.

    The reference contains "the" once, so at most one of the four hypothesis
    unigrams may be credited: 1/4 = 25%. The overall score stays non-trivial
    only because the smoothing floor lifts the empty higher orders, which is
    exactly what sacreBLEU does too.
    """
    result = corpus_bleu(["the the the the"], ["the cat sat down"])

    assert result.precisions[0] == pytest.approx(25.0)
    assert result.score == pytest.approx(
        sacrebleu_score(["the the the the"], ["the cat sat down"])["bleu"], abs=0.1
    )


def test_no_overlap_at_any_order_scores_zero():
    """Smoothing must not award credit to a hypothesis that shares nothing.

    sacreBLEU stops early when no n-gram matches at all; the from-scratch
    implementation has to make the same exception or the two disagree by
    several BLEU on such corpora.
    """
    result = corpus_bleu(["completely different words here"],
                         ["nothing at all matches"])

    assert result.score == 0.0
    assert result.precisions == [0.0, 0.0, 0.0, 0.0]


def test_brevity_penalty_punishes_truncation():
    result = corpus_bleu(["the cat"], ["the cat sat on the mat quietly"])
    assert result.brevity_penalty < 1.0
    assert result.length_ratio < 1.0


def test_no_penalty_when_output_is_long_enough():
    result = corpus_bleu(
        ["the cat sat on the mat quietly today"], ["the cat sat on the mat"]
    )
    assert result.brevity_penalty == 1.0


@pytest.mark.parametrize(
    "hypotheses,references",
    [
        (["the cat sat on the mat", "Estoy muy cansado hoy"],
         ["the cat is on the mat", "Estoy muy cansado hoy"]),
        (["a b c d e f", "g h i j", "k l m"], ["a b c x y z", "g h q j", "k l m"]),
        (["completely different words here"], ["nothing at all matches"]),
    ],
)
def test_matches_sacrebleu(hypotheses, references):
    """The from-scratch implementation must agree with the reference one.

    Agreement is the whole justification for quoting our own numbers alongside
    sacreBLEU's in the report.
    """
    ours = corpus_bleu(hypotheses, references).score
    theirs = sacrebleu_score(hypotheses, references)["bleu"]
    assert ours == pytest.approx(theirs, abs=0.1)


def test_tokenizer_separates_punctuation():
    assert tokenize_13a("Hola, Tom.") == ["Hola", ",", "Tom", "."]


def test_sentence_bleu_orders_hypotheses_sensibly():
    reference = "the cat sat on the mat"
    good = sentence_bleu("the cat sat on the mat", reference)
    fair = sentence_bleu("the cat sat on a mat", reference)
    poor = sentence_bleu("dogs run quickly away", reference)

    assert good > fair > poor


# --- error analysis ---------------------------------------------------------


def test_classifier_detects_repetition():
    categories = classify("I am tired", "Estoy cansado", "Estoy estoy estoy estoy")
    assert "repetition" in categories


def test_classifier_detects_truncation():
    categories = classify(
        "The quick brown fox jumps over the lazy dog",
        "El rapido zorro marron salta sobre el perro perezoso",
        "El zorro",
    )
    assert "truncation" in categories


def test_classifier_detects_number_mismatch():
    categories = classify("I have 42 books", "Tengo 42 libros", "Tengo 24 libros")
    assert "number_mismatch" in categories


def test_perfect_translation_is_not_flagged_as_a_failure():
    categories = classify("I am tired", "Estoy cansado", "Estoy cansado")
    assert categories == ["other"]


# --- loss -------------------------------------------------------------------


def test_zero_smoothing_matches_torch_cross_entropy():
    vocab = 50
    logits = torch.randn(4, 6, vocab)
    targets = torch.randint(1, vocab, (4, 6))
    targets[0, 4:] = PAD_ID

    ours, _ = LabelSmoothedCrossEntropy(smoothing=0.0)(logits, targets)
    theirs = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab), targets.reshape(-1), ignore_index=PAD_ID
    )
    assert ours.item() == pytest.approx(theirs.item(), abs=1e-5)


def test_uniform_logits_give_log_vocab():
    vocab = 1000
    logits = torch.zeros(2, 5, vocab)
    targets = torch.randint(1, vocab, (2, 5))

    _, stats = LabelSmoothedCrossEntropy(smoothing=0.0)(logits, targets)

    assert stats["nll"] == pytest.approx(math.log(vocab), abs=1e-4)
    assert perplexity(stats["nll"]) == pytest.approx(vocab, rel=1e-3)


def test_padding_is_excluded_from_loss_and_from_the_token_count():
    vocab = 20
    logits = torch.randn(1, 6, vocab)
    targets = torch.tensor([[3, 4, 5, PAD_ID, PAD_ID, PAD_ID]])

    _, stats = LabelSmoothedCrossEntropy(smoothing=0.1)(logits, targets)
    assert stats["tokens"] == 3


def test_smoothing_raises_the_loss_but_not_the_nll():
    vocab = 100
    logits = torch.randn(2, 4, vocab) * 5
    targets = torch.randint(1, vocab, (2, 4))

    plain, plain_stats = LabelSmoothedCrossEntropy(smoothing=0.0)(logits, targets)
    smoothed, smoothed_stats = LabelSmoothedCrossEntropy(smoothing=0.1)(logits, targets)

    assert smoothed.item() > plain.item()
    assert smoothed_stats["nll"] == pytest.approx(plain_stats["nll"], abs=1e-5)


def test_smoothing_must_be_a_valid_probability():
    with pytest.raises(ValueError):
        LabelSmoothedCrossEntropy(smoothing=1.0)


# --- schedule ---------------------------------------------------------------


def test_inverse_sqrt_peaks_at_the_end_of_warmup():
    warmup = 400
    curve = schedule_preview("inverse_sqrt", d_model=512, warmup_steps=warmup,
                             total_steps=4_000)

    peak = max(range(len(curve)), key=lambda i: curve[i]) + 1
    assert abs(peak - warmup) <= 2
    assert curve[0] < curve[warmup - 1]      # rising during warmup
    assert curve[-1] < curve[warmup - 1]     # decaying afterwards


def test_schedule_decays_as_inverse_square_root():
    curve = schedule_preview("inverse_sqrt", d_model=512, warmup_steps=100,
                             total_steps=10_000)
    # Quadrupling the step count should roughly halve the learning rate.
    assert curve[999] / curve[3999] == pytest.approx(2.0, rel=0.02)


# --- decoding ---------------------------------------------------------------


def test_greedy_decode_starts_with_bos_and_terminates():
    model = TranslationTransformer(
        ModelConfig(vocab_size=50, d_model=32, num_heads=4, d_ff=64,
                    num_encoder_layers=1, num_decoder_layers=1,
                    dropout=0.0, attention_dropout=0.0)
    ).eval()

    source = torch.randint(4, 50, (2, 6))
    output = greedy_decode(model, source, config=DecodeConfig(max_length_cap=12))

    assert output.shape[0] == 2
    assert (output[:, 0] == BOS_ID).all()
    assert output.shape[1] <= 13


def test_decoding_never_exceeds_the_length_budget():
    model = TranslationTransformer(
        ModelConfig(vocab_size=50, d_model=32, num_heads=4, d_ff=64,
                    num_encoder_layers=1, num_decoder_layers=1,
                    dropout=0.0, attention_dropout=0.0)
    ).eval()

    source = torch.randint(4, 50, (1, 4))
    config = DecodeConfig(max_length_ratio=1.0, max_length_offset=2,
                          max_length_cap=100)
    output = greedy_decode(model, source, config=config)

    assert output.shape[1] <= 4 * 1.0 + 2 + 1
