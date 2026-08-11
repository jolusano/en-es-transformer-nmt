"""Cleaning, splitting, tokenisation and batching."""

from __future__ import annotations

import pytest

from nmt.constants import BOS_ID, EOS_ID, PAD_ID, TAG_EN, TAG_ES, UNK_ID
from nmt.data.cleaning import CleaningRules, clean_pairs, normalise
from nmt.data.dataset import (
    Example,
    TokenBucketSampler,
    TranslationDataset,
    collate_batch,
)
from nmt.data.splitting import SplitSizes, UnionFind, split_pairs
from nmt.data.tokenizer import WordTokenizer

# --- normalisation ----------------------------------------------------------


def test_nfc_composition_unifies_accents():
    """Decomposed and pre-composed accents must become one string."""
    precomposed = "español"
    decomposed = "español"

    assert precomposed != decomposed
    assert normalise(precomposed) == normalise(decomposed)


def test_typographic_variants_are_folded():
    assert normalise("don’t") == "don't"
    assert normalise("“quoted”") == '"quoted"'
    assert normalise("dash—here") == "dash-here"


def test_whitespace_is_collapsed_and_trimmed():
    assert normalise("  too   many\tspaces \n") == "too many spaces"


def test_zero_width_characters_are_removed():
    assert normalise("wo​rd") == "word"


# --- filtering --------------------------------------------------------------


def test_cleaning_removes_the_expected_categories():
    pairs = [
        ("Hello there.", "Hola."),                       # kept
        ("", "Vacío."),                                  # empty
        ("1997.", "1997."),                              # no letters
        ("Visit http://x.com", "Visita http://x.com"),   # url
        ("Tom", "Tom"),                                  # identical
        # Both sides reach the four-token threshold, so the ratio test applies.
        ("I am very tired today",
         "Estoy muy cansado hoy porque he trabajado durante toda la noche sin "
         "descansar ni un solo momento de verdad"),
        ("Hello there.", "Hola."),                       # duplicate
    ]
    kept, report = clean_pairs(pairs, CleaningRules())

    assert kept == [("Hello there.", "Hola.")]
    assert report.removed["duplicate_pair"] == 1
    assert report.removed["identical_sides"] == 1
    assert report.removed["contains_url_or_email"] == 1
    assert report.removed["no_alphabetic_content"] == 1
    assert report.removed["empty_after_normalisation"] == 1
    assert report.removed["length_ratio"] == 1


def test_length_ratio_is_not_applied_to_short_sentences():
    """"Yes." / "Si, lo hare pronto." is a legitimate 1:4 pair."""
    kept, _ = clean_pairs([("Yes.", "Si, lo hare pronto.")], CleaningRules())
    assert len(kept) == 1


# --- splitting --------------------------------------------------------------


def test_union_find_merges_transitively():
    forest = UnionFind()
    for item in "abcd":
        forest.add(item)
    forest.union("a", "b")
    forest.union("b", "c")

    assert forest.find("a") == forest.find("c")
    assert forest.find("a") != forest.find("d")


def test_split_has_no_leakage_on_a_graph_shaped_corpus():
    """The whole point of component-aware splitting.

    "I am tired" links to two Spanish sentences; both must land in the same
    split or the model memorises one and is tested on the other.
    """
    pairs = [
        ("I am tired.", "Estoy cansado."),
        ("I am tired.", "Estoy cansada."),
        ("I am tired.", "Tengo sueño."),
    ]
    pairs += [(f"Sentence {i}.", f"Frase {i}.") for i in range(300)]

    splits, report = split_pairs(pairs, SplitSizes(validation=0.1, test=0.1), seed=0)

    assert sum(report.leakage_check.values()) == 0
    assert sum(len(v) for v in splits.values()) == len(pairs)

    # All three translations of "I am tired." are in exactly one split.
    holders = [name for name, rows in splits.items()
               if any(en == "I am tired." for en, _ in rows)]
    assert len(holders) == 1


def test_naive_random_split_would_leak():
    """Demonstrates the failure the component split exists to prevent."""
    import random

    pairs = [("I am tired.", f"Traduccion {i}.") for i in range(20)]
    rng = random.Random(0)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    train, test = shuffled[:14], shuffled[14:]

    train_sources = {en for en, _ in train}
    leaked = sum(1 for en, _ in test if en in train_sources)
    assert leaked > 0, "the naive split really does leak on graph-shaped data"


# --- tokenisation -----------------------------------------------------------


@pytest.fixture
def word_tokenizer() -> WordTokenizer:
    sentences = [
        "the cat sat on the mat", "el gato se sento en la alfombra",
        "I am tired", "estoy cansado", "the dog runs", "el perro corre",
    ] * 3
    return WordTokenizer.train(sentences, vocab_size=100, min_frequency=1)


def test_reserved_ids_are_pinned(word_tokenizer):
    assert word_tokenizer.piece_to_id("<pad>") == PAD_ID
    assert word_tokenizer.piece_to_id("<unk>") == UNK_ID
    assert word_tokenizer.piece_to_id("<s>") == BOS_ID
    assert word_tokenizer.piece_to_id("</s>") == EOS_ID
    assert word_tokenizer.piece_to_id(TAG_EN) != UNK_ID
    assert word_tokenizer.piece_to_id(TAG_ES) != UNK_ID


def test_unknown_words_map_to_unk(word_tokenizer):
    assert word_tokenizer.piece_to_id("supercalifragilistic") == UNK_ID


def test_encode_source_adds_tag_and_eos(word_tokenizer):
    ids = word_tokenizer.encode_source("the cat", TAG_ES)

    assert ids[0] == word_tokenizer.piece_to_id(TAG_ES)
    assert ids[-1] == EOS_ID


def test_encode_target_is_bracketed_by_bos_and_eos(word_tokenizer):
    ids = word_tokenizer.encode_target("the cat")

    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID


def test_decode_drops_special_tokens(word_tokenizer):
    ids = word_tokenizer.encode_source("the cat", TAG_EN)
    decoded = word_tokenizer.decode(ids)

    assert "<" not in decoded
    assert "cat" in decoded


def test_detokenizer_attaches_punctuation_correctly():
    assert WordTokenizer.detokenize(["Hola", ",", "Tom", "."]) == "Hola, Tom."
    assert WordTokenizer.detokenize(["¿", "Como", "estas", "?"]) == "¿Como estas?"


# --- dataset ----------------------------------------------------------------


def test_dataset_materialises_both_directions(word_tokenizer):
    pairs = [("the cat sat", "el gato se sento"), ("I am tired", "estoy cansado")]
    dataset = TranslationDataset(pairs, word_tokenizer)

    assert len(dataset) == 2 * len(pairs)
    assert {example.direction for example in dataset.examples} == {"en-es", "es-en"}


def test_collate_shifts_target_correctly(word_tokenizer):
    examples = [
        Example([10, 11, EOS_ID], [BOS_ID, 20, 21, EOS_ID], "en-es", "a", "b"),
        Example([10, EOS_ID], [BOS_ID, 20, EOS_ID], "en-es", "a", "b"),
    ]
    batch = collate_batch(examples)

    # decoder_input starts at <s>; labels are the same sequence shifted left.
    assert batch.decoder_input[0].tolist() == [BOS_ID, 20, 21]
    assert batch.labels[0].tolist() == [20, 21, EOS_ID]
    assert batch.source_padding_mask[1].tolist() == [True, True, False]


def test_collate_slices_are_contiguous():
    """Regression test for a real, silent bug.

    ``decoder_input`` and ``labels`` are slices of one padded tensor and share
    its storage at a non-zero offset. On PyTorch's MPS backend, moving such a
    view to the device ignored the offset, so ``decoder_input`` arrived holding
    the *labels*: the model was fed the answer and reported ~95% token accuracy
    at random initialisation while BLEU stayed at zero.
    """
    examples = [
        Example([10, 11, EOS_ID], [BOS_ID, 20, 21, EOS_ID], "en-es", "a", "b"),
        Example([10, EOS_ID], [BOS_ID, 20, EOS_ID], "en-es", "a", "b"),
    ]
    batch = collate_batch(examples)

    assert batch.decoder_input.is_contiguous()
    assert batch.labels.is_contiguous()
    assert batch.decoder_input.storage_offset() == 0
    assert batch.labels.storage_offset() == 0


def test_token_bucket_sampler_respects_the_budget():
    lengths = [5] * 40 + [50] * 10
    sampler = TokenBucketSampler(lengths, max_tokens=100, shuffle=False)

    for batch in sampler:
        longest = max(lengths[i] for i in batch)
        assert len(batch) * longest <= 100 or len(batch) == 1

    assert sorted(i for batch in sampler for i in batch) == list(range(len(lengths)))
