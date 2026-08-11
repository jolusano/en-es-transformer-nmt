"""Automated error analysis.

The brief asks for at least twenty poor translations to be examined and their
failure modes named.  Reading 11,000 test translations by hand is not
practical, so this module does the triage: it ranks every test sentence by
sentence-level BLEU, tags each one with the failure categories it exhibits, and
writes the worst cases out for manual inspection alongside aggregate counts.

The detectors are deliberately simple and conservative -- they are a *reading
aid*, not a verdict.  Each flagged example is quoted in full in the report so
the classification can be checked.  Categories:

``repetition``
    An n-gram repeated more often than it is in the reference.  The classic
    degenerate-decoding loop.

``truncation`` / ``over_generation``
    Output much shorter or longer than the reference.  Truncation usually
    means an early ``</s>``; over-generation often accompanies repetition.

``unknown_token``
    The output contains ``<unk>``. Only possible for the word-level models --
    and quantifying this gap is a large part of the point of the subword
    comparison.

``copied_source``
    The output overlaps the *source* more than the reference: the model gave
    up and passed the input through. Frequent with rare proper nouns.

``number_mismatch``
    Digits in the reference are missing or altered in the output. Worth
    separating because it is a *semantic* error that BLEU under-penalises, and
    it is the kind of mistake that makes a translation system unusable in
    practice regardless of its score.

``no_content_overlap``
    Not a single content word shared with the reference -- either a complete
    failure or a legitimate paraphrase that BLEU cannot see. Both are worth
    looking at, and the report distinguishes them by hand.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from nmt.evaluation.bleu import ngram_counts, sentence_bleu, tokenize_13a

_DIGITS = re.compile(r"\d+")

#: Function words carry little content, so overlap is measured without them.
_STOPWORDS = {
    # English
    "the", "a", "an", "of", "to", "in", "is", "it", "and", "that", "for",
    "on", "with", "as", "at", "by", "i", "you", "he", "she", "we", "they",
    "was", "are", "be", "this", "have", "has", "do", "does", "did", "not",
    # Spanish
    "el", "la", "los", "las", "un", "una", "de", "del", "que", "y", "en",
    "es", "se", "no", "al", "lo", "su", "por", "con", "para", "yo",
    "tu", "ella", "nosotros", "ellos", "son", "ser", "esta", "este",
}


@dataclass
class ErrorExample:
    """One analysed translation."""

    index: int
    direction: str
    source: str
    reference: str
    hypothesis: str
    sentence_bleu: float
    source_length: int
    reference_length: int
    hypothesis_length: int
    categories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _max_repeated_ngram(tokens: Sequence[str], order: int = 2) -> int:
    """Highest repeat count of any n-gram in ``tokens``."""
    if len(tokens) < order:
        return 0
    counts = ngram_counts(tokens, order)
    return max(counts.values()) if counts else 0


def _content_overlap(hypothesis: Sequence[str], reference: Sequence[str]) -> float:
    """Fraction of the reference's content words present in the hypothesis."""
    reference_content = {
        token.lower() for token in reference if token.lower() not in _STOPWORDS
    } - {".", ",", "!", "?", "¿", "¡"}
    if not reference_content:
        return 1.0
    hypothesis_lower = {token.lower() for token in hypothesis}
    return len(reference_content & hypothesis_lower) / len(reference_content)


def classify(
    source: str, reference: str, hypothesis: str
) -> list[str]:
    """Return every failure category the translation exhibits."""
    categories: list[str] = []

    source_tokens = tokenize_13a(source)
    reference_tokens = tokenize_13a(reference)
    hypothesis_tokens = tokenize_13a(hypothesis)

    if not hypothesis_tokens:
        return ["empty_output"]

    # --- repetition -------------------------------------------------------
    hypothesis_repeats = _max_repeated_ngram(hypothesis_tokens, 2)
    reference_repeats = _max_repeated_ngram(reference_tokens, 2)
    if hypothesis_repeats >= 2 and hypothesis_repeats > reference_repeats:
        categories.append("repetition")

    # --- length -----------------------------------------------------------
    ratio = len(hypothesis_tokens) / max(1, len(reference_tokens))
    if ratio < 0.6:
        categories.append("truncation")
    elif ratio > 1.7:
        categories.append("over_generation")

    # --- unknown tokens ---------------------------------------------------
    if "<unk>" in hypothesis:
        categories.append("unknown_token")

    # --- source copying ---------------------------------------------------
    source_overlap = _content_overlap(hypothesis_tokens, source_tokens)
    reference_overlap = _content_overlap(hypothesis_tokens, reference_tokens)
    if source_overlap > 0.5 and source_overlap > reference_overlap:
        categories.append("copied_source")

    # --- numbers ----------------------------------------------------------
    reference_numbers = set(_DIGITS.findall(reference))
    if reference_numbers and reference_numbers != set(_DIGITS.findall(hypothesis)):
        categories.append("number_mismatch")

    # --- content ----------------------------------------------------------
    if reference_overlap == 0.0:
        categories.append("no_content_overlap")

    if not categories:
        categories.append("other")

    return categories


def analyse(
    sources: Sequence[str],
    references: Sequence[str],
    hypotheses: Sequence[str],
    direction: str,
    *,
    worst_k: int = 30,
) -> dict[str, object]:
    """Score, tag and rank every translation in one direction.

    Returns
    -------
    dict
        ``worst`` holds the ``worst_k`` lowest-scoring examples for manual
        inspection; ``category_counts`` and ``bleu_by_length`` hold the
        aggregate views the report plots.
    """
    examples: list[ErrorExample] = []

    for index, (source, reference, hypothesis) in enumerate(
        zip(sources, references, hypotheses)
    ):
        examples.append(
            ErrorExample(
                index=index,
                direction=direction,
                source=source,
                reference=reference,
                hypothesis=hypothesis,
                sentence_bleu=sentence_bleu(hypothesis, reference),
                source_length=len(tokenize_13a(source)),
                reference_length=len(tokenize_13a(reference)),
                hypothesis_length=len(tokenize_13a(hypothesis)),
                categories=classify(source, reference, hypothesis),
            )
        )

    category_counts: Counter[str] = Counter()
    for example in examples:
        category_counts.update(example.categories)

    # BLEU stratified by source length: the plot that shows whether the model
    # degrades on long sentences, which is the transformer-vs-LSTM claim.
    buckets = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 1_000)]
    by_length: list[dict[str, object]] = []
    for low, high in buckets:
        subset = [e for e in examples if low <= e.source_length <= high]
        if not subset:
            continue
        by_length.append(
            {
                "bucket": f"{low}-{high if high < 1000 else '+'}",
                "low": low,
                "high": high,
                "sentences": len(subset),
                "mean_sentence_bleu": sum(e.sentence_bleu for e in subset) / len(subset),
                "mean_length_ratio": sum(
                    e.hypothesis_length / max(1, e.reference_length) for e in subset
                )
                / len(subset),
            }
        )

    ranked = sorted(examples, key=lambda e: e.sentence_bleu)

    return {
        "direction": direction,
        "sentences": len(examples),
        "category_counts": dict(category_counts.most_common()),
        "category_rates": {
            name: count / len(examples) for name, count in category_counts.items()
        },
        "bleu_by_length": by_length,
        "mean_sentence_bleu": (
            sum(e.sentence_bleu for e in examples) / len(examples) if examples else 0.0
        ),
        "worst": [e.to_dict() for e in ranked[:worst_k]],
        # A few good ones too: a report that only shows failures gives no sense
        # of what the system does well.
        "best": [e.to_dict() for e in ranked[-10:][::-1]],
    }


def to_markdown(analysis: dict[str, object], *, limit: int = 20) -> str:
    """Render the worst examples as a Markdown table for the report appendix."""
    lines = [
        f"### Worst {limit} translations — {analysis['direction']}",
        "",
        "| # | sBLEU | Source | Reference | Model output | Failure modes |",
        "|---|-------|--------|-----------|--------------|---------------|",
    ]

    def escape(text: str) -> str:
        return text.replace("|", "\\|")

    for rank, example in enumerate(analysis["worst"][:limit], start=1):  # type: ignore[index]
        lines.append(
            f"| {rank} | {example['sentence_bleu']:.1f} | {escape(example['source'])} "
            f"| {escape(example['reference'])} | {escape(example['hypothesis'])} "
            f"| {', '.join(example['categories'])} |"
        )

    lines.extend(["", "**Failure-mode frequency across the whole test set**", ""])
    lines.append("| Category | Sentences | Rate |")
    lines.append("|----------|-----------|------|")
    for name, count in analysis["category_counts"].items():  # type: ignore[union-attr]
        rate = analysis["category_rates"][name]  # type: ignore[index]
        lines.append(f"| {name} | {count} | {rate:.1%} |")

    return "\n".join(lines) + "\n"
