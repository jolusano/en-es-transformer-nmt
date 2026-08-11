"""BLEU, implemented from the definition and cross-checked against sacreBLEU.

BLEU (Papineni et al., 2002) scores a hypothesis by how much of its n-gram
content appears in the reference:

.. math::
    \\mathrm{BLEU} = BP \\cdot \\exp\\!\\left(\\sum_{n=1}^{N} w_n \\log p_n\\right)

with uniform weights :math:`w_n = 1/N` and :math:`N = 4`.

Modified precision, :math:`p_n`
    The fraction of the hypothesis' n-grams that occur in the reference, where
    each reference n-gram can only be matched as many times as it occurs there.
    That *clipping* is what stops "the the the the" from scoring a perfect
    unigram precision against "the cat sat on the mat".  Counts are pooled over
    the whole corpus before the ratio is taken -- BLEU is a corpus-level metric,
    and averaging per-sentence BLEU gives a different (and lower) number.

Brevity penalty, :math:`BP`
    Precision alone is trivially maximised by emitting one word that is
    certainly in the reference.  There is no recall term in BLEU, so a length
    penalty stands in for it:

    .. math::
        BP = \\begin{cases}
            1 & c > r \\\\
            e^{1 - r/c} & c \\leq r
        \\end{cases}

    with *c* the total hypothesis length and *r* the total reference length.

Why implement it rather than only calling sacreBLEU
---------------------------------------------------
Two reasons.  First, the arithmetic above is course material and writing it out
demonstrates it is understood.  Second, and more practically, BLEU is
notoriously sensitive to tokenisation -- the same system can differ by several
points depending on how the text was split before counting -- which is exactly
why sacreBLEU exists.  The project reports **sacreBLEU** numbers, computed with
its ``13a`` tokenisation, as the headline figures so they are comparable to
published work; this implementation runs alongside as an assertion that the two
agree, and a test in ``tests/test_bleu.py`` pins the agreement to 0.1 BLEU.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

# --- Tokenisation for scoring ----------------------------------------------

#: A faithful reimplementation of the `mteval-v13a` tokenisation that
#: sacreBLEU uses by default: separate punctuation from words, normalise a few
#: entities, collapse whitespace. Scoring on raw whitespace tokens would let
#: "casa." and "casa" count as different unigrams.
_PUNCT_SPACING = [
    (re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), r" \1 "),
    (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),
    (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),
    (re.compile(r"([0-9])(-)"), r"\1 \2 "),
]


def tokenize_13a(text: str) -> list[str]:
    """Tokenise for scoring, following the ``13a`` convention."""
    text = text.replace("<skipped>", "").replace("-\n", "").replace("\n", " ")
    text = text.replace("&quot;", '"').replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">")
    for pattern, replacement in _PUNCT_SPACING:
        text = pattern.sub(replacement, text)
    return text.split()


def ngram_counts(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    """Count all n-grams of the given order."""
    return Counter(
        tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1)
    )


@dataclass
class BleuScore:
    """A BLEU score together with the quantities it is built from."""

    score: float = 0.0
    precisions: list[float] = field(default_factory=list)
    brevity_penalty: float = 0.0
    length_ratio: float = 0.0
    hypothesis_length: int = 0
    reference_length: int = 0
    sentences: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def __str__(self) -> str:
        precisions = "/".join(f"{p:.1f}" for p in self.precisions)
        return (
            f"BLEU = {self.score:.2f}  {precisions}  "
            f"(BP = {self.brevity_penalty:.3f}, ratio = {self.length_ratio:.3f}, "
            f"hyp_len = {self.hypothesis_length}, ref_len = {self.reference_length})"
        )


def corpus_bleu(
    hypotheses: Sequence[str],
    references: Sequence[str],
    *,
    max_order: int = 4,
    smooth: bool = True,
    effective_order: bool = False,
) -> BleuScore:
    """Corpus-level BLEU with clipped n-gram precision and brevity penalty.

    Parameters
    ----------
    smooth
        Exponential smoothing for orders with zero matches (the ``exp`` method
        of mteval-v13a, which is sacreBLEU's default). Without it a corpus
        containing no matching 4-gram scores exactly zero, which makes
        early-training checkpoints indistinguishable from one another.
    effective_order
        Average only over the n-gram orders the hypothesis is long enough to
        have. Leave this ``False`` for corpus scoring -- sacreBLEU does, and
        turning it on would make the headline numbers incomparable to published
        work. Turn it on for *sentence* scoring, where sentences shorter than
        ``max_order`` are otherwise forced to zero; :func:`sentence_bleu` does.
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"got {len(hypotheses)} hypotheses but {len(references)} references"
        )

    matches = [0] * max_order
    totals = [0] * max_order
    hypothesis_length = reference_length = 0

    for hypothesis, reference in zip(hypotheses, references):
        hypothesis_tokens = tokenize_13a(hypothesis)
        reference_tokens = tokenize_13a(reference)

        hypothesis_length += len(hypothesis_tokens)
        reference_length += len(reference_tokens)

        for order in range(1, max_order + 1):
            hypothesis_ngrams = ngram_counts(hypothesis_tokens, order)
            reference_ngrams = ngram_counts(reference_tokens, order)

            # Clipping: an n-gram can be credited at most as often as it
            # appears in the reference.
            overlap = {
                ngram: min(count, reference_ngrams[ngram])
                for ngram, count in hypothesis_ngrams.items()
                if ngram in reference_ngrams
            }
            matches[order - 1] += sum(overlap.values())
            totals[order - 1] += max(0, len(hypothesis_tokens) - order + 1)

    # Exponential smoothing, as in mteval-v13a and sacreBLEU's default. It is
    # applied *only* to orders with no matches at all: an order that matched
    # something is reported as its true ratio. Each successive zero order
    # halves the value assigned to it, so a hypothesis that fails at 3-grams
    # and 4-grams is penalised more at order 4 than at order 3.
    #
    # (An earlier version of this function added one to every numerator and
    # denominator unconditionally. That is a legitimate smoothing scheme in its
    # own right, but it inflates non-zero precisions and put this function
    # about 5 BLEU above sacreBLEU on short corpora.)
    if hypothesis_length == 0:
        brevity_penalty = 0.0
    elif hypothesis_length > reference_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - reference_length / hypothesis_length)

    # Early stop when nothing matched at any order. Smoothing exists to keep a
    # *partially* correct hypothesis distinguishable from a worse one; applying
    # it to a hypothesis with no overlap at all would award a positive score to
    # output sharing not one word with the reference. sacreBLEU makes the same
    # exception, and without it the two implementations disagree by ~8 BLEU on
    # such corpora.
    if not any(matches):
        return BleuScore(
            score=0.0,
            precisions=[0.0] * max_order,
            brevity_penalty=brevity_penalty,
            length_ratio=(
                hypothesis_length / reference_length if reference_length else 0.0
            ),
            hypothesis_length=hypothesis_length,
            reference_length=reference_length,
            sentences=len(hypotheses),
        )

    precisions: list[float] = []
    smooth_factor = 1.0
    for order in range(max_order):
        numerator, denominator = matches[order], totals[order]
        if denominator == 0:
            precisions.append(0.0)
        elif numerator == 0 and smooth:
            smooth_factor *= 2.0
            precisions.append(100.0 / (smooth_factor * denominator))
        else:
            precisions.append(100.0 * numerator / denominator)

    # With `effective_order`, orders that the hypothesis is too short to have
    # any of are dropped from the geometric mean rather than contributing a
    # zero. A 3-token sentence has no 4-grams at all, so under the fixed
    # 4-order mean it scores 0 no matter how perfect it is -- which made the
    # error analysis rank *correct* translations of short sentences as the
    # worst in the corpus. This is what sacreBLEU's sentence_bleu() does, and
    # why it is off for corpus scoring (where every order has counts anyway)
    # and on for sentence scoring.
    order_count = (
        max(1, sum(1 for total in totals if total > 0)) if effective_order else max_order
    )
    usable = precisions[:order_count]

    if usable and min(usable) > 0:
        log_mean = sum(math.log(p) for p in usable) / order_count
        score = brevity_penalty * math.exp(log_mean)
    else:
        score = 0.0

    return BleuScore(
        score=score,
        precisions=precisions,
        brevity_penalty=brevity_penalty,
        length_ratio=hypothesis_length / reference_length if reference_length else 0.0,
        hypothesis_length=hypothesis_length,
        reference_length=reference_length,
        sentences=len(hypotheses),
    )


def sentence_bleu(hypothesis: str, reference: str, *, max_order: int = 4) -> float:
    """Smoothed BLEU for a single sentence, with effective order.

    Only meaningful for *ranking* sentences against each other -- which is what
    the error analysis uses it for, to surface the worst translations. It is
    not comparable to a corpus BLEU number.

    ``effective_order`` is essential here and must not be dropped. Tatoeba's
    median sentence is six tokens and a quarter are under four, and a sentence
    with fewer than four tokens has no 4-grams: scored at fixed order it gets
    zero however good it is. Without this, the "worst translations" ranking
    fills up with short sentences the model translated perfectly.
    """
    return corpus_bleu(
        [hypothesis], [reference],
        max_order=max_order, smooth=True, effective_order=True,
    ).score


# --- Reference implementations ---------------------------------------------


def sacrebleu_score(
    hypotheses: Sequence[str], references: Sequence[str]
) -> dict[str, object]:
    """Score with sacreBLEU -- the numbers quoted in the report.

    Also returns chrF2, a character-n-gram F-score. chrF is worth reporting
    alongside BLEU for Spanish specifically: Spanish is morphologically richer
    than English, so a translation can get a word's stem right and its
    inflection wrong. BLEU counts that as a total miss; chrF gives partial
    credit, and the gap between the two metrics is itself informative.
    """
    import sacrebleu

    # Instantiating the metric (rather than calling the convenience function)
    # gives access to the reproducibility signature, which records the
    # tokenisation and smoothing actually used. Quoting it in the report is
    # what makes the BLEU number comparable to anyone else's.
    metric = sacrebleu.metrics.BLEU()
    bleu = metric.corpus_score(list(hypotheses), [list(references)])
    chrf = sacrebleu.corpus_chrf(list(hypotheses), [list(references)])

    return {
        "bleu": bleu.score,
        "precisions": list(bleu.precisions),
        "brevity_penalty": bleu.bp,
        "length_ratio": bleu.sys_len / bleu.ref_len if bleu.ref_len else 0.0,
        "hypothesis_length": bleu.sys_len,
        "reference_length": bleu.ref_len,
        "chrf2": chrf.score,
        "signature": str(metric.get_signature()),
    }


def compare_implementations(
    hypotheses: Sequence[str], references: Sequence[str]
) -> dict[str, object]:
    """Run both implementations and report their difference.

    Included in every evaluation run so that the agreement claimed in the
    report is verified on the actual output rather than asserted.
    """
    ours = corpus_bleu(hypotheses, references)
    reference = sacrebleu_score(hypotheses, references)

    return {
        "from_scratch": ours.to_dict(),
        "sacrebleu": reference,
        "absolute_difference": abs(ours.score - reference["bleu"]),
    }
