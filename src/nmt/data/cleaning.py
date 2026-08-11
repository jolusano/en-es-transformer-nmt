"""Text normalisation and corpus filtering.

Every rule here is a modelling decision, so each one is documented with the
reason it exists.  The guiding principle is *remove noise, preserve meaning*:
we normalise away distinctions the model should not have to spend capacity on
(five kinds of apostrophe, decomposed accents, stray control characters) while
leaving intact everything a reader would consider part of the sentence
(casing, punctuation, inverted Spanish marks).

Two decisions deserve emphasis because they are frequently done differently:

**Casing is preserved.**  Lowercasing shrinks the vocabulary and usually buys a
point of BLEU on a small corpus, but it makes the system unable to produce
correctly-cased output, which would then need a separate truecasing model at
inference.  Since the deliverable is an interactive application whose output a
human reads, correct casing is part of the product.

**Punctuation is preserved.**  Spanish inverted marks (``¿``/``¡``) are
genuine grammatical signal, and translating punctuation is part of translating.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field

# --- Character-level normalisation tables ----------------------------------

#: Typographic variants collapsed onto a single ASCII form.  Tatoeba is
#: crowd-sourced, so the same sentence appears with curly and straight quotes
#: depending on the contributor's keyboard; without this the tokeniser learns
#: several embeddings for what is linguistically one token.
_CHARACTER_MAP = {
    # apostrophes / single quotes
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "′": "'", "´": "'", "`": "'",
    # double quotes
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "″": '"', "«": '"', "»": '"',
    # dashes
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    # spaces
    " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", "　": " ",
    # invisible characters that carry no meaning but split tokens
    "​": "", "‌": "", "‍": "", "﻿": "", "­": "",
    # miscellaneous
    "…": "...", "⁄": "/",
}

_TRANSLATION_TABLE = str.maketrans(_CHARACTER_MAP)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")
_HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
_URL_OR_EMAIL = re.compile(r"(https?://|www\.|\S+@\S+\.\S+)", re.IGNORECASE)

#: Characters that should never appear in either language.  Their presence
#: means the row is mislabelled (CJK, Cyrillic, Greek text filed as English).
_FOREIGN_SCRIPT = re.compile(
    r"[Ѐ-ӿ֐-׿؀-ۿऀ-ॿ"
    r"぀-ヿ㐀-䶿一-鿿가-힯]"
)


def normalise(text: str) -> str:
    """Apply the character-level normalisation pipeline to one sentence.

    Order matters:

    1. **NFC composition** first.  Spanish accents arrive both pre-composed
       (``ñ`` = U+00F1) and decomposed (``n`` + U+0303).  They render
       identically but are different strings, so without NFC the tokeniser
       treats ``español`` as two distinct types.
    2. **Control-character removal**, before whitespace collapsing, so that a
       stripped control character does not weld two words together.
    3. **Typographic folding** via the translation table.
    4. **Whitespace collapsing** last, since steps 2-3 can introduce runs of
       spaces.
    """
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_CHARS.sub(" ", text)
    text = text.translate(_TRANSLATION_TABLE)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


@dataclass(frozen=True)
class CleaningRules:
    """Thresholds for the corpus-level filters.

    Defaults were chosen after inspecting the length distributions in
    ``notebooks/01_data_exploration.ipynb``; the report reproduces the plots
    that motivate each number.
    """

    min_tokens: int = 1
    #: Tatoeba is a corpus of short everyday sentences: 99.5% of pairs are
    #: below 30 whitespace tokens.  Capping at 64 discards a negligible tail
    #: while bounding the O(n^2) cost of self-attention and the padding waste
    #: in each batch.
    max_tokens: int = 64
    min_chars: int = 2
    max_chars: int = 400
    #: A faithful translation does not change length by more than about a
    #: factor of three.  Larger ratios are almost always a misaligned link
    #: (a one-word sentence pointed at a full paragraph).
    max_length_ratio: float = 3.0
    #: The ratio test is unreliable on very short sentences ("Yes." vs "Si, lo
    #: haré."), so it is only applied once both sides reach this length.
    ratio_min_tokens: int = 4
    drop_identical: bool = True
    drop_urls: bool = True
    drop_foreign_script: bool = True
    require_letters: bool = True


@dataclass
class CleaningReport:
    """How many pairs each filter removed, and why."""

    input_pairs: int = 0
    kept_pairs: int = 0
    removed: Counter = field(default_factory=Counter)
    rules: dict[str, object] = field(default_factory=dict)
    examples_removed: dict[str, list[tuple[str, str]]] = field(default_factory=dict)

    def record(self, reason: str, en: str, es: str, *, keep_example: int = 3) -> None:
        """Count a rejection and keep a few illustrative cases per reason."""
        self.removed[reason] += 1
        bucket = self.examples_removed.setdefault(reason, [])
        if len(bucket) < keep_example:
            bucket.append((en[:120], es[:120]))

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["removed"] = dict(self.removed)
        payload["removed_total"] = self.input_pairs - self.kept_pairs
        payload["retention_rate"] = (
            self.kept_pairs / self.input_pairs if self.input_pairs else 0.0
        )
        return payload


def _rejection_reason(en: str, es: str, rules: CleaningRules) -> str | None:
    """Return the name of the first rule that rejects the pair, else ``None``."""
    if not en or not es:
        return "empty_after_normalisation"

    if rules.require_letters and not (_HAS_LETTER.search(en) and _HAS_LETTER.search(es)):
        # Pure digits or punctuation ("1997.", "!!!") carry no translatable
        # content and only teach the model to copy.
        return "no_alphabetic_content"

    if rules.drop_urls and (_URL_OR_EMAIL.search(en) or _URL_OR_EMAIL.search(es)):
        return "contains_url_or_email"

    if rules.drop_foreign_script and (
        _FOREIGN_SCRIPT.search(en) or _FOREIGN_SCRIPT.search(es)
    ):
        return "unexpected_script"

    if not (rules.min_chars <= len(en) <= rules.max_chars):
        return "english_char_length"
    if not (rules.min_chars <= len(es) <= rules.max_chars):
        return "spanish_char_length"

    en_tokens = en.split()
    es_tokens = es.split()

    if not (rules.min_tokens <= len(en_tokens) <= rules.max_tokens):
        return "english_token_length"
    if not (rules.min_tokens <= len(es_tokens) <= rules.max_tokens):
        return "spanish_token_length"

    if rules.drop_identical and en.casefold() == es.casefold():
        # Usually proper nouns or interjections filed as translations of
        # themselves ("Tom." / "Tom."). They teach an identity mapping that
        # competes with the translation objective.
        return "identical_sides"

    if min(len(en_tokens), len(es_tokens)) >= rules.ratio_min_tokens:
        ratio = max(len(en_tokens), len(es_tokens)) / min(len(en_tokens), len(es_tokens))
        if ratio > rules.max_length_ratio:
            return "length_ratio"

    return None


def clean_pairs(
    pairs: list[tuple[str, str]] | list[tuple[int, int, str, str]],
    rules: CleaningRules | None = None,
) -> tuple[list[tuple[str, str]], CleaningReport]:
    """Normalise and filter a list of sentence pairs.

    Accepts either ``(en, es)`` tuples or ``(en_id, es_id, en, es)`` tuples;
    ids are dropped from the output but the filtering is identical.

    Duplicate removal is exact and case-sensitive, performed on the *normalised*
    pair.  Near-duplicate removal is deliberately not attempted: Tatoeba's
    alternative translations of the same sentence are legitimate paraphrases
    and removing them would discard real supervision.  What must not happen is
    those paraphrases straddling the train/test boundary, and that is handled
    by the grouped split in :mod:`nmt.data.splitting` rather than here.
    """
    rules = rules or CleaningRules()
    report = CleaningReport(input_pairs=len(pairs), rules=asdict(rules))

    kept: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for row in pairs:
        raw_en, raw_es = (row[2], row[3]) if len(row) == 4 else (row[0], row[1])
        en = normalise(str(raw_en))
        es = normalise(str(raw_es))

        reason = _rejection_reason(en, es, rules)
        if reason is not None:
            report.record(reason, en or str(raw_en), es or str(raw_es))
            continue

        key = (en, es)
        if key in seen:
            report.record("duplicate_pair", en, es)
            continue

        seen.add(key)
        kept.append(key)

    report.kept_pairs = len(kept)
    return kept, report
