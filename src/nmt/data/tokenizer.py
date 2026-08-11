"""Tokenisation, with two interchangeable strategies behind one interface.

The project trains the same architecture under two vocabulary regimes so that
the report can compare them on equal terms:

``SubwordTokenizer``
    A **joint** SentencePiece BPE model fitted on the concatenation of the
    English and Spanish training text.  Sharing one vocabulary across both
    languages is what makes a single bidirectional model possible: the decoder
    emits from the same inventory whichever way it is translating, and cognates
    ("nation"/"nación", "important"/"importante") share subword units and
    therefore share statistical strength.  Being open-vocabulary, it never
    emits ``<unk>``.

``WordTokenizer``
    A frequency-truncated whitespace/punctuation vocabulary.  It exists because
    pre-trained word vectors are distributed as word-level tables: to initialise
    embeddings from MUSE, the units the model embeds must *be* words.  The cost
    is a closed vocabulary and an out-of-vocabulary rate that the report
    quantifies.

Both classes expose the same six methods, so every downstream component --
dataset, model, decoder, Gradio app -- is written once and works with either.
"""

from __future__ import annotations

import json
import re
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from nmt.constants import (
    BOS_ID,
    BOS_TOKEN,
    DIRECTION_TAGS,
    EOS_ID,
    EOS_TOKEN,
    PAD_ID,
    PAD_TOKEN,
    RESERVED_TOKENS,
    UNK_ID,
    UNK_TOKEN,
)
from nmt.utils.io import ensure_dir, read_json, write_json
from nmt.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseTokenizer(ABC):
    """Common interface for every tokenisation strategy."""

    kind: str = "base"

    # --- required of subclasses --------------------------------------------

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert a raw string to token ids (no BOS/EOS added)."""

    @abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        """Convert token ids back to a string, dropping special tokens."""

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Convert a raw string to token *strings* (for inspection/plots)."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Number of entries in the vocabulary."""

    @abstractmethod
    def id_to_piece(self, index: int) -> str:
        """Surface form of a token id."""

    @abstractmethod
    def piece_to_id(self, piece: str) -> int:
        """Token id of a surface form (``UNK_ID`` when absent)."""

    @abstractmethod
    def save(self, directory: Path | str) -> Path:
        """Persist the tokeniser so inference can reload it exactly."""

    # --- shared behaviour ---------------------------------------------------

    def encode_source(self, text: str, tag: str) -> list[int]:
        """Encode a source sentence with its direction tag and EOS.

        The tag is prepended as a single reserved token rather than being
        spliced into the text, so the tokeniser can never split it into
        subwords.  The encoder sees ``[<2xx>] + tokens + [</s>]``; no BOS is
        needed on the source side because the encoder is not autoregressive.
        """
        return [self.piece_to_id(tag), *self.encode(text), EOS_ID]

    def encode_target(self, text: str) -> list[int]:
        """Encode a target sentence as ``[<s>] + tokens + [</s>]``.

        Teacher forcing then feeds ``ids[:-1]`` to the decoder and supervises
        against ``ids[1:]`` -- the "shift right" of the architecture diagram.
        """
        return [BOS_ID, *self.encode(text), EOS_ID]

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    def __repr__(self) -> str:
        return f"{type(self).__name__}(vocab_size={self.vocab_size})"


# ---------------------------------------------------------------------------
# Subword (SentencePiece)
# ---------------------------------------------------------------------------


class SubwordTokenizer(BaseTokenizer):
    """Joint SentencePiece model shared by both languages."""

    kind = "subword"

    def __init__(self, model_path: Path | str) -> None:
        import sentencepiece as spm

        self.model_path = Path(model_path)
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(str(self.model_path))
        self._verify_special_ids()

    def _verify_special_ids(self) -> None:
        """Fail loudly if the reserved ids drifted from the expected layout.

        Every mask, loss and decoder in the project assumes ``PAD_ID == 0`` and
        friends.  A silent mismatch would train a model that appears to work
        but pads with a real token.
        """
        expected = {
            PAD_ID: PAD_TOKEN,
            UNK_ID: UNK_TOKEN,
            BOS_ID: BOS_TOKEN,
            EOS_ID: EOS_TOKEN,
        }
        for index, piece in expected.items():
            actual = self._sp.IdToPiece(index)
            if actual != piece:
                raise ValueError(
                    f"SentencePiece id {index} is {actual!r}, expected {piece!r}. "
                    "Retrain the tokeniser with SubwordTokenizer.train()."
                )
        for tag in DIRECTION_TAGS:
            if self._sp.PieceToId(tag) == UNK_ID:
                raise ValueError(f"direction tag {tag!r} missing from the vocabulary")

    # --- training -----------------------------------------------------------

    @classmethod
    def train(
        cls,
        sentences: Iterable[str],
        directory: Path | str,
        *,
        vocab_size: int = 16_000,
        model_type: str = "bpe",
        character_coverage: float = 1.0,
        prefix: str = "joint_bpe",
    ) -> SubwordTokenizer:
        """Fit a joint SentencePiece model on ``sentences``.

        Parameters
        ----------
        sentences
            Every training-side sentence from *both* languages, interleaved.
            Fitting on the union rather than per language is what produces the
            shared inventory the bidirectional model needs.
        vocab_size
            16k is the operating point chosen for this corpus.  Tatoeba's
            training half contains on the order of 10^7 tokens; the usual
            32k-40k NMT vocabularies are sized for corpora two orders of
            magnitude larger, and at that size the rare half of the vocabulary
            would be updated too seldom to learn a useful embedding.  16k keeps
            the tied embedding matrix at 16k x 512 = 8.4M parameters -- around
            a quarter of the model -- while holding the average token count per
            sentence low.  The report shows the vocabulary-size sweep behind
            this choice.
        model_type
            ``"bpe"`` (default) or ``"unigram"``.
        character_coverage
            1.0 is correct for two Latin-script languages; the 0.9995 figure
            often quoted is meant for languages with huge character sets.
        """
        import sentencepiece as spm

        directory = ensure_dir(directory)
        corpus_file = directory / f"{prefix}_corpus.txt"

        count = 0
        with corpus_file.open("w", encoding="utf-8") as handle:
            for sentence in sentences:
                cleaned = sentence.replace("\n", " ").strip()
                if cleaned:
                    handle.write(cleaned + "\n")
                    count += 1

        if count == 0:
            corpus_file.unlink(missing_ok=True)
            raise ValueError(
                "no training sentences were supplied to SubwordTokenizer.train(); "
                "this usually means an earlier pipeline stage produced an empty "
                "corpus -- check data/interim/pairs_raw.tsv"
            )
        logger.info("Fitting SentencePiece on %s sentences", f"{count:,}")

        model_prefix = directory / prefix
        spm.SentencePieceTrainer.Train(
            input=str(corpus_file),
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            # Pin the reserved ids to the layout the rest of the code assumes.
            pad_id=PAD_ID,
            unk_id=UNK_ID,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            pad_piece=PAD_TOKEN,
            unk_piece=UNK_TOKEN,
            bos_piece=BOS_TOKEN,
            eos_piece=EOS_TOKEN,
            # Direction tags must survive as atomic units.
            user_defined_symbols=list(DIRECTION_TAGS),
            # Digits are split individually: Tatoeba contains many one-off
            # numbers, and splitting stops the vocabulary filling with dates.
            split_digits=True,
            # Punctuation gets its own piece rather than fusing with the word
            # before it, which keeps "casa" and "casa." on one embedding.
            split_by_unicode_script=True,
            byte_fallback=True,
            normalization_rule_name="identity",  # we normalise in nmt.data.cleaning
            train_extremely_large_corpus=False,
            num_threads=8,
        )
        corpus_file.unlink(missing_ok=True)

        tokenizer = cls(model_prefix.with_suffix(".model"))
        write_json(
            directory / f"{prefix}_meta.json",
            {
                "kind": cls.kind,
                "model_file": model_prefix.with_suffix(".model").name,
                "vocab_size": tokenizer.vocab_size,
                "model_type": model_type,
                "character_coverage": character_coverage,
                "training_sentences": count,
            },
        )
        return tokenizer

    # --- interface ----------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        return self._sp.EncodeAsIds(text)

    def tokenize(self, text: str) -> list[str]:
        return self._sp.EncodeAsPieces(text)

    def decode(self, ids: Sequence[int]) -> str:
        cleaned = [
            int(i)
            for i in ids
            if int(i) not in (PAD_ID, BOS_ID, EOS_ID)
            and self._sp.IdToPiece(int(i)) not in DIRECTION_TAGS
        ]
        return self._sp.DecodeIds(cleaned)

    @property
    def vocab_size(self) -> int:
        return int(self._sp.GetPieceSize())

    def id_to_piece(self, index: int) -> str:
        return self._sp.IdToPiece(int(index))

    def piece_to_id(self, piece: str) -> int:
        return int(self._sp.PieceToId(piece))

    def save(self, directory: Path | str) -> Path:
        directory = ensure_dir(directory)
        destination = directory / self.model_path.name
        if destination.resolve() != self.model_path.resolve():
            destination.write_bytes(self.model_path.read_bytes())
        return destination


# ---------------------------------------------------------------------------
# Word level
# ---------------------------------------------------------------------------

#: Words (letters, digits, internal apostrophes/hyphens) or single punctuation
#: marks.  Written to work for both languages: ``\w`` is Unicode-aware, so
#: accented characters and ``ñ`` stay inside words.
_WORD_PATTERN = re.compile(r"\w+(?:[''-]\w+)*|[^\w\s]", re.UNICODE)

#: Punctuation that attaches to the preceding word when detokenising.
_ATTACH_LEFT = set(".,;:!?%)]}>»'\"")
#: Punctuation that attaches to the following word.
_ATTACH_RIGHT = set("¿¡([{<«$#@")


class WordTokenizer(BaseTokenizer):
    """Closed word-level vocabulary, required for pre-trained word vectors."""

    kind = "word"

    def __init__(self, itos: list[str], *, lowercase_lookup: bool = True) -> None:
        self._itos = list(itos)
        self._stoi = {piece: index for index, piece in enumerate(self._itos)}
        #: MUSE vectors are lowercase-only.  When a cased word is missing we
        #: retry its lowercase form before giving up on ``<unk>``, which
        #: recovers most sentence-initial capitals at no vocabulary cost.
        self.lowercase_lookup = lowercase_lookup

    # --- training -----------------------------------------------------------

    @classmethod
    def train(
        cls,
        sentences: Iterable[str],
        *,
        vocab_size: int = 32_000,
        min_frequency: int = 2,
    ) -> WordTokenizer:
        """Build the vocabulary from token frequencies in ``sentences``.

        Parameters
        ----------
        vocab_size
            Upper bound including reserved tokens.  32k is twice the subword
            budget because a word vocabulary must cover two languages'
            *surface forms* rather than shared subword units, and Spanish
            inflection alone (each verb contributing dozens of forms) consumes
            entries quickly.
        min_frequency
            Singletons are dropped.  A word seen once cannot train a useful
            300-dimensional embedding, and mapping it to ``<unk>`` at training
            time is what teaches the model to handle unknown words at test
            time instead of never having seen the symbol.
        """
        counter: Counter[str] = Counter()
        for sentence in sentences:
            counter.update(_WORD_PATTERN.findall(sentence))

        budget = vocab_size - len(RESERVED_TOKENS)
        frequent = [
            piece
            for piece, count in counter.most_common()
            if count >= min_frequency
        ][:budget]

        itos = [*RESERVED_TOKENS, *frequent]
        logger.info(
            "Word vocabulary: %s types kept out of %s observed (min_frequency=%d)",
            f"{len(itos):,}",
            f"{len(counter):,}",
            min_frequency,
        )
        return cls(itos)

    # --- interface ----------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        return _WORD_PATTERN.findall(text)

    def encode(self, text: str) -> list[int]:
        return [self.piece_to_id(piece) for piece in self.tokenize(text)]

    def piece_to_id(self, piece: str) -> int:
        index = self._stoi.get(piece)
        if index is not None:
            return index
        if self.lowercase_lookup:
            index = self._stoi.get(piece.lower())
            if index is not None:
                return index
        return UNK_ID

    def id_to_piece(self, index: int) -> str:
        index = int(index)
        return self._itos[index] if 0 <= index < len(self._itos) else UNK_TOKEN

    def decode(self, ids: Sequence[int]) -> str:
        pieces = [
            self.id_to_piece(i)
            for i in ids
            if int(i) not in (PAD_ID, BOS_ID, EOS_ID)
            and self.id_to_piece(i) not in DIRECTION_TAGS
        ]
        return self.detokenize(pieces)

    @staticmethod
    def detokenize(pieces: Sequence[str]) -> str:
        """Rejoin tokens into a readable sentence.

        Word-level tokenisation is lossy about spacing, so detokenisation is a
        heuristic: attach closing punctuation to the left, opening punctuation
        (including Spanish ``¿`` and ``¡``) to the right.  This matters for
        BLEU, which is computed on detokenised text.
        """
        out: list[str] = []
        glue_to_previous = False  # set when the previous piece was "¿", "(", ...

        for piece in pieces:
            if out and (glue_to_previous or piece in _ATTACH_LEFT):
                out[-1] += piece
            else:
                out.append(piece)
            glue_to_previous = piece in _ATTACH_RIGHT

        return " ".join(out).strip()

    @property
    def vocab_size(self) -> int:
        return len(self._itos)

    @property
    def itos(self) -> list[str]:
        """The index-to-string table (used to build the embedding matrix)."""
        return self._itos

    def coverage(self, sentences: Iterable[str]) -> dict[str, float]:
        """Measure the out-of-vocabulary rate on held-out text.

        Reported in the paper as the concrete cost of the closed vocabulary.
        """
        total = unknown = 0
        unknown_types: Counter[str] = Counter()
        for sentence in sentences:
            for piece in self.tokenize(sentence):
                total += 1
                if self.piece_to_id(piece) == UNK_ID:
                    unknown += 1
                    unknown_types[piece] += 1
        return {
            "tokens": total,
            "unk_tokens": unknown,
            "unk_rate": unknown / total if total else 0.0,
            "unk_types": len(unknown_types),
            "most_common_unk": unknown_types.most_common(20),
        }

    # --- persistence --------------------------------------------------------

    def save(self, directory: Path | str, *, prefix: str = "word") -> Path:
        directory = ensure_dir(directory)
        path = directory / f"{prefix}_vocab.json"
        write_json(
            path,
            {
                "kind": self.kind,
                "lowercase_lookup": self.lowercase_lookup,
                "itos": self._itos,
            },
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> WordTokenizer:
        payload = read_json(path)
        return cls(payload["itos"], lowercase_lookup=payload.get("lowercase_lookup", True))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def load_tokenizer(path: Path | str) -> BaseTokenizer:
    """Load whichever tokeniser lives at ``path``.

    Accepts a ``.model`` file (SentencePiece), a ``*_vocab.json`` file
    (word level), or a directory containing exactly one of the two.  Keeping
    the dispatch here means a checkpoint only has to record a path, and the
    Gradio app does not need to know which experiment produced it.
    """
    path = Path(path)

    if path.is_dir():
        for candidate in sorted(path.glob("*_vocab.json")):
            return WordTokenizer.load(candidate)
        for candidate in sorted(path.glob("*.model")):
            return SubwordTokenizer(candidate)
        raise FileNotFoundError(f"no tokeniser found in {path}")

    if path.suffix == ".model":
        return SubwordTokenizer(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("kind") == WordTokenizer.kind:
            return WordTokenizer.load(path)
        model_file = path.parent / payload["model_file"]
        return SubwordTokenizer(model_file)

    raise ValueError(f"unrecognised tokeniser file: {path}")


def normalise_for_scoring(text: str) -> str:
    """Light normalisation applied to both hypothesis and reference.

    Only NFC composition and whitespace collapsing -- never lowercasing or
    punctuation stripping, since BLEU is reported case-sensitively.
    """
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()
