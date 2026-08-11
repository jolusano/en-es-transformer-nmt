"""Pre-trained cross-lingual word vectors for embedding initialisation.

One of the two experiments the report compares initialises the embedding matrix
from **MUSE** vectors (Conneau et al., 2018, *Word Translation Without Parallel
Data*) instead of from a random draw.

Why MUSE specifically, and not plain monolingual fastText: the model has a
*single shared* embedding table serving English and Spanish on both the encoder
and the decoder side.  Concatenating two independently-trained monolingual
tables would place "cat" and "gato" in unrelated coordinate systems, and the
initialisation would be actively misleading -- worse than random, because the
model would first have to *unlearn* the accidental geometry.  MUSE vectors for
the two languages are aligned into a **shared** space by a learned orthogonal
transform, so translation-equivalent words already start close together.  That
is a meaningful prior for a shared table, and it is the hypothesis the
experiment tests.

Format of a ``.vec`` file (text, frequency-sorted, ~628 MB per language)::

    200000 300
    , -0.0112864 -0.00206967 ...
    the 0.0324 -0.0141 ...

Words are lowercase.  :class:`~nmt.data.tokenizer.WordTokenizer` performs a
lowercase fallback on lookup, which is what makes the two halves line up.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from nmt.constants import RESERVED_TOKENS
from nmt.utils.io import project_root
from nmt.utils.logging_utils import get_logger

logger = get_logger(__name__)

MUSE_URLS = {
    "en": "https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec",
    "es": "https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.es.vec",
}


@dataclass
class EmbeddingInitReport:
    """Coverage statistics for the pre-trained initialisation.

    ``covered_fraction`` is the headline number: it says what proportion of the
    vocabulary actually received a pre-trained vector rather than a random one,
    and therefore how much of the hypothesis is really being tested.
    """

    vocab_size: int = 0
    dimension: int = 0
    covered: int = 0
    covered_by_lowercase: int = 0
    randomly_initialised: int = 0
    reserved: int = 0
    covered_fraction: float = 0.0
    per_language_hits: dict[str, int] = field(default_factory=dict)
    sample_missing: list[str] = field(default_factory=list)
    source: str = "MUSE (wiki.multi.{lang}.vec)"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_vectors(
    path: Path | str,
    *,
    wanted: set[str] | None = None,
    max_words: int | None = None,
) -> tuple[dict[str, np.ndarray], int]:
    """Stream a ``.vec`` file, keeping only the words we need.

    Parameters
    ----------
    wanted
        When given, only these words are retained.  The file is 628 MB per
        language but our vocabulary is at most ~32k entries, so filtering
        during the scan keeps peak memory in the tens of megabytes instead of
        several gigabytes.
    max_words
        Optional early stop.  The file is frequency-sorted, so truncating to
        the first *n* lines keeps the *most common* words.

    Returns
    -------
    (dict, int)
        Word -> vector, and the declared dimensionality.
    """
    path = Path(path)
    vectors: dict[str, np.ndarray] = {}

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().split()
        declared_count, dimension = int(header[0]), int(header[1])

        for line_number, line in enumerate(handle):
            if max_words is not None and line_number >= max_words:
                break
            # Words containing a space would break a naive split, so the vector
            # is parsed from the right: the last `dimension` fields are floats.
            parts = line.rstrip().rsplit(" ", dimension)
            if len(parts) != dimension + 1:
                continue
            word = parts[0]
            if wanted is not None and word not in wanted:
                continue
            vectors[word] = np.asarray(parts[1:], dtype=np.float32)

    logger.info(
        "%s: kept %s / %s vectors (dim=%d)",
        path.name,
        f"{len(vectors):,}",
        f"{declared_count:,}",
        dimension,
    )
    return vectors, dimension


def build_embedding_matrix(
    itos: Sequence[str],
    *,
    vector_paths: dict[str, Path | str] | None = None,
    dimension: int = 300,
    seed: int = 42,
    scale_to_match: bool = True,
) -> tuple[np.ndarray, EmbeddingInitReport]:
    """Build a ``(len(itos), dimension)`` matrix from aligned word vectors.

    Words absent from both MUSE tables -- reserved symbols, rare inflections,
    proper nouns -- are drawn from ``N(0, sigma^2)`` where ``sigma`` matches the
    empirical standard deviation of the vectors that *were* found.  Sampling at
    the pre-trained scale matters: the default ``N(0, 1)`` initialisation would
    give unknown words vectors around 30x longer than known ones, and those
    words would dominate the attention logits at the start of training.

    ``<pad>`` is forced to exactly zero so a padded position contributes
    nothing before masking is applied.
    """
    vector_paths = vector_paths or {
        lang: project_root() / "data" / "raw" / "embeddings" / f"wiki.multi.{lang}.vec"
        for lang in ("en", "es")
    }

    wanted = set(itos) | {word.lower() for word in itos}

    tables: dict[str, dict[str, np.ndarray]] = {}
    observed_dimension = dimension
    for lang, path in vector_paths.items():
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}. Download it with:\n"
                f"  curl -L -o {path} {MUSE_URLS.get(lang, '<url>')}"
            )
        tables[lang], observed_dimension = load_vectors(path, wanted=wanted)

    if observed_dimension != dimension:
        logger.warning(
            "Requested dimension %d but vectors are %d-dimensional; using %d.",
            dimension,
            observed_dimension,
            observed_dimension,
        )
        dimension = observed_dimension

    report = EmbeddingInitReport(vocab_size=len(itos), dimension=dimension)
    reserved = set(RESERVED_TOKENS)

    # Statistics of the pre-trained cloud, used to scale the random fallback.
    found_stack = [
        vector
        for table in tables.values()
        for vector in list(table.values())[:20_000]
    ]
    sigma = float(np.std(np.stack(found_stack))) if found_stack else 0.02

    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, sigma, size=(len(itos), dimension)).astype(np.float32)

    for index, word in enumerate(itos):
        if word in reserved:
            report.reserved += 1
            continue

        hit: np.ndarray | None = None
        hit_language = None
        used_lowercase = False

        for lang, table in tables.items():
            candidate = table.get(word)
            if candidate is None:
                candidate = table.get(word.lower())
                if candidate is not None:
                    used_lowercase = True
            if candidate is not None:
                # A word present in both tables (cognates, shared punctuation,
                # names) gets the mean of the two aligned vectors, which is
                # itself a valid point in the shared space.
                hit = candidate if hit is None else (hit + candidate) / 2.0
                hit_language = lang if hit_language is None else "both"

        if hit is None:
            report.randomly_initialised += 1
            if len(report.sample_missing) < 40:
                report.sample_missing.append(word)
            continue

        matrix[index] = hit
        report.covered += 1
        report.covered_by_lowercase += int(used_lowercase)
        report.per_language_hits[hit_language or "?"] = (
            report.per_language_hits.get(hit_language or "?", 0) + 1
        )

    # <pad> must contribute nothing to any sum.
    matrix[0] = 0.0

    if scale_to_match:
        # Transformers scale embeddings by sqrt(d_model) before adding the
        # positional signal; MUSE vectors have norm ~0.3 while a freshly
        # initialised embedding has norm ~sqrt(d) * 0.02. Normalising the whole
        # matrix to unit average norm puts the pre-trained run on the same
        # footing as the random one, so the comparison isolates *geometry*
        # rather than scale.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        mean_norm = float(np.mean(norms[norms > 0]))
        if mean_norm > 0:
            matrix /= mean_norm
        matrix[0] = 0.0

    report.covered_fraction = report.covered / max(
        1, len(itos) - report.reserved
    )
    logger.info(
        "Pre-trained initialisation covers %s / %s non-reserved types (%.1f%%)",
        f"{report.covered:,}",
        f"{len(itos) - report.reserved:,}",
        100 * report.covered_fraction,
    )
    return matrix, report


def nearest_neighbours(
    matrix: np.ndarray,
    itos: Sequence[str],
    query: str,
    *,
    k: int = 10,
) -> list[tuple[str, float]]:
    """Cosine nearest neighbours of ``query`` in an embedding table.

    Used in the report to show, qualitatively, that the MUSE space really is
    cross-lingual at initialisation (the neighbours of "house" should include
    "casa"), and to show what the learned-from-scratch table looks like after
    training.
    """
    stoi = {word: index for index, word in enumerate(itos)}
    if query not in stoi:
        raise KeyError(f"{query!r} is not in the vocabulary")

    normalised = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    similarity = normalised @ normalised[stoi[query]]
    order = np.argsort(-similarity)[: k + 1]

    return [(itos[i], float(similarity[i])) for i in order if itos[i] != query][:k]


def iter_vocabulary_words(itos: Iterable[str]) -> Iterable[str]:
    """Yield vocabulary entries that are plausibly real words.

    Filters out reserved symbols and pure punctuation, so coverage statistics
    are not flattered by the fact that "," is in every embedding table.
    """
    reserved = set(RESERVED_TOKENS)
    for word in itos:
        if word in reserved:
            continue
        if not any(character.isalpha() for character in word):
            continue
        yield word


def main() -> None:
    """``python -m nmt.data.embeddings`` -- build the initialisation matrix.

    Writes a ``.npy`` array aligned to the word tokeniser's index order, plus a
    JSON coverage report that the paper quotes.
    """
    import argparse

    import numpy as np

    from nmt.data.tokenizer import WordTokenizer
    from nmt.utils.io import write_json
    from nmt.utils.logging_utils import setup_logging

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=project_root() / "artifacts" / "tokenizers" / "word_vocab.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root() / "artifacts" / "tokenizers" / "muse_embeddings.npy",
    )
    parser.add_argument(
        "--vectors-dir",
        type=Path,
        default=project_root() / "data" / "raw" / "embeddings",
    )
    parser.add_argument("--dimension", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    tokenizer = WordTokenizer.load(args.tokenizer)

    matrix, report = build_embedding_matrix(
        tokenizer.itos,
        vector_paths={
            lang: args.vectors_dir / f"wiki.multi.{lang}.vec" for lang in ("en", "es")
        },
        dimension=args.dimension,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, matrix)
    logger.info("Wrote %s with shape %s", args.output, matrix.shape)

    # Qualitative check: in a genuinely cross-lingual space, an English word's
    # neighbours should include its Spanish translation.
    probes = ["house", "water", "book", "run", "happy"]
    neighbours: dict[str, list] = {}
    for probe in probes:
        try:
            neighbours[probe] = nearest_neighbours(matrix, tokenizer.itos, probe, k=8)
        except KeyError:
            continue
    report_payload = report.to_dict()
    report_payload["nearest_neighbours_at_init"] = neighbours

    results = project_root() / "artifacts" / "results" / "embedding_init_report.json"
    write_json(results, report_payload)
    logger.info("Coverage report -> %s", results)


if __name__ == "__main__":
    main()
