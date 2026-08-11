"""Corpus exploration figures, built from ``artifacts/results/dataset_stats.json``.

Every number plotted here is read from the JSON the preprocessing pipeline
wrote, so re-running ``python -m nmt.data.build`` and then this module is what
refreshes the report's data section. Nothing is transcribed by hand.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from nmt.utils.io import project_root, read_json
from nmt.viz.style import (
    AXIS,
    COLOR,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SERIES,
    figure,
    save,
    use_style,
)


def _read_split(path: Path) -> list[tuple[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader, None)
        return [(row[0], row[1]) for row in reader if len(row) >= 2]


def plot_length_distributions(
    processed_dir: Path, output: Path, *, max_length: int = 30
) -> list[Path]:
    """Sentence-length histograms and the English/Spanish length relationship.

    The left panel is the distribution that motivates the ``max_tokens=64``
    filter and the token-bucketing sampler; the right panel shows that the two
    sides track each other closely, which is what makes a length-ratio filter a
    sensible way to catch misaligned pairs.
    """
    use_style()
    pairs = _read_split(processed_dir / "train.tsv")
    english = np.array([len(s.split()) for s, _ in pairs])
    spanish = np.array([len(t.split()) for _, t in pairs])

    fig, axes = figure(9.0, 3.5, ncols=2, gridspec_kw={"width_ratios": [1.25, 1.0]})

    bins = np.arange(0, max_length + 2) - 0.5
    axes[0].hist(english, bins=bins, color=COLOR["en"], alpha=0.72,
                 label="English", edgecolor="none")
    axes[0].hist(spanish, bins=bins, color=COLOR["es"], alpha=0.62,
                 label="Spanish", edgecolor="none")
    axes[0].set_xlabel("sentence length (whitespace tokens)")
    axes[0].set_ylabel("sentence pairs")
    axes[0].set_title("Length distribution", fontsize=9.5)
    axes[0].legend()
    axes[0].set_xlim(0, max_length)

    median_en = float(np.median(english))
    axes[0].axvline(median_en, color=INK_MUTED, linewidth=1.0, linestyle=(0, (3, 2)))
    axes[0].annotate(
        f"median {median_en:.0f} tokens\n95th percentile {np.percentile(english, 95):.0f}",
        xy=(median_en, axes[0].get_ylim()[1] * 0.72),
        xytext=(median_en + 5.5, axes[0].get_ylim()[1] * 0.80),
        fontsize=7.4, color=INK_SECONDARY,
        arrowprops={"arrowstyle": "-", "color": AXIS, "linewidth": 0.8,
                    "connectionstyle": "arc3,rad=0.2"},
    )

    # --- joint distribution ----------------------------------------------
    limit = 26
    heat, xedges, yedges = np.histogram2d(
        english, spanish, bins=[np.arange(1, limit), np.arange(1, limit)]
    )
    image = axes[1].imshow(
        np.log1p(heat.T), origin="lower", aspect="equal", cmap="Blues",
        extent=(1, limit - 1, 1, limit - 1), interpolation="nearest",
    )
    axes[1].plot([1, limit - 1], [1, limit - 1], color=SERIES[1], linewidth=1.2,
                 linestyle=(0, (4, 2)), label="equal length")
    axes[1].set_xlabel("English tokens")
    axes[1].set_ylabel("Spanish tokens")
    axes[1].set_title("Source vs target length", fontsize=9.5)
    axes[1].grid(False)
    axes[1].legend(loc="upper left")

    bar = fig.colorbar(image, ax=axes[1], fraction=0.042, pad=0.02)
    bar.set_label("log(1 + pairs)", fontsize=7.4)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7)

    correlation = float(np.corrcoef(english, spanish)[0, 1])
    axes[1].text(
        0.97, 0.05, f"r = {correlation:.3f}", transform=axes[1].transAxes,
        ha="right", fontsize=8, color=INK_PRIMARY, fontweight="semibold",
    )

    fig.suptitle("Tatoeba EN–ES: sentences are short and length-matched",
                 fontsize=11.5, x=0.02, ha="left")
    return save(fig, output)


def plot_vocabulary_growth(stats: dict, output: Path) -> list[Path]:
    """Type counts, coverage curves and the OOV cost of a closed vocabulary.

    This is the figure that justifies subword tokenisation. Spanish has far
    more surface forms than English for the same amount of text -- inflection,
    clitics, gendered agreement -- so a word-level vocabulary of any practical
    size leaves roughly twice the out-of-vocabulary rate on the Spanish side.
    """
    use_style()
    vocabulary = stats["splits"]["train"]["vocabulary"]

    fig, axes = figure(9.0, 3.4, ncols=3, gridspec_kw={"width_ratios": [0.8, 1.2, 1.0]})

    # --- panel 1: type counts --------------------------------------------
    languages = ["English", "Spanish"]
    types = [vocabulary["en"]["types"], vocabulary["es"]["types"]]
    bars = axes[0].bar(languages, types, color=[COLOR["en"], COLOR["es"]], width=0.55)
    axes[0].set_ylabel("distinct word types")
    axes[0].set_title("Vocabulary size", fontsize=9.5)
    for bar_patch, value in zip(bars, types):
        axes[0].text(bar_patch.get_x() + bar_patch.get_width() / 2,
                     value + max(types) * 0.02, f"{value:,}",
                     ha="center", fontsize=8, color=INK_PRIMARY,
                     fontweight="semibold")
    axes[0].set_ylim(0, max(types) * 1.22)
    ratio = types[1] / types[0]
    axes[0].text(0.5, 0.96, f"{ratio:.2f}x more Spanish types",
                 transform=axes[0].transAxes, ha="center", va="top",
                 fontsize=7.6, color=INK_SECONDARY)

    # --- panel 2: coverage curves ----------------------------------------
    cutoffs = [1_000, 5_000, 10_000, 20_000, 32_000]
    for language, colour, label in (("en", COLOR["en"], "English"),
                                    ("es", COLOR["es"], "Spanish")):
        coverage = [vocabulary[language][f"coverage_top_{c}"] * 100 for c in cutoffs]
        axes[1].plot(cutoffs, coverage, color=colour, marker="o", label=label)
        # Label only the endpoints, placed clear of the marker: the first to
        # the right and below, the last to the left and above, so the two
        # language curves never write over each other.
        axes[1].annotate(f"{coverage[0]:.1f}%", (cutoffs[0], coverage[0]),
                         textcoords="offset points", xytext=(7, -9),
                         fontsize=7, color=colour, ha="left")
        axes[1].annotate(f"{coverage[-1]:.1f}%", (cutoffs[-1], coverage[-1]),
                         textcoords="offset points", xytext=(-6, -11),
                         fontsize=7, color=colour, ha="right")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("vocabulary size (most frequent types)")
    axes[1].set_ylabel("% of running tokens covered")
    axes[1].set_title("Coverage of running text", fontsize=9.5)
    axes[1].set_ylim(70, 102)
    axes[1].legend(loc="lower right")

    # --- panel 3: OOV rates ----------------------------------------------
    oov = stats["splits"]["test"]["word_oov"]
    rates = [oov["en"]["unk_rate"] * 100, oov["es"]["unk_rate"] * 100]
    bars = axes[2].bar(languages, rates, color=[COLOR["en"], COLOR["es"]], width=0.55)
    axes[2].set_ylabel("% of test tokens mapped to <unk>")
    axes[2].set_title("Out-of-vocabulary rate\n(32k word vocabulary)", fontsize=9.5)
    for bar_patch, value in zip(bars, rates):
        axes[2].text(bar_patch.get_x() + bar_patch.get_width() / 2, value + 0.08,
                     f"{value:.2f}%", ha="center", fontsize=8,
                     color=INK_PRIMARY, fontweight="semibold")
    axes[2].set_ylim(0, max(rates) * 1.42)
    axes[2].text(0.5, 0.94, "subword tokenisation\nremoves this entirely",
                 transform=axes[2].transAxes, ha="center", va="top",
                 fontsize=7.4, color=INK_SECONDARY, linespacing=1.5)

    fig.suptitle("Spanish morphology is the reason this project uses subwords",
                 fontsize=11.5, x=0.02, ha="left")
    return save(fig, output)


def plot_subword_effect(stats: dict, output: Path) -> list[Path]:
    """What BPE does to sequence lengths, and the fertility it costs."""
    use_style()
    train = stats["splits"]["train"]

    fig, axes = figure(9.0, 3.3, ncols=2)

    labels = ["English\nwords", "English\nsubwords", "Spanish\nwords", "Spanish\nsubwords"]
    medians = [
        train["length_words"]["en"]["median"],
        train["length_subwords"]["en"]["median"],
        train["length_words"]["es"]["median"],
        train["length_subwords"]["es"]["median"],
    ]
    p95 = [
        train["length_words"]["en"]["p95"],
        train["length_subwords"]["en"]["p95"],
        train["length_words"]["es"]["p95"],
        train["length_subwords"]["es"]["p95"],
    ]
    colours = [COLOR["en"], COLOR["en"], COLOR["es"], COLOR["es"]]
    positions = np.arange(4)

    axes[0].bar(positions - 0.19, medians, width=0.36, color=colours,
                label="median", alpha=0.95)
    axes[0].bar(positions + 0.19, p95, width=0.36, color=colours,
                label="95th percentile", alpha=0.45)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels, fontsize=7.6)
    axes[0].set_ylabel("tokens per sentence")
    axes[0].set_title("Segmentation lengthens sequences", fontsize=9.5)
    axes[0].legend()
    for position, value in zip(positions - 0.19, medians):
        axes[0].text(position, value + 0.3, f"{value:.0f}", ha="center", fontsize=7.4,
                     color=INK_PRIMARY)
    for position, value in zip(positions + 0.19, p95):
        axes[0].text(position, value + 0.3, f"{value:.0f}", ha="center", fontsize=7.4,
                     color=INK_SECONDARY)

    fertility = [train["subword_fertility"]["en"], train["subword_fertility"]["es"]]
    bars = axes[1].bar(["English", "Spanish"], fertility,
                       color=[COLOR["en"], COLOR["es"]], width=0.5)
    axes[1].axhline(1.0, color=INK_MUTED, linewidth=1.0, linestyle=(0, (3, 2)))
    axes[1].set_ylabel("subword tokens per word")
    axes[1].set_title("Fertility of the 16k joint BPE model", fontsize=9.5)
    axes[1].set_ylim(0, max(fertility) * 1.4)
    for bar_patch, value in zip(bars, fertility):
        axes[1].text(bar_patch.get_x() + bar_patch.get_width() / 2, value + 0.03,
                     f"{value:.2f}", ha="center", fontsize=8.5,
                     color=INK_PRIMARY, fontweight="semibold")
    axes[1].text(0.5, 0.95,
                 "1.00 would mean every word is a single token;\n"
                 "the excess is the price paid for zero <unk>",
                 transform=axes[1].transAxes, va="top",
                 ha="center", fontsize=7.3, color=INK_SECONDARY, linespacing=1.5)

    fig.suptitle("The subword trade: longer sequences, open vocabulary",
                 fontsize=11.5, x=0.02, ha="left")
    return save(fig, output)


def plot_corpus_construction(stats: dict, output: Path) -> list[Path]:
    """Where the data came from and how the splits were made."""
    use_style()
    fig, axes = figure(9.0, 3.3, ncols=2, gridspec_kw={"width_ratios": [1.1, 1.0]})

    # --- panel 1: pipeline funnel ----------------------------------------
    pairing = stats.get("pairing", {})
    cleaning = stats.get("cleaning", {})
    stages = [
        ("English sentences\nin Tatoeba", pairing.get("english_sentences", 0)),
        ("Spanish sentences", pairing.get("spanish_sentences", 0)),
        ("EN–ES pairs\nafter the join", pairing.get("pairs_found", 0)),
        ("after cleaning\nand dedup", cleaning.get("kept_pairs", 0)),
    ]
    names = [name for name, _ in stages]
    values = [value for _, value in stages]
    colours = [INK_MUTED, INK_MUTED, SERIES[0], SERIES[2]]

    bars = axes[0].barh(range(len(stages)), values, color=colours, height=0.6)
    axes[0].set_yticks(range(len(stages)))
    axes[0].set_yticklabels(names, fontsize=7.6)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("sentences / pairs")
    axes[0].set_title("From raw exports to a usable bitext", fontsize=9.5)
    axes[0].grid(axis="x")
    axes[0].grid(axis="y", visible=False)
    for bar_patch, value in zip(bars, values):
        axes[0].text(value + max(values) * 0.015,
                     bar_patch.get_y() + bar_patch.get_height() / 2,
                     f"{value:,}", va="center", fontsize=7.6, color=INK_PRIMARY)
    axes[0].set_xlim(0, max(values) * 1.22)

    # Two decimals: at one decimal this rounds to "100.0%", which reads as
    # "nothing was removed" and contradicts the caption's 99.96%.
    retention = cleaning.get("retention_rate", 0) * 100
    axes[0].text(
        0.98, 0.06,
        f"cleaning kept {retention:.2f}% — Tatoeba is a curated,\n"
        "already-clean corpus, so the filters mostly confirm that",
        transform=axes[0].transAxes, ha="right", va="bottom",
        fontsize=7.2, color=INK_SECONDARY, linespacing=1.5,
    )

    # --- panel 2: split sizes --------------------------------------------
    split = stats.get("split", {})
    split_pairs = split.get("split_pairs", {})
    names = ["train", "validation", "test"]
    values = [split_pairs.get(name, 0) for name in names]
    bars = axes[1].bar(names, values, color=[SERIES[0], SERIES[1], SERIES[2]], width=0.55)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("sentence pairs (log scale)")
    for bar_patch, value in zip(bars, values):
        axes[1].text(bar_patch.get_x() + bar_patch.get_width() / 2, value * 1.15,
                     f"{value:,}", ha="center", fontsize=8,
                     color=INK_PRIMARY, fontweight="semibold")
    axes[1].set_ylim(1e3, max(values) * 4)

    leakage = sum(split.get("leakage_check", {}).values())
    axes[1].set_title(
        "Splits, made by connected component\n"
        f"{split.get('components', 0):,} components · "
        f"{leakage} sentences shared across splits",
        fontsize=9.5,
    )

    fig.suptitle("Corpus construction", fontsize=11.5, x=0.02, ha="left")
    return save(fig, output)


def plot_token_frequency(stats: dict, output: Path) -> list[Path]:
    """Zipf curve and the most frequent tokens in each language."""
    use_style()
    vocabulary = stats["splits"]["train"]["vocabulary"]
    fig, axes = figure(9.0, 3.3, ncols=2, gridspec_kw={"width_ratios": [1.0, 1.15]})

    for language, colour, label in (("en", COLOR["en"], "English"),
                                    ("es", COLOR["es"], "Spanish")):
        counts = [count for _, count in vocabulary[language]["most_common"]]
        ranks = np.arange(1, len(counts) + 1)
        axes[0].plot(ranks, counts, color=colour, marker="o", markersize=3.5,
                     label=label, linewidth=1.6)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("rank")
    axes[0].set_ylabel("frequency")
    axes[0].set_title("Top-25 tokens follow Zipf's law", fontsize=9.5)
    axes[0].legend()

    top = 12
    english_tokens = vocabulary["en"]["most_common"][:top]
    spanish_tokens = vocabulary["es"]["most_common"][:top]
    positions = np.arange(top)

    axes[1].barh(positions + 0.2, [c for _, c in english_tokens], height=0.38,
                 color=COLOR["en"], label="English")
    axes[1].barh(positions - 0.2, [c for _, c in spanish_tokens], height=0.38,
                 color=COLOR["es"], label="Spanish")
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels(
        [f"{e}  /  {s}" for (e, _), (s, _) in zip(english_tokens, spanish_tokens)],
        fontsize=7.2,
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("occurrences in the training split")
    axes[1].set_title("Most frequent tokens (English / Spanish)", fontsize=9.5)
    axes[1].legend(loc="lower right")
    axes[1].grid(axis="x")
    axes[1].grid(axis="y", visible=False)

    fig.suptitle("Token frequency", fontsize=11.5, x=0.02, ha="left")
    return save(fig, output)


def draw_all(
    output_dir: Path | str | None = None,
    *,
    stats_path: Path | str | None = None,
    processed_dir: Path | str | None = None,
) -> dict[str, list[Path]]:
    """Render every corpus figure."""
    root = project_root()
    output_dir = Path(output_dir or root / "reports" / "figures")
    stats_path = Path(stats_path or root / "artifacts" / "results" / "dataset_stats.json")
    processed_dir = Path(processed_dir or root / "data" / "processed")

    stats = read_json(stats_path)
    return {
        "data_lengths": plot_length_distributions(processed_dir, output_dir / "data_lengths"),
        "data_vocabulary": plot_vocabulary_growth(stats, output_dir / "data_vocabulary"),
        "data_subwords": plot_subword_effect(stats, output_dir / "data_subwords"),
        "data_construction": plot_corpus_construction(stats, output_dir / "data_construction"),
        "data_frequency": plot_token_frequency(stats, output_dir / "data_frequency"),
    }
