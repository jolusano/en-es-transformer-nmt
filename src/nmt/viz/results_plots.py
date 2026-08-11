"""Evaluation figures, read from ``evaluation_test.json``.

Covers the three things the rubric asks the evaluation section to show: the
headline scores against a baseline, how quality varies with sentence length,
and the distribution of failure modes found by the error analysis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nmt.constants import DIRECTION_NAMES, DIRECTIONS
from nmt.utils.io import read_json
from nmt.viz.style import (
    COLOR,
    INK_MUTED,
    INK_SECONDARY,
    SERIES,
    figure,
    save,
    use_style,
)

#: Compact aliases for axes where the full label would not fit. Used by the
#: decoding figure, whose eight groups put two system names side by side.
SHORT_LABELS = {
    "bpe_scratch": "BPE 16k",
    "word_random": "Word random",
    "word_muse": "Word MUSE",
    "lstm_baseline": "LSTM",
}

#: Human-readable names for the experiment identifiers.
RUN_LABELS = {
    "bpe_scratch": "BPE 16k, learned",
    "word_random": "Word 32k, random",
    "word_muse": "Word 32k, MUSE",
    "lstm_baseline": "LSTM + attention",
}


def plot_bleu_comparison(reports: dict[str, dict], output: Path) -> list[Path]:
    """Grouped BLEU and chrF2 per direction, across every system."""
    use_style()
    runs = list(reports)
    # Taller than the other two-panel figures: the shared legend occupies a
    # strip above the axes rather than sitting inside them.
    fig, axes = figure(9.2, 3.8, ncols=2)

    width = 0.8 / max(1, len(runs))
    positions = np.arange(len(DIRECTIONS))

    for metric, axis, label in (
        ("bleu", axes[0], "BLEU"),
        ("chrf2", axes[1], "chrF2"),
    ):
        for run_index, run in enumerate(runs):
            summary = reports[run]["summary"]
            values = [summary[metric][direction] for direction in DIRECTIONS]
            offset = (run_index - (len(runs) - 1) / 2) * width
            bars = axis.bar(
                positions + offset, values, width=width * 0.9,
                color=COLOR.get(run, SERIES[run_index % len(SERIES)]),
                label=RUN_LABELS.get(run, run),
            )
            for bar_patch, value in zip(bars, values):
                axis.text(
                    bar_patch.get_x() + bar_patch.get_width() / 2,
                    value + max(values) * 0.02,
                    f"{value:.1f}", ha="center", fontsize=6.8, color=INK_SECONDARY,
                )

        axis.set_xticks(positions)
        axis.set_xticklabels([DIRECTION_NAMES[d] for d in DIRECTIONS], fontsize=8)
        axis.set_ylabel(label)
        axis.set_title(f"{label} on the held-out test set", fontsize=9.5)
        # Headroom for the value labels, which sit just above each bar.
        axis.set_ylim(0, max(axis.get_ylim()[1], 1.0) * 1.16)

    # One legend for the whole figure, placed in the margin between the title
    # and the axes. Per-axes placement put it on top of the bars, which is the
    # one thing a legend must never do.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper right", bbox_to_anchor=(0.995, 1.005),
        ncols=len(runs), fontsize=7.4, frameon=False,
        handlelength=1.1, columnspacing=1.1, handletextpad=0.45,
    )

    fig.suptitle("Translation quality by system and direction",
                 fontsize=11.5, x=0.02, ha="left")
    return save(fig, output)


def plot_bleu_by_length(reports: dict[str, dict], output: Path) -> list[Path]:
    """Sentence-level BLEU stratified by source length.

    Intended to test the claim that a recurrent model degrades faster than a
    transformer as sentences lengthen, since it must carry information across a
    number of sequential steps proportional to the length while a transformer
    connects any two positions in one attention hop.

    On this corpus the claim is *not* borne out: the transformer's advantage is
    widest on the shortest sentences and narrows monotonically. That is reported
    as a negative result rather than explained away, but the test is weak here --
    83% of the test set is ten tokens or fewer and the longest bucket holds 25
    sentences, so the long-sentence estimates carry very little weight. See the
    evaluation section of the report.
    """
    use_style()
    fig, axes = figure(9.2, 3.3, ncols=2, sharey=True)

    for axis, direction in zip(axes, DIRECTIONS):
        for run_index, (run, report) in enumerate(reports.items()):
            analysis = report["directions"][direction]["error_analysis"]
            buckets = analysis["bleu_by_length"]
            if not buckets:
                continue
            labels = [bucket["bucket"] for bucket in buckets]
            values = [bucket["mean_sentence_bleu"] for bucket in buckets]
            axis.plot(
                range(len(values)), values,
                color=COLOR.get(run, SERIES[run_index % len(SERIES)]),
                marker="o", markersize=4.5,
                label=RUN_LABELS.get(run, run),
            )
            axis.set_xticks(range(len(labels)))
            axis.set_xticklabels(labels, fontsize=7.6)

        axis.set_xlabel("source length (tokens)")
        axis.set_title(DIRECTION_NAMES[direction], fontsize=9.5)

    axes[0].set_ylabel("mean sentence BLEU")
    axes[0].legend(fontsize=7.4)

    fig.suptitle("Quality against sentence length", fontsize=11.5, x=0.02, ha="left")
    return save(fig, output)


def plot_error_categories(report: dict, output: Path, *, run: str = "") -> list[Path]:
    """Failure-mode census produced by the automated error analysis."""
    use_style()
    fig, axes = figure(9.2, 3.4, ncols=2, sharey=True)

    for axis, direction in zip(axes, DIRECTIONS):
        analysis = report["directions"][direction]["error_analysis"]
        rates = analysis["category_rates"]
        if not rates:
            continue
        ordered = sorted(rates.items(), key=lambda item: item[1], reverse=True)
        names = [name.replace("_", " ") for name, _ in ordered]
        values = [rate * 100 for _, rate in ordered]

        colour = COLOR["en-es"] if direction == "en-es" else COLOR["es-en"]
        bars = axis.barh(range(len(names)), values, color=colour, height=0.62)
        axis.set_yticks(range(len(names)))
        axis.set_yticklabels(names, fontsize=7.8)
        axis.invert_yaxis()
        axis.set_xlabel("% of test sentences exhibiting the pattern")
        axis.set_title(DIRECTION_NAMES[direction], fontsize=9.5)
        axis.grid(axis="x")
        axis.grid(axis="y", visible=False)

        for bar_patch, value in zip(bars, values):
            axis.text(value + max(values) * 0.02,
                      bar_patch.get_y() + bar_patch.get_height() / 2,
                      f"{value:.1f}%", va="center", fontsize=7.2, color=INK_SECONDARY)
        axis.set_xlim(0, max(values) * 1.2)

    title = "Failure modes" + (f" — {RUN_LABELS.get(run, run)}" if run else "")
    fig.suptitle(title + "   (categories are not mutually exclusive)",
                 fontsize=11.5, x=0.02, ha="left")
    return save(fig, output)


def plot_decoding_comparison(reports: dict[str, dict], output: Path) -> list[Path]:
    """Greedy against beam search, isolating the contribution of the search."""
    use_style()
    runs = [run for run, report in reports.items() if "greedy_bleu" in report]
    if not runs:
        return []

    fig, ax = figure(7.4, 3.2)
    positions = np.arange(len(runs) * len(DIRECTIONS))
    labels: list[str] = []
    greedy_values: list[float] = []
    beam_values: list[float] = []

    for run in runs:
        for direction in DIRECTIONS:
            labels.append(f"{SHORT_LABELS.get(run, run)}\n{direction}")
            greedy_values.append(reports[run]["greedy_bleu"][direction])
            beam_values.append(reports[run]["summary"]["bleu"][direction])

    ax.bar(positions - 0.19, greedy_values, width=0.36, color=INK_MUTED, label="greedy")
    ax.bar(positions + 0.19, beam_values, width=0.36, color=SERIES[0],
           label="beam search")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7.4)
    ax.set_ylabel("BLEU")
    ax.set_title("What the search contributes", fontsize=11)
    # Above the axes and flush right: inside, matplotlib's "best" position
    # dropped it onto the tallest bars; flush left it ran into the title, which
    # this theme also sets left-aligned.
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.015), ncols=2,
              fontsize=8, frameon=False)
    ax.set_ylim(0, max(beam_values) * 1.18)

    for position, greedy, beam in zip(positions, greedy_values, beam_values):
        gain = beam - greedy
        ax.text(position, max(greedy, beam) + max(beam_values) * 0.03,
                f"{gain:+.2f}", ha="center", fontsize=7.2,
                color=SERIES[2] if gain > 0 else SERIES[1], fontweight="semibold")

    return save(fig, output)


def plot_length_ratio(reports: dict[str, dict], output: Path) -> list[Path]:
    """Hypothesis/reference length ratio -- the truncation diagnostic.

    A ratio below 1 means the system systematically produces output shorter
    than the reference, which BLEU punishes through its brevity penalty. It is
    the signature of a length-penalty setting that is too weak.
    """
    use_style()
    fig, ax = figure(7.4, 3.1)

    runs = list(reports)
    positions = np.arange(len(DIRECTIONS))
    width = 0.8 / max(1, len(runs))

    for run_index, run in enumerate(runs):
        ratios = [
            reports[run]["directions"][direction]["metrics"]["sacrebleu"]["length_ratio"]
            for direction in DIRECTIONS
        ]
        offset = (run_index - (len(runs) - 1) / 2) * width
        ax.bar(positions + offset, ratios, width=width * 0.9,
               color=COLOR.get(run, SERIES[run_index % len(SERIES)]),
               label=RUN_LABELS.get(run, run))

    ax.axhline(1.0, color=INK_MUTED, linewidth=1.1, linestyle=(0, (4, 2)))
    ax.text(len(DIRECTIONS) - 0.45, 1.015, "reference length", fontsize=7.2,
            color=INK_MUTED, ha="right")
    ax.set_xticks(positions)
    ax.set_xticklabels([DIRECTION_NAMES[d] for d in DIRECTIONS], fontsize=8)
    ax.set_ylabel("hypothesis / reference length")
    ax.set_title("Are the translations the right length?", fontsize=11)
    ax.legend(fontsize=7.4)
    ax.set_ylim(0, 1.35)

    return save(fig, output)


def load_reports(results_dir: Path | str, *, split: str = "test") -> dict[str, dict]:
    """Collect every ``evaluation_<split>.json`` under ``results_dir``."""
    results_dir = Path(results_dir)
    reports: dict[str, dict] = {}
    for directory in sorted(results_dir.iterdir()):
        path = directory / f"evaluation_{split}.json"
        if directory.is_dir() and path.exists():
            reports[directory.name] = read_json(path)
    return reports


def draw_all(results_dir: Path | str, output_dir: Path | str,
             *, split: str = "test") -> dict[str, list[Path]]:
    """Render every evaluation figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = load_reports(results_dir, split=split)
    if not reports:
        return {}

    produced = {
        "eval_bleu": plot_bleu_comparison(reports, output_dir / "eval_bleu"),
        "eval_length": plot_bleu_by_length(reports, output_dir / "eval_length"),
        "eval_length_ratio": plot_length_ratio(reports, output_dir / "eval_length_ratio"),
    }
    decoding = plot_decoding_comparison(reports, output_dir / "eval_decoding")
    if decoding:
        produced["eval_decoding"] = decoding

    for run, report in reports.items():
        produced[f"eval_errors_{run}"] = plot_error_categories(
            report, output_dir / f"eval_errors_{run}", run=run
        )

    return produced
