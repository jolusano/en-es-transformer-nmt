"""Training-dynamics figures, read from ``training_summary.json`` / ``train_log.jsonl``.

These are the curves the brief asks to be presented and discussed. The design
choice worth noting: **loss and BLEU are plotted on separate axes, never as a
dual-axis chart**. Two y-scales on one frame invite the reader to compare the
slopes of two quantities that share no units, and the apparent crossing point
is an artefact of how the axes were scaled. Separate panels say the same thing
without the illusion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nmt.training.scheduler import schedule_preview
from nmt.utils.io import read_json, read_jsonl
from nmt.viz.style import (
    AXIS,
    COLOR,
    INK_MUTED,
    INK_SECONDARY,
    SERIES,
    figure,
    save,
    use_style,
)


def _history(summary: dict) -> list[dict]:
    return summary.get("history", [])


def plot_training_curves(summary: dict, output: Path) -> list[Path]:
    """Loss, perplexity and validation BLEU across epochs.

    The gap between the training and validation curves is the overfitting
    story; the point where validation loss turns up while validation BLEU is
    still climbing is why the project keeps two checkpoints (best loss and best
    BLEU) rather than one.
    """
    use_style()
    history = _history(summary)
    if not history:
        raise ValueError("training summary contains no epoch history")

    epochs = [record["epoch"] for record in history]
    train_loss = [record["train"]["loss"] for record in history]
    validation_loss = [record["validation"]["loss"] for record in history]
    train_ppl = [record["train"]["perplexity"] for record in history]
    validation_ppl = [record["validation"]["perplexity"] for record in history]
    bleu = [record["validation"].get("bleu") for record in history]
    has_bleu = any(value is not None for value in bleu)

    columns = 3 if has_bleu else 2
    fig, axes = figure(9.4, 3.2, ncols=columns)

    axes[0].plot(epochs, train_loss, color=COLOR["train"], label="training")
    axes[0].plot(epochs, validation_loss, color=COLOR["validation"], label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("label-smoothed cross entropy")
    axes[0].set_title("Loss", fontsize=9.5)
    axes[0].legend()

    best_index = int(np.argmin(validation_loss))
    axes[0].scatter([epochs[best_index]], [validation_loss[best_index]],
                    s=42, facecolor="none", edgecolor=COLOR["validation"],
                    linewidth=1.6, zorder=5)
    axes[0].annotate(
        f"best validation loss\n{validation_loss[best_index]:.3f} at epoch {epochs[best_index]}",
        xy=(epochs[best_index], validation_loss[best_index]),
        xytext=(0.34, 0.74), textcoords="axes fraction",
        fontsize=7.2, color=INK_SECONDARY,
        arrowprops={"arrowstyle": "-", "color": AXIS, "linewidth": 0.8,
                    "connectionstyle": "arc3,rad=0.25"},
    )

    axes[1].plot(epochs, train_ppl, color=COLOR["train"], label="training")
    axes[1].plot(epochs, validation_ppl, color=COLOR["validation"], label="validation")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("perplexity")
    axes[1].set_title("Perplexity", fontsize=9.5)
    axes[1].set_yscale("log")
    axes[1].legend()

    if has_bleu:
        values = [v for v in bleu if v is not None]
        bleu_epochs = [e for e, v in zip(epochs, bleu) if v is not None]
        axes[2].plot(bleu_epochs, values, color=SERIES[2])
        axes[2].set_xlabel("epoch")
        axes[2].set_ylabel("BLEU (validation sample, greedy)")
        axes[2].set_title("Translation quality", fontsize=9.5)

        peak = int(np.argmax(values))
        axes[2].scatter([bleu_epochs[peak]], [values[peak]], s=42, facecolor="none",
                        edgecolor=SERIES[2], linewidth=1.6, zorder=5)
        axes[2].annotate(
            f"best BLEU {values[peak]:.2f}\nat epoch {bleu_epochs[peak]}",
            xy=(bleu_epochs[peak], values[peak]),
            xytext=(0.26, 0.22), textcoords="axes fraction",
            fontsize=7.2, color=INK_SECONDARY,
            arrowprops={"arrowstyle": "-", "color": AXIS, "linewidth": 0.8,
                        "connectionstyle": "arc3,rad=-0.25"},
        )

    name = summary.get("name", "run")
    fig.suptitle(f"Training dynamics — {name}", fontsize=11.5, x=0.02, ha="left")
    return save(fig, output)


def plot_learning_rate(log_records: list[dict], output: Path,
                       *, summary: dict | None = None) -> list[Path]:
    """The realised learning-rate schedule, with the analytic curve behind it."""
    use_style()
    steps = [r["step"] for r in log_records if r.get("event") == "step"]
    rates = [r["learning_rate"] for r in log_records if r.get("event") == "step"]

    fig, axes = figure(9.0, 3.1, ncols=2)

    if steps:
        axes[0].plot(steps, rates, color=SERIES[0], linewidth=1.8)
        peak = int(np.argmax(rates))
        axes[0].scatter([steps[peak]], [rates[peak]], s=40, facecolor="none",
                        edgecolor=SERIES[0], linewidth=1.5, zorder=5)
        axes[0].annotate(
            f"peak {rates[peak]:.2e}\nat step {steps[peak]:,}",
            xy=(steps[peak], rates[peak]), xytext=(0.42, 0.78),
            textcoords="axes fraction", fontsize=7.2, color=INK_SECONDARY,
            arrowprops={"arrowstyle": "-", "color": AXIS, "linewidth": 0.8,
                        "connectionstyle": "arc3,rad=0.2"},
        )
    axes[0].set_xlabel("optimiser step")
    axes[0].set_ylabel("learning rate")
    axes[0].set_title("Realised schedule", fontsize=9.5)

    warmup = 4_000
    d_model = 512
    if summary:
        warmup = summary["config"]["optim"]["warmup_steps"]
        d_model = summary["config"]["model"]["d_model"]

    total = max(20_000, max(steps) if steps else 20_000)
    curve = schedule_preview("inverse_sqrt", d_model=d_model,
                             warmup_steps=warmup, total_steps=total)
    axes[1].plot(range(1, total + 1), curve, color=SERIES[0], linewidth=1.8)
    axes[1].axvline(warmup, color=INK_MUTED, linewidth=1.0, linestyle=(0, (3, 2)))
    axes[1].text(warmup * 1.08, max(curve) * 0.55,
                 f"warmup ends\nstep {warmup:,}", fontsize=7.2, color=INK_SECONDARY,
                 linespacing=1.5)
    axes[1].set_xlabel("optimiser step")
    axes[1].set_ylabel("relative learning rate")
    axes[1].set_title(r"Inverse-square-root schedule", fontsize=9.5)
    axes[1].text(
        0.38, 0.33,
        "Linear ramp, then decay as 1/√t.\n"
        "Early gradients are mostly noise because\n"
        "attention starts near-uniform — large steps\n"
        "then are large in an arbitrary direction.",
        transform=axes[1].transAxes, fontsize=7.0, color=INK_MUTED,
        linespacing=1.6, va="top",
    )

    fig.suptitle("Learning-rate schedule", fontsize=11.5, x=0.02, ha="left")
    return save(fig, output)


def plot_gradient_norms(log_records: list[dict], output: Path) -> list[Path]:
    """Gradient norm over training, showing where clipping bound the update."""
    use_style()
    steps = [r["step"] for r in log_records if r.get("event") == "step"]
    norms = [r.get("grad_norm", 0.0) for r in log_records if r.get("event") == "step"]

    fig, ax = figure(7.2, 3.1)
    ax.plot(steps, norms, color=SERIES[0], linewidth=1.0, alpha=0.75)

    if len(norms) > 25:
        window = max(5, len(norms) // 60)
        smoothed = np.convolve(norms, np.ones(window) / window, mode="valid")
        ax.plot(steps[window - 1:], smoothed, color=SERIES[3], linewidth=2.0,
                label=f"moving average ({window} steps)")
        ax.legend(loc="upper right")

    ax.axhline(1.0, color=SERIES[1], linewidth=1.2, linestyle=(0, (4, 2)))
    ax.text(max(steps) * 0.02 if steps else 0, 1.06, "clipping threshold",
            fontsize=7.4, color=SERIES[1])
    ax.set_xlabel("optimiser step")
    ax.set_ylabel("global gradient norm")
    ax.set_yscale("log")
    ax.set_title("Gradient norms and the clipping threshold", fontsize=11)
    return save(fig, output)


def plot_run_comparison(summaries: dict[str, dict], output: Path) -> list[Path]:
    """Validation loss and BLEU for every experiment on shared axes."""
    use_style()
    fig, axes = figure(9.0, 3.6, ncols=2)

    for name, summary in summaries.items():
        history = _history(summary)
        if not history:
            continue
        epochs = [r["epoch"] for r in history]
        losses = [r["validation"]["loss"] for r in history]
        colour = COLOR.get(name, SERIES[0])
        axes[0].plot(epochs, losses, color=colour, label=name)

        bleu = [(r["epoch"], r["validation"].get("bleu")) for r in history]
        bleu = [(e, v) for e, v in bleu if v is not None]
        if bleu:
            axes[1].plot([e for e, _ in bleu], [v for _, v in bleu],
                         color=colour, label=name)

    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("validation loss")
    axes[0].set_title("Validation loss", fontsize=9.5)

    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation BLEU (greedy sample)")
    axes[1].set_title("Validation BLEU", fontsize=9.5)

    # Shared legend above the panels: inside either one it covered the curves.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.995, 1.005),
               ncols=len(labels), fontsize=7.4, frameon=False,
               handlelength=1.2, columnspacing=1.2, handletextpad=0.45)

    fig.suptitle("All experiments", fontsize=11.5, x=0.02, ha="left")
    return save(fig, output)


def draw_all(results_dir: Path | str, output_dir: Path | str,
             *, runs: list[str] | None = None) -> dict[str, list[Path]]:
    """Render the training figures for every run found in ``results_dir``."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    produced: dict[str, list[Path]] = {}
    summaries: dict[str, dict] = {}

    #: Never plotted: a two-epoch, 2.7M-parameter pipeline check whose curves
    #: would appear beside the real runs as an unexplained stub.
    excluded = {"smoke"}

    candidates = runs or [
        path.name for path in sorted(results_dir.iterdir())
        if path.is_dir()
        and path.name not in excluded
        and (path / "training_summary.json").exists()
    ]

    for run in candidates:
        summary_path = results_dir / run / "training_summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        if not summary.get("history"):
            continue
        summaries[run] = summary

        produced[f"train_curves_{run}"] = plot_training_curves(
            summary, output_dir / f"train_curves_{run}"
        )

        log_path = results_dir / run / "train_log.jsonl"
        if log_path.exists():
            records = read_jsonl(log_path)
            produced[f"train_lr_{run}"] = plot_learning_rate(
                records, output_dir / f"train_lr_{run}", summary=summary
            )
            if any(r.get("grad_norm") for r in records):
                produced[f"train_grad_{run}"] = plot_gradient_norms(
                    records, output_dir / f"train_grad_{run}"
                )

    if len(summaries) > 1:
        produced["train_comparison"] = plot_run_comparison(
            summaries, output_dir / "train_comparison"
        )

    return produced
