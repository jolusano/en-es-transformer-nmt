"""Attention visualisation.

Cross-attention weights are the closest thing a transformer has to an explicit
word alignment: the decoder position generating "casa" tends to place most of
its probability mass on the source position holding "house". Plotting them is
the standard way to show that the model has learned something structured rather
than memorised the corpus, and it is also a genuinely useful debugging tool --
a diffuse, structureless map usually means the model has collapsed to a
language-model prior and is ignoring the source.

The maps are shown per head as well as averaged, because averaging hides the
specialisation: individual heads pick up distinct jobs (one tracks the previous
token, another the sentence boundary, another the aligned content word), and
the mean of eight such heads looks blander than any of them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from nmt.viz.style import (
    INK_MUTED,
    SEQUENTIAL,
    figure,
    save,
    use_style,
)


def _clean(pieces: list[str]) -> list[str]:
    """Strip SentencePiece's word-boundary marker for display."""
    return [piece.replace("▁", "") or "_" for piece in pieces]


def predicted_labels(output_tokens: list[str]) -> list[str]:
    """Row labels for a cross-attention map.

    Cross-attention is captured on the decoder *input*, which is the output
    shifted right, so row ``i`` is the state that predicts token ``i + 1``.
    Labelling rows with the input tokens is off by one and renders the map
    unreadable -- it appears to claim that ``<s>`` attends to a content word.
    Dropping the leading ``<s>`` aligns each row with the token it produces.
    """
    return _clean(output_tokens[1:])


def plot_alignment(
    attention: np.ndarray | torch.Tensor,
    source_tokens: list[str],
    target_tokens: list[str],
    output: Path,
    *,
    title: str = "Cross-attention alignment",
    subtitle: str | None = None,
) -> list[Path]:
    """Heatmap of one ``(target_len, source_len)`` attention matrix."""
    use_style()
    if isinstance(attention, torch.Tensor):
        attention = attention.detach().cpu().numpy()

    source_tokens = _clean(source_tokens)
    target_tokens = predicted_labels(target_tokens)
    attention = attention[: len(target_tokens), : len(source_tokens)]
    target_tokens = target_tokens[: attention.shape[0]]

    width = max(4.5, 0.42 * len(source_tokens) + 2.4)
    height = max(3.0, 0.36 * len(target_tokens) + 1.8)
    fig, ax = figure(width, height)

    image = ax.imshow(attention, cmap=SEQUENTIAL, aspect="auto",
                      vmin=0.0, vmax=float(max(attention.max(), 1e-6)))
    ax.set_xticks(range(len(source_tokens)))
    ax.set_xticklabels(source_tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(target_tokens)))
    ax.set_yticklabels(target_tokens, fontsize=8)
    ax.set_xlabel("source tokens")
    ax.set_ylabel("token being generated")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Mark the arg-max source position for each generated token: the model's
    # implicit alignment decision.
    for row in range(attention.shape[0]):
        column = int(attention[row].argmax())
        ax.add_patch(
            __import__("matplotlib").patches.Rectangle(
                (column - 0.5, row - 0.5), 1, 1,
                fill=False, edgecolor="#eb6834", linewidth=1.6,
            )
        )

    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("attention weight", fontsize=7.5)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7)

    ax.set_title(title, fontsize=11)
    if subtitle:
        ax.text(0.0, 1.04, subtitle, transform=ax.transAxes, fontsize=7.6,
                color=INK_MUTED, va="bottom")

    return save(fig, output)


def plot_head_grid(
    attention: np.ndarray | torch.Tensor,
    source_tokens: list[str],
    target_tokens: list[str],
    output: Path,
    *,
    title: str = "Cross-attention, per head",
) -> list[Path]:
    """Small multiples: one panel per head of one layer.

    ``attention`` is ``(heads, target_len, source_len)``.
    """
    use_style()
    if isinstance(attention, torch.Tensor):
        attention = attention.detach().cpu().numpy()

    heads = attention.shape[0]
    columns = min(4, heads)
    rows = (heads + columns - 1) // columns

    source_tokens = _clean(source_tokens)
    target_tokens = predicted_labels(target_tokens)

    fig, axes = figure(2.2 * columns + 0.8, 2.0 * rows + 1.0,
                       nrows=rows, ncols=columns)
    axes = np.atleast_1d(axes).ravel()

    for head in range(heads):
        ax = axes[head]
        matrix = attention[head][: len(target_tokens), : len(source_tokens)]
        ax.imshow(matrix, cmap=SEQUENTIAL, aspect="auto", vmin=0.0,
                  vmax=float(max(matrix.max(), 1e-6)))
        ax.set_title(f"head {head + 1}", fontsize=8.5, pad=4)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        if head % columns == 0:
            ax.set_yticks(range(len(target_tokens)))
            ax.set_yticklabels(target_tokens, fontsize=6.2)
        else:
            ax.set_yticks([])

        if head >= heads - columns:
            ax.set_xticks(range(len(source_tokens)))
            ax.set_xticklabels(source_tokens, rotation=90, fontsize=6.2)
        else:
            ax.set_xticks([])

    for extra in range(heads, len(axes)):
        axes[extra].set_visible(False)

    fig.suptitle(title, fontsize=11.5, x=0.02, ha="left")
    return save(fig, output)


def plot_layer_progression(
    layers: list[np.ndarray | torch.Tensor],
    source_tokens: list[str],
    target_tokens: list[str],
    output: Path,
) -> list[Path]:
    """Head-averaged cross-attention at each decoder layer.

    Early layers are typically broad; later layers sharpen onto specific source
    positions as the decoder settles on what it is translating.
    """
    use_style()
    source_tokens = _clean(source_tokens)
    target_tokens = predicted_labels(target_tokens)

    matrices = []
    for layer in layers:
        if isinstance(layer, torch.Tensor):
            layer = layer.detach().cpu().numpy()
        if layer.ndim == 3:
            layer = layer.mean(axis=0)
        matrices.append(layer[: len(target_tokens), : len(source_tokens)])

    fig, axes = figure(2.3 * len(matrices) + 0.8, 3.0, ncols=len(matrices))
    axes = np.atleast_1d(axes)

    for index, (ax, matrix) in enumerate(zip(axes, matrices)):
        ax.imshow(matrix, cmap=SEQUENTIAL, aspect="auto", vmin=0.0,
                  vmax=float(max(matrix.max(), 1e-6)))
        ax.set_title(f"layer {index + 1}", fontsize=9)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(range(len(source_tokens)))
        ax.set_xticklabels(source_tokens, rotation=90, fontsize=6.4)
        if index == 0:
            ax.set_yticks(range(len(target_tokens)))
            ax.set_yticklabels(target_tokens, fontsize=6.6)
        else:
            ax.set_yticks([])

        entropy = float(
            -(matrix * np.log(matrix + 1e-9)).sum(axis=-1).mean()
        )
        ax.set_xlabel(f"mean entropy {entropy:.2f}", fontsize=7)

    fig.suptitle("Cross-attention sharpens with depth", fontsize=11.5,
                 x=0.02, ha="left")
    return save(fig, output)


def attention_from_translator(translator, text: str, direction: str):
    """Run one sentence and pull out its attention tensors.

    Returns ``(result, maps, source_pieces, target_pieces)`` where ``maps`` is
    the raw ``attention_maps()`` dictionary of the model.
    """
    result = translator.translate(text, direction, return_attention=True)

    source_pieces = result.source_tokens
    target_pieces = result.output_tokens
    maps = translator.model.attention_maps()

    return result, maps, source_pieces, target_pieces
