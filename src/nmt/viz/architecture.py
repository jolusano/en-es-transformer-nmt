"""Vector diagrams of the architecture.

Drawn with matplotlib primitives rather than imported as bitmaps so that every
figure is (a) vector output that stays sharp in the PDF, (b) reproducible from
source, and (c) labelled with *this project's* actual shapes and hyper-
parameters instead of the generic ones from the paper.

Figures produced here
---------------------
``transformer_architecture``  the whole encoder-decoder, annotated with tensor
                              shapes at every stage
``attention_mechanism``       scaled dot-product attention as a sequence of
                              matrix operations, with shapes
``multi_head_split``          how one d_model stream becomes h subspaces
``masking``                   padding, causal, and their combination
``positional_encoding``       the sinusoid table and its frequency structure
``residual_norm``             pre-LN against post-LN
``beam_search``               the search tree for a worked example
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from nmt.viz.style import (
    AXIS,
    BLOCK_EDGE,
    BLOCK_FILL,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SEQUENTIAL,
    SERIES,
    figure,
    save,
    use_style,
)

# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------


def _box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    kind: str = "io",
    *,
    fontsize: float = 8.0,
    weight: str = "normal",
    sublabel: str | None = None,
    alpha: float = 1.0,
) -> tuple[float, float]:
    """Draw a rounded, labelled block. Returns its centre."""
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0,rounding_size=0.10",
        linewidth=1.1,
        edgecolor=BLOCK_EDGE.get(kind, INK_MUTED),
        facecolor=BLOCK_FILL.get(kind, "#f0efec"),
        alpha=alpha,
        zorder=3,
    )
    ax.add_patch(patch)

    centre_x, centre_y = x + width / 2, y + height / 2
    text_y = centre_y + (0.09 if sublabel else 0.0)
    ax.text(
        centre_x,
        text_y,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK_PRIMARY,
        fontweight=weight,
        zorder=4,
    )
    if sublabel:
        ax.text(
            centre_x,
            centre_y - 0.15,
            sublabel,
            ha="center",
            va="center",
            fontsize=fontsize - 1.7,
            color=INK_MUTED,
            zorder=4,
        )
    return centre_x, centre_y


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK_SECONDARY,
    style: str = "-|>",
    dashed: bool = False,
    curve: float = 0.0,
    linewidth: float = 1.1,
    zorder: int = 2,
) -> None:
    """Draw a connector between two points."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=11,
            linewidth=linewidth,
            linestyle=(0, (4, 2.5)) if dashed else "solid",
            color=color,
            connectionstyle=f"arc3,rad={curve}",
            shrinkA=1.5,
            shrinkB=1.5,
            zorder=zorder,
        )
    )


def _residual(ax, x: float, y_from: float, y_to: float, x_offset: float, *, color: str) -> None:
    """Draw a residual connection looping around a sublayer."""
    ax.plot(
        [x, x - x_offset, x - x_offset, x],
        [y_from, y_from, y_to, y_to],
        color=color,
        linewidth=1.0,
        linestyle=(0, (3, 2)),
        zorder=1,
        solid_capstyle="round",
    )
    _arrow(ax, (x - x_offset, y_to), (x, y_to), color=color, linewidth=1.0)


def _bare_axes(ax) -> None:
    """Strip an axes down to a blank canvas."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)


# ---------------------------------------------------------------------------
# 1. Full architecture
# ---------------------------------------------------------------------------


def draw_transformer_architecture(path: Path | str) -> list[Path]:
    """The complete model, annotated with the shapes this project actually uses.

    Read it bottom to top on each side. The encoder (left) turns the tagged
    source sentence into one vector per source position; the decoder (right)
    generates the target one token at a time, attending both to what it has
    already produced and, through cross-attention, to the encoder's output.
    """
    use_style()
    fig, ax = figure(9.0, 10.4)
    _bare_axes(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)

    encoder_x, decoder_x = 1.0, 5.9
    block_width = 2.9

    # --- column headings --------------------------------------------------
    ax.text(encoder_x + block_width / 2, 11.62, "ENCODER", ha="center", fontsize=10,
            fontweight="bold", color=BLOCK_EDGE["encoder"])
    ax.text(decoder_x + block_width / 2, 11.62, "DECODER", ha="center", fontsize=10,
            fontweight="bold", color=BLOCK_EDGE["decoder"])

    # --- encoder input ----------------------------------------------------
    ax.text(encoder_x + block_width / 2, 0.28,
            "source:  <2es>  I   am   tired   .   </s>",
            ha="center", fontsize=7.8, color=INK_SECONDARY, family="monospace")
    ax.text(encoder_x + block_width / 2, 0.02,
            "the direction tag tells one model which way to translate",
            ha="center", fontsize=6.8, color=INK_MUTED, style="italic")

    _box(ax, encoder_x, 0.75, block_width, 0.6, "Input Embedding", "embedding",
         sublabel="shared table, 16k x 512")
    _arrow(ax, (encoder_x + block_width / 2, 0.42), (encoder_x + block_width / 2, 0.75))

    _box(ax, encoder_x, 1.62, block_width, 0.52, "+  Positional Encoding", "embedding",
         sublabel="sinusoidal, fixed")
    _arrow(ax, (encoder_x + block_width / 2, 1.35), (encoder_x + block_width / 2, 1.62))

    # --- encoder stack ----------------------------------------------------
    stack_bottom, stack_top = 2.55, 6.55
    ax.add_patch(FancyBboxPatch(
        (encoder_x - 0.32, stack_bottom), block_width + 0.64, stack_top - stack_bottom,
        boxstyle="round,pad=0,rounding_size=0.12", linewidth=1.0,
        edgecolor=BLOCK_EDGE["encoder"], facecolor="#f4f8fd", zorder=1))
    ax.text(encoder_x - 0.52, (stack_bottom + stack_top) / 2, "4x", fontsize=11,
            fontweight="bold", color=BLOCK_EDGE["encoder"], ha="center", va="center",
            rotation=90)

    _box(ax, encoder_x, 2.85, block_width, 0.62, "Multi-Head Self-Attention", "attention",
         sublabel="8 heads x 64 dims, bidirectional")
    _box(ax, encoder_x, 3.80, block_width, 0.45, "Add  &  LayerNorm", "norm")
    _box(ax, encoder_x, 4.60, block_width, 0.62, "Feed-Forward Network", "feedforward",
         sublabel="512 -> 2048 -> 512,  ReLU")
    _box(ax, encoder_x, 5.55, block_width, 0.45, "Add  &  LayerNorm", "norm")

    centre = encoder_x + block_width / 2
    _arrow(ax, (centre, 2.14), (centre, 2.85))
    _arrow(ax, (centre, 3.47), (centre, 3.80))
    _arrow(ax, (centre, 4.25), (centre, 4.60))
    _arrow(ax, (centre, 5.22), (centre, 5.55))
    _arrow(ax, (centre, 6.00), (centre, 6.55))
    _residual(ax, encoder_x, 2.75, 4.02, 0.20, color=BLOCK_EDGE["encoder"])
    _residual(ax, encoder_x, 4.42, 5.77, 0.20, color=BLOCK_EDGE["encoder"])

    ax.text(encoder_x + block_width + 0.42, 6.75, "memory\n(6, 512)", fontsize=7.2,
            color=INK_MUTED, ha="center", va="center")

    # --- decoder input ----------------------------------------------------
    ax.text(decoder_x + block_width / 2, 0.28,
            "target so far:  <s>  Estoy  cansado  .",
            ha="center", fontsize=7.8, color=INK_SECONDARY, family="monospace")
    ax.text(decoder_x + block_width / 2, 0.02,
            "shifted right: position t predicts token t+1",
            ha="center", fontsize=6.8, color=INK_MUTED, style="italic")

    _box(ax, decoder_x, 0.75, block_width, 0.6, "Output Embedding", "embedding",
         sublabel="same shared table")
    _arrow(ax, (decoder_x + block_width / 2, 0.42), (decoder_x + block_width / 2, 0.75))
    _box(ax, decoder_x, 1.62, block_width, 0.52, "+  Positional Encoding", "embedding",
         sublabel="sinusoidal, fixed")
    _arrow(ax, (decoder_x + block_width / 2, 1.35), (decoder_x + block_width / 2, 1.62))

    # --- decoder stack ----------------------------------------------------
    d_bottom, d_top = 2.55, 8.15
    ax.add_patch(FancyBboxPatch(
        (decoder_x - 0.32, d_bottom), block_width + 0.64, d_top - d_bottom,
        boxstyle="round,pad=0,rounding_size=0.12", linewidth=1.0,
        edgecolor=BLOCK_EDGE["decoder"], facecolor="#fdf6f2", zorder=1))
    ax.text(decoder_x + block_width + 0.52, (d_bottom + d_top) / 2, "4x", fontsize=11,
            fontweight="bold", color=BLOCK_EDGE["decoder"], ha="center", va="center",
            rotation=-90)

    _box(ax, decoder_x, 2.85, block_width, 0.62, "Masked Multi-Head Self-Attention",
         "attention", fontsize=7.4, sublabel="cannot see the future")
    _box(ax, decoder_x, 3.80, block_width, 0.45, "Add  &  LayerNorm", "norm")
    _box(ax, decoder_x, 4.60, block_width, 0.62, "Multi-Head Cross-Attention",
         "attention", sublabel="Q from decoder,  K,V from encoder")
    _box(ax, decoder_x, 5.55, block_width, 0.45, "Add  &  LayerNorm", "norm")
    _box(ax, decoder_x, 6.35, block_width, 0.62, "Feed-Forward Network", "feedforward",
         sublabel="512 -> 2048 -> 512,  ReLU")
    _box(ax, decoder_x, 7.30, block_width, 0.45, "Add  &  LayerNorm", "norm")

    dcentre = decoder_x + block_width / 2
    for start, end in ((2.14, 2.85), (3.47, 3.80), (4.25, 4.60),
                       (5.22, 5.55), (6.00, 6.35), (6.97, 7.30), (7.75, 8.35)):
        _arrow(ax, (dcentre, start), (dcentre, end))
    _residual(ax, decoder_x, 2.75, 4.02, 0.20, color=BLOCK_EDGE["decoder"])
    _residual(ax, decoder_x, 4.42, 5.77, 0.20, color=BLOCK_EDGE["decoder"])
    _residual(ax, decoder_x, 6.25, 7.52, 0.20, color=BLOCK_EDGE["decoder"])

    # --- cross-attention link --------------------------------------------
    ax.plot([centre, centre, decoder_x - 0.55],
            [6.55, 7.05, 7.05],
            color=BLOCK_EDGE["encoder"], linewidth=1.3, linestyle=(0, (5, 3)), zorder=2)
    ax.plot([decoder_x - 0.55, decoder_x - 0.55],
            [7.05, 4.91], color=BLOCK_EDGE["encoder"], linewidth=1.3,
            linestyle=(0, (5, 3)), zorder=2)
    _arrow(ax, (decoder_x - 0.55, 4.91), (decoder_x, 4.91),
           color=BLOCK_EDGE["encoder"], linewidth=1.3)
    ax.text(decoder_x - 0.72, 5.9, "K, V", fontsize=8, color=BLOCK_EDGE["encoder"],
            ha="right", va="center", fontweight="bold")
    ax.text((centre + decoder_x) / 2, 7.20,
            "the only path from source to target",
            fontsize=7, color=BLOCK_EDGE["encoder"], ha="center", style="italic")

    # --- output head ------------------------------------------------------
    _box(ax, decoder_x, 8.35, block_width, 0.52, "Linear  (tied to embedding)", "output",
         fontsize=7.6, sublabel="512 -> 16,000")
    _box(ax, decoder_x, 9.10, block_width, 0.52, "Softmax", "output")
    _arrow(ax, (dcentre, 8.87), (dcentre, 9.10))
    _arrow(ax, (dcentre, 9.62), (dcentre, 10.05))

    _box(ax, decoder_x, 10.05, block_width, 0.6, "next-token distribution", "io",
         fontsize=7.6, sublabel="P(token | source, target so far)")

    ax.text(decoder_x + block_width / 2, 10.95,
            "Estoy  cansado  .  </s>", ha="center", fontsize=8,
            color=SERIES[1], family="monospace", fontweight="bold")
    _arrow(ax, (dcentre, 10.65), (dcentre, 10.88), color=SERIES[1])

    # --- legend + specification panel -------------------------------------
    # Both blocks sit inside one hairline panel so the empty upper-left of the
    # canvas reads as a deliberate key rather than as stray floating text.
    ax.add_patch(FancyBboxPatch(
        (0.72, 7.62), 3.45, 2.85,
        boxstyle="round,pad=0,rounding_size=0.10",
        linewidth=0.9, edgecolor=GRIDLINE, facecolor="#f6f5f1", zorder=0))

    legend_y = 9.42
    ax.text(0.88, legend_y + 0.68, "Reading the diagram", fontsize=8.2,
            fontweight="semibold", color=INK_PRIMARY)
    for offset, (line, text) in enumerate([
        ("solid", "data flow"),
        ("dashed", "residual connection"),
        ("cross", "encoder to decoder"),
    ]):
        y = legend_y + 0.34 - offset * 0.29
        if line == "cross":
            ax.plot([0.90, 1.30], [y, y], color=BLOCK_EDGE["encoder"],
                    linewidth=1.3, linestyle=(0, (5, 3)))
        else:
            ax.plot([0.90, 1.30], [y, y], color=INK_SECONDARY, linewidth=1.1,
                    linestyle="solid" if line == "solid" else (0, (3, 2)))
        ax.text(1.40, y, text, fontsize=7.2, color=INK_SECONDARY, va="center")

    ax.text(0.88, 8.74,
            "37.6M parameters\n"
            "4 encoder + 4 decoder layers\n"
            "d_model 512,  8 heads,  d_ff 2048\n"
            "one shared 16k vocabulary\n"
            "embeddings tied 3 ways",
            fontsize=7.2, color=INK_MUTED, va="top", linespacing=1.7)

    ax.set_title(r"Bidirectional EN$\leftrightarrow$ES translation transformer",
                 fontsize=12, pad=16)
    return save(fig, path)


# ---------------------------------------------------------------------------
# 2. Attention mechanism
# ---------------------------------------------------------------------------


def draw_attention_mechanism(path: Path | str) -> list[Path]:
    """Scaled dot-product attention as a chain of matrix operations."""
    use_style()
    fig, ax = figure(9.2, 4.3)
    _bare_axes(ax)
    ax.set_xlim(0, 11.5)
    # Cropped to the drawn content: the explanatory paragraphs bottom out
    # around y = 0.75, so extending the axis to 0 would leave a band of empty
    # surface under the figure in the PDF.
    ax.set_ylim(0.62, 5.85)

    y = 3.5
    height = 0.62

    stages = [
        (0.15, 1.15, "Q", "attention", "(L, 64)\nqueries"),
        (1.75, 1.15, r"$K^{\top}$", "attention", "(64, L)\nkeys"),
        (4.10, 1.35, "scale\n" + r"$\div \sqrt{64}$", "norm", None),
        (5.90, 1.25, "mask", "norm", "set blocked\npositions to −∞"),
        (7.55, 1.35, "softmax", "feedforward", "rows sum to 1"),
        (9.55, 1.15, "× V", "attention", "(L, 64)\nvalues"),
    ]

    positions = []
    for x, width, label, kind, sub in stages:
        cx, cy = _box(ax, x, y, width, height, label, kind, fontsize=9, weight="semibold")
        positions.append((x, width, cx, cy))
        if sub:
            ax.text(cx, y - 0.30, sub, ha="center", va="top", fontsize=7,
                    color=INK_MUTED, linespacing=1.5)

    # matmul symbol between Q and K
    ax.text(1.55, y + height / 2, "×", fontsize=13, ha="center", va="center",
            color=INK_SECONDARY)
    ax.text(3.55, y + height / 2, "=", fontsize=12, ha="center", va="center",
            color=INK_SECONDARY)
    ax.text(3.55, y + height + 0.30, "scores  (L, L)", fontsize=7.2, ha="center",
            color=INK_MUTED)

    for index in range(2, len(positions) - 1):
        x0 = positions[index][0] + positions[index][1]
        x1 = positions[index + 1][0]
        _arrow(ax, (x0, y + height / 2), (x1, y + height / 2))
    _arrow(ax, (2.90, y + height / 2), (4.10, y + height / 2))

    ax.text(10.95, y + height / 2, "= context", fontsize=8.5, va="center",
            color=INK_PRIMARY, fontweight="semibold")

    # --- the formula ------------------------------------------------------
    ax.text(5.75, 5.35,
            r"$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}"
            r"{\sqrt{d_k}}\right)V$",
            fontsize=15, ha="center", va="center", color=INK_PRIMARY)

    # --- why the scaling --------------------------------------------------
    ax.text(0.15, 2.35, r"Why divide by $\sqrt{d_k}$?", fontsize=8.5,
            fontweight="semibold", color=INK_PRIMARY)
    ax.text(0.15, 2.05,
            "A dot product of two 64-dimensional unit-variance vectors has variance 64,\n"
            "so raw scores swing over roughly ±8. Softmax of values that spread out is\n"
            "almost one-hot, and its gradient p(1−p) collapses to zero — the layer stops\n"
            "learning. Dividing by √64 = 8 restores unit variance and keeps softmax in\n"
            "its responsive range.",
            fontsize=7.6, va="top", color=INK_SECONDARY, linespacing=1.75)

    ax.text(6.55, 2.35, "Read it as a soft dictionary lookup",
            fontsize=8.5, fontweight="semibold", color=INK_PRIMARY)
    ax.text(6.55, 2.05,
            "Each query is compared against every key by dot product.\n"
            "A hard lookup would take the single best-matching key;\n"
            "softmax takes a weighted blend instead — which is what\n"
            "makes the whole operation differentiable, and therefore\n"
            "trainable end to end.",
            fontsize=7.6, va="top", color=INK_SECONDARY, linespacing=1.75)

    ax.set_title("Scaled dot-product attention", fontsize=12, pad=12)
    return save(fig, path)


# ---------------------------------------------------------------------------
# 3. Multi-head split
# ---------------------------------------------------------------------------


def draw_multi_head_split(path: Path | str) -> list[Path]:
    """How one 512-wide stream becomes eight 64-wide attention subspaces."""
    use_style()
    fig, ax = figure(8.6, 4.9)
    _bare_axes(ax)
    ax.set_xlim(0, 10)
    # Negative floor leaves a band under the diagram for the caption.
    ax.set_ylim(-0.85, 5.4)

    _box(ax, 3.6, 4.35, 2.8, 0.55, "x   (L, 512)", "io", fontsize=9, weight="semibold")

    heads = 8
    width = 0.86
    gap = 0.20
    total = heads * width + (heads - 1) * gap
    start = (10 - total) / 2

    for head in range(heads):
        x = start + head * (width + gap)
        _box(ax, x, 2.55, width, 0.85, f"head\n{head + 1}", "attention", fontsize=7.2)
        ax.text(x + width / 2, 2.35, "64", ha="center", va="top", fontsize=6.6,
                color=INK_MUTED)
        _arrow(ax, (5.0, 4.35), (x + width / 2, 3.40), curve=0.0,
               color=AXIS, linewidth=0.8)
        _arrow(ax, (x + width / 2, 2.55), (5.0, 1.62), color=AXIS, linewidth=0.8)

    _box(ax, 3.6, 1.05, 2.8, 0.55, r"concat  $\rightarrow$  $W^{O}$", "output", fontsize=9,
         weight="semibold")
    _box(ax, 3.6, 0.20, 2.8, 0.50, "output   (L, 512)", "io", fontsize=8.5)
    _arrow(ax, (5.0, 1.05), (5.0, 0.70))

    ax.text(0.05, 4.62,
            "Each head gets its own\n" + r"$W^Q, W^K, W^V$" + " projections\ninto a 64-dim subspace.",
            fontsize=7.4, color=INK_SECONDARY, va="center", linespacing=1.7)
    ax.text(9.95, 4.62,
            "512 = 8 × 64, so\nthe split is free:\nsame total compute.",
            fontsize=7.4, color=INK_SECONDARY, va="center", ha="right", linespacing=1.7)

    ax.text(5.0, -0.22,
            "One softmax can attend to essentially one place. Translating \"the red house\" as \"la casa roja\" needs the\n"
            "noun tracked for gender agreement and the adjective tracked for reordering — at the same time.",
            fontsize=7.4, ha="center", va="top", color=INK_MUTED, linespacing=1.7)

    ax.set_title("Multi-head attention: h parallel subspaces at one head's cost",
                 fontsize=11.5, pad=12)
    return save(fig, path)


# ---------------------------------------------------------------------------
# 4. Masking
# ---------------------------------------------------------------------------


def draw_masking(path: Path | str) -> list[Path]:
    """Padding, causal and combined masks, drawn as grids."""
    use_style()
    fig, axes = figure(9.0, 3.6, ncols=3)

    tokens = ["<s>", "Estoy", "can-", "sado", ".", "<pad>"]
    size = len(tokens)
    real = 5  # last position is padding

    padding = np.ones((size, size))
    padding[:, real:] = 0.0

    causal = np.tril(np.ones((size, size)))
    combined = padding * causal

    panels = [
        (padding, "Padding mask", "hide filler positions\nadded to square off the batch"),
        (causal, "Causal mask", "hide the future\nposition t sees only ≤ t"),
        (combined, "Combined (decoder self-attention)", "a key is usable only if it is\nboth real and not in the future"),
    ]

    for ax, (matrix, title, subtitle) in zip(axes, panels):
        ax.imshow(matrix, cmap=SEQUENTIAL, vmin=0, vmax=1.35, aspect="equal")
        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=6.8)
        ax.set_yticklabels(tokens, fontsize=6.8)
        ax.set_title(title, fontsize=9, pad=6)
        ax.grid(False)
        ax.set_xlabel("key (attended to)", fontsize=7.2)
        if ax is axes[0]:
            ax.set_ylabel("query (attending)", fontsize=7.2)

        for row in range(size):
            for column in range(size):
                value = matrix[row, column]
                ax.text(column, row, "1" if value else "0", ha="center", va="center",
                        fontsize=6.4,
                        color="#ffffff" if value else INK_MUTED)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.5, -0.30, subtitle, transform=ax.transAxes, ha="center", va="top",
                fontsize=7, color=INK_MUTED, linespacing=1.6)

    fig.suptitle("Attention masks:  1 = may attend,  0 = blocked",
                 fontsize=11.5, x=0.02, ha="left", y=1.04)
    return save(fig, path)


# ---------------------------------------------------------------------------
# 5. Positional encoding
# ---------------------------------------------------------------------------


def draw_positional_encoding(path: Path | str, *, d_model: int = 128,
                             max_length: int = 60) -> list[Path]:
    """The sinusoidal table, plus the waves that generate it."""
    use_style()
    fig, axes = figure(9.0, 3.9, ncols=2, gridspec_kw={"width_ratios": [1.35, 1.0]})

    position = np.arange(max_length)[:, None]
    index = np.arange(0, d_model, 2)
    frequency = np.exp(-math.log(10_000.0) * index / d_model)
    table = np.zeros((max_length, d_model))
    table[:, 0::2] = np.sin(position * frequency)
    table[:, 1::2] = np.cos(position * frequency)

    image = axes[0].imshow(table.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                           interpolation="nearest")
    axes[0].set_xlabel("position in the sentence")
    axes[0].set_ylabel("embedding dimension")
    axes[0].set_title("The encoding table", fontsize=9.5)
    axes[0].grid(False)
    bar = fig.colorbar(image, ax=axes[0], fraction=0.035, pad=0.02)
    bar.set_label("value", fontsize=7.5)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7, color=AXIS)

    for dimension, colour in zip((0, 6, 20, 60), SERIES):
        axes[1].plot(np.arange(max_length), table[:, dimension], color=colour,
                     linewidth=1.7, label=f"dim {dimension}")
    axes[1].set_xlabel("position")
    axes[1].set_ylabel("value")
    # A sinusoid is read by its shape, not off a value axis, so the full grid
    # only competes with the four curves -- but zero is a genuine reference
    # here, so it stays as a hairline behind them.
    axes[1].grid(False)
    axes[1].axhline(0.0, color=GRIDLINE, linewidth=0.6, zorder=0)
    axes[1].set_title("Four of the underlying waves", fontsize=9.5)
    axes[1].legend(ncols=2, loc="lower right")
    axes[1].set_ylim(-1.45, 1.75)

    axes[1].text(0.0, 1.62,
                 "Wavelengths run from 2π up to 10000·2π: fast dimensions separate\n"
                 "neighbouring words, slow ones separate distant regions — like a\n"
                 "counter written in continuous values.",
                 fontsize=6.9, color=INK_MUTED, va="top", linespacing=1.6)

    fig.suptitle("Sinusoidal positional encoding", fontsize=11.5, x=0.02, ha="left")
    return save(fig, path)


# ---------------------------------------------------------------------------
# 6. Pre-LN vs post-LN
# ---------------------------------------------------------------------------


def draw_residual_norm(path: Path | str) -> list[Path]:
    """Pre-layer-norm against post-layer-norm."""
    use_style()
    fig, ax = figure(8.4, 4.7)
    _bare_axes(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.35, 5.0)

    # --- post-LN (left) ---------------------------------------------------
    ax.text(2.1, 4.62, "Post-LN   (original paper)", fontsize=9.5,
            fontweight="semibold", ha="center", color=INK_PRIMARY)
    _box(ax, 1.35, 0.35, 1.5, 0.42, "x", "io", fontsize=8.5)
    _box(ax, 1.35, 1.45, 1.5, 0.52, "Sublayer", "attention", fontsize=8)
    _box(ax, 1.35, 2.45, 1.5, 0.42, r"$\oplus$", "norm", fontsize=10)
    _box(ax, 1.35, 3.35, 1.5, 0.52, "LayerNorm", "norm", fontsize=8)
    for a, b in ((0.77, 1.45), (1.97, 2.45), (2.87, 3.35)):
        _arrow(ax, (2.1, a), (2.1, b))
    ax.plot([1.35, 0.85, 0.85, 1.35], [0.56, 0.56, 2.66, 2.66],
            color=INK_MUTED, linewidth=1.0, linestyle=(0, (3, 2)))
    _arrow(ax, (0.85, 2.66), (1.35, 2.66), color=INK_MUTED, linewidth=1.0)
    ax.text(2.1, 4.05, "x ← LayerNorm(x + Sublayer(x))", fontsize=7.4,
            ha="center", color=INK_SECONDARY, family="monospace")

    # --- pre-LN (right) ---------------------------------------------------
    ax.text(6.9, 4.62, "Pre-LN   (used here)", fontsize=9.5,
            fontweight="semibold", ha="center", color=BLOCK_EDGE["encoder"])
    _box(ax, 6.15, 0.35, 1.5, 0.42, "x", "io", fontsize=8.5)
    _box(ax, 6.15, 1.45, 1.5, 0.52, "LayerNorm", "norm", fontsize=8)
    _box(ax, 6.15, 2.45, 1.5, 0.52, "Sublayer", "attention", fontsize=8)
    _box(ax, 6.15, 3.45, 1.5, 0.42, r"$\oplus$", "norm", fontsize=10)
    for a, b in ((0.77, 1.45), (1.97, 2.45), (2.97, 3.45)):
        _arrow(ax, (6.9, a), (6.9, b))
    ax.plot([6.15, 5.62, 5.62, 6.15], [0.56, 0.56, 3.66, 3.66],
            color=BLOCK_EDGE["encoder"], linewidth=1.5)
    _arrow(ax, (5.62, 3.66), (6.15, 3.66), color=BLOCK_EDGE["encoder"], linewidth=1.5)
    ax.text(6.9, 4.05, "x ← x + Sublayer(LayerNorm(x))", fontsize=7.4,
            ha="center", color=INK_SECONDARY, family="monospace")

    ax.text(5.30, 2.0, "un-normalised\nresidual highway", fontsize=7,
            color=BLOCK_EDGE["encoder"], ha="right", va="center", linespacing=1.6)

    ax.text(5.0, -0.28,
            "In post-LN every residual addition passes through a normalisation on its way to the next layer, so the gradient reaching\n"
            "layer 1 is attenuated once per layer above it — which is why post-LN needs a long, carefully tuned warmup to avoid diverging.\n"
            "In pre-LN the residual path runs from input to output untouched, gradients arrive undamped, and training is stable at higher\n"
            "learning rates with a shorter warmup. That is why this project uses it: a Colab run that diverges at step 300 is an hour lost.",
            fontsize=7.3, va="top", ha="center", color=INK_SECONDARY, linespacing=1.9)

    ax.set_title("Where to put the layer normalisation", fontsize=11.5, pad=12)
    return save(fig, path)


# ---------------------------------------------------------------------------
# 7. Beam search
# ---------------------------------------------------------------------------


def draw_beam_search(path: Path | str) -> list[Path]:
    """The beam-search tree for a short worked example."""
    use_style()
    fig, ax = figure(8.8, 4.4)
    _bare_axes(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)

    levels = [
        [("<s>", 0.00, True)],
        [("Estoy", -0.22, True), ("Tengo", -1.10, True), ("Soy", -2.40, False)],
        [("cansado", -0.35, True), ("muy", -1.55, True),
         ("sueño", -1.42, True), ("cansada", -2.90, False)],
        [(".", -0.41, True), ("hoy", -1.90, True), ("</s>", -2.20, False)],
    ]

    x_positions = [0.9, 3.0, 5.6, 8.3]
    node_centres: list[list[tuple[float, float]]] = []

    for level_index, (level, x) in enumerate(zip(levels, x_positions)):
        centres = []
        count = len(level)
        for node_index, (token, score, kept) in enumerate(level):
            y = 4.65 - node_index * (2.95 / max(1, count - 1) if count > 1 else 0) - (0 if count > 1 else 1.6)
            colour = SERIES[0] if kept else INK_MUTED
            fill = "#dce9f9" if kept else "#f0efec"
            ax.add_patch(FancyBboxPatch(
                (x - 0.62, y - 0.20), 1.30, 0.42,
                boxstyle="round,pad=0,rounding_size=0.09",
                linewidth=1.3 if kept else 0.8,
                edgecolor=colour, facecolor=fill,
                alpha=1.0 if kept else 0.55, zorder=3))
            ax.text(x + 0.03, y + 0.055, token, ha="center", va="center",
                    fontsize=7.6, color=INK_PRIMARY if kept else INK_MUTED,
                    zorder=4, fontweight="semibold" if kept else "normal")
            ax.text(x + 0.03, y - 0.30, f"{score:+.2f}", ha="center", va="top",
                    fontsize=6.5, color=INK_MUTED, zorder=4)
            centres.append((x + 0.68, y))
        node_centres.append(centres)

        if level_index > 0:
            for parent_x, parent_y in node_centres[level_index - 1]:
                for node_index, (_, _, kept) in enumerate(level):
                    child_x, child_y = x - 0.62, centres[node_index][1]
                    ax.plot([parent_x, child_x], [parent_y, child_y],
                            color=SERIES[0] if kept else GRIDLINE,
                            linewidth=1.0 if kept else 0.7,
                            alpha=0.85 if kept else 0.6, zorder=1)

    ax.text(5.0, 0.80, "beam size 2:  keep the two best partial hypotheses at every step",
            fontsize=8, color=INK_PRIMARY, fontweight="semibold", ha="center")
    ax.text(5.0, 0.52,
            "Scores are cumulative log-probabilities — every one is negative, so a longer hypothesis is penalised purely for being\n"
            "longer. Dividing by length^0.6 before the final comparison corrects that; without it beam search systematically truncates.",
            fontsize=7.3, va="top", ha="center", color=INK_SECONDARY, linespacing=1.85)

    ax.text(9.95, 5.05, "greyed = pruned", fontsize=7.2, color=INK_MUTED, ha="right")
    ax.set_title("Beam search over the target vocabulary", fontsize=11.5, pad=12)
    return save(fig, path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


ARCHITECTURE_FIGURES = {
    "transformer_architecture": draw_transformer_architecture,
    "attention_mechanism": draw_attention_mechanism,
    "multi_head_split": draw_multi_head_split,
    "masking": draw_masking,
    "positional_encoding": draw_positional_encoding,
    "residual_norm": draw_residual_norm,
    "beam_search": draw_beam_search,
}


def draw_all(output_dir: Path | str) -> dict[str, list[Path]]:
    """Render every architecture figure into ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        name: function(output_dir / name)
        for name, function in ARCHITECTURE_FIGURES.items()
    }
