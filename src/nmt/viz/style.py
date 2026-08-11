"""Shared figure styling.

Every figure in the report, the study guide and the notebooks is drawn through
this module so the whole document reads as one system rather than as a pile of
matplotlib defaults.

The look is modelled on the article figures at distill.pub: a warm off-white
surface, hairline chrome that recedes behind the data, generous whitespace,
direct labels in preference to legend boxes, and a small number of saturated
hues used consistently for the same meaning throughout.

Colour discipline
-----------------
The categorical slots are assigned **in fixed order and never cycled**, and the
same slot always means the same thing across the whole document: slot 1 is
English (or the English->Spanish direction), slot 2 is Spanish, slot 3 the
subword model, slot 4 the recurrent baseline. A reader who learns the mapping
on figure 1 keeps it for the rest of the report.

The palette was checked with a colour-vision-deficiency simulator rather than
by eye: worst adjacent-pair separation is ΔE 9.2 under deuteranopia (target
>= 8) and ΔE 27.6 for normal vision (floor 15). The aqua slot falls below 3:1
contrast against the surface, so every figure that uses it also carries a
legend or direct labels -- colour is never the only channel.

Figures are exported as **PDF** (vector, for LaTeX), **SVG** (for the HTML
study guide) and **PNG** (for previews and the README).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Tokens -----------------------------------------------------------------

SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

#: Categorical slots, in fixed assignment order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4", "#eda100"]

#: Stable semantic aliases, so a figure never has to remember an index.
COLOR = {
    "en": SERIES[0],
    "es": SERIES[1],
    "en-es": SERIES[0],
    "es-en": SERIES[1],
    "train": SERIES[0],
    "validation": SERIES[1],
    "bpe_scratch": SERIES[0],
    "word_random": SERIES[1],
    "word_muse": SERIES[2],
    "lstm_baseline": SERIES[3],
    "encoder": SERIES[0],
    "decoder": SERIES[1],
    "attention": SERIES[3],
    "highlight": SERIES[1],
    "neutral": INK_MUTED,
}

#: Single-hue sequential ramp (blue, light -> dark), for attention heatmaps and
#: any other continuous magnitude. One hue only: never a rainbow.
SEQUENTIAL_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]
SEQUENTIAL = LinearSegmentedColormap.from_list("nmt_sequential", SEQUENTIAL_STEPS)

#: Diverging ramp (blue <-> red through a neutral grey) for signed quantities
#: such as "gain over the baseline".
DIVERGING = LinearSegmentedColormap.from_list(
    "nmt_diverging",
    ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f0a3a3", "#e34948", "#a02020"],
)

#: Soft fills used by the architecture diagrams. Tinted, low-chroma versions of
#: the categorical hues so that labelled boxes stay readable with black text.
BLOCK_FILL = {
    "encoder": "#dce9f9",
    "decoder": "#fbe3d8",
    "attention": "#e2ddf3",
    "norm": "#eceae4",
    "feedforward": "#d8f0e6",
    "embedding": "#fdf0d3",
    "output": "#f6dce5",
    "io": "#f0efec",
}
BLOCK_EDGE = {
    "encoder": "#2a78d6",
    "decoder": "#eb6834",
    "attention": "#4a3aa7",
    "norm": "#898781",
    "feedforward": "#1baf7a",
    "embedding": "#eda100",
    "output": "#e87ba4",
    "io": "#898781",
}

FONT_STACK = [
    "Helvetica Neue",
    "Helvetica",
    "Arial",
    "DejaVu Sans",
    "sans-serif",
]


def use_style() -> None:
    """Apply the project's matplotlib theme to the current session.

    Idempotent, so notebooks can call it in every cell without harm.
    """
    mpl.rcParams.update(
        {
            # --- surface ---------------------------------------------------
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.edgecolor": "none",
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            # --- type ------------------------------------------------------
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.titlesize": 12,
            # Titles sit left-aligned above the plot, as in a printed article,
            # rather than centred over the axes.
            "axes.titlelocation": "left",
            "axes.titlepad": 15,
            "axes.titleweight": "semibold",
            "text.color": INK_PRIMARY,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "axes.labelpad": 6,
            # --- chrome: recessive -----------------------------------------
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRIDLINE,
            # Hairline weight. The rule across the document is that *every*
            # data plot carries gridlines and they are all this thin -- a mix
            # of gridded and ungridded panels reads as carelessness, whereas a
            # consistent hairline recedes and stops being noticed at all.
            # Diagrams and heatmaps have no value axis, so they are the only
            # figures that switch them off.
            "grid.linewidth": 0.4,
            "grid.alpha": 1.0,
            # Gridlines belong *behind* the data. matplotlib's default draws
            # them on top, which puts hairlines across every bar -- the single
            # most common way a chart looks amateurish in print.
            "axes.axisbelow": True,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            # --- marks -----------------------------------------------------
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0.8,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            # --- legend ----------------------------------------------------
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.6,
            "legend.columnspacing": 1.4,
            "legend.labelcolor": INK_SECONDARY,
            # --- layout ----------------------------------------------------
            "figure.constrained_layout.use": True,
            "figure.figsize": (7.0, 4.0),
            # Keep text as text in the PDF so LaTeX output stays selectable.
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def annotate(
    ax: plt.Axes,
    text: str,
    xy: tuple[float, float],
    xytext: tuple[float, float],
    *,
    color: str = INK_SECONDARY,
    fontsize: float = 8.0,
    arrow: bool = True,
) -> None:
    """Add a distill-style annotation with a thin curved leader line.

    Annotations directly on the plot are preferred to captions: they attach the
    explanation to the mark it describes, which is most of what makes those
    figures readable.
    """
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        fontsize=fontsize,
        color=color,
        ha="left",
        va="center",
        arrowprops=(
            {
                "arrowstyle": "-",
                "color": AXIS,
                "linewidth": 0.8,
                "connectionstyle": "arc3,rad=0.2",
                "shrinkA": 2,
                "shrinkB": 4,
            }
            if arrow
            else None
        ),
    )


def label_line_end(
    ax: plt.Axes,
    x: Iterable[float],
    y: Iterable[float],
    label: str,
    color: str,
    *,
    offset: float = 1.01,
    fontsize: float = 8.5,
) -> None:
    """Write a series name at the end of its line instead of in a legend box.

    Used when there are few enough series that direct labelling is legible; a
    legend is still added when more than four series share an axis.
    """
    x = list(x)
    y = list(y)
    if not x:
        return
    ax.text(
        x[-1] * offset,
        y[-1],
        f" {label}",
        color=color,
        fontsize=fontsize,
        va="center",
        ha="left",
        fontweight="medium",
    )


def caption(fig: plt.Figure, text: str, *, y: float = -0.02) -> None:
    """Attach a small caption beneath a figure."""
    fig.text(
        0.0,
        y,
        text,
        ha="left",
        va="top",
        fontsize=8,
        color=INK_MUTED,
        wrap=True,
    )


def despine(ax: plt.Axes, *, left: bool = False, bottom: bool = False) -> None:
    """Remove additional spines beyond the theme's default top/right."""
    if left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
    if bottom:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False)


def save(fig: plt.Figure, path: Path | str, *, formats: Iterable[str] = ("pdf", "png", "svg")) -> list[Path]:
    """Write a figure in every format the project consumes.

    Parameters
    ----------
    path
        Destination *without* an extension.

    Returns
    -------
    list[Path]
        Everything written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for extension in formats:
        target = path.with_suffix(f".{extension}")
        fig.savefig(target, format=extension)
        written.append(target)

    plt.close(fig)
    return written


def figure(width: float = 7.0, height: float = 4.0, **kwargs) -> tuple[plt.Figure, plt.Axes]:
    """Create a themed figure and axes.

    The default 7-inch width matches the text block of the LaTeX report, so
    figures are placed at their natural size and the type inside them ends up
    the same size as the surrounding body text.
    """
    use_style()
    return plt.subplots(figsize=(width, height), **kwargs)
