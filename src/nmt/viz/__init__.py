"""Figure generation.

Every figure in the report and the study guide is produced by this package, so
regenerating the document is a matter of re-running

    python -m nmt.viz.make_figures
"""

from nmt.viz.style import (
    COLOR,
    DIVERGING,
    SEQUENTIAL,
    SERIES,
    annotate,
    caption,
    figure,
    label_line_end,
    save,
    use_style,
)

__all__ = [
    "COLOR",
    "DIVERGING",
    "SEQUENTIAL",
    "SERIES",
    "annotate",
    "caption",
    "figure",
    "label_line_end",
    "save",
    "use_style",
]
