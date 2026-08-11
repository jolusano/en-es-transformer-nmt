"""Interactive translation application.

    python app/app.py                                   # auto-discovers a checkpoint
    python app/app.py --checkpoint path/to/best_bleu.pt
    python app/app.py --share                           # temporary public link

Satisfies the brief's inference requirement: it loads the saved model once at
start-up, accepts any sentence the user types, translates it in either
direction without retraining, and shows the result immediately.

Beyond the minimum, the interface exposes the parts of the system the report
discusses so a viewer can interrogate the model rather than only use it:

* **direction switch** -- the same weights serve both ways; the app changes
  only the direction tag prepended to the source,
* **decoding controls** -- greedy against beam search, beam width and length
  penalty, so the effect described in the evaluation section can be reproduced
  live,
* **attention view** -- the cross-attention alignment for the sentence just
  translated,
* **tokenisation view** -- how the sentence was split, which is what makes an
  unexpected translation of a rare word explicable.
"""

from __future__ import annotations

import argparse
import html
import sys
import time
from pathlib import Path

# Allow `python app/app.py` from anywhere without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gradio as gr  # noqa: E402

from nmt.constants import DIRECTION_NAMES  # noqa: E402
from nmt.inference.search import DecodeConfig  # noqa: E402
from nmt.inference.translator import Translator  # noqa: E402
from nmt.utils.io import project_root  # noqa: E402
from nmt.utils.logging_utils import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

#: Major version of the installed Gradio, used to route the handful of kwargs
#: that moved between 4.x/5.x and 6.x.
_GRADIO_MAJOR = int(gr.__version__.split(".")[0])

DIRECTION_CHOICES = {
    "English → Spanish": "en-es",
    "Spanish → English": "es-en",
}

EXAMPLES = [
    ["I am very tired today.", "English → Spanish"],
    ["Where is the nearest train station?", "English → Spanish"],
    ["She has been studying Spanish for three years.", "English → Spanish"],
    ["The red house on the corner belongs to my grandmother.", "English → Spanish"],
    ["¿Cuántos años tienes?", "Spanish → English"],
    ["Me gustaría reservar una mesa para dos personas.", "Spanish → English"],
    ["No sé si podré ir a la fiesta mañana.", "Spanish → English"],
    ["El libro que me prestaste era muy interesante.", "Spanish → English"],
]

CSS = """
/* Gradio applies its own light/dark theme through component classes, not
   through `prefers-color-scheme`, so a media query here would never fire.
   Secondary text therefore *inherits* the theme's ink and is dimmed with
   opacity, which adapts to either scheme; the token chips are the exception --
   their wash is a fixed brand colour, so they pin their own foreground to keep
   contrast (~16:1) rather than inheriting near-white ink onto a pale chip,
   which is what made them unreadable before. */
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    max-width: 1180px !important;
}

#title h1 { margin-bottom: 0.15rem; font-weight: 650; letter-spacing: -0.01em; }
#subtitle p { font-size: 0.95rem; margin: 0 0 0.15rem 0; opacity: 0.78; }
#byline p   { font-size: 0.82rem; margin: 0 0 0.9rem 0; opacity: 0.58; }

.token-row { line-height: 2.15; margin-bottom: 2px; }
.token-chip {
    display: inline-block;
    padding: 2px 8px;
    margin: 2px 3px;
    border-radius: 6px;
    background: #dce9f9;
    color: #0b0b0b;
    border: 1px solid #2a78d6;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.78rem;
    white-space: nowrap;
}
.token-chip.special {
    background: #fbe3d8;
    color: #0b0b0b;
    border-color: #eb6834;
    font-weight: 600;
}

.token-caption {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    margin: 12px 0 3px 0;
    opacity: 0.85;
}
.stat { font-size: 0.85rem; margin: 4px 0; opacity: 0.72; }
.hint { font-size: 0.82rem; font-style: italic; margin-top: 8px; opacity: 0.62; }
"""


def find_checkpoint(explicit: Path | None = None) -> Path:
    """Locate a checkpoint to serve.

    Prefers the BLEU-selected checkpoint of the main experiment, then any other
    run, so that a fresh clone with a single trained model just works.
    """
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"no checkpoint at {explicit}")
        return explicit

    root = project_root() / "artifacts" / "checkpoints"
    preferred = ["bpe_scratch", "word_muse", "word_random", "lstm_baseline"]

    for run in preferred:
        for filename in ("best_bleu.pt", "best_loss.pt", "last.pt"):
            candidate = root / run / filename
            if candidate.exists():
                return candidate

    for candidate in sorted(root.glob("*/best_bleu.pt")):
        return candidate
    for candidate in sorted(root.glob("*/*.pt")):
        return candidate

    raise FileNotFoundError(
        f"No checkpoint found under {root}.\n"
        "Train one first:  python -m nmt.training.train --config configs/bpe_scratch.yaml\n"
        "or pass --checkpoint /path/to/model.pt"
    )


def render_tokens(pieces: list[str]) -> str:
    """Render a token list as HTML chips, marking reserved symbols."""
    chips = []
    for piece in pieces:
        display = piece.replace("▁", "␣")
        special = piece.startswith("<") and piece.endswith(">")
        chips.append(
            f'<span class="token-chip{" special" if special else ""}">'
            f"{html.escape(display)}</span>"
        )
    return "<div class='token-row'>" + "".join(chips) + "</div>"


def build_interface(translator: Translator, checkpoint: Path) -> gr.Blocks:
    """Assemble the Gradio app around a loaded translator."""

    def translate(
        text: str,
        direction_label: str,
        strategy: str,
        beam_size: int,
        length_penalty: float,
        show_attention: bool,
    ):
        if not text or not text.strip():
            return "", "", "", None

        direction = DIRECTION_CHOICES[direction_label]

        # Rebuild the decode config per request so the sliders take effect
        # without reloading the model.
        translator.decode_config = DecodeConfig(
            strategy=strategy,
            beam_size=int(beam_size),
            length_penalty=float(length_penalty),
        )

        started = time.time()
        result = translator.translate(
            text, direction, return_attention=show_attention
        )
        elapsed = time.time() - started

        stats = (
            f'<p class="stat">{DIRECTION_NAMES[direction]} &middot; '
            f"{strategy}{f' (beam {int(beam_size)})' if strategy == 'beam' else ''} "
            f"&middot; {result.num_source_tokens} source tokens &rarr; "
            f"{result.num_output_tokens} generated &middot; "
            f"{elapsed * 1000:.0f} ms</p>"
        )

        tokens_html = (
            "<div class='token-caption'>Source tokenisation</div>"
            + render_tokens(result.source_tokens)
            + "<div class='token-caption'>Generated tokens</div>"
            + render_tokens(result.output_tokens)
            + "<div class='hint'>Orange chips are reserved symbols: the "
              "direction tag that tells the model which way to translate, and "
              "the end-of-sentence marker. &#9251; marks a word boundary.</div>"
        )

        figure = None
        if show_attention and result.attention is not None and result.attention.numel():
            figure = _attention_figure(result)

        return result.translation, stats, tokens_html, figure

    def _attention_figure(result):
        """Render the alignment heatmap for the sentence just translated.

        Drawn on a *transparent* background with mid-grey ink rather than the
        report's light surface: the app follows the viewer's system theme, and
        a hard white panel dropped into a dark interface is the one thing that
        makes an otherwise tidy UI look unfinished. Mid-grey labels stay legible
        against either background, and the sequential blue ramp works on both.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from nmt.viz.style import INK_MUTED, SEQUENTIAL, use_style

        use_style()
        attention = result.attention.numpy()
        source = [t.replace("\u2581", "") or "_" for t in result.source_tokens]
        target = [t.replace("\u2581", "") or "_" for t in result.output_tokens]
        attention = attention[: len(target), : len(source)]

        fig, ax = plt.subplots(
            figsize=(max(4.2, 0.44 * len(source) + 1.7),
                     max(2.8, 0.36 * len(target) + 1.5))
        )
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        ax.imshow(attention, cmap=SEQUENTIAL, aspect="auto")
        ax.set_xticks(range(len(source)))
        ax.set_xticklabels(source, rotation=45, ha="right", fontsize=8, color=INK_MUTED)
        ax.set_yticks(range(len(target)))
        ax.set_yticklabels(target, fontsize=8, color=INK_MUTED)
        ax.set_xlabel("source token", fontsize=8.5, color=INK_MUTED)
        ax.set_ylabel("generated token", fontsize=8.5, color=INK_MUTED)
        ax.tick_params(colors=INK_MUTED, length=0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Ring the strongest source position for each generated token: the
        # model's implicit alignment decision, and the thing worth looking at.
        for row in range(attention.shape[0]):
            column = int(attention[row].argmax())
            ax.add_patch(plt.Rectangle((column - 0.5, row - 0.5), 1, 1,
                                       fill=False, edgecolor="#eb6834", linewidth=1.5))

        fig.tight_layout()
        return fig

    def swap(direction_label: str, text: str, translation: str):
        """Flip the direction and move the translation into the input box."""
        other = (
            "Spanish → English"
            if direction_label == "English → Spanish"
            else "English → Spanish"
        )
        return other, (translation or text), ""

    # Gradio 6 moved `theme` and `css` from the Blocks constructor to launch().
    # Passing them in the wrong place is a warning on 6.x and a TypeError on
    # 4.x, so the version is checked once and the kwargs routed accordingly.
    blocks_kwargs = {"title": "EN <-> ES Neural Machine Translation"}
    if _GRADIO_MAJOR < 6:
        blocks_kwargs["css"] = CSS
        blocks_kwargs["theme"] = gr.themes.Soft(primary_hue="blue")

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.Markdown(
            "# English ↔ Spanish Neural Machine Translation", elem_id="title"
        )
        gr.Markdown(
            "A single transformer, trained from scratch on the Tatoeba corpus, "
            "translating in **both** directions.",
            elem_id="subtitle",
        )
        gr.Markdown(
            "AIG230 Final Project · Group 7 — Jose Luis Sanchez Noriega & Bikash Subedi",
            elem_id="byline",
        )

        with gr.Row():
            with gr.Column(scale=1):
                direction = gr.Radio(
                    choices=list(DIRECTION_CHOICES),
                    value="English → Spanish",
                    label="Direction",
                )
                source = gr.Textbox(
                    label="Input sentence",
                    placeholder="Type a sentence and press Translate…",
                    lines=4,
                )
                with gr.Row():
                    translate_button = gr.Button("Translate", variant="primary")
                    swap_button = gr.Button("Swap direction")
                    clear_button = gr.ClearButton(value="Clear")

                with gr.Accordion("Decoding settings", open=False):
                    strategy = gr.Radio(
                        choices=["beam", "greedy"], value="beam",
                        label="Search strategy",
                        info="Beam explores several hypotheses and usually scores "
                             "1–2 BLEU higher; greedy is faster.",
                    )
                    beam_size = gr.Slider(
                        1, 10, value=4, step=1, label="Beam width",
                        info="Hypotheses kept at each step.",
                    )
                    length_penalty = gr.Slider(
                        0.0, 1.5, value=0.6, step=0.05, label="Length penalty α",
                        info="Scores are divided by length^α. Lower values "
                             "produce shorter output.",
                    )
                    show_attention = gr.Checkbox(
                        value=True, label="Show attention alignment",
                        info="Plots which source words the model looked at for "
                             "each word it generated. Costs one extra forward "
                             "pass, so turn it off if you want minimum latency.",
                    )

            with gr.Column(scale=1):
                # Kept to the kwargs that are stable across Gradio 4/5/6: the
                # copy-button argument was renamed between major versions and
                # is not worth pinning a version over.
                output = gr.Textbox(label="Translation", lines=4)
                stats = gr.HTML()
                with gr.Accordion("How the sentence was tokenised", open=False):
                    tokens = gr.HTML()
                attention_plot = gr.Plot(label="Cross-attention alignment")
                gr.Markdown(
                    "<p class='hint'>Each row is a generated token and each "
                    "column a source token; darker means the model attended "
                    "there more. Try <i>The red house is very big.</i> — the "
                    "adjective and noun swap order in Spanish, and the "
                    "alignment crosses over.</p>"
                )

        gr.Examples(examples=EXAMPLES, inputs=[source, direction], label="Try one of these")

        gr.Markdown(
            f"<p class='stat'>Serving <code>{checkpoint.parent.name}/{checkpoint.name}</code> "
            f"on <code>{translator.device}</code> &middot; "
            f"vocabulary {translator.tokenizer.vocab_size:,} &middot; "
            "the model is loaded once at start-up and is never retrained here.</p>"
        )

        inputs = [source, direction, strategy, beam_size, length_penalty, show_attention]
        outputs = [output, stats, tokens, attention_plot]

        translate_button.click(translate, inputs=inputs, outputs=outputs)
        source.submit(translate, inputs=inputs, outputs=outputs)
        swap_button.click(swap, inputs=[direction, source, output],
                          outputs=[direction, source, output])
        clear_button.add([source, output, stats, tokens, attention_plot])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--share", action="store_true",
        help="create a temporary public gradio.live link",
    )
    args = parser.parse_args()

    setup_logging()
    checkpoint = find_checkpoint(args.checkpoint)
    logger.info("Loading %s", checkpoint)

    translator = Translator.from_checkpoint(
        checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device,
        decode_config=DecodeConfig(strategy="beam", beam_size=4, length_penalty=0.6),
    )

    demo = build_interface(translator, checkpoint)

    launch_kwargs: dict = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
    }
    if _GRADIO_MAJOR >= 6:
        launch_kwargs["css"] = CSS
        launch_kwargs["theme"] = gr.themes.Soft(primary_hue="blue")

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
