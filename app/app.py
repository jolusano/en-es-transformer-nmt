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
.gradio-container { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
#title h1 { margin-bottom: 0.1rem; }
#subtitle { color: #52514e; font-size: 0.92rem; margin-top: 0; }
.token-chip {
    display: inline-block; padding: 2px 7px; margin: 2px;
    border-radius: 5px; background: #dce9f9; border: 1px solid #2a78d6;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.78rem;
}
.token-chip.special { background: #fbe3d8; border-color: #eb6834; }
.stat { color: #52514e; font-size: 0.85rem; }
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
    return "<div>" + "".join(chips) + "</div>"


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
            "<p class='stat'><b>Source tokenisation</b></p>"
            + render_tokens(result.source_tokens)
            + "<p class='stat' style='margin-top:8px'><b>Generated tokens</b></p>"
            + render_tokens(result.output_tokens)
        )

        figure = None
        if show_attention and result.attention is not None and result.attention.numel():
            figure = _attention_figure(result)

        return result.translation, stats, tokens_html, figure

    def _attention_figure(result):
        """Render the alignment heatmap for the sentence just translated."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from nmt.viz.style import SEQUENTIAL, use_style

        use_style()
        attention = result.attention.numpy()
        source = [t.replace("▁", "") or "_" for t in result.source_tokens]
        target = [t.replace("▁", "") or "_" for t in result.output_tokens]
        attention = attention[: len(target), : len(source)]

        fig, ax = plt.subplots(
            figsize=(max(4.0, 0.42 * len(source) + 1.6),
                     max(2.6, 0.34 * len(target) + 1.4))
        )
        ax.imshow(attention, cmap=SEQUENTIAL, aspect="auto")
        ax.set_xticks(range(len(source)))
        ax.set_xticklabels(source, rotation=45, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(target)))
        ax.set_yticklabels(target, fontsize=7.5)
        ax.set_xlabel("source")
        ax.set_ylabel("generated")
        ax.set_title("Cross-attention alignment", fontsize=10)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
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
            "translating in **both** directions. "
            "AIG230 Final Project — Group 7: Jose Luis Sanchez Noriega & Bikash Subedi.",
            elem_id="subtitle",
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
                        value=False, label="Show attention alignment",
                        info="Re-runs the model to capture cross-attention.",
                    )

            with gr.Column(scale=1):
                # Kept to the kwargs that are stable across Gradio 4/5/6: the
                # copy-button argument was renamed between major versions and
                # is not worth pinning a version over.
                output = gr.Textbox(label="Translation", lines=4)
                stats = gr.HTML()
                with gr.Accordion("How the sentence was tokenised", open=False):
                    tokens = gr.HTML()
                attention_plot = gr.Plot(label="Cross-attention")

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
