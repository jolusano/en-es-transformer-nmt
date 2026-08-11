"""High-level translation interface used by the app, the notebooks and eval.

:class:`Translator` is the single place where "a checkpoint on disk" becomes
"a function from a string to a string".  Wrapping it once means the Gradio app,
the BLEU evaluation and the error analysis all exercise exactly the same code
path -- so a number in the report cannot disagree with what a user sees in the
demo.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nmt.config import ModelConfig
from nmt.constants import DIRECTION_TO_TAG, DIRECTIONS
from nmt.data.cleaning import normalise
from nmt.data.tokenizer import BaseTokenizer, load_tokenizer
from nmt.inference.search import DecodeConfig, decode
from nmt.model.baseline_lstm import AttentionSeq2Seq
from nmt.model.transformer import TranslationTransformer
from nmt.utils.devices import resolve_device
from nmt.utils.io import project_root
from nmt.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class TranslationResult:
    """One translation plus the diagnostics the app displays."""

    source: str
    translation: str
    direction: str
    source_tokens: list[str]
    output_tokens: list[str]
    num_source_tokens: int
    num_output_tokens: int
    seconds: float
    #: ``(target_len, source_len)`` averaged cross-attention, when requested.
    attention: Any | None = None


def _resolve_tokenizer(stored: str | None) -> Path:
    """Find the tokeniser a checkpoint was trained with.

    A checkpoint records an *absolute* path, which is correct on the machine
    that produced it and meaningless anywhere else -- a model trained in Colab
    points at ``/content/...``, so a checkpoint shared with anyone else would
    fail to load. When the recorded path is missing we look for a tokeniser of
    the same file name in this repository's ``artifacts/tokenizers/``, which is
    where the fitted tokenisers are committed.
    """
    local_dir = project_root() / "artifacts" / "tokenizers"

    if stored:
        stored_path = Path(stored)
        if stored_path.exists():
            return stored_path

        candidate = local_dir / stored_path.name
        if candidate.exists():
            logger.info(
                "Checkpoint records %s, which does not exist here; using %s",
                stored_path,
                candidate,
            )
            return candidate

    # Nothing recorded, or nothing matching: fall back to whatever is present.
    if local_dir.is_dir() and any(local_dir.iterdir()):
        logger.warning(
            "Could not resolve the checkpoint's tokeniser (%s); falling back to "
            "%s. Verify the output looks like real language -- a mismatched "
            "tokeniser produces fluent-looking nonsense rather than an error.",
            stored or "not recorded",
            local_dir,
        )
        return local_dir

    raise FileNotFoundError(
        f"cannot locate the tokeniser for this checkpoint (recorded: {stored!r}). "
        f"Pass tokenizer_path=..., or run `python -m nmt.data.build` to fit one."
    )


def build_model_from_config(config: ModelConfig) -> nn.Module:
    """Instantiate whichever architecture the config names."""
    if config.architecture == "transformer":
        return TranslationTransformer(config)
    if config.architecture == "lstm":
        return AttentionSeq2Seq(
            config.vocab_size,
            embedding_dim=config.effective_embedding_dim,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.dropout,
            tie_embeddings=config.tie_embeddings,
        )
    raise ValueError(f"unknown architecture {config.architecture!r}")


class Translator:
    """Loads a checkpoint and translates text in either direction."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: BaseTokenizer,
        *,
        device: torch.device | None = None,
        decode_config: DecodeConfig | None = None,
        max_length: int = 128,
    ) -> None:
        self.device = device or resolve_device("auto")
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.decode_config = decode_config or DecodeConfig()
        self.max_length = max_length

    # --- construction -------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        *,
        tokenizer_path: Path | str | None = None,
        device: str | torch.device = "auto",
        decode_config: DecodeConfig | None = None,
    ) -> Translator:
        """Load a trained model without needing to know how it was configured.

        The checkpoint carries its own ``model_config`` and the path of the
        tokeniser it was trained with, so a single file argument is enough.
        """
        checkpoint_path = Path(checkpoint_path)
        device = resolve_device(device) if isinstance(device, str) else device
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)

        config = ModelConfig(**payload["model_config"])
        model = build_model_from_config(config)
        model.load_state_dict(payload["model_state"])

        if tokenizer_path is None:
            tokenizer_path = _resolve_tokenizer(payload.get("tokenizer_path"))
        tokenizer = load_tokenizer(tokenizer_path)

        if tokenizer.vocab_size != config.vocab_size:
            raise ValueError(
                f"tokeniser has {tokenizer.vocab_size} entries but the model was "
                f"trained with {config.vocab_size}; these do not belong together"
            )

        logger.info(
            "Loaded %s (%s, epoch %s) on %s",
            checkpoint_path.name,
            config.architecture,
            payload.get("epoch", "?"),
            device,
        )
        return cls(model, tokenizer, device=device, decode_config=decode_config)

    # --- translation --------------------------------------------------------

    def _encode_source(self, text: str, direction: str) -> list[int]:
        if direction not in DIRECTION_TO_TAG:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
        ids = self.tokenizer.encode_source(normalise(text), DIRECTION_TO_TAG[direction])
        if len(ids) > self.max_length:
            ids = ids[: self.max_length - 1] + [self.tokenizer.eos_id]
        return ids

    def translate(
        self,
        text: str,
        direction: str,
        *,
        return_attention: bool = False,
    ) -> TranslationResult:
        """Translate one sentence.

        Parameters
        ----------
        direction
            ``"en-es"`` or ``"es-en"``.
        return_attention
            Also return the cross-attention map, averaged over heads and
            layers, for the alignment display in the app.
        """
        import time

        started = time.time()
        ids = self._encode_source(text, direction)
        source = torch.tensor([ids], dtype=torch.long, device=self.device)

        output = decode(
            self.model, source, config=self.decode_config, device=self.device
        )
        output_ids = output[0].tolist()
        translation = self.tokenizer.decode(output_ids)

        attention = None
        if return_attention and isinstance(self.model, TranslationTransformer):
            attention = self._cross_attention(source, output)

        return TranslationResult(
            source=text,
            translation=translation,
            direction=direction,
            source_tokens=[self.tokenizer.id_to_piece(i) for i in ids],
            output_tokens=[
                self.tokenizer.id_to_piece(i)
                for i in output_ids
                if i != self.tokenizer.pad_id
            ],
            num_source_tokens=len(ids),
            num_output_tokens=sum(1 for i in output_ids if i != self.tokenizer.pad_id),
            seconds=time.time() - started,
            attention=attention,
        )

    @torch.no_grad()
    def _cross_attention(
        self, source: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        """Re-run the model storing attention, and average the cross maps.

        Averaging over heads and layers is a simplification -- individual heads
        specialise, and the report shows them separately -- but a single map is
        what makes a readable alignment picture in the app.
        """
        self.model(source, output[:, :-1], store_attention=True)
        maps = self.model.attention_maps()["cross"]
        if not maps:
            return torch.empty(0)
        # Each map is (batch, heads, target_len, source_len).
        stacked = torch.stack([layer.mean(dim=1) for layer in maps])
        return stacked.mean(dim=0)[0].cpu()

    def translate_batch(
        self,
        texts: Sequence[str],
        direction: str,
        *,
        batch_size: int = 32,
        progress: bool = False,
    ) -> list[str]:
        """Translate many sentences, batched.

        Sentences are sorted by length before batching and restored to the
        original order afterwards. Batching similar lengths together cuts the
        padding waste that otherwise dominates decoding time, and it matters
        here because evaluation decodes 11,000 sentences per model.
        """
        encoded = [self._encode_source(text, direction) for text in texts]
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
        translations: dict[int, str] = {}

        iterator = range(0, len(order), batch_size)
        if progress:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc=f"decoding {direction}", unit="batch")

        for start in iterator:
            indices = order[start : start + batch_size]
            width = max(len(encoded[i]) for i in indices)
            source = torch.full(
                (len(indices), width),
                self.tokenizer.pad_id,
                dtype=torch.long,
                device=self.device,
            )
            for row, index in enumerate(indices):
                source[row, : len(encoded[index])] = torch.tensor(
                    encoded[index], device=self.device
                )

            output = decode(
                self.model, source, config=self.decode_config, device=self.device
            )
            for row, index in enumerate(indices):
                translations[index] = self.tokenizer.decode(output[row].tolist())

        return [translations[i] for i in range(len(texts))]

    def translate_pairs(
        self, pairs: Iterable[tuple[str, str]], direction: str, **kwargs: Any
    ) -> tuple[list[str], list[str], list[str]]:
        """Translate one side of a bitext, returning (sources, hypotheses, references)."""
        pairs = list(pairs)
        if direction == "en-es":
            sources = [pair[0] for pair in pairs]
            references = [pair[1] for pair in pairs]
        else:
            sources = [pair[1] for pair in pairs]
            references = [pair[0] for pair in pairs]

        hypotheses = self.translate_batch(sources, direction, **kwargs)
        return sources, hypotheses, references
