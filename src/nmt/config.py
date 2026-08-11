"""Typed configuration objects, loadable from the YAML files in ``configs/``.

Every experiment in the report is fully described by one YAML file.  Keeping
configuration in data rather than in code means the exact settings behind a
number can be recovered from the checkpoint, and that reproducing a run is a
matter of pointing the trainer at the same file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from nmt.utils.io import project_root


def _filter_known(cls: type, payload: dict[str, Any]) -> dict[str, Any]:
    """Drop keys the dataclass does not declare, warning about none of them.

    YAML files carry documentation keys (``description``, ``notes``) that are
    for humans; silently ignoring them keeps the files readable.
    """
    known = {f.name for f in fields(cls)}
    return {key: value for key, value in payload.items() if key in known}


@dataclass
class ModelConfig:
    """Architecture hyper-parameters.

    Defaults are the operating point used for the main experiment.  The
    reasoning behind each is set out in the report; in brief:

    ``d_model=512``, ``num_heads=8``, ``d_ff=2048``
        The proportions of "Transformer base" (Vaswani et al., 2017), kept so
        the design sits on a well-understood point of the design space.
        ``d_ff = 4 * d_model`` and ``d_k = d_model / heads = 64`` are both the
        standard ratios.

    ``num_encoder_layers=4``, ``num_decoder_layers=4``
        *Reduced* from the paper's 6+6.  The paper trained on WMT14
        English-German, roughly 4.5M sentence pairs; after cleaning, this
        corpus has about two orders of magnitude less text per direction.
        Depth is the first thing to cut when data is the binding constraint:
        6+6 overfits here, and 4+4 halves both the step time and the memory,
        which is what makes the run fit in a free Colab session.

    ``dropout=0.1``
        The paper's value.  Raised to 0.2-0.3 in the small-data ablation.

    ``norm_first=True``
        Pre-layer-normalisation; see :mod:`nmt.model.layers` for why.
    """

    #: ``"transformer"`` or ``"lstm"``.  The recurrent option builds the
    #: Bahdanau-attention baseline in :mod:`nmt.model.baseline_lstm`, which
    #: shares this loop, this data and this vocabulary so the comparison in the
    #: evaluation section isolates the architecture.
    architecture: str = "transformer"

    vocab_size: int = 16_000
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 2_048
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation: str = "relu"
    positional_encoding: str = "sinusoidal"
    max_position: int = 512
    norm_first: bool = True
    tie_embeddings: bool = True
    #: Embedding width.  ``None`` means "same as ``d_model``".  The pre-trained
    #: experiments set this to 300 to match the MUSE vectors, and the model
    #: inserts learned projections between the 300-d embedding space and the
    #: 512-d residual stream so that the rest of the architecture is
    #: byte-for-byte identical across experiments.
    embedding_dim: int | None = None
    #: Path to a ``.npy`` embedding matrix used to initialise the table.
    pretrained_embeddings: str | None = None
    #: Keep pre-trained vectors fixed for this many epochs before unfreezing.
    freeze_embeddings_epochs: int = 0

    # --- recurrent baseline only -------------------------------------------
    #: Hidden width of the LSTM baseline.  Tuned so the baseline lands within a
    #: few percent of the transformer's parameter count, so that any BLEU gap
    #: cannot be explained away by capacity.
    lstm_hidden_size: int = 512
    lstm_num_layers: int = 2

    @property
    def effective_embedding_dim(self) -> int:
        return self.embedding_dim or self.d_model

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataConfig:
    """Where the corpus lives and how it is batched."""

    processed_dir: str = "data/processed"
    tokenizer: str = "artifacts/tokenizers/joint_bpe.model"
    max_length: int = 128
    #: Batches are capped by token count rather than sentence count; see
    #: :class:`nmt.data.dataset.TokenBucketSampler`.
    max_tokens_per_batch: int = 8_192
    eval_batch_size: int = 64
    num_workers: int = 2
    #: Cap on training pairs, for quick smoke tests.  ``None`` uses all of them.
    limit_train_pairs: int | None = None


@dataclass
class OptimConfig:
    """Optimiser, schedule and regularisation.

    ``label_smoothing=0.1``
        Replaces the one-hot target with a distribution that keeps 0.9 on the
        correct token and spreads 0.1 over the rest.  Translation is a
        one-to-many problem -- several outputs are equally correct -- so a loss
        that demands probability 1.0 on one particular token is asking for
        something false.  It costs perplexity and gains BLEU, which is the
        trade the paper reports and this project reproduces.

    ``warmup_steps=4000``
        The learning rate rises linearly for this many steps then decays with
        the inverse square root of the step count.  Early in training the
        attention distributions are near-uniform and the gradient direction is
        mostly noise; taking large steps then is what produces the divergence
        that transformers are notorious for.  Pre-LN tolerates a shorter warmup
        than post-LN, but some is still worth having.

    ``betas=(0.9, 0.98)``, ``eps=1e-9``
        The paper's Adam settings.  The higher second-moment beta (0.98 rather
        than the usual 0.999) makes the optimiser adapt faster to the changing
        gradient scale of the warmup phase.
    """

    optimizer: str = "adamw"
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1.0e-9
    scheduler: str = "inverse_sqrt"
    warmup_steps: int = 4_000
    label_smoothing: float = 0.1
    max_grad_norm: float = 1.0
    #: Accumulate gradients over this many batches before stepping, to
    #: simulate a larger batch than the GPU can hold.
    gradient_accumulation_steps: int = 1


@dataclass
class TrainConfig:
    """Run length, checkpointing and early stopping."""

    epochs: int = 20
    seed: int = 42
    device: str = "auto"
    amp: bool = True
    log_every: int = 100
    #: Run validation this many times per epoch (1 = only at the end).
    evals_per_epoch: int = 1
    #: Stop when validation loss has not improved for this many evaluations.
    early_stopping_patience: int = 5
    keep_last_checkpoints: int = 2
    #: Compute BLEU on a sample of the validation set during training, so the
    #: curve the report plots is a translation-quality curve, not only a loss.
    validation_bleu_samples: int = 500
    checkpoint_dir: str = "artifacts/checkpoints"
    results_dir: str = "artifacts/results"


@dataclass
class ExperimentConfig:
    """A complete, self-describing experiment."""

    name: str = "baseline"
    description: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # --- serialisation ------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path | str) -> ExperimentConfig:
        """Load an experiment description from a YAML file."""
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentConfig:
        return cls(
            name=payload.get("name", "baseline"),
            description=payload.get("description", ""),
            model=ModelConfig(**_filter_known(ModelConfig, payload.get("model", {}))),
            data=DataConfig(**_filter_known(DataConfig, payload.get("data", {}))),
            optim=OptimConfig(**_filter_known(OptimConfig, payload.get("optim", {}))),
            train=TrainConfig(**_filter_known(TrainConfig, payload.get("train", {}))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "model": asdict(self.model),
            "data": asdict(self.data),
            "optim": asdict(self.optim),
            "train": asdict(self.train),
        }

    def to_yaml(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return path

    # --- resolved paths -----------------------------------------------------

    def resolve(self, root: Path | None = None) -> ExperimentConfig:
        """Turn the relative paths in the YAML into absolute ones."""
        root = root or project_root()
        self.data.processed_dir = str(root / self.data.processed_dir)
        self.data.tokenizer = str(root / self.data.tokenizer)
        self.train.checkpoint_dir = str(root / self.train.checkpoint_dir / self.name)
        self.train.results_dir = str(root / self.train.results_dir / self.name)
        if self.model.pretrained_embeddings:
            self.model.pretrained_embeddings = str(
                root / self.model.pretrained_embeddings
            )
        return self
