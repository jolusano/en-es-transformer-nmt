"""Datasets, batching and mask construction.

The central idea: **one pair yields two training examples.**  A cleaned pair
``("I am tired.", "Estoy cansado.")`` becomes

    <2es> I am tired. </s>       ->  <s> Estoy cansado. </s>
    <2en> Estoy cansado. </s>    ->  <s> I am tired. </s>

so the corpus doubles and every parameter is trained on both directions.  This
is the whole reason a single set of weights can serve both ways.

Batching uses **token-count buckets** rather than a fixed number of sentences.
Tatoeba's length distribution is heavily right-skewed: a fixed sentence count
would produce batches whose padding waste swings between 5% and 60% depending
on whether a long sentence happened to be drawn.  Grouping similar lengths
together and capping each batch by *tokens* keeps memory per step roughly
constant, which is what lets the model train inside Colab's memory budget.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from nmt.constants import DIRECTION_TO_TAG, DIRECTIONS, PAD_ID
from nmt.data.tokenizer import BaseTokenizer
from nmt.utils.seed import worker_init_fn


@dataclass
class Example:
    """One directional training example after tokenisation."""

    source_ids: list[int]
    target_ids: list[int]
    direction: str
    source_text: str
    target_text: str

    def __len__(self) -> int:
        """Length used for bucketing: the longer of the two sides."""
        return max(len(self.source_ids), len(self.target_ids))


class TranslationDataset(Dataset):
    """Materialises both translation directions from a list of pairs.

    Parameters
    ----------
    pairs
        ``(english, spanish)`` tuples.
    tokenizer
        Any :class:`~nmt.data.tokenizer.BaseTokenizer`.
    directions
        Which directions to generate.  Training uses both; evaluation passes a
        single direction so BLEU can be reported per direction.
    max_length
        Sequences longer than this (after tokenisation, including the tag and
        the end-of-sentence marker) are truncated.  Truncation is preferred to
        dropping because it only affects a fraction of a percent of the corpus
        and keeps the two directions symmetric in count.
    """

    def __init__(
        self,
        pairs: Sequence[tuple[str, str]],
        tokenizer: BaseTokenizer,
        *,
        directions: Sequence[str] = DIRECTIONS,
        max_length: int = 128,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.directions = tuple(directions)
        self.examples: list[Example] = []

        for english, spanish in pairs:
            for direction in self.directions:
                source, target = (
                    (english, spanish) if direction == "en-es" else (spanish, english)
                )
                source_ids = tokenizer.encode_source(source, DIRECTION_TO_TAG[direction])
                target_ids = tokenizer.encode_target(target)

                if len(source_ids) > max_length:
                    # Keep the tag (position 0) and the final EOS.
                    source_ids = source_ids[: max_length - 1] + [tokenizer.eos_id]
                if len(target_ids) > max_length:
                    target_ids = target_ids[: max_length - 1] + [tokenizer.eos_id]

                self.examples.append(
                    Example(source_ids, target_ids, direction, source, target)
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]

    @property
    def lengths(self) -> list[int]:
        return [len(example) for example in self.examples]

    def token_count(self) -> dict[str, int]:
        """Total source/target tokens, used to size the learning-rate schedule."""
        return {
            "source_tokens": sum(len(e.source_ids) for e in self.examples),
            "target_tokens": sum(len(e.target_ids) for e in self.examples),
            "examples": len(self.examples),
        }


@dataclass
class Batch:
    """A padded, masked mini-batch ready for the model.

    Attributes
    ----------
    source
        ``(batch, source_len)`` encoder input ids.
    decoder_input
        ``(batch, target_len - 1)`` -- the target shifted right, i.e. beginning
        with ``<s>`` and stopping before the final token.
    labels
        ``(batch, target_len - 1)`` -- the target shifted left, the next-token
        supervision aligned with ``decoder_input``.
    source_padding_mask, target_padding_mask
        ``True`` at real (non-pad) positions.
    """

    source: torch.Tensor
    decoder_input: torch.Tensor
    labels: torch.Tensor
    source_padding_mask: torch.Tensor
    target_padding_mask: torch.Tensor
    directions: list[str]
    source_texts: list[str]
    target_texts: list[str]

    def to(self, device: torch.device) -> Batch:
        """Move every tensor to ``device``, leaving the metadata alone."""
        return Batch(
            source=self.source.to(device, non_blocking=True),
            decoder_input=self.decoder_input.to(device, non_blocking=True),
            labels=self.labels.to(device, non_blocking=True),
            source_padding_mask=self.source_padding_mask.to(device, non_blocking=True),
            target_padding_mask=self.target_padding_mask.to(device, non_blocking=True),
            directions=self.directions,
            source_texts=self.source_texts,
            target_texts=self.target_texts,
        )

    @property
    def num_target_tokens(self) -> int:
        """Non-pad label positions -- the denominator for per-token loss."""
        return int(self.target_padding_mask.sum().item())

    def __len__(self) -> int:
        return self.source.size(0)


def _pad(sequences: list[list[int]], pad_value: int = PAD_ID) -> torch.Tensor:
    """Right-pad a ragged list of id lists into a ``(batch, max_len)`` tensor."""
    width = max(len(sequence) for sequence in sequences)
    padded = torch.full((len(sequences), width), pad_value, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        padded[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return padded


def collate_batch(examples: Sequence[Example]) -> Batch:
    """Pad a list of examples and build the teacher-forcing tensors.

    The shift is done here rather than in the model so that the model's
    ``forward`` receives exactly what it is supposed to condition on, and the
    off-by-one that plagues seq2seq implementations is visible in one place:

    ``target_ids``    ``<s>  Estoy cansado . </s>``
    ``decoder_input`` ``<s>  Estoy cansado .``
    ``labels``        ``Estoy cansado . </s>``
    """
    source = _pad([example.source_ids for example in examples])
    target = _pad([example.target_ids for example in examples])

    # `.contiguous()` is load-bearing, not tidiness. Both of these are slices
    # of `target` and therefore share its storage with a non-zero offset. On
    # PyTorch's MPS backend, moving such a view to the device ignores the
    # offset: `decoder_input` arrives on the GPU holding the *labels*, so the
    # model is fed the answer and reports ~95% token accuracy at random
    # initialisation while BLEU stays at zero. Materialising each slice into
    # its own buffer here makes the transfer correct on every backend.
    # See tests/test_dataset.py::test_collate_slices_are_contiguous.
    decoder_input = target[:, :-1].contiguous()
    labels = target[:, 1:].contiguous()

    return Batch(
        source=source,
        decoder_input=decoder_input,
        labels=labels,
        source_padding_mask=source.ne(PAD_ID),
        target_padding_mask=labels.ne(PAD_ID),
        directions=[example.direction for example in examples],
        source_texts=[example.source_text for example in examples],
        target_texts=[example.target_text for example in examples],
    )


class TokenBucketSampler(Sampler[list[int]]):
    """Yield batches of similar-length examples capped by total token count.

    Algorithm
    ---------
    1. Shuffle all indices (so epochs differ).
    2. Take *pools* of ``bucket_multiplier x`` the nominal batch size and sort
       each pool by length.  Sorting locally rather than globally preserves
       randomness across epochs while still grouping similar lengths.
    3. Emit batches from each sorted pool, closing a batch when
       ``len(batch) x longest_in_batch`` would exceed ``max_tokens``.
    4. Shuffle the completed batch order, so the model does not systematically
       see short sentences first within an epoch (which biases early gradients).
    """

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        max_tokens: int = 8192,
        bucket_multiplier: int = 50,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        self.lengths = list(lengths)
        self.max_tokens = max_tokens
        self.bucket_multiplier = bucket_multiplier
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self._cached_length: int | None = None

    def set_epoch(self, epoch: int) -> None:
        """Re-seed the shuffle so each epoch sees a different batch layout."""
        self.epoch = epoch

    def _build_batches(self) -> list[list[int]]:
        indices = list(range(len(self.lengths)))
        rng = random.Random(self.seed + self.epoch)

        if self.shuffle:
            rng.shuffle(indices)
        else:
            indices.sort(key=lambda i: self.lengths[i])

        average_length = max(1, sum(self.lengths) // max(1, len(self.lengths)))
        pool_size = max(1, self.max_tokens // average_length) * self.bucket_multiplier

        batches: list[list[int]] = []
        for start in range(0, len(indices), pool_size):
            pool = sorted(
                indices[start : start + pool_size], key=lambda i: self.lengths[i]
            )
            batch: list[int] = []
            longest = 0
            for index in pool:
                candidate_longest = max(longest, self.lengths[index])
                if batch and (len(batch) + 1) * candidate_longest > self.max_tokens:
                    batches.append(batch)
                    batch, longest = [index], self.lengths[index]
                else:
                    batch.append(index)
                    longest = candidate_longest
            if batch:
                batches.append(batch)

        if self.drop_last and len(batches) > 1:
            batches.pop()
        if self.shuffle:
            rng.shuffle(batches)

        return batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._build_batches()
        self._cached_length = len(batches)
        yield from batches

    def __len__(self) -> int:
        if self._cached_length is None:
            self._cached_length = len(self._build_batches())
        return self._cached_length


def build_dataloader(
    dataset: TranslationDataset,
    *,
    max_tokens: int = 8192,
    batch_size: int | None = None,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = False,
) -> DataLoader:
    """Wrap ``dataset`` in a DataLoader.

    Pass ``batch_size`` to use plain fixed-size batches (simpler, used for
    evaluation where throughput matters less and reproducible ordering is
    convenient); leave it ``None`` to use token bucketing for training.
    """
    if batch_size is not None:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_batch,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            pin_memory=pin_memory,
        )

    sampler = TokenBucketSampler(
        dataset.lengths, max_tokens=max_tokens, shuffle=shuffle, seed=seed
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_batch,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=pin_memory,
    )


def padding_waste(dataset: TranslationDataset, *, max_tokens: int = 8192) -> dict:
    """Quantify how much of each batch is padding under the bucketing scheme.

    Reported in the paper to justify the sampler: the same figure computed with
    a fixed sentence count is several times worse.
    """
    sampler = TokenBucketSampler(dataset.lengths, max_tokens=max_tokens, shuffle=False)
    real = padded = 0
    batch_sizes: list[int] = []

    for batch in sampler:
        longest = max(dataset.lengths[i] for i in batch)
        real += sum(dataset.lengths[i] for i in batch)
        padded += longest * len(batch)
        batch_sizes.append(len(batch))

    return {
        "batches": len(batch_sizes),
        "mean_batch_size": sum(batch_sizes) / max(1, len(batch_sizes)),
        "min_batch_size": min(batch_sizes) if batch_sizes else 0,
        "max_batch_size": max(batch_sizes) if batch_sizes else 0,
        "real_tokens": real,
        "padded_tokens": padded,
        "padding_fraction": 1.0 - real / padded if padded else 0.0,
    }


def estimate_steps_per_epoch(
    dataset: TranslationDataset, *, max_tokens: int = 8192
) -> int:
    """Approximate optimiser steps per epoch, for sizing the LR warmup."""
    total = sum(dataset.lengths)
    return max(1, math.ceil(total / max_tokens))
