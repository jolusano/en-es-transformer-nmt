"""Autoregressive decoding: greedy and beam search.

Training and inference are asymmetric in a way worth stating plainly, because
it explains most of what goes wrong with a translation model.  During training
the decoder is given the *gold* prefix at every position (teacher forcing), so
all positions can be computed in one parallel pass.  At inference there is no
gold prefix -- the model must consume its own previous outputs.  A single bad
token early on therefore conditions everything after it, a mismatch known as
*exposure bias*, and it is why a model with excellent teacher-forced loss can
still produce degenerate output.

Two decoders are implemented.

Greedy
    Take the highest-probability token at each step.  Fast, and the right
    default for the interactive application where latency is visible to the
    user.  It is myopic: a token that looks best locally may leave no good
    continuation, and greedy cannot revise it.

Beam search
    Keep the ``beam_size`` highest-scoring partial hypotheses at every step and
    extend all of them.  It explores a wider slice of the search space and
    typically buys 1-2 BLEU over greedy.  Two details matter:

    *Length normalisation.*  A hypothesis' score is a sum of log-probabilities,
    every one of which is negative, so longer hypotheses score worse purely for
    being longer.  Unnormalised beam search consequently produces
    systematically truncated translations.  Dividing by ``length^alpha`` with
    ``alpha ~ 0.6`` (Wu et al., 2016) corrects this.

    *Finished-hypothesis handling.*  A beam that has emitted ``</s>`` is
    complete and must be set aside rather than extended, otherwise it keeps
    accumulating probability mass and crowds the beam.

Both decoders run **batched** over sentences.  There is no key/value cache: the
decoder re-runs over the whole prefix at each step, which is O(n^2) work
instead of O(n).  That is a deliberate trade -- the cache would roughly double
the size of the decoder code and obscure the architecture the report is
explaining, and at these sentence lengths (median 8 subword tokens) the wall
clock cost is negligible.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from nmt.constants import BOS_ID, EOS_ID, PAD_ID


@dataclass
class DecodeConfig:
    """Decoding hyper-parameters."""

    strategy: str = "beam"
    beam_size: int = 4
    length_penalty: float = 0.6
    #: Stop after ``source_len * ratio + offset`` tokens.  Translation length
    #: is roughly proportional to source length, so a relative cap is both
    #: safer and tighter than a fixed one.
    max_length_ratio: float = 1.8
    max_length_offset: int = 10
    max_length_cap: int = 256
    #: Forbid repeating any n-gram of this size.  0 disables it.  Repetition
    #: loops are a classic failure of under-trained NMT models, and the error
    #: analysis quantifies how often this triggers.
    no_repeat_ngram_size: int = 0


def _max_length(source: torch.Tensor, config: DecodeConfig) -> int:
    """Length budget for a batch, derived from the longest source sentence."""
    source_length = int(source.ne(PAD_ID).sum(dim=1).max().item())
    return min(
        config.max_length_cap,
        int(source_length * config.max_length_ratio) + config.max_length_offset,
    )


def _banned_by_ngram(sequence: list[int], size: int) -> set[int]:
    """Tokens that would complete a repeated n-gram if emitted next."""
    if size <= 0 or len(sequence) < size:
        return set()
    prefix = tuple(sequence[-(size - 1):]) if size > 1 else ()
    banned = set()
    for start in range(len(sequence) - size + 1):
        window = tuple(sequence[start : start + size])
        if window[:-1] == prefix:
            banned.add(window[-1])
    return banned


@torch.no_grad()
def greedy_decode(
    model: nn.Module,
    source: torch.Tensor,
    *,
    config: DecodeConfig | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Decode a batch by always taking the arg-max token.

    Parameters
    ----------
    source
        ``(batch, source_len)`` ids including the direction tag.

    Returns
    -------
    torch.Tensor
        ``(batch, generated_len)`` ids beginning with ``<s>``, padded after
        each sequence's ``</s>``.
    """
    config = config or DecodeConfig()
    device = device or next(model.parameters()).device
    model.eval()

    source = source.to(device)
    batch = source.size(0)
    limit = _max_length(source, config)

    generated = torch.full((batch, 1), BOS_ID, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)

    for _ in range(limit):
        logits = model(source, generated)
        next_token = logits[:, -1].argmax(dim=-1)

        # Once a sequence has emitted </s>, keep it padded so its content
        # cannot change and its logits stop mattering.
        next_token = torch.where(
            finished, torch.full_like(next_token, PAD_ID), next_token
        )
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        finished |= next_token.eq(EOS_ID)
        if bool(finished.all()):
            break

    return generated


@torch.no_grad()
def beam_search_decode(
    model: nn.Module,
    source: torch.Tensor,
    *,
    config: DecodeConfig | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Decode a batch with length-normalised beam search.

    The batch and beam axes are flattened into one dimension of size
    ``batch * beam`` so that all hypotheses of all sentences are advanced in a
    single forward pass.

    Returns
    -------
    torch.Tensor
        ``(batch, length)`` -- the best hypothesis per sentence, right-padded.
    """
    config = config or DecodeConfig()
    device = device or next(model.parameters()).device
    model.eval()

    source = source.to(device)
    batch = source.size(0)
    beam = max(1, config.beam_size)
    limit = _max_length(source, config)

    # (batch, src) -> (batch * beam, src), grouped so rows [i*beam:(i+1)*beam]
    # all belong to sentence i.
    expanded_source = source.repeat_interleave(beam, dim=0)

    tokens = torch.full((batch * beam, 1), BOS_ID, dtype=torch.long, device=device)

    # Only the first beam of each sentence starts alive; the others are given
    # -inf so that step 1 expands a single hypothesis rather than `beam`
    # identical copies of it.
    scores = torch.full((batch, beam), float("-inf"), device=device)
    scores[:, 0] = 0.0
    scores = scores.view(-1)

    finished_sequences: list[list[tuple[float, list[int]]]] = [[] for _ in range(batch)]
    alive = torch.ones(batch * beam, dtype=torch.bool, device=device)

    for _step in range(limit):
        logits = model(expanded_source, tokens)
        log_probs = torch.log_softmax(logits[:, -1].float(), dim=-1)
        vocab_size = log_probs.size(-1)

        # Never generate <pad> or a second <s>.
        log_probs[:, PAD_ID] = float("-inf")
        log_probs[:, BOS_ID] = float("-inf")

        if config.no_repeat_ngram_size > 0:
            for row in range(tokens.size(0)):
                for banned in _banned_by_ngram(
                    tokens[row].tolist(), config.no_repeat_ngram_size
                ):
                    log_probs[row, banned] = float("-inf")

        # Dead beams must not contribute candidates.
        log_probs = log_probs.masked_fill(~alive.unsqueeze(1), float("-inf"))

        candidate_scores = (scores.unsqueeze(1) + log_probs).view(batch, beam * vocab_size)
        top_scores, top_indices = candidate_scores.topk(beam, dim=-1)

        beam_index = top_indices // vocab_size      # which hypothesis it extends
        token_index = top_indices % vocab_size      # which token it appends

        # Reorder the running sequences to follow the selected parents.
        flat_parent = (
            beam_index + torch.arange(batch, device=device).unsqueeze(1) * beam
        ).view(-1)
        tokens = torch.cat(
            [tokens[flat_parent], token_index.view(-1, 1)], dim=1
        )
        scores = top_scores.view(-1)
        alive = alive[flat_parent]

        # Retire any hypothesis that just emitted </s>.
        just_finished = token_index.view(-1).eq(EOS_ID) & alive
        for row in torch.nonzero(just_finished, as_tuple=False).flatten().tolist():
            sentence = row // beam
            length = tokens.size(1) - 1  # exclude the leading <s>
            normalised = float(scores[row].item()) / (length**config.length_penalty)
            finished_sequences[sentence].append((normalised, tokens[row].tolist()))

        alive &= ~just_finished
        scores = scores.masked_fill(~alive, float("-inf"))

        # Stop when every sentence has collected `beam` complete hypotheses.
        if all(len(candidates) >= beam for candidates in finished_sequences):
            break
        if not bool(alive.any()):
            break

    # Fall back to the best incomplete hypothesis if a sentence never finished
    # (it hit the length cap), so every input always gets an output.
    results: list[list[int]] = []
    for sentence in range(batch):
        candidates = finished_sequences[sentence]
        if not candidates:
            rows = slice(sentence * beam, (sentence + 1) * beam)
            best_row = int(scores[rows].argmax().item()) + sentence * beam
            length = max(1, tokens.size(1) - 1)
            candidates = [
                (float(scores[best_row].item()) / (length**config.length_penalty),
                 tokens[best_row].tolist())
            ]
        results.append(max(candidates, key=lambda item: item[0])[1])

    width = max(len(sequence) for sequence in results)
    output = torch.full((batch, width), PAD_ID, dtype=torch.long, device=device)
    for row, sequence in enumerate(results):
        output[row, : len(sequence)] = torch.tensor(sequence, device=device)
    return output


def decode(
    model: nn.Module,
    source: torch.Tensor,
    *,
    config: DecodeConfig | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Dispatch to the decoder named in ``config.strategy``."""
    config = config or DecodeConfig()
    if config.strategy == "greedy" or config.beam_size <= 1:
        return greedy_decode(model, source, config=config, device=device)
    if config.strategy == "beam":
        return beam_search_decode(model, source, config=config, device=device)
    raise ValueError(f"unknown decoding strategy {config.strategy!r}")
