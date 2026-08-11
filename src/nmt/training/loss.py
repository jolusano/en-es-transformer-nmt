"""Label-smoothed cross entropy.

Plain cross entropy asks the model to put probability 1.0 on the reference
token and 0.0 on all 15,999 others.  For translation that target is simply
false: "I'm tired" is equally well rendered as "Estoy cansado" or "Tengo
sueño", and at the token level a dozen alternatives are often defensible.
Training towards a one-hot target therefore pushes the model to be confident
about a choice the data does not actually support, and the only way to drive
the loss down is to make the logit gap enormous -- which shows up as
overconfident, badly calibrated output and, empirically, worse BLEU.

Label smoothing (Szegedy et al., 2016; adopted for NMT by Vaswani et al., 2017)
replaces the target with

.. math::
    q(k) = \\begin{cases}
        1 - \\varepsilon + \\dfrac{\\varepsilon}{V'} & k = y \\\\[6pt]
        \\dfrac{\\varepsilon}{V'} & k \\neq y
    \\end{cases}

with :math:`\\varepsilon = 0.1` and :math:`V'` the number of candidate tokens.
The model is now asked for 0.9 on the reference and a small mass everywhere
else, a target it can actually reach, which caps the logit gap and keeps
gradients flowing.

Two implementation details that are easy to get wrong and both matter:

* ``<pad>`` positions must be excluded from the loss *and* from the
  normalising token count -- otherwise the reported loss depends on how much
  padding happened to be in the batch.
* ``<pad>`` must also be excluded from the smoothing *distribution*.  Spreading
  probability mass onto a token the model must never emit teaches it to emit
  that token.  This is why ``V'`` above is ``vocab_size - 1``, not
  ``vocab_size``.

Note that label smoothing deliberately raises the reported cross entropy: the
smoothed target has non-zero entropy, so the loss cannot reach zero even for a
perfect model.  The report plots the *unsmoothed* per-token cross entropy
alongside it so that perplexity remains interpretable.
"""

from __future__ import annotations

import torch
from torch import nn

from nmt.constants import PAD_ID


class LabelSmoothedCrossEntropy(nn.Module):
    """Cross entropy against a smoothed target distribution.

    Parameters
    ----------
    smoothing
        :math:`\\varepsilon`.  ``0.0`` reduces exactly to standard cross
        entropy, which the unit tests check.
    ignore_index
        Positions with this label contribute nothing.
    """

    def __init__(self, *, smoothing: float = 0.1, ignore_index: int = PAD_ID) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Parameters
        ----------
        logits
            ``(batch, seq_len, vocab_size)`` raw scores.
        targets
            ``(batch, seq_len)`` gold token ids.

        Returns
        -------
        (loss, stats)
            ``loss`` is the mean smoothed cross entropy over non-pad positions.
            ``stats`` additionally carries the *unsmoothed* negative
            log-likelihood, from which perplexity is computed, and the token
            count used as the denominator.
        """
        vocab_size = logits.size(-1)
        flat_logits = logits.reshape(-1, vocab_size)
        flat_targets = targets.reshape(-1)

        keep = flat_targets.ne(self.ignore_index)
        num_tokens = int(keep.sum().item())
        if num_tokens == 0:
            zero = logits.sum() * 0.0
            return zero, {"loss": 0.0, "nll": 0.0, "tokens": 0, "accuracy": 0.0}

        log_probs = torch.log_softmax(flat_logits.float(), dim=-1)

        # Negative log-likelihood of the reference token (no smoothing).
        nll = -log_probs.gather(1, flat_targets.clamp_min(0).unsqueeze(1)).squeeze(1)
        nll = nll[keep]

        # Mean log-probability over the vocabulary, excluding <pad>, which is
        # the smoothing term's contribution.
        smooth_sum = log_probs.sum(dim=-1) - log_probs[:, self.ignore_index]
        smooth = -smooth_sum[keep] / (vocab_size - 1)

        loss = (1.0 - self.smoothing) * nll + self.smoothing * smooth
        loss = loss.mean()

        with torch.no_grad():
            predictions = flat_logits.argmax(dim=-1)
            correct = (predictions == flat_targets)[keep].sum().item()

        return loss, {
            "loss": float(loss.item()),
            "nll": float(nll.mean().item()),
            "tokens": num_tokens,
            "accuracy": correct / num_tokens,
        }


def perplexity(nll: float) -> float:
    """Convert mean per-token negative log-likelihood to perplexity.

    Perplexity is the exponential of the cross entropy and reads as "the model
    is as uncertain as if it were choosing uniformly among this many tokens".
    Computed from the *unsmoothed* NLL, since the smoothed loss has an
    irreducible floor that would make the number incomparable across
    smoothing settings.
    """
    # Guard against overflow when the model is still at chance early in
    # training and the NLL is around log(16000) ~ 9.7.
    return float(torch.exp(torch.tensor(min(nll, 20.0))).item())
