"""
Cumulative-link (proportional-odds) ordinal head for the ordinal transformer experiment (v1).

Stage order (the repository's encoder, verified by test_ordinal.py): CN=0 < SMC=1 < EMCI=2 < MCI=3 < LMCI=4 < AD=5.

The parameterisation is heads.OrdinalHead's, reused unchanged:
  * one severity score g(H) = Linear(d_model, 1)
  * thresholds tau_0 < tau_1 < ... < tau_4, strictly increasing by construction (tau_0 plus softplus increments)
  * P(y <= k | H) = sigmoid(tau_k - g(H)); class probabilities are the CDF differences
    (heads.OrdinalHead.probabilities), so they are non-negative and sum to 1.

What changes is the loss. Here it is the exact negative log-likelihood of the cumulative-link model,
-log P(y = k), weighted per sample exactly like the reference head's weighted cross-entropy. The reference loss is
the NLL of a softmax over six unordered classes; this is the NLL of an ordered model, so the output structure is the
only difference. The auxiliary multiclass term of heads 'ordinal_fixed' is NOT used.

Numerically stable log P(y = k). With a = tau_{k-1} - g and b = tau_k - g (b > a):
    sigmoid(b) - sigmoid(a) = sigmoid(b) * sigmoid(-a) * (1 - exp(a - b))
    log P(y = k) = logsigmoid(b) + logsigmoid(-a) + log(-expm1(a - b))
tau_{-1} = -inf and tau_{K-1} = +inf are represented by -/+BIG, which makes their terms exactly 0.

Initialisation: the thresholds are set so that at g = 0 every stage has probability 1/6, matching the reference
head's zero-bias softmax start. No label or label count is used.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "model"))
from heads import OrdinalHead  # noqa: E402

BIG = 1e4


class CumulativeLinkHead(OrdinalHead):
    kind = "ordinal_cumlink"

    def __init__(self, d_model: int, n_stages: int = 6, dropout: float = 0.3):
        super().__init__(d_model, n_stages=n_stages, dropout=dropout)
        self.init_uniform()

    @torch.no_grad()
    def init_uniform(self):
        """Thresholds at logit(k/K), k = 1..K-1: uniform class probabilities when g = 0. Uses no labels."""
        return self.init_from_prior(torch.ones(self.n_stages))

    def log_probs(self, out):
        """out = tau - g, shape [B, K-1]  ->  log P(y = k), shape [B, K]."""
        B = out.shape[0]
        lo = torch.cat([out.new_full((B, 1), -BIG), out], dim=1)     # a_k = tau_{k-1} - g
        hi = torch.cat([out, out.new_full((B, 1), BIG)], dim=1)      # b_k = tau_k - g
        return F.logsigmoid(hi) + F.logsigmoid(-lo) + torch.log(-torch.expm1(lo - hi))

    def loss(self, out, y, weight=None):
        """Exact cumulative-link NLL; with class weights it is normalised like F.cross_entropy(weight=...)."""
        nll = -self.log_probs(out).gather(1, y.unsqueeze(1)).squeeze(1)
        if weight is None:
            return nll.mean()
        w = weight[y]
        return (nll * w).sum() / w.sum()

    def decode_argmax(self, out):
        """Primary decision rule (same as the reference head): most probable stage."""
        return self.probabilities(out).argmax(dim=-1)

    def decode_median(self, out):
        """Secondary rule: the median stage (number of thresholds exceeded); minimises expected absolute error."""
        return self.decode(out)


def median_decode(p):
    """Median stage of (bagged) class probabilities p [N, K]: the first k with cumulative probability >= 0.5."""
    cum = torch.as_tensor(p).cumsum(dim=-1)
    return (cum < 0.5).sum(dim=-1).clamp_max(cum.shape[-1] - 1)
