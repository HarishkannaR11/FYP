"""
Prediction heads for the MultiAttentionTransformer.

The encoder in multi_attention_transformer.py deliberately contains no
classification head: it returns a whole-brain embedding. This module adds the
two heads needed to turn that embedding into a stage prediction, plus
MC-dropout uncertainty.

    MulticlassHead   plain 6-way softmax. The baseline head: it treats the six
                     stages as unordered, so confusing CN with AD costs exactly
                     as much as confusing LMCI with AD.

    OrdinalHead      cumulative-link model over the ordered stages, which is
                     what the project proposes:

                         P(y <= k | H) = sigmoid(tau_k - g(H)),
                         tau_0 < tau_1 < ... < tau_4

                     One scalar severity score g(H) and five ordered cut
                     points. Monotonicity of the thresholds is enforced by
                     construction -- tau_k = tau_0 + sum of softplus
                     increments -- rather than hoped for, so the cumulative
                     probabilities can never cross.

Having both is deliberate. The ordinal formulation is a claim about the
problem, and a claim has to be testable: with both heads available, the
ordinal head can be compared against the unordered one on identical splits
instead of being assumed to help.

Uncertainty
-----------
`predict_mc` keeps dropout active at inference and draws S samples. The mean
of the sampled probabilities is the prediction; the entropy of that mean is
the predictive uncertainty. This is MC-dropout, a cheap approximation to a
Bayesian posterior -- it is NOT a full Bayesian treatment and is labelled as
such wherever it is reported.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_STAGES = 6


class MulticlassHead(nn.Module):
    """Unordered 6-way classification."""

    kind = "multiclass"

    def __init__(self, d_model: int, n_stages: int = N_STAGES, dropout: float = 0.3):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_stages)
        self.n_stages = n_stages

    def forward(self, h):
        return self.fc(self.drop(h))                      # [B, n_stages]

    def loss(self, out, y, weight=None):
        return F.cross_entropy(out, y, weight=weight)

    def probabilities(self, out):
        return F.softmax(out, dim=-1)

    def decode(self, out):
        """Point prediction. For an unordered softmax, argmax is correct."""
        return out.argmax(dim=-1)


class OrdinalHead(nn.Module):
    """Cumulative-link ordinal model with monotone thresholds."""

    kind = "ordinal"

    def __init__(self, d_model: int, n_stages: int = N_STAGES, dropout: float = 0.3,
                 mean_reduce: bool = False, aux_rank: int = 0,
                 aux_weight: float = 0.0):
        """
        mean_reduce  average the n_stages-1 BCE terms instead of summing them,
                     so the loss has the same scale as cross-entropy (fix for
                     the measured 1.64x scale difference).
        aux_rank     width of an auxiliary projection trained alongside the
                     ordinal score. The ordinal score alone sends a RANK-1
                     gradient to the encoder (measured: effective rank 1.00,
                     top singular value 100%); an auxiliary head restores a
                     higher-rank teaching signal. It is used for TRAINING ONLY
                     -- inference still decodes the cumulative-link model, so
                     the ordinal formulation is preserved.
        aux_weight   weight on the auxiliary cross-entropy term.
        """
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.score = nn.Linear(d_model, 1)                # g(H), the severity score
        self.tau0 = nn.Parameter(torch.zeros(1))
        # increments are softplus-ed, so tau is strictly increasing by construction
        self.raw_steps = nn.Parameter(torch.zeros(n_stages - 2))
        self.n_stages = n_stages
        self.mean_reduce = mean_reduce
        self.aux_weight = aux_weight
        self.aux = nn.Linear(d_model, aux_rank) if aux_rank else None
        self._last_aux = None

    @torch.no_grad()
    def init_from_prior(self, class_counts):
        """Place the thresholds so the initial prediction matches the training
        class distribution. By default the head starts at P(CN)=0.53 against a
        uniform 0.167, so the first gradients are spent undoing the
        initialisation rather than learning."""
        c = torch.as_tensor(class_counts, dtype=torch.float)
        p = c / c.sum()
        cum = torch.cumsum(p, 0)[:-1].clamp(1e-4, 1 - 1e-4)
        tau = torch.log(cum / (1 - cum))                  # logit of the CDF
        self.tau0.copy_(tau[:1])
        steps = (tau[1:] - tau[:-1]).clamp_min(1e-3)
        self.raw_steps.copy_(torch.log(torch.expm1(steps).clamp_min(1e-6)))
        return self

    def thresholds(self):
        steps = F.softplus(self.raw_steps) + 1e-4
        return torch.cat([self.tau0, self.tau0 + torch.cumsum(steps, 0)])

    def forward(self, h):
        hd = self.drop(h)
        g = self.score(hd).squeeze(-1)                    # [B]
        self._last_aux = self.aux(hd) if self.aux is not None else None
        return self.thresholds().unsqueeze(0) - g.unsqueeze(1)   # [B, n_stages-1]

    def loss(self, out, y, weight=None):
        """Binary cross-entropy on each cumulative indicator 1[y <= k]."""
        k = torch.arange(self.n_stages - 1, device=y.device).unsqueeze(0)
        target = (y.unsqueeze(1) <= k).float()
        per = F.binary_cross_entropy_with_logits(out, target, reduction="none")
        per = per.mean(dim=1) if self.mean_reduce else per.sum(dim=1)
        if weight is not None:
            w = weight[y]
            base = (per * w).sum() / w.sum()
        else:
            base = per.mean()
        if self.aux is not None and self.aux_weight > 0 and self._last_aux is not None:
            base = base + self.aux_weight * F.cross_entropy(
                self._last_aux, y, weight=weight)
        return base

    def decode(self, out):
        """Point prediction for a cumulative-link model: the number of
        thresholds the severity score exceeds, i.e. sum_k 1[P(y<=k) < 0.5].

        argmax over the class probabilities is NOT valid here. For a
        cumulative-link model the interior classes occupy narrow probability
        windows and can never win an argmax: sweeping the score across its
        whole range, classes 1..K-2 win argmax on 0.0% of it, so only the two
        extreme stages are reachable. That is what made the head appear to
        'predict AD for everything'."""
        return (torch.sigmoid(out) < 0.5).sum(dim=-1)

    def probabilities(self, out):
        cum = torch.sigmoid(out)                          # P(y <= k), [B, K-1]
        ones = torch.ones_like(cum[:, :1])
        zeros = torch.zeros_like(cum[:, :1])
        cum = torch.cat([zeros, cum, ones], dim=1)        # P(y <= -1)=0, P(y<=K-1)=1
        p = cum[:, 1:] - cum[:, :-1]
        return p.clamp_min(0)                             # guard float error only


class StagePredictor(nn.Module):
    """Encoder + head. The encoder is used exactly as tested; nothing in
    multi_attention_transformer.py is modified."""

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, regional, fc, local_mask):
        h = self.encoder(regional, fc, local_mask)["brain_embedding"]
        return self.head(h)

    @torch.no_grad()
    def predict_mc(self, regional, fc, local_mask, n_samples: int = 30):
        """MC-dropout. Returns (mean probabilities, predictive entropy).

        Dropout is forced on; everything else stays in eval mode."""
        was_training = self.training
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        acc = None
        for _ in range(n_samples):
            p = self.head.probabilities(self(regional, fc, local_mask))
            acc = p if acc is None else acc + p
        self.train(was_training)
        p = acc / n_samples
        p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
        entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)
        return p, entropy


def build_head(kind: str, d_model: int, dropout: float = 0.3, **kw):
    if kind == "multiclass":
        return MulticlassHead(d_model, dropout=dropout)
    if kind == "ordinal":
        return OrdinalHead(d_model, dropout=dropout, **kw)
    if kind == "ordinal_fixed":
        # all three evidence-driven fixes together
        return OrdinalHead(d_model, dropout=dropout, mean_reduce=True,
                           aux_rank=kw.pop("aux_rank", N_STAGES),
                           aux_weight=kw.pop("aux_weight", 0.5), **kw)
    raise ValueError(f"unknown head: {kind!r}")
