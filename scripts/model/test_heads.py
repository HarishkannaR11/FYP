"""
Tests for the prediction heads.

The point of the synthetic-recovery tests is diagnostic. When the ordinal head
collapsed to a single class on the real cohort, there were two candidate
explanations: the head is implemented wrongly, or 81 training subjects carry
too little signal. These tests settle the first question. If the head recovers
a known ordering from synthetic data where the ordering is real, then a
collapse on the real data is a statement about the data, not about the code.
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heads import MulticlassHead, OrdinalHead, StagePredictor, build_head

N_STAGES = 6
torch.set_num_threads(1)


# --------------------------------------------------------------- shape / math
def test_multiclass_shapes():
    h = MulticlassHead(16)
    out = h(torch.randn(5, 16))
    assert out.shape == (5, N_STAGES)
    p = h.probabilities(out)
    assert torch.allclose(p.sum(1), torch.ones(5), atol=1e-6)


def test_ordinal_shapes():
    h = OrdinalHead(16)
    out = h(torch.randn(5, 16))
    assert out.shape == (5, N_STAGES - 1)
    p = h.probabilities(out)
    assert p.shape == (5, N_STAGES)
    assert torch.allclose(p.sum(1), torch.ones(5), atol=1e-5)
    assert (p >= 0).all()


def test_ordinal_thresholds_are_strictly_increasing():
    h = OrdinalHead(16)
    with torch.no_grad():
        h.raw_steps.copy_(torch.randn(N_STAGES - 2) * 3)
    t = h.thresholds()
    assert t.shape == (N_STAGES - 1,)
    assert (t[1:] > t[:-1]).all(), "cumulative probabilities could cross"


def test_ordinal_cumulative_probabilities_are_monotone():
    """P(y<=k) must be non-decreasing in k for any input."""
    h = OrdinalHead(16)
    with torch.no_grad():
        h.raw_steps.copy_(torch.randn(N_STAGES - 2))
    cum = torch.sigmoid(h(torch.randn(32, 16)))
    assert (cum[:, 1:] - cum[:, :-1] >= -1e-6).all()


def test_ordinal_score_orders_predictions():
    """A higher severity score must push probability mass to later stages."""
    h = OrdinalHead(4)
    with torch.no_grad():
        h.score.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        h.score.bias.zero_()
    h.eval()
    lo = h.probabilities(h(torch.tensor([[-3.0, 0, 0, 0]])))
    hi = h.probabilities(h(torch.tensor([[3.0, 0, 0, 0]])))
    assert (lo @ torch.arange(N_STAGES, dtype=torch.float)
            < hi @ torch.arange(N_STAGES, dtype=torch.float)), \
        "a larger score should mean a later predicted stage"


def test_losses_are_finite_and_positive():
    y = torch.randint(0, N_STAGES, (8,))
    for kind in ("multiclass", "ordinal"):
        h = build_head(kind, 16)
        out = h(torch.randn(8, 16))
        loss = h.loss(out, y)
        assert torch.isfinite(loss) and loss > 0


def test_class_weighting_changes_the_loss():
    y = torch.tensor([0, 0, 0, 0, 5, 5])
    w = torch.tensor([0.5, 1.0, 1.0, 1.0, 1.0, 4.0])
    for kind in ("multiclass", "ordinal"):
        h = build_head(kind, 16)
        out = h(torch.randn(6, 16))
        assert not torch.isclose(h.loss(out, y), h.loss(out, y, weight=w))


def test_gradients_reach_every_parameter():
    y = torch.randint(0, N_STAGES, (8,))
    for kind in ("multiclass", "ordinal"):
        h = build_head(kind, 16)
        h.loss(h(torch.randn(8, 16)), y).backward()
        for n, p in h.named_parameters():
            assert p.grad is not None, f"{kind}.{n} got no gradient"
            assert torch.isfinite(p.grad).all(), f"{kind}.{n} has non-finite grad"


# ------------------------------------------------------- synthetic recovery
def _synthetic(n=600, d=16, noise=0.35, seed=0):
    """Embeddings carrying a genuine monotone severity signal along one axis."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, N_STAGES, (n,), generator=g)
    x = torch.randn(n, d, generator=g) * noise
    x[:, 0] += y.float()                       # the ordering lives on axis 0
    return x, y


@pytest.mark.parametrize("kind", ["multiclass", "ordinal"])
def test_head_recovers_a_real_signal(kind):
    """Both heads must learn an ordering that genuinely exists in the data.

    This is the control for the real-cohort result: the head is given a clean
    monotone signal and must find it."""
    torch.manual_seed(0)
    x, y = _synthetic()
    xtr, ytr, xte, yte = x[:450], y[:450], x[450:], y[450:]
    h = build_head(kind, 16, dropout=0.0)
    opt = torch.optim.AdamW(h.parameters(), lr=5e-2)
    for _ in range(400):
        opt.zero_grad()
        h.loss(h(xtr), ytr).backward()
        opt.step()
    h.eval()
    with torch.no_grad():
        pred = h.probabilities(h(xte)).argmax(1)
    acc = (pred == yte).float().mean().item()
    mae = (pred - yte).abs().float().mean().item()
    assert acc > 0.60, f"{kind}: accuracy {acc:.3f} on a clean signal"
    assert mae < 0.60, f"{kind}: MAE {mae:.3f} stages on a clean signal"


def test_ordinal_respects_distance_better_than_multiclass():
    """On an ordered problem with noise, the ordinal head should make smaller
    distance errors than the unordered head. This is the claim that justifies
    using it at all, so it is tested rather than assumed."""
    torch.manual_seed(0)
    x, y = _synthetic(n=900, noise=1.2, seed=1)
    xtr, ytr, xte, yte = x[:650], y[:650], x[650:], y[650:]
    mae = {}
    for kind in ("multiclass", "ordinal"):
        torch.manual_seed(0)
        h = build_head(kind, 16, dropout=0.0)
        opt = torch.optim.AdamW(h.parameters(), lr=5e-2)
        for _ in range(500):
            opt.zero_grad()
            h.loss(h(xtr), ytr).backward()
            opt.step()
        h.eval()
        with torch.no_grad():
            pred = h.probabilities(h(xte)).argmax(1)
        mae[kind] = (pred - yte).abs().float().mean().item()
    assert mae["ordinal"] <= mae["multiclass"] + 1e-6, \
        f"ordinal MAE {mae['ordinal']:.3f} vs multiclass {mae['multiclass']:.3f}"


def test_degenerate_input_collapses_to_one_class():
    """Control in the other direction: with NO signal, a head predicting a
    single class is the correct behaviour, not a bug. This is what the real
    cohort produced, so the test documents that the two are indistinguishable
    from the head's point of view."""
    torch.manual_seed(0)
    n, d = 300, 16
    x = torch.randn(n, d)
    y = torch.randint(0, N_STAGES, (n,))       # labels unrelated to x
    h = build_head("ordinal", d, dropout=0.0)
    opt = torch.optim.AdamW(h.parameters(), lr=5e-2)
    for _ in range(300):
        opt.zero_grad()
        h.loss(h(x), y).backward()
        opt.step()
    h.eval()
    with torch.no_grad():
        pred = h.probabilities(h(x)).argmax(1)
    assert len(torch.unique(pred)) <= 2, \
        "with no signal the head should concentrate on very few classes"


# ------------------------------------------------------------ MC-dropout
def test_mc_dropout_returns_valid_distribution_and_entropy():
    class Enc(torch.nn.Module):
        def forward(self, regional, fc, local_mask):
            return {"brain_embedding": regional.mean(1)}

    m = StagePredictor(Enc(), build_head("multiclass", 3, dropout=0.5))
    reg = torch.randn(7, 246, 3)
    fc = torch.zeros(7, 246, 246)
    mask = torch.ones(246, 246, dtype=torch.bool)
    p, ent = m.predict_mc(reg, fc, mask, n_samples=20)
    assert p.shape == (7, N_STAGES)
    assert torch.allclose(p.sum(1), torch.ones(7), atol=1e-5)
    assert (ent >= 0).all() and (ent <= np.log(N_STAGES) + 1e-6).all()


def test_mc_dropout_leaves_module_mode_unchanged():
    class Enc(torch.nn.Module):
        def forward(self, regional, fc, local_mask):
            return {"brain_embedding": regional.mean(1)}

    m = StagePredictor(Enc(), build_head("multiclass", 3, dropout=0.5))
    m.train()
    m.predict_mc(torch.randn(2, 246, 3), torch.zeros(2, 246, 246),
                 torch.ones(246, 246, dtype=torch.bool), n_samples=3)
    assert m.training, "predict_mc must restore the previous mode"


def test_mc_dropout_is_stochastic_but_deterministic_under_a_seed():
    class Enc(torch.nn.Module):
        def forward(self, regional, fc, local_mask):
            return {"brain_embedding": regional.mean(1)}

    m = StagePredictor(Enc(), build_head("multiclass", 3, dropout=0.5))
    reg, fc = torch.randn(4, 246, 3), torch.zeros(4, 246, 246)
    mask = torch.ones(246, 246, dtype=torch.bool)
    torch.manual_seed(1)
    a, _ = m.predict_mc(reg, fc, mask, n_samples=5)
    torch.manual_seed(1)
    b, _ = m.predict_mc(reg, fc, mask, n_samples=5)
    assert torch.allclose(a, b), "seeded MC-dropout must reproduce"
