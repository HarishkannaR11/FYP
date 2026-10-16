"""
Independent validation of the cumulative-link ordinal decoder.

These tests exist because the previous decision rule (argmax over class
probabilities) made four of the six stages mathematically unreachable, which
is what produced the "predicts AD for every subject" result. The tests below
check the decoder across its whole valid score range rather than on a handful
of random inputs, so the same class of failure cannot return unnoticed.
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heads import OrdinalHead

N = 6
torch.set_num_threads(1)


def _head_with_prior(counts=(1, 1, 1, 1, 1, 1), d=8):
    h = OrdinalHead(d, dropout=0.0)
    h.init_from_prior(torch.tensor(counts, dtype=torch.float))
    h.eval()
    return h


def _sweep(head, lo=-12.0, hi=12.0, n=6001, d=8):
    """Drive the severity score across its range by using a one-hot input and
    setting the score weight, so g sweeps lo..hi exactly."""
    with torch.no_grad():
        head.score.weight.zero_()
        head.score.weight[0, 0] = 1.0
        head.score.bias.zero_()
    x = torch.zeros(n, d)
    x[:, 0] = torch.linspace(lo, hi, n)
    out = head(x)
    return out, head.decode(out), head.probabilities(out)


# ------------------------------------------------------- reachability
def test_all_six_classes_reachable_across_the_score_range():
    head = _head_with_prior()
    _, dec, _ = _sweep(head)
    reached = sorted(set(int(v) for v in dec))
    assert reached == list(range(N)), (
        f"only classes {reached} are reachable; a decoder that cannot emit "
        "every stage cannot predict every stage")


def test_argmax_is_not_used_as_the_decision_rule():
    """Documents the defect: argmax reaches only the two extreme classes."""
    head = _head_with_prior()
    _, dec, probs = _sweep(head)
    arg = probs.argmax(dim=-1)
    assert len(set(int(v) for v in arg)) < N, (
        "this test encodes the known argmax pathology; if argmax now reaches "
        "all classes the test needs revisiting")
    assert len(set(int(v) for v in dec)) == N
    assert not torch.equal(arg, dec), "decode() must differ from argmax"


@pytest.mark.parametrize("counts", [
    (20, 20, 20, 16, 20, 20),          # the real cohort
    (50, 10, 10, 10, 10, 10),          # skewed toward CN
    (10, 10, 10, 10, 10, 50),          # skewed toward AD
])
def test_all_classes_reachable_under_different_priors(counts):
    head = _head_with_prior(counts)
    _, dec, _ = _sweep(head)
    assert sorted(set(int(v) for v in dec)) == list(range(N))


# ------------------------------------------------------- monotonicity
def test_decoded_stage_is_monotone_non_increasing_in_the_score():
    """A higher severity score must never decode to an earlier stage."""
    head = _head_with_prior()
    _, dec, _ = _sweep(head)
    d = dec.numpy()
    assert np.all(np.diff(d) >= 0), "decoded stage must be monotone in g"


def test_decoder_matches_its_definition():
    """decode() == number of thresholds whose cumulative probability < 0.5."""
    head = _head_with_prior()
    out, dec, _ = _sweep(head, n=501)
    manual = (torch.sigmoid(out) < 0.5).sum(dim=-1)
    assert torch.equal(dec, manual)


def test_extremes_decode_to_the_extreme_stages():
    head = _head_with_prior()
    out, dec, _ = _sweep(head, lo=-40.0, hi=40.0, n=201)
    assert int(dec[0]) == 0, "a very low severity score must decode to CN"
    assert int(dec[-1]) == N - 1, "a very high severity score must decode to AD"


# ------------------------------------------------------- probabilities
def test_probabilities_remain_a_valid_distribution_across_the_range():
    head = _head_with_prior()
    _, _, p = _sweep(head, n=2001)
    assert torch.isfinite(p).all()
    assert (p >= 0).all()
    assert torch.allclose(p.sum(-1), torch.ones(p.shape[0]), atol=1e-5)


def test_cumulative_probabilities_are_monotone_across_the_range():
    head = _head_with_prior()
    out, _, _ = _sweep(head, n=2001)
    cum = torch.sigmoid(out)
    assert (cum[:, 1:] - cum[:, :-1] >= -1e-6).all(), \
        "P(y<=k) must be non-decreasing in k everywhere"


def test_expected_stage_increases_monotonically_with_the_score():
    head = _head_with_prior()
    _, _, p = _sweep(head, n=1001)
    exp = (p * torch.arange(N, dtype=torch.float)).sum(-1)
    assert (exp.diff() >= -1e-5).all()


def test_decode_and_probability_mode_agree_at_the_extremes():
    """Where argmax is valid -- the extremes -- the two rules must agree."""
    head = _head_with_prior()
    _, dec, p = _sweep(head, lo=-40.0, hi=40.0, n=101)
    assert int(dec[0]) == int(p[0].argmax())
    assert int(dec[-1]) == int(p[-1].argmax())


# ------------------------------------------------------- prior init
def test_prior_initialisation_reproduces_the_class_distribution():
    counts = torch.tensor([20., 20., 20., 16., 20., 20.])
    head = OrdinalHead(8, dropout=0.0).init_from_prior(counts)
    with torch.no_grad():
        head.score.weight.zero_()
        head.score.bias.zero_()
    p = head.probabilities(head(torch.zeros(1, 8)))[0]
    expect = counts / counts.sum()
    assert torch.allclose(p, expect, atol=1e-3), \
        f"initial distribution {p.tolist()} != prior {expect.tolist()}"


def test_prior_initialisation_keeps_thresholds_ordered():
    for counts in [(20, 20, 20, 16, 20, 20), (1, 2, 3, 4, 5, 6),
                   (100, 1, 1, 1, 1, 100)]:
        head = OrdinalHead(8).init_from_prior(
            torch.tensor(counts, dtype=torch.float))
        t = head.thresholds()
        assert (t[1:] > t[:-1]).all(), f"thresholds unordered for {counts}"


def test_default_initialisation_was_biased_toward_cn():
    """Records the measured defect that init_from_prior corrects."""
    head = OrdinalHead(8, dropout=0.0)
    with torch.no_grad():
        head.score.weight.zero_()
        head.score.bias.zero_()
    p = head.probabilities(head(torch.zeros(1, 8)))[0]
    assert float(p[0]) > 0.4, "default init was measured at P(CN)=0.53"
    fixed = OrdinalHead(8, dropout=0.0).init_from_prior(
        torch.ones(N))
    with torch.no_grad():
        fixed.score.weight.zero_()
        fixed.score.bias.zero_()
    pf = fixed.probabilities(fixed(torch.zeros(1, 8)))[0]
    assert abs(float(pf[0]) - 1 / N) < 1e-3, "prior init should start uniform"


# ------------------------------------------------------- loss reduction
def test_mean_reduce_rescales_without_changing_the_optimum():
    torch.manual_seed(0)
    h = torch.randn(64, 8)
    y = torch.randint(0, N, (64,))
    a = OrdinalHead(8, dropout=0.0, mean_reduce=False)
    b = OrdinalHead(8, dropout=0.0, mean_reduce=True)
    b.load_state_dict(a.state_dict())
    la, lb = a.loss(a(h), y), b.loss(b(h), y)
    assert torch.allclose(la / (N - 1), lb, atol=1e-6), \
        "mean_reduce must be exactly the summed loss divided by K-1"
