"""
Unit tests for the Multi-Attention Transformer architecture.

Covers forward-pass shapes, local-mask behaviour, finiteness, gradient flow
and input validation. No training, no data, no performance claims.

Run:  python -m pytest scripts/model/test_multi_attention_transformer.py -v
  or: python scripts/model/test_multi_attention_transformer.py
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_attention_transformer import (N_ROI, MultiAttentionTransformer,  # noqa: E402
                                         TransformerConfig, build_model)

D_MODEL, HEADS, LAYERS = 32, 4, 2


def tiny_model(**kw):
    kw.setdefault("d_model", D_MODEL)
    kw.setdefault("num_heads", HEADS)
    kw.setdefault("num_layers", LAYERS)
    kw.setdefault("dropout", 0.0)
    return build_model(**kw)


def band_mask(width: int = 1) -> torch.Tensor:
    """Symmetric band mask: each ROI may attend to itself and `width`
    neighbours either side. A stand-in for a caller-supplied neighbourhood;
    it encodes no anatomical claim."""
    m = torch.eye(N_ROI, dtype=torch.bool)
    for k in range(1, width + 1):
        off = torch.ones(N_ROI - k, dtype=torch.bool)
        m = m | torch.diag(off, k) | torch.diag(off, -k)
    return m


def sym_fc(b: int) -> torch.Tensor:
    fc = torch.randn(b, N_ROI, N_ROI)
    fc = (fc + fc.transpose(1, 2)) / 2
    fc.diagonal(dim1=1, dim2=2).zero_()
    return fc


def inputs(b: int = 2):
    return torch.randn(b, N_ROI, 3), sym_fc(b), band_mask()


# --------------------------------------------------------------- shapes
@pytest.mark.parametrize("batch", [1, 2, 5])
def test_forward_shapes(batch):
    m = tiny_model().eval()
    reg, fc, mask = inputs(batch)
    with torch.no_grad():
        o = m(reg, fc, mask)
    assert o["roi_embeddings"].shape == (batch, N_ROI, D_MODEL)
    assert o["brain_embedding"].shape == (batch, D_MODEL)
    assert o["branch_weights"].shape == (batch, N_ROI, 3)
    assert len(o["branch_weights_per_layer"]) == LAYERS


def test_batch_size_one():
    """Batch size 1 must not collapse a dimension."""
    m = tiny_model().eval()
    reg, fc, mask = inputs(1)
    with torch.no_grad():
        o = m(reg, fc, mask)
    assert o["roi_embeddings"].dim() == 3
    assert o["brain_embedding"].dim() == 2


def test_unbatched_fc_accepted():
    m = tiny_model().eval()
    reg, _, mask = inputs(1)
    fc2d = sym_fc(1)[0]
    with torch.no_grad():
        o = m(reg, fc2d, mask)
    assert o["roi_embeddings"].shape == (1, N_ROI, D_MODEL)


@pytest.mark.parametrize("pooling", ["attention", "mean"])
def test_pooling_modes(pooling):
    m = tiny_model(pooling=pooling).eval()
    reg, fc, mask = inputs(2)
    with torch.no_grad():
        o = m(reg, fc, mask)
    assert o["brain_embedding"].shape == (2, D_MODEL)
    if pooling == "attention":
        w = o["pool_weights"]
        assert w.shape == (2, N_ROI)
        assert torch.allclose(w.sum(1), torch.ones(2), atol=1e-5)
    else:
        assert o["pool_weights"] is None


# ----------------------------------------------------------- mask behaviour
def test_masked_edges_get_exactly_zero_probability():
    m = tiny_model().eval()
    reg, fc, mask = inputs(1)
    with torch.no_grad():
        o = m(reg, fc, mask, return_attention=True)
    for layer_maps in o["attention"]:
        a = layer_maps["local"]                       # [B, H, N, N]
        disallowed = ~mask.to(torch.bool)
        assert torch.all(a[..., disallowed] == 0), \
            "masked local-attention edges must receive zero probability"


def test_allowed_rows_sum_to_one():
    m = tiny_model().eval()
    reg, fc, mask = inputs(1)
    with torch.no_grad():
        o = m(reg, fc, mask, return_attention=True)
    a = o["attention"][0]["local"]
    rows = a.sum(-1)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5)


def test_empty_mask_row_does_not_produce_nan():
    """An ROI with no permitted neighbours must not poison the forward pass."""
    m = tiny_model().eval()
    reg, fc, mask = inputs(1)
    mask = mask.clone()
    mask[7, :] = False                                # ROI 7 has no neighbours
    with torch.no_grad():
        o = m(reg, fc, mask, return_attention=True)
    assert torch.isfinite(o["roi_embeddings"]).all()
    assert torch.isfinite(o["brain_embedding"]).all()
    a = o["attention"][0]["local"]
    assert torch.all(a[:, :, 7, :] == 0), \
        "a neighbourless ROI should receive an all-zero local distribution"


def test_fully_empty_mask_is_survivable():
    m = tiny_model().eval()
    reg, fc, _ = inputs(1)
    mask = torch.zeros(N_ROI, N_ROI, dtype=torch.bool)
    with torch.no_grad():
        o = m(reg, fc, mask)
    assert torch.isfinite(o["roi_embeddings"]).all()


def test_mask_actually_changes_the_output():
    """A different neighbourhood must give a different representation,
    otherwise the local branch is not contributing."""
    torch.manual_seed(0)
    m = tiny_model().eval()
    reg, fc, _ = inputs(1)
    with torch.no_grad():
        a = m(reg, fc, band_mask(1))["roi_embeddings"]
        b = m(reg, fc, band_mask(8))["roi_embeddings"]
    assert not torch.allclose(a, b, atol=1e-6)


# --------------------------------------------------------------- branches
def test_all_three_branches_contribute():
    """Gate weights are a softmax, so every branch has strictly positive
    weight at every ROI."""
    m = tiny_model().eval()
    reg, fc, mask = inputs(2)
    with torch.no_grad():
        w = m(reg, fc, mask)["branch_weights"]
    assert (w > 0).all()
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)


def test_fc_changes_the_output_once_connectivity_scales_are_nonzero():
    """alpha starts at zero by design, so the network branch is initially
    FC-agnostic. With non-zero scales the connectivity must matter."""
    torch.manual_seed(0)
    m = tiny_model().eval()
    for layer in m.layers:
        torch.nn.init.constant_(layer.net.alpha_pos, 0.5)
        torch.nn.init.constant_(layer.net.alpha_neg, -0.3)
    reg, fc, mask = inputs(1)
    with torch.no_grad():
        a = m(reg, fc, mask)["roi_embeddings"]
        b = m(reg, sym_fc(1), mask)["roi_embeddings"]
    assert not torch.allclose(a, b, atol=1e-6)


def test_negative_connectivity_is_not_discarded():
    """A sign flip of FC must change the result: anti-correlation carries
    information and is not treated as absence of a relationship."""
    torch.manual_seed(0)
    m = tiny_model().eval()
    for layer in m.layers:
        torch.nn.init.constant_(layer.net.alpha_pos, 0.5)
        torch.nn.init.constant_(layer.net.alpha_neg, -0.3)
    reg, fc, mask = inputs(1)
    with torch.no_grad():
        a = m(reg, fc, mask)["roi_embeddings"]
        b = m(reg, -fc, mask)["roi_embeddings"]
    assert not torch.allclose(a, b, atol=1e-6)


# --------------------------------------------------------------- gradients
def test_gradient_flows_to_every_parameter():
    m = tiny_model().train()
    reg, fc, mask = inputs(2)
    out = m(reg, fc, mask)
    (out["brain_embedding"].pow(2).mean()
     + out["roi_embeddings"].pow(2).mean()).backward()
    missing = [n for n, p in m.named_parameters()
               if p.requires_grad and (p.grad is None or
                                       not torch.isfinite(p.grad).all())]
    assert not missing, f"no finite gradient for: {missing[:10]}"


def test_gradient_reaches_each_branch():
    m = tiny_model().train()
    reg, fc, mask = inputs(2)
    m(reg, fc, mask)["brain_embedding"].sum().backward()
    for i, layer in enumerate(m.layers):
        for name, mod in (("global", layer.glob), ("local", layer.loc),
                          ("network", layer.net)):
            g = mod.q.weight.grad
            assert g is not None and torch.isfinite(g).all(), \
                f"layer {i} {name} branch received no gradient"
            assert g.abs().sum() > 0, f"layer {i} {name} gradient is all zero"


def test_outputs_are_finite_under_extreme_inputs():
    m = tiny_model().eval()
    mask = band_mask()
    reg = torch.randn(2, N_ROI, 3) * 1e3
    fc = sym_fc(2) * 20.0
    with torch.no_grad():
        o = m(reg, fc, mask)
    assert torch.isfinite(o["roi_embeddings"]).all()
    assert torch.isfinite(o["brain_embedding"]).all()


# -------------------------------------------------------------- validation
def test_rejects_wrong_roi_count():
    m = tiny_model()
    with pytest.raises(ValueError, match="regional must be"):
        m(torch.randn(1, 100, 3), sym_fc(1), band_mask())


def test_rejects_wrong_feature_count():
    m = tiny_model()
    with pytest.raises(ValueError, match="regional must be"):
        m(torch.randn(1, N_ROI, 4), sym_fc(1), band_mask())


def test_rejects_nan_and_inf():
    m = tiny_model()
    reg, fc, mask = inputs(1)
    bad = reg.clone(); bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        m(bad, fc, mask)
    bad_fc = fc.clone(); bad_fc[0, 0, 1] = float("inf")
    with pytest.raises(ValueError, match="NaN or Inf"):
        m(reg, bad_fc, mask)


def test_rejects_wrong_fc_shape():
    m = tiny_model()
    reg, _, mask = inputs(1)
    with pytest.raises(ValueError, match="fc must be"):
        m(reg, torch.randn(1, 100, 100), mask)


def test_config_rejects_indivisible_dims():
    with pytest.raises(ValueError, match="divisible"):
        TransformerConfig(d_model=30, num_heads=4)


def test_roi_order_matters():
    """Learnable ROI identity embeddings mean the ordering is meaningful;
    permuting ROIs must change the representation."""
    torch.manual_seed(0)
    m = tiny_model().eval()
    reg, fc, mask = inputs(1)
    perm = torch.randperm(N_ROI)
    with torch.no_grad():
        a = m(reg, fc, mask)["brain_embedding"]
        b = m(reg[:, perm], fc[:, perm][:, :, perm], mask)["brain_embedding"]
    assert not torch.allclose(a, b, atol=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))
