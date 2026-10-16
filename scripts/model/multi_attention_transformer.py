"""
Multi-Attention Transformer encoder over Brainnetome-246 ROI features.

Scope: architecture only. This module defines the encoder that turns one
acquisition's regional biomarkers and connectivity matrix into ROI-level and
whole-brain embeddings. It contains no classification head, no training
loop, no meta-learning, no Bayesian layer and no data loading.

Input per acquisition
    regional   [B, 246, 3]    mALFF, mReHo, DC_z, in that channel order
    fc         [B, 246, 246]  Fisher-z functional connectivity, signed
    local_mask [246, 246]     caller-supplied neighbourhood mask

Output
    roi_embeddings  [B, 246, d_model]
    brain_embedding [B, d_model]
    branch_weights  [B, 246, 3] per layer, plus a summary mean

The three regional biomarkers are the model's input channels. FC_Strength is
deliberately absent: it equals DC/245 exactly and carries no independent
information, so it is not an input channel.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn

try:
    from attention_modules import (BranchFusion, FeedForward, GlobalAttention,
                                   LocalAttention, NetworkAttention)
except ModuleNotFoundError:          # python -I strips the script directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from attention_modules import (BranchFusion, FeedForward, GlobalAttention,
                                   LocalAttention, NetworkAttention)

N_ROI = 246
N_REGIONAL_FEATURES = 3          # mALFF, mReHo, DC_z
FEATURE_ORDER = ("mALFF", "mReHo", "DC_z")


@dataclass
class TransformerConfig:
    """Configuration for the encoder. All fields have defensible defaults."""
    n_roi: int = N_ROI
    n_features: int = N_REGIONAL_FEATURES
    d_model: int = 128
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    ffn_expansion: int = 4
    pooling: str = "attention"          # "attention" | "mean"
    validate_inputs: bool = True

    def __post_init__(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads "
                f"({self.num_heads})")
        if self.pooling not in ("attention", "mean"):
            raise ValueError(f"unknown pooling: {self.pooling!r}")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")


class MultiAttentionLayer(nn.Module):
    """One encoder layer: three parallel attention branches, gated fusion,
    then a position-wise feed-forward network. Pre-norm residual form, which
    is the more stable of the two for deeper stacks.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        d, h, p = cfg.d_model, cfg.num_heads, cfg.dropout
        self.norm_attn = nn.LayerNorm(d)
        self.glob = GlobalAttention(d, h, p)
        self.loc = LocalAttention(d, h, p)
        self.net = NetworkAttention(d, h, p)
        self.fuse = BranchFusion(d, p)
        self.norm_ffn = nn.LayerNorm(d)
        self.ffn = FeedForward(d, cfg.ffn_expansion, p)

    def forward(self, x, fc, local_mask, return_attention=False):
        h = self.norm_attn(x)
        hg, ag = self.glob(h)
        hl, al = self.loc(h, local_mask)
        hn, an = self.net(h, fc)
        fused, w = self.fuse(hg, hl, hn)
        x = x + fused                                   # residual 1
        x = x + self.ffn(self.norm_ffn(x))              # residual 2
        maps = {"global": ag, "local": al, "network": an} if return_attention else None
        return x, w, maps


class AttentionPooling(nn.Module):
    r"""Pool 246 ROI embeddings into one brain embedding.

    .. math::
        a_i = \frac{\exp(w^\top \tanh(V h_i))}{\sum_j \exp(w^\top \tanh(V h_j))},
        \qquad z = \sum_i a_i h_i

    A learnable query scores each ROI and the embeddings are averaged under
    the resulting distribution. This is reported rather than left implicit
    because the pooling choice determines which ROIs drive the whole-brain
    representation.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.v = nn.Linear(d_model, d_model)
        self.w = nn.Linear(d_model, 1, bias=False)

    def forward(self, x):
        a = torch.softmax(self.w(torch.tanh(self.v(x))), dim=1)  # [B, N, 1]
        return (a * x).sum(dim=1), a.squeeze(-1)


class MultiAttentionTransformer(nn.Module):
    """Encoder stack over 246 ROI tokens.

    Token construction: the three regional biomarkers of ROI *i* are
    projected to ``d_model`` and added to a learnable identity embedding for
    ROI *i*. The identity embedding is what makes the ROI ordering
    meaningful to the model, so the caller must present ROIs in a consistent
    order across every acquisition (Brainnetome index 1..246).
    """

    def __init__(self, cfg: TransformerConfig | None = None, **kwargs):
        super().__init__()
        self.cfg = cfg or TransformerConfig(**kwargs)
        c = self.cfg
        self.input_proj = nn.Linear(c.n_features, c.d_model)
        self.roi_embedding = nn.Parameter(torch.zeros(c.n_roi, c.d_model))
        nn.init.normal_(self.roi_embedding, std=0.02)
        self.input_drop = nn.Dropout(c.dropout)
        self.layers = nn.ModuleList(
            MultiAttentionLayer(c) for _ in range(c.num_layers))
        self.norm_out = nn.LayerNorm(c.d_model)
        self.pool = AttentionPooling(c.d_model) if c.pooling == "attention" else None

    # ------------------------------------------------------------ checks
    def _validate(self, regional, fc, local_mask):
        c = self.cfg
        if regional.dim() != 3:
            raise ValueError(
                f"regional must be [B, {c.n_roi}, {c.n_features}], "
                f"got {tuple(regional.shape)}")
        b, n, f = regional.shape
        if n != c.n_roi or f != c.n_features:
            raise ValueError(
                f"regional must be [B, {c.n_roi}, {c.n_features}] "
                f"({', '.join(FEATURE_ORDER)}), got {tuple(regional.shape)}")
        if fc.dim() == 2:
            fc = fc.unsqueeze(0)
        if fc.shape[-2:] != (c.n_roi, c.n_roi):
            raise ValueError(
                f"fc must be [B, {c.n_roi}, {c.n_roi}], got {tuple(fc.shape)}")
        if fc.shape[0] not in (1, b):
            raise ValueError(
                f"fc batch {fc.shape[0]} incompatible with regional batch {b}")
        if local_mask.shape[-2:] != (c.n_roi, c.n_roi):
            raise ValueError(
                f"local_mask must be [{c.n_roi}, {c.n_roi}], "
                f"got {tuple(local_mask.shape)}")
        if not torch.isfinite(regional).all():
            raise ValueError("regional contains NaN or Inf")
        if not torch.isfinite(fc).all():
            raise ValueError("fc contains NaN or Inf")
        if not torch.isfinite(local_mask.to(torch.float32)).all():
            raise ValueError("local_mask contains NaN or Inf")
        return fc

    # ----------------------------------------------------------- forward
    def forward(self, regional: torch.Tensor, fc: torch.Tensor,
                local_mask: torch.Tensor, return_attention: bool = False):
        """
        Parameters
        ----------
        regional : [B, 246, 3]  mALFF, mReHo, DC_z
        fc : [B, 246, 246] or [246, 246]  signed Fisher-z connectivity
        local_mask : [246, 246] or [B, 246, 246]  True/1 = edge permitted
        return_attention : also return per-layer attention maps (large)

        Returns
        -------
        dict with
            roi_embeddings  [B, 246, d_model]
            brain_embedding [B, d_model]
            branch_weights  [B, 246, 3]  mean over layers, order (g, l, n)
            branch_weights_per_layer  list of [B, 246, 3]
            pool_weights    [B, 246] or None
            attention       dict per layer, only if return_attention
        """
        if self.cfg.validate_inputs:
            fc = self._validate(regional, fc, local_mask)
        elif fc.dim() == 2:
            fc = fc.unsqueeze(0)

        x = self.input_proj(regional) + self.roi_embedding.unsqueeze(0)
        x = self.input_drop(x)

        per_layer, maps = [], []
        for layer in self.layers:
            x, w, m = layer(x, fc, local_mask, return_attention)
            per_layer.append(w)
            if return_attention:
                maps.append(m)

        roi = self.norm_out(x)
        if self.pool is not None:
            brain, pool_w = self.pool(roi)
        else:
            brain, pool_w = roi.mean(dim=1), None

        out = {
            "roi_embeddings": roi,
            "brain_embedding": brain,
            "branch_weights": torch.stack(per_layer, 0).mean(0),
            "branch_weights_per_layer": per_layer,
            "pool_weights": pool_w,
            "branch_order": ("global", "local", "network"),
        }
        if return_attention:
            out["attention"] = maps
        return out

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(**kwargs) -> MultiAttentionTransformer:
    """Convenience constructor: ``build_model(d_model=256, num_layers=6)``."""
    return MultiAttentionTransformer(TransformerConfig(**kwargs))


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_model(d_model=64, num_heads=4, num_layers=2)
    mask = torch.eye(N_ROI, dtype=torch.bool)
    mask |= torch.diag(torch.ones(N_ROI - 1, dtype=torch.bool), 1)
    mask |= torch.diag(torch.ones(N_ROI - 1, dtype=torch.bool), -1)
    reg = torch.randn(2, N_ROI, 3)
    fcm = torch.randn(2, N_ROI, N_ROI)
    fcm = (fcm + fcm.transpose(1, 2)) / 2
    fcm.diagonal(dim1=1, dim2=2).zero_()
    o = m(reg, fcm, mask)
    print(f"parameters      : {m.n_parameters():,}")
    print(f"roi_embeddings  : {tuple(o['roi_embeddings'].shape)}")
    print(f"brain_embedding : {tuple(o['brain_embedding'].shape)}")
    print(f"branch_weights  : {tuple(o['branch_weights'].shape)}  "
          f"mean (g,l,n) = "
          f"{o['branch_weights'].mean((0, 1)).detach().tolist()}")
