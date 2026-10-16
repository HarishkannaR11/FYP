"""
The three attention branches of the Multi-Attention Transformer.

Each branch answers a different question about the same 246 ROI tokens:

    GlobalAttention   which regions interact, with no prior restriction
    LocalAttention    which *neighbouring* regions interact, restricted to a
                      caller-supplied neighbourhood mask
    NetworkAttention  which regions interact given their measured functional
                      connectivity

Architecture only. No anatomical information is invented here: the local
neighbourhood mask and the functional-connectivity matrix are both inputs.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = float("-inf")


def _split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """[B, N, D] -> [B, H, N, Dh]"""
    b, n, d = x.shape
    return x.view(b, n, n_heads, d // n_heads).transpose(1, 2)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    """[B, H, N, Dh] -> [B, N, D]"""
    b, h, n, dh = x.shape
    return x.transpose(1, 2).contiguous().view(b, n, h * dh)


class _MHAProjections(nn.Module):
    """Shared Q/K/V/output projections for the three branches."""

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads "
                f"({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def qkv(self, x):
        return (_split_heads(self.q(x), self.num_heads),
                _split_heads(self.k(x), self.num_heads),
                _split_heads(self.v(x), self.num_heads))

    def scores(self, q, k):
        return torch.matmul(q, k.transpose(-2, -1)) * self.scale

    def apply(self, attn, v):
        ctx = _merge_heads(torch.matmul(self.attn_drop(attn), v))
        return self.proj_drop(self.out(ctx))


class GlobalAttention(_MHAProjections):
    r"""Unrestricted multi-head self-attention over all ROIs.

    .. math::
        A = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right)

    Every ROI may attend to every other ROI. This is the branch that can
    discover long-range interactions not present in either the anatomical
    neighbourhood mask or the measured connectivity matrix.
    """

    def forward(self, x: torch.Tensor):
        q, k, v = self.qkv(x)
        attn = F.softmax(self.scores(q, k), dim=-1)
        return self.apply(attn, v), attn


class LocalAttention(_MHAProjections):
    r"""Self-attention restricted to a supplied neighbourhood mask.

    .. math::
        A_{ij} = \frac{\exp(s_{ij})\,M_{ij}}{\sum_{k} \exp(s_{ik})\,M_{ik}},
        \qquad s_{ij} = \frac{q_i^\top k_j}{\sqrt{d_h}}

    where :math:`M \in \{0,1\}^{N\times N}` is the caller-supplied mask.
    Disallowed edges are filled with :math:`-\infty` before the softmax, so
    they receive **exactly** zero probability rather than a small residual.

    A row of ``M`` with no permitted neighbours would make the softmax
    undefined (``exp(-inf)`` everywhere, 0/0). Such rows are detected and
    given an all-zero attention distribution instead, so this branch
    contributes nothing for that ROI and the token survives through the
    residual connection rather than becoming NaN.

    The mask is an input. No neighbourhood is inferred or invented here.
    """

    def forward(self, x: torch.Tensor, local_mask: torch.Tensor):
        q, k, v = self.qkv(x)
        s = self.scores(q, k)                              # [B, H, N, N]

        allowed = local_mask.to(torch.bool)
        if allowed.dim() == 2:
            allowed = allowed.unsqueeze(0).unsqueeze(0)     # broadcast B, H
        elif allowed.dim() == 3:
            allowed = allowed.unsqueeze(1)                  # per-sample mask

        # rows with no permitted neighbour at all
        empty = ~allowed.any(dim=-1, keepdim=True)          # [.., N, 1]

        s = s.masked_fill(~allowed, NEG_INF)
        # neutralise fully-masked rows so softmax never sees an all -inf row
        s = torch.where(empty.expand_as(s), torch.zeros_like(s), s)
        attn = F.softmax(s, dim=-1)
        # ...then discard their (meaningless, uniform) distribution
        attn = torch.where(empty.expand_as(attn),
                           torch.zeros_like(attn), attn)
        return self.apply(attn, v), attn


class NetworkAttention(_MHAProjections):
    r"""Attention biased by the measured functional-connectivity matrix.

    .. math::
        s_{ij} = \frac{q_i^\top k_j}{\sqrt{d_h}}
                 + \alpha^{+}_h \,\mathrm{relu}(C_{ij})
                 + \alpha^{-}_h \,\mathrm{relu}(-C_{ij})

    where :math:`C` is the Fisher-z transformed 246x246 connectivity matrix
    and :math:`\alpha^{+}_h, \alpha^{-}_h` are per-head learnable scalars.

    **Why positive and negative connectivity are handled separately.** A
    single signed bias (:math:`\alpha C_{ij}`) forces one assumption: that
    anti-correlation means *less* relevant. In resting-state fMRI an
    anti-correlation is a real relationship, not an absence of one, so that
    assumption is not safe to hard-code. Splitting the bias into a positive
    and a negative channel, each with its own unconstrained learnable scale,
    lets the model decide whether strong anti-correlation should raise or
    lower attention. Both scales start at zero, so the branch begins as
    ordinary self-attention and any reliance on connectivity is learned.

    The connectivity matrix is an input and is never modified.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__(d_model, num_heads, dropout)
        # start at zero: no connectivity bias until it is learned
        self.alpha_pos = nn.Parameter(torch.zeros(num_heads))
        self.alpha_neg = nn.Parameter(torch.zeros(num_heads))

    def forward(self, x: torch.Tensor, fc: torch.Tensor):
        q, k, v = self.qkv(x)
        s = self.scores(q, k)                               # [B, H, N, N]

        if fc.dim() == 2:
            fc = fc.unsqueeze(0)
        c = fc.unsqueeze(1)                                 # [B, 1, N, N]
        pos = F.relu(c)
        neg = F.relu(-c)
        ap = self.alpha_pos.view(1, -1, 1, 1)
        an = self.alpha_neg.view(1, -1, 1, 1)
        attn = F.softmax(s + ap * pos + an * neg, dim=-1)
        return self.apply(attn, v), attn


class BranchFusion(nn.Module):
    r"""Learnable gated fusion of the three branch outputs.

    .. math::
        g_i = \mathrm{softmax}\big(W\,[h^{g}_i \Vert h^{l}_i \Vert h^{n}_i]\big)
        \in \Delta^2,
        \qquad
        h_i = W_o \sum_{b \in \{g,l,n\}} g_{i,b}\, h^{b}_i

    The gate is produced per ROI from the concatenated branch outputs and is
    a softmax, so all three weights are strictly positive: every branch
    contributes to the fused representation at every ROI, and the relative
    contributions are learned rather than fixed.
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.gate = nn.Linear(3 * d_model, 3)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, hg: torch.Tensor, hl: torch.Tensor, hn: torch.Tensor):
        cat = torch.cat([hg, hl, hn], dim=-1)               # [B, N, 3D]
        w = F.softmax(self.gate(cat), dim=-1)               # [B, N, 3]
        stacked = torch.stack([hg, hl, hn], dim=-2)         # [B, N, 3, D]
        fused = (w.unsqueeze(-1) * stacked).sum(dim=-2)     # [B, N, D]
        return self.drop(self.proj(fused)), w


class FeedForward(nn.Module):
    """Position-wise feed-forward network, applied per ROI token."""

    def __init__(self, d_model: int, expansion: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, expansion * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
