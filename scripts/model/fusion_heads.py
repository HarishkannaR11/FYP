"""
Comorbidity fusion (Block 6): fuse the transformer's brain embedding with a
patient's comorbidity profile before the stage head.

    brain embedding z  [B, d]   from MultiAttentionTransformer
    comorbidity c      [B, n]   age, sex, APOE4, GDS, medical-history flags
                                (scaled on the training subjects)

    e   = ClinicalEncoder(c)                          [B, d]
    g   = sigmoid(w^T [z || e] + b)                   [B, 1]   per-patient gate
    h   = LayerNorm(g * P_z z + (1 - g) * P_e e)      [B, d]
    out = head(h)

The gate g is the share of the fused representation that comes from the
brain (1) versus the comorbidity profile (0). It is learned per patient, so
the representation is patient-specific, and it is reported for
explainability. Nothing in multi_attention_transformer.py or heads.py is
modified; the encoder and head are used exactly as built there.
"""
import torch
import torch.nn as nn


class ClinicalEncoder(nn.Module):
    """Small MLP from the comorbidity vector to the embedding size."""

    def __init__(self, n_in: int, d_model: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, d_model), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_model, d_model))

    def forward(self, c):
        return self.net(c)


class GatedFusion(nn.Module):
    """Per-patient gate between the brain and comorbidity embeddings."""

    def __init__(self, d_model: int, dropout: float = 0.3):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 1)
        self.proj_brain = nn.Linear(d_model, d_model)
        self.proj_clin = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, z, e):
        g = torch.sigmoid(self.gate(torch.cat([z, e], dim=-1)))          # [B, 1]
        h = g * self.proj_brain(z) + (1 - g) * self.proj_clin(e)
        return self.drop(self.norm(h)), g


class FusionPredictor(nn.Module):
    """Transformer encoder + comorbidity encoder + gated fusion + stage head."""

    def __init__(self, encoder, head, n_clinical: int, d_model: int, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.clinical = ClinicalEncoder(n_clinical, d_model, dropout)
        self.fusion = GatedFusion(d_model, dropout)
        self.head = head

    def forward_full(self, regional, fc, local_mask, clin, return_attention=False):
        enc = self.encoder(regional, fc, local_mask, return_attention=return_attention)
        h, g = self.fusion(enc["brain_embedding"], self.clinical(clin))
        return self.head(h), g, enc

    def forward(self, regional, fc, local_mask, clin):
        return self.forward_full(regional, fc, local_mask, clin)[0]

    @torch.no_grad()
    def predict_mc(self, regional, fc, local_mask, clin, n_samples: int = 30):
        """MC-dropout, as StagePredictor.predict_mc. Returns (mean probs, entropy)."""
        was_training = self.training
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        acc = None
        for _ in range(n_samples):
            p = self.head.probabilities(self(regional, fc, local_mask, clin))
            acc = p if acc is None else acc + p
        self.train(was_training)
        p = acc / n_samples
        p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
        entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)
        return p, entropy
