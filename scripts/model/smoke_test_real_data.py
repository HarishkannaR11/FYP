"""
End-to-end smoke test: real selected-dataset features through the
Multi-Attention Transformer. Verifies plumbing only. Nothing is trained
and no result is produced.

The local attention mask used here is all-permitted. That is NOT an
anatomical choice -- with every edge allowed, the local branch degenerates
to global attention. It exists so the forward pass can be exercised before
a real neighbourhood mask is supplied.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from multi_attention_transformer import build_model  # noqa: E402

OUT = os.path.join(ROOT, "derivatives", "model_ready")
BATCH = 4


def main():
    reg = np.load(os.path.join(OUT, "X_regional.npy"), mmap_mode="r")
    fc = np.load(os.path.join(OUT, "X_fc.npy"), mmap_mode="r")
    y = np.load(os.path.join(OUT, "y.npy"))
    subs = np.load(os.path.join(OUT, "subjects.npy"))
    print(f"bundle : regional {reg.shape}  fc {fc.shape}  y {y.shape}")

    r = torch.from_numpy(np.ascontiguousarray(reg[:BATCH])).float()
    f = torch.from_numpy(np.ascontiguousarray(fc[:BATCH])).float()
    mask = torch.ones(246, 246, dtype=torch.bool)   # smoke test only

    m = build_model(d_model=64, num_heads=4, num_layers=2).eval()
    with torch.no_grad():
        o = m(r, f, mask)

    bw = o["branch_weights"].mean((0, 1)).tolist()
    print(f"parameters      : {m.n_parameters():,}")
    print(f"roi_embeddings  : {tuple(o['roi_embeddings'].shape)}")
    print(f"brain_embedding : {tuple(o['brain_embedding'].shape)}")
    print(f"branch_weights  : {tuple(o['branch_weights'].shape)}  "
          f"mean (g,l,n) = {[round(v, 3) for v in bw]}")
    print(f"pool_weights    : {tuple(o['pool_weights'].shape)}  "
          f"sums to {o['pool_weights'].sum(1).tolist()[:2]}")
    print(f"all finite      : {bool(torch.isfinite(o['brain_embedding']).all())}")
    print(f"batch subjects  : {subs[:BATCH].tolist()}")
    print(f"batch labels    : {y[:BATCH].tolist()}")
    print("\nforward pass OK. Nothing trained.")


if __name__ == "__main__":
    main()
