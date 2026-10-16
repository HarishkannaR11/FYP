"""
Step 2: why does the ordinal head collapse while multiclass trains?

Each hypothesis is stated, tested in isolation, and recorded with its result.
Nothing here touches the held-out test set; everything uses the training pool.

H1  The loss is mis-specified (targets, signs, decoding).
H2  The ordinal loss has a different scale from cross-entropy, so the same
    learning rate is not the same step size.
H3  The encoder receives a RANK-1 gradient. The ordinal head projects the
    whole embedding onto ONE scalar severity score, so dL/dh is always
    parallel to a single weight vector. The multiclass head has six output
    rows, so the encoder receives up to a six-dimensional signal. A rank-1
    channel is a far weaker teaching signal for a 196k-parameter encoder.
H4  The head cannot overfit even a tiny, deliberately learnable subset --
    which would indicate an optimisation failure rather than absent signal.
H5  Initialisation places the model in a region where gradients vanish.
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heads import OrdinalHead, MulticlassHead, StagePredictor, build_head
from multi_attention_transformer import build_model
from train_transformer import Scaler, evaluate, load_all

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "derivatives", "model_training")
STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
N = 6
torch.set_num_threads(1)

findings = []
_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def record(h, hypothesis, intervention, result, verdict):
    findings.append({"hypothesis": h, "statement": hypothesis,
                     "intervention": intervention, "result": result,
                     "verdict": verdict})
    log(f"\n[{h}] {hypothesis}")
    log(f"     intervention: {intervention}")
    log(f"     result      : {result}")
    log(f"     VERDICT     : {verdict}")


# ----------------------------------------------------------------- H1
def h1_loss_specification():
    torch.manual_seed(0)
    head = OrdinalHead(8)
    checks = []

    # target construction: 1[y <= k] for k = 0..4
    y = torch.tensor([0, 3, 5])
    k = torch.arange(N - 1).unsqueeze(0)
    tgt = (y.unsqueeze(1) <= k).float()
    expect = torch.tensor([[1., 1., 1., 1., 1.],
                           [0., 0., 0., 1., 1.],
                           [0., 0., 0., 0., 0.]])
    checks.append(("cumulative targets", bool(torch.equal(tgt, expect))))

    # sign convention: out = tau_k - g, so P(y<=k) = sigmoid(tau_k - g)
    with torch.no_grad():
        head.score.weight.zero_()
        head.score.bias.fill_(0.0)
    out = head(torch.zeros(1, 8))
    checks.append(("out equals thresholds when g=0",
                   bool(torch.allclose(out[0], head.thresholds(), atol=1e-6))))

    # raising g must lower every P(y<=k): more severity, later stage
    with torch.no_grad():
        head.score.bias.fill_(2.0)
    out2 = head(torch.zeros(1, 8))
    checks.append(("higher score lowers P(y<=k)",
                   bool((torch.sigmoid(out2) < torch.sigmoid(out)).all())))

    # decoding: probabilities sum to 1, are non-negative, and the expected
    # stage increases with g
    def exp_stage(bias):
        with torch.no_grad():
            head.score.bias.fill_(bias)
        p = head.probabilities(head(torch.zeros(1, 8)))
        return float(p @ torch.arange(N, dtype=torch.float)), p

    e_lo, p_lo = exp_stage(-4.0)
    e_hi, p_hi = exp_stage(4.0)
    checks.append(("probabilities sum to 1",
                   bool(torch.allclose(p_lo.sum(), torch.tensor(1.0), atol=1e-5))))
    checks.append(("non-negative probabilities", bool((p_lo >= 0).all())))
    checks.append(("expected stage increases with score", e_hi > e_lo))

    ok = all(v for _, v in checks)
    record("H1", "the ordinal loss is mis-specified",
           "unit-checked target construction, sign convention, monotonicity "
           "and decoding in isolation",
           "; ".join(f"{n}={v}" for n, v in checks),
           "REJECTED -- the formulation is correct" if ok
           else "SUPPORTED -- formulation is wrong")
    return ok


# ----------------------------------------------------------------- H2
def h2_loss_scale():
    torch.manual_seed(0)
    h = torch.randn(32, 16)
    y = torch.randint(0, N, (32,))
    mc, od = MulticlassHead(16, dropout=0.0), OrdinalHead(16, dropout=0.0)
    lmc = mc.loss(mc(h), y)
    lod = od.loss(od(h), y)
    gmc = torch.autograd.grad(lmc, list(mc.parameters()), retain_graph=True)
    god = torch.autograd.grad(lod, list(od.parameters()), retain_graph=True)
    nmc = float(torch.sqrt(sum(g.pow(2).sum() for g in gmc)))
    nod = float(torch.sqrt(sum(g.pow(2).sum() for g in god)))
    ratio = lod.item() / lmc.item()
    record("H2", "the ordinal loss has a different scale from cross-entropy",
           "computed both losses and their head-gradient norms on identical "
           "random embeddings",
           f"loss CE={lmc.item():.4f} ordinal={lod.item():.4f} "
           f"(ratio {ratio:.2f}x, ordinal sums {N - 1} BCE terms); "
           f"head grad norm CE={nmc:.4f} ordinal={nod:.4f}",
           f"SUPPORTED -- ordinal loss is {ratio:.1f}x CE, so an identical "
           f"learning rate is not an identical step size"
           if ratio > 1.5 else "REJECTED -- scales are comparable")
    return ratio


# ----------------------------------------------------------------- H3
def h3_gradient_rank():
    """The decisive structural test: what rank of signal reaches the encoder?"""
    torch.manual_seed(0)
    d, B = 32, 24
    results = {}
    for kind in ("multiclass", "ordinal"):
        head = build_head(kind, d, dropout=0.0)
        h = torch.randn(B, d, requires_grad=True)
        y = torch.randint(0, N, (B,))
        loss = head.loss(head(h), y)
        (gh,) = torch.autograd.grad(loss, h)
        # effective rank of the gradient arriving at the embedding
        sv = torch.linalg.svdvals(gh)
        sv = sv / sv.sum()
        eff_rank = float(torch.exp(-(sv * torch.log(sv.clamp_min(1e-12))).sum()))
        results[kind] = {"grad_norm": float(gh.norm()),
                         "numerical_rank": int(torch.linalg.matrix_rank(gh)),
                         "effective_rank": eff_rank,
                         "top_sv_share": float(sv[0])}
    record("H3", "the encoder receives a rank-1 gradient from the ordinal head",
           "measured the rank and effective rank of dL/d(embedding) for both "
           "heads on identical inputs",
           f"multiclass: numerical rank {results['multiclass']['numerical_rank']}, "
           f"effective rank {results['multiclass']['effective_rank']:.2f}, "
           f"top singular value holds {results['multiclass']['top_sv_share']:.1%}; "
           f"ordinal: numerical rank {results['ordinal']['numerical_rank']}, "
           f"effective rank {results['ordinal']['effective_rank']:.2f}, "
           f"top singular value holds {results['ordinal']['top_sv_share']:.1%}",
           "SUPPORTED -- the ordinal gradient is rank-1 while multiclass is "
           "higher rank, so the encoder is taught through a single direction"
           if results["ordinal"]["numerical_rank"] == 1
           and results["multiclass"]["numerical_rank"] > 1
           else "REJECTED")
    return results


# ----------------------------------------------------------------- H4
def h4_overfit_tiny(reg, fc, y, mask_t, pool, n_per_class=2, epochs=250):
    """Can each head memorise a deliberately tiny subset? A model that cannot
    overfit 12 subjects has an optimisation problem, not a data problem."""
    rng = np.random.default_rng(0)
    pick = []
    for g in range(N):
        idx = pool[y[pool] == g]
        pick.extend(rng.choice(idx, size=min(n_per_class, idx.size),
                               replace=False).tolist())
    pick = np.array(sorted(pick))
    sc = Scaler().fit(reg, pick)
    X = torch.from_numpy(sc.transform(reg[pick]))
    Fm = torch.from_numpy(fc[pick])
    yt = torch.from_numpy(y[pick])
    out = {}
    for kind in ("multiclass", "ordinal"):
        torch.manual_seed(0)
        enc = build_model(d_model=16, num_heads=2, num_layers=1, dropout=0.0)
        model = StagePredictor(enc, build_head(kind, 16, dropout=0.0))
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
        model.train()
        for ep in range(epochs):
            opt.zero_grad()
            loss = model.head.loss(model(X, Fm, mask_t), yt)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p = model.head.probabilities(model(X, Fm, mask_t)).numpy()
        m = evaluate(p, y[pick])
        out[kind] = {"train_accuracy": m["accuracy"],
                     "balanced_accuracy": m["balanced_accuracy"],
                     "final_loss": float(loss),
                     "distinct_classes_predicted": m["n_predicted_classes"]}
    record("H4", "the head cannot overfit a deliberately learnable subset",
           f"trained both heads to convergence on {len(pick)} training-pool "
           f"subjects ({n_per_class}/class), no dropout, no weight decay, "
           f"{epochs} epochs, identical encoder and seed",
           f"multiclass: train acc {out['multiclass']['train_accuracy']:.2f}, "
           f"loss {out['multiclass']['final_loss']:.4f}, "
           f"{out['multiclass']['distinct_classes_predicted']} classes predicted; "
           f"ordinal: train acc {out['ordinal']['train_accuracy']:.2f}, "
           f"loss {out['ordinal']['final_loss']:.4f}, "
           f"{out['ordinal']['distinct_classes_predicted']} classes predicted",
           "SUPPORTED -- ordinal cannot memorise even 12 subjects, so this is "
           "an optimisation failure, not absent signal"
           if out["ordinal"]["train_accuracy"] < 0.75 <= out["multiclass"]["train_accuracy"]
           else ("REJECTED -- both heads can memorise the subset, so the "
                 "collapse on the full data is about signal, not optimisation"
                 if out["ordinal"]["train_accuracy"] >= 0.75
                 else "INCONCLUSIVE -- neither head memorised the subset"))
    return out, pick


# ----------------------------------------------------------------- H5
def h5_initialisation():
    torch.manual_seed(0)
    head = OrdinalHead(32, dropout=0.0)
    t = head.thresholds()
    p0 = head.probabilities(head(torch.zeros(1, 32)))[0]
    record("H5", "initialisation places the model where gradients vanish",
           "inspected initial thresholds and the implied class distribution "
           "at a zero embedding",
           f"tau = {[round(float(v), 3) for v in t]}; implied P(stage) = "
           f"{[round(float(v), 3) for v in p0]} (uniform would be "
           f"{1 / N:.3f}); P(CN) = {float(p0[0]):.3f}",
           "SUPPORTED -- init is strongly biased toward CN, far from the "
           "uniform prior, so early gradients fight the initialisation"
           if float(p0[0]) > 0.4 else "REJECTED -- init is near-uniform")
    return [round(float(v), 4) for v in p0]


def main():
    os.makedirs(OUT, exist_ok=True)
    log("=" * 78)
    log("STEP 2 -- ORDINAL COLLAPSE DIAGNOSIS")
    log("=" * 78)
    log("All experiments use the training pool only. The held-out test set is "
        "not loaded, not indexed and not evaluated.")

    h1 = h1_loss_specification()
    ratio = h2_loss_scale()
    h3 = h3_gradient_rank()
    p0 = h5_initialisation()

    reg, fc, y, subjects, mask, pool, test, fold, smeta = load_all()
    mask_t = torch.from_numpy(mask)
    h4, pick = h4_overfit_tiny(reg, fc, y, mask_t, pool)

    log("\n" + "=" * 78)
    log("SUMMARY")
    log("=" * 78)
    for f in findings:
        log(f"  {f['hypothesis']}: {f['verdict'].split(' -- ')[0]:12s} "
            f"{f['statement']}")

    out = {"findings": findings,
           "loss_scale_ratio_ordinal_over_ce": ratio,
           "gradient_rank": h3,
           "overfit_subset": h4,
           "overfit_subject_indices": [int(i) for i in pick],
           "ordinal_init_distribution": p0,
           "split_fingerprint": smeta["split_fingerprint"],
           "test_set_touched": False}
    p = os.path.join(OUT, "ordinal_diagnosis.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"\nwrote {os.path.relpath(p, ROOT)}")
    with open(os.path.join(OUT, "ordinal_diagnosis.log"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")


if __name__ == "__main__":
    main()
