"""
Correctness tests for the ordinal head and the label order (run before any training).

Writes derivatives/experiments/ordinal_transformer_v1/tests/test_results.json and .log; exits non-zero on failure.
Run:  python scripts/experiments/ordinal_transformer_v1/test_ordinal.py
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "model"))
import feature_selection_cv as H  # noqa: E402  (sets thread env vars first)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from ordinal_head import CumulativeLinkHead, median_decode  # noqa: E402

ROOT = H.ROOT
OUT = os.path.join(ROOT, "derivatives", "experiments", "ordinal_transformer_v1", "tests")
ORDER = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
K = 6
results, lines = [], []


def check(name, ok, detail=""):
    results.append({"test": name, "pass": bool(ok), "detail": detail})
    lines.append(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    print(lines[-1], flush=True)


def head_with(tau0, raw_steps, dtype=torch.float64):
    h = CumulativeLinkHead(4).to(dtype)
    with torch.no_grad():
        h.tau0.copy_(torch.tensor([tau0], dtype=dtype))
        h.raw_steps.copy_(torch.as_tensor(raw_steps, dtype=dtype))
    return h


def out_of(h, g):
    return h.thresholds().unsqueeze(0) - g.unsqueeze(1)


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # 1 label order, from the data files themselves
    meta = json.load(open(os.path.join(ROOT, "derivatives", "model_ready", "metadata.json")))
    y = np.load(os.path.join(ROOT, "derivatives", "model_ready", "y.npy"))
    st = np.load(os.path.join(ROOT, "derivatives", "model_ready", "stages.npy")).astype(str)
    pairs = sorted(set(zip(y.tolist(), st.tolist())))
    check("label order CN<SMC<EMCI<MCI<LMCI<AD in metadata, y.npy/stages.npy and the shared STAGES constant",
          meta["stage_order"] == ORDER and pairs == list(enumerate(ORDER)) and list(H.STAGES) == ORDER,
          f"metadata {meta['stage_to_index']}; y->stage {pairs}")

    # 2 valid probabilities, monotone CDF and thresholds, including extreme scores and thresholds
    ok, worst = True, 0.0
    for _ in range(200):
        h = head_with(rng.normal(0, 3), rng.normal(0, 3, K - 2))
        g = torch.as_tensor(rng.normal(0, 30, 64))
        g[:4] = torch.tensor([-1e3, -60.0, 60.0, 1e3], dtype=torch.float64)
        o = out_of(h, g)
        p = h.probabilities(o)
        tau = h.thresholds()
        cdf = torch.sigmoid(o)
        worst = max(worst, float((p.sum(1) - 1).abs().max()))
        ok &= bool((p >= 0).all() and (p.sum(1) - 1).abs().max() < 1e-9 and (torch.diff(tau) > 0).all()
                   and (torch.diff(cdf, dim=1) >= 0).all() and torch.isfinite(p).all())
    check("probabilities non-negative, sum to 1; thresholds strictly increasing; CDF monotone (incl. |g| up to 1e3)",
          ok, f"max |sum-1| {worst:.1e}")

    # 3 stable log-probabilities agree with log of the CDF differences, and stay finite where the naive form fails
    h = head_with(-1.0, [0.3, -0.5, 0.2, 0.8])
    g = torch.linspace(-12, 12, 97, dtype=torch.float64)
    o = out_of(h, g)
    diff = float((h.log_probs(o) - torch.log(h.probabilities(o))).abs().max())
    gx = torch.tensor([-80.0, 80.0], dtype=torch.float64)
    lp_x = h.log_probs(out_of(h, gx))
    naive = torch.log(h.probabilities(out_of(h, gx).float()).double())
    check("stable log P(y=k) equals log of CDF differences (|g| <= 12) and is finite at |g| = 80",
          diff < 1e-9 and bool(torch.isfinite(lp_x).all()),
          f"max abs diff {diff:.1e}; naive float32 log gives -inf in {int(torch.isinf(naive).sum())} of 12 entries")
    check("exp(log P) sums to 1", float((h.log_probs(o).exp().sum(1) - 1).abs().max()) < 1e-9)

    # 4 the loss is the exact (weighted) NLL
    yy = torch.as_tensor(rng.integers(0, K, 97))
    w = torch.as_tensor(rng.uniform(0.5, 2, K))
    lp = torch.log(h.probabilities(o))
    manual_u = float(-lp.gather(1, yy[:, None]).mean())
    manual_w = float((-lp.gather(1, yy[:, None]).squeeze(1) * w[yy]).sum() / w[yy].sum())
    check("loss = mean -log P(y) (unweighted) and sum w_y(-log P(y)) / sum w_y (weighted, as F.cross_entropy)",
          abs(float(h.loss(o, yy)) - manual_u) < 1e-9 and abs(float(h.loss(o, yy, w)) - manual_w) < 1e-9)

    # 5 target encoding: the gradient pushes the severity score towards the true stage
    signs = []
    for k in range(K):
        for g0, want in ((-20.0, -1), (20.0, 1)):
            gg = torch.tensor([g0], dtype=torch.float64, requires_grad=True)
            hk = head_with(-2.0, [0.0, 0.0, 0.0, 0.0])
            L = hk.loss(out_of(hk, gg), torch.tensor([k]))
            L.backward()
            if (k == 0 and want == -1) or (k == K - 1 and want == 1):
                continue                                  # already on the correct side: gradient ~0 there
            signs.append(np.sign(float(gg.grad)) == want)  # gradient descent then moves g towards stage k
    check("gradient wrt the severity score has the right sign for every stage (score too low -> increase it, too high "
          "-> decrease it)", all(signs), f"{sum(signs)}/{len(signs)} cases")

    # 6 finite-difference gradient check (double precision) over score, tau0 and threshold increments
    hg = head_with(0.2, [0.1, -0.3, 0.4, 0.0])
    yv = torch.tensor([0, 1, 2, 3, 4, 5, 2, 4])
    wv = torch.tensor([1.0, 1.2, 0.8, 1.5, 1.0, 0.9], dtype=torch.float64)

    def f(g, tau0, raw):
        steps = torch.nn.functional.softplus(raw) + 1e-4
        tau = torch.cat([tau0, tau0 + torch.cumsum(steps, 0)])
        return hg.loss(tau.unsqueeze(0) - g.unsqueeze(1), yv, wv)

    inputs = (torch.as_tensor(rng.normal(0, 2, 8), dtype=torch.float64).requires_grad_(),
              torch.tensor([0.2], dtype=torch.float64, requires_grad=True),
              torch.as_tensor(rng.normal(0, 1, K - 2), dtype=torch.float64).requires_grad_())
    check("torch.autograd.gradcheck of the weighted loss wrt scores, tau0 and threshold increments",
          torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-7))

    # 7 uniform initialisation without labels
    h0 = CumulativeLinkHead(4).double()
    p0 = h0.probabilities(out_of(h0, torch.zeros(1, dtype=torch.float64)))
    check("initial thresholds give uniform stage probabilities (1/6) at g = 0", float((p0 - 1 / K).abs().max()) < 1e-3,
          f"{np.round(p0.detach().numpy().ravel(), 4).tolist()}")

    # 8 edge cases: batch of one, all-one-class batch, boundary stages, float32 path
    h32 = CumulativeLinkHead(4)
    o1 = h32(torch.randn(1, 4))
    ok = o1.shape == (1, K - 1) and h32.probabilities(o1).shape == (1, K)
    for k in (0, K - 1):
        ok &= bool(torch.isfinite(h32.loss(h32(torch.randn(5, 4)), torch.full((5,), k))))
    ok &= bool(torch.isfinite(h32.loss(h32(torch.randn(7, 4)), torch.full((7,), 3), torch.ones(K))))
    check("edge cases: batch of one, single-class batches at both boundary stages, weights, float32", ok)

    # 9 decision rules
    # cumulative of row 1: .45 .47 .49 .51 -> median 3 although argmax is 5; row 2: .1 .2 .5 -> 2; row 3 -> 5
    pp = torch.tensor([[0.45, 0.02, 0.02, 0.02, 0.02, 0.47], [0.1, 0.1, 0.3, 0.3, 0.1, 0.1], [0, 0, 0, 0, 0, 1.0]],
                      dtype=torch.float64)
    md = median_decode(pp).tolist()
    hm = head_with(-1.0, [0.3, -0.5, 0.2, 0.8])
    om = out_of(hm, torch.linspace(-6, 6, 50, dtype=torch.float64))
    same = bool(torch.equal(hm.decode_median(om), median_decode(hm.probabilities(om))))
    check("median rule: first stage with cumulative probability >= 0.5, equal to threshold counting for one model",
          md == [3, 2, 5] and same, f"median of examples {md}")

    # 10 order recovery on synthetic data with a known latent ordering
    n, d = 1200, 5
    X = torch.as_tensor(rng.normal(0, 1, (n, d)))
    wtrue = torch.as_tensor(rng.normal(0, 1, d))
    s = X @ wtrue + torch.as_tensor(rng.logistic(0, 0.5, n))
    cuts = torch.quantile(s, torch.tensor([1, 2, 3, 4, 5], dtype=torch.float64) / 6)
    ys = torch.bucketize(s, cuts)
    lin = torch.nn.Linear(d, 1).double()
    hs = CumulativeLinkHead(d).double()
    opt = torch.optim.Adam(list(lin.parameters()) + [hs.tau0, hs.raw_steps], lr=0.05)
    tr, te = slice(0, 900), slice(900, n)
    for _ in range(800):
        opt.zero_grad()
        L = hs.loss(hs.thresholds().unsqueeze(0) - lin(X[tr]).squeeze(1).unsqueeze(1), ys[tr])
        L.backward()
        opt.step()
    with torch.no_grad():
        g_te = lin(X[te]).squeeze(1)
        o_te = hs.thresholds().unsqueeze(0) - g_te.unsqueeze(1)
        pred = hs.decode_argmax(o_te)
        oracle = torch.bucketize(X[te] @ wtrue, cuts)                     # noise-free latent score
        acc, acc_oracle = float((pred == ys[te]).double().mean()), float((oracle == ys[te]).double().mean())
        rho = spearmanr(g_te.numpy(), ys[te].numpy()).correlation
        mae = float((pred - ys[te]).abs().double().mean())
    check("synthetic ordered data: learned thresholds increasing, severity score ordered with the true stage, accuracy "
          "close to the noise-free oracle", bool((torch.diff(hs.thresholds()) > 0).all()) and rho > 0.85
          and acc > acc_oracle - 0.08,
          f"Spearman {rho:.3f}; accuracy {acc:.3f} vs oracle {acc_oracle:.3f}; MAE {mae:.2f} stages")

    n_fail = sum(not r["pass"] for r in results)
    lines.append(f"\n{len(results)} tests, {n_fail} failed")
    print(lines[-1])
    json.dump({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "torch": torch.__version__, "results": results},
              open(os.path.join(OUT, "test_results.json"), "w"), indent=2)
    open(os.path.join(OUT, "test_results.log"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
