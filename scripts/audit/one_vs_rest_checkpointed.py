"""
One-vs-rest multivariate permutation summary, with durable checkpointing.

Statistically identical to scripts/audit/one_vs_rest_summary.py. Two changes:

1. ALGEBRAIC SPEEDUP (same statistic, not an approximation)
   For a two-group split the PERMANOVA pseudo-F reduces to a form in which
   only one quantity varies across permutations.

   With grand mean m, group means m_g and m_r, sizes n_g + n_r = n:

       m_g - m = (n_r/n)(m_g - m_r)        m_r - m = (n_g/n)(m_r - m_g)
       SSB = n_g||m_g-m||^2 + n_r||m_r-m||^2
           = (n_g n_r / n) * ||m_g - m_r||^2
       SST = SSB + SSW  (constant across permutations)
       =>  SSW = SST - SSB
       =>  F   = SSB / ((SST - SSB) / (n-2))

   So each permutation needs only s_g = sum of the stage group's rows;
   m_g = s_g/n_g and m_r = (S - s_g)/n_r with S the fixed total sum. That
   touches n_g rows instead of making several passes over all n, and SST is
   computed once. The returned F is the same number to floating point.

2. CHECKPOINTING
   Every completed test is appended atomically (tmp file + os.replace) to a
   JSONL checkpoint carrying the cohort fingerprint, permutation count and
   seed. On restart, entries are revalidated against the current cohort and
   configuration; only matching entries are reused.

Usage:
    python scripts/audit/one_vs_rest_checkpointed.py <out_dir> [--verify-only]
"""
import argparse
import csv
import hashlib
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")

STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
FEATS = ["mALFF", "mReHo", "DC_z", "FC"]
N_PERM = 9999
SEED = 20261013
ALPHA = 0.05
Q_FLOOR = 1.0 / (N_PERM + 1)


def log(m=""):
    print(m, flush=True)


def bh(p):
    p = np.asarray(p, float)
    n = p.size
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def cohort_fingerprint(y, subs):
    """Identifies the exact cohort; a different cohort invalidates reuse."""
    h = hashlib.sha256()
    h.update(",".join(sorted(subs)).encode())
    h.update(b"|")
    h.update(",".join(map(str, y.tolist())).encode())
    return h.hexdigest()[:16]


def pseudo_f_fast(Z, mask, S, SST, n):
    """Two-group PERMANOVA pseudo-F. See module docstring for the identity."""
    n_g = int(mask.sum())
    n_r = n - n_g
    if n_g < 2 or n_r < 2:
        return np.nan
    s_g = Z[mask].sum(0)
    diff = s_g / n_g - (S - s_g) / n_r
    ssb = (n_g * n_r / n) * float(diff @ diff)
    ssw = SST - ssb
    if ssw <= 0:
        return np.nan
    return ssb / (ssw / (n - 2))


def pseudo_f_reference(Z, mask):
    """Literal definition, used only to prove the fast form agrees."""
    n = Z.shape[0]
    g, r = Z[mask], Z[~mask]
    if len(g) < 2 or len(r) < 2:
        return np.nan
    grand = Z.mean(0)
    ssb = len(g) * ((g.mean(0) - grand) ** 2).sum() + \
          len(r) * ((r.mean(0) - grand) ** 2).sum()
    ssw = ((g - g.mean(0)) ** 2).sum() + ((r - r.mean(0)) ** 2).sum()
    if ssw <= 0:
        return np.nan
    return ssb / (ssw / (n - 2))


def load_cohort():
    reg = np.load(os.path.join(SRC, "X_regional.npy"))
    fc = np.load(os.path.join(SRC, "X_fc.npy"), mmap_mode="r")
    y = np.load(os.path.join(SRC, "y.npy"))
    subs = np.load(os.path.join(SRC, "subjects.npy"))
    iu = np.triu_indices(246, 1)
    X = {n: reg[:, :, i].astype(np.float64) for i, n in enumerate(FEATS[:3])}
    X["FC"] = np.asarray(fc[:, iu[0], iu[1]], dtype=np.float64)
    del reg, fc
    return X, y, subs


def zscore(A):
    sd = A.std(0)
    Z = (A - A.mean(0)) / np.where(sd > 0, sd, 1.0)
    return np.nan_to_num(Z)


# ------------------------------------------------------------ checkpoint
def ckpt_path(out_dir):
    return os.path.join(out_dir, "one_vs_rest_checkpoint.jsonl")


def load_ckpt(out_dir, fingerprint):
    """Reuse only entries matching this cohort AND this configuration."""
    p = ckpt_path(out_dir)
    done, rejected = {}, 0
    if not os.path.isfile(p):
        return done, 0
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                rejected += 1
                continue
            if (r.get("cohort_fingerprint") != fingerprint
                    or r.get("n_permutations") != N_PERM
                    or r.get("seed") != SEED
                    or r.get("status") != "COMPLETED"
                    or r.get("p_value") is None):
                rejected += 1
                continue
            done[(r["biomarker"], r["stage"])] = r      # last valid wins
    return done, rejected


def append_ckpt(out_dir, rec):
    """Atomic append: rewrite via tmp + os.replace so no partial line survives."""
    p = ckpt_path(out_dir)
    tmp = p + ".tmp"
    old = ""
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            old = fh.read()
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(old)
        if old and not old.endswith("\n"):
            fh.write("\n")
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)


def main(out_dir, verify_only=False):
    os.makedirs(out_dir, exist_ok=True)
    X, y, subs = load_cohort()
    n = len(y)
    fp = cohort_fingerprint(y, subs)

    log("=" * 74)
    log("ONE-VS-REST (checkpointed)")
    log("=" * 74)
    log(f"cohort {n} acquisitions / {len(set(subs))} subjects   "
        f"fingerprint {fp}")
    for i, s in enumerate(STAGES):
        log(f"   {s:5s} n = {int((y == i).sum()):3d}")
    log(f"permutations {N_PERM}   seed {SEED}   dims " +
        ", ".join(f"{k}={v.shape[1]}" for k, v in X.items()))

    # --- prove the fast statistic equals the literal definition -----------
    log("\nvalidating the fast statistic against the literal definition:")
    rng0 = np.random.default_rng(0)
    worst = 0.0
    for bm in FEATS:
        Z = zscore(X[bm])
        S, n_ = Z.sum(0), Z.shape[0]
        g = Z.mean(0)
        SST = float(((Z - g) ** 2).sum())
        for _ in range(3):
            m = rng0.permutation(y == 0)
            a = pseudo_f_fast(Z, m, S, SST, n_)
            b = pseudo_f_reference(Z, m)
            worst = max(worst, abs(a - b) / max(abs(b), 1e-12))
        log(f"   {bm:6s} max relative difference {worst:.3e}")
    if worst > 1e-9:
        raise SystemExit(f"FAST STATISTIC DISAGREES (rel {worst:.3e}) -- aborting")
    log(f"   agreement within {worst:.3e} -- the speedup is exact\n")
    if verify_only:
        return

    done, rejected = load_ckpt(out_dir, fp)
    log(f"checkpoint: {len(done)} reusable, {rejected} rejected "
        f"(wrong cohort/config/incomplete)\n")

    rows = []
    for bm in FEATS:
        Z = zscore(X[bm])
        S = Z.sum(0)
        SST = float(((Z - Z.mean(0)) ** 2).sum())
        for si, s in enumerate(STAGES):
            if (bm, s) in done:
                rows.append(done[(bm, s)])
                log(f"  {bm:6s} {s:5s}  SKIP (checkpointed)  "
                    f"p={done[(bm,s)]['p_value']:.4f}")
                continue
            t0 = time.time()
            mask = (y == si)
            try:
                obs = pseudo_f_fast(Z, mask, S, SST, n)
                rng = np.random.default_rng(SEED)
                null = np.empty(N_PERM)
                for k in range(N_PERM):
                    null[k] = pseudo_f_fast(Z, rng.permutation(mask), S, SST, n)
                p = (np.sum(null >= obs) + 1) / (N_PERM + 1)
                rec = dict(biomarker=bm, stage=s, n_stage=int(mask.sum()),
                           n_rest=int((~mask).sum()), pseudo_F=float(obs),
                           null_mean=float(np.nanmean(null)),
                           null_p95=float(np.nanpercentile(null, 95)),
                           p_value=float(p), n_permutations=N_PERM, seed=SEED,
                           cohort_fingerprint=fp, n_features=int(Z.shape[1]),
                           status="COMPLETED", error=None,
                           seconds=round(time.time() - t0, 1))
            except Exception as e:                           # noqa: BLE001
                rec = dict(biomarker=bm, stage=s, n_stage=int(mask.sum()),
                           n_rest=int((~mask).sum()), pseudo_F=None,
                           null_mean=None, null_p95=None, p_value=None,
                           n_permutations=N_PERM, seed=SEED,
                           cohort_fingerprint=fp, n_features=int(Z.shape[1]),
                           status="FAILED", error=f"{type(e).__name__}: {e}",
                           seconds=round(time.time() - t0, 1))
            append_ckpt(out_dir, rec)
            rows.append(rec)
            if rec["status"] == "COMPLETED":
                log(f"  {bm:6s} {s:5s}  n={rec['n_stage']:3d} vs "
                    f"{rec['n_rest']:3d}  F={rec['pseudo_F']:7.4f}  "
                    f"null={rec['null_mean']:6.4f}  p={rec['p_value']:.4f}  "
                    f"{rec['seconds']}s")
            else:
                log(f"  {bm:6s} {s:5s}  FAILED: {rec['error']}")

    ok = [r for r in rows if r["status"] == "COMPLETED"]
    log(f"\ncompleted {len(ok)}/24")
    if len(ok) < 24:
        log("INCOMPLETE -- not applying FDR and not writing the summary figure.")
        for r in rows:
            if r["status"] != "COMPLETED":
                log(f"   pending/failed: {r['biomarker']} {r['stage']} {r['error']}")
        return rows

    q = bh([r["p_value"] for r in rows])
    for r, qq in zip(rows, q):
        r["q_value_BH"] = float(qq)
        r["significant_q05"] = bool(qq < ALPHA)
    with open(os.path.join(out_dir, "one_vs_rest_statistics.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    for name, key in [("one_vs_rest_p_values.csv", "p_value"),
                      ("one_vs_rest_q_values.csv", "q_value_BH")]:
        M = np.array([r[key] for r in rows]).reshape(len(FEATS), len(STAGES))
        with open(os.path.join(out_dir, name), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["biomarker"] + STAGES)
            for i, bm in enumerate(FEATS):
                w.writerow([bm] + [f"{v:.6g}" for v in M[i]])
    log(f"\nwrote statistics/p/q CSVs to {os.path.relpath(out_dir, ROOT)}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    main(a.out_dir, a.verify_only)
