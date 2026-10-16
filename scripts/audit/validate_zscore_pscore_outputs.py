"""
Validation for the Z-score / P-score heatmap outputs.

Every check recomputes from the source data or the source CSVs. Nothing is
copied from a log. A failed check is reported as FAIL and the script exits
non-zero; nothing is repaired silently.
"""
import csv
import hashlib
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")
V2 = os.path.join(ROOT, "derivatives", "INTRA_INTER_CLASS_HEATMAPS", "v2_cohort116")
OUT = os.path.join(V2, "zscore_and_pscore_heatmaps")
CSVD = os.path.join(OUT, "csv")

STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
BIOMARKERS = ["mALFF", "mReHo", "DC_z", "FC"]
EXPECTED = {"CN": 20, "SMC": 20, "EMCI": 20, "MCI": 16, "LMCI": 20, "AD": 20}
N_ROI = 246
DDOF = 1

# Sibling analyses under v2_cohort116 each own their outputs and carry their
# own validator; they are out of scope for this one's untouched-files check.
SIBLING_ANALYSES = ("diagnostic_group_hierarchical",)


def in_sibling(path):
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(V2))
    return rel.split(os.sep)[0] in SIBLING_ANALYSES


results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""),
          flush=True)


def read_mat(path, cols=STAGES):
    rows = list(csv.DictReader(open(path)))
    M = np.array([[float(r[c]) for c in cols] for r in rows], dtype=np.float64)
    return M, rows


# ---------------------------------------------------------------- cohort
y = np.load(os.path.join(SRC, "y.npy"))
subjects = np.load(os.path.join(SRC, "subjects.npy")).astype(str)
meta = json.load(open(os.path.join(SRC, "metadata.json")))

check("cohort has 116 acquisitions", len(y) == 116, f"n = {len(y)}")
check("cohort has 116 unique subjects", len(set(subjects.tolist())) == 116,
      f"unique = {len(set(subjects.tolist()))}")
check("one acquisition per subject", len(subjects) == len(set(subjects.tolist())))
counts = {s: int((y == i).sum()) for i, s in enumerate(STAGES)}
check("stage counts match the expected cohort", counts == EXPECTED, str(counts))
check("stage order comes from the data, not reassigned",
      meta["stage_order"] == STAGES, str(meta["stage_order"]))
check("stage_to_index is the project order",
      meta["stage_to_index"] == {s: i for i, s in enumerate(STAGES)})

# ------------------------------------------------- Z-score matrices match source
Xr = np.load(os.path.join(SRC, "X_regional.npy")).astype(np.float64)
fc_u = np.load(os.path.join(SRC, "X_fc_roi_unsigned.npy")).astype(np.float64)
feats = {"mALFF": Xr[:, :, 0].T, "mReHo": Xr[:, :, 1].T,
         "DC_z": Xr[:, :, 2].T, "FC": fc_u.T}

for bm in BIOMARKERS:
    path = os.path.join(CSVD, f"zscore_{bm}_roi_by_subject.csv")
    rows = list(csv.DictReader(open(path)))
    check(f"{bm}: Z-score CSV has 246 ROI rows", len(rows) == N_ROI, f"{len(rows)} rows")
    cols = [c for c in rows[0] if c not in ("ROI_ID", "ROI_NAME", "zero_variance")]
    check(f"{bm}: Z-score CSV has 116 subject columns", len(cols) == 116,
          f"{len(cols)} columns")
    check(f"{bm}: Z-score CSV columns are the cohort subjects, in order",
          cols == list(subjects))

    M = feats[bm]
    expect = (M - M.mean(1, keepdims=True)) / M.std(1, ddof=DDOF, keepdims=True)
    got = np.array([[float(r[c]) if r[c] != "" else np.nan for c in cols] for r in rows])
    # CSV is stored to 6 decimal places
    check(f"{bm}: stored Z-scores reproduce from the source features",
          np.allclose(got, expect, atol=2e-6, equal_nan=True),
          f"max diff {np.nanmax(np.abs(got - expect)):.2e}")
    check(f"{bm}: per-ROI Z-score mean is 0 and sd is 1",
          np.allclose(expect.mean(1), 0, atol=1e-10)
          and np.allclose(expect.std(1, ddof=DDOF), 1, atol=1e-10))
    zv = sum(int(r["zero_variance"]) for r in rows)
    check(f"{bm}: zero-variance ROIs are flagged, not silently zeroed",
          zv == int((M.std(1, ddof=DDOF) == 0).sum()), f"{zv} flagged")

    # clustering order mappings
    sp = os.path.join(CSVD, f"clustering_order_{bm}_subjects.csv")
    srows = list(csv.DictReader(open(sp)))
    check(f"{bm}: subject order CSV maps all 116 display columns", len(srows) == 116)
    oidx = [int(r["original_index"]) for r in srows]
    check(f"{bm}: subject order is a permutation of 0..115",
          sorted(oidx) == list(range(116)))
    check(f"{bm}: subject order CSV carries correct subject, stage and index",
          all(r["subject_id"] == subjects[int(r["original_index"])]
              and r["stage"] == STAGES[y[int(r["original_index"])]]
              for r in srows))
    rp = os.path.join(CSVD, f"clustering_order_{bm}_rois.csv")
    rrows = list(csv.DictReader(open(rp)))
    check(f"{bm}: ROI order CSV maps all 246 display rows", len(rrows) == N_ROI)
    check(f"{bm}: ROI order is a permutation of 0..245",
          sorted(int(r["original_index"]) for r in rrows) == list(range(N_ROI)))

# ------------------------------------------------- ROI x stage statistics
for bm in BIOMARKERS:
    P, _ = read_mat(os.path.join(CSVD, f"roi_stage_rawp_{bm}.csv"))
    Q, _ = read_mat(os.path.join(CSVD, f"roi_stage_qvalues_{bm}.csv"))
    G, _ = read_mat(os.path.join(CSVD, f"roi_stage_effectsize_hedges_g_{bm}.csv"))
    NLQ, _ = read_mat(os.path.join(CSVD, f"roi_stage_neglog10q_{bm}.csv"))
    NLP, _ = read_mat(os.path.join(CSVD, f"roi_stage_neglog10p_{bm}.csv"))
    check(f"{bm}: ROI x stage matrices are 246 x 6",
          P.shape == Q.shape == G.shape == (N_ROI, 6), str(P.shape))
    check(f"{bm}: ROI x stage family is 1476", P.size == 1476)

    srcP, _ = read_mat(os.path.join(V2, f"{bm}_roi_stage_pvalues.csv"))
    srcQ, _ = read_mat(os.path.join(V2, f"{bm}_roi_stage_qvalues.csv"))
    srcG, _ = read_mat(os.path.join(V2, f"{bm}_roi_stage_effectsizes.csv"))
    check(f"{bm}: copied p-values match the source analysis exactly",
          np.array_equal(P, srcP))
    check(f"{bm}: copied q-values match the source analysis exactly",
          np.array_equal(Q, srcQ))
    check(f"{bm}: copied effect sizes match the source analysis exactly",
          np.array_equal(G, srcG))

    check(f"{bm}: q >= p everywhere (p and q are not interchanged)",
          bool((Q >= P - 1e-9).all()),
          f"min(q - p) = {float((Q - P).min()):.2e}")
    check(f"{bm}: -log10(q) matches the q matrix",
          np.allclose(NLQ, -np.log10(np.maximum(Q, 1e-12)), atol=1e-5))
    check(f"{bm}: -log10(p) matches the p matrix",
          np.allclose(NLP, -np.log10(np.maximum(P, 1e-12)), atol=1e-5))
    check(f"{bm}: p and q values are in [0, 1]",
          bool(((P >= 0) & (P <= 1) & (Q >= 0) & (Q <= 1)).all()))
    check(f"{bm}: no value is missing or NaN",
          bool(np.isfinite(P).all() and np.isfinite(Q).all() and np.isfinite(G).all()))
    check(f"{bm}: no ROI x stage cell claims q < 0.05", int((Q < 0.05).sum()) == 0,
          f"min q = {Q.min():.4f}")

# ------------------------------------------------- overall 4 x 6
rows = list(csv.DictReader(open(os.path.join(CSVD, "overall_summary_4x6.csv"))))
check("overall summary has exactly 24 rows", len(rows) == 24, f"{len(rows)} rows")
check("overall summary covers 4 biomarkers x 6 stages",
      sorted({r["biomarker"] for r in rows}) == sorted(BIOMARKERS)
      and sorted({r["stage"] for r in rows}) == sorted(STAGES))

src = {(r["biomarker"], r["stage"]): r for r in
       csv.DictReader(open(os.path.join(V2, "one_vs_rest_statistics.csv")))}
ok_p = ok_q = True
for r in rows:
    s = src[(r["biomarker"], r["stage"])]
    ok_p &= abs(float(r["p_value_permutation"]) - float(s["p_value"])) < 1e-9
    ok_q &= abs(float(r["q_value_BH_24family"]) - float(s["q_value_BH"])) < 1e-9
check("overall p-values match one_vs_rest_statistics.csv", ok_p)
check("overall q-values match one_vs_rest_statistics.csv", ok_q)
check("overall summary claims no significance",
      all(r["significant_q05"] == "False" for r in rows))
check("all 24 source tests are COMPLETED",
      all(s["status"] == "COMPLETED" for s in src.values()))
check("all 24 source tests share one cohort fingerprint",
      len({s["cohort_fingerprint"] for s in src.values()}) == 1,
      list({s["cohort_fingerprint"] for s in src.values()})[0])

# ------------------------------------------------- figures exist and are valid
EXPECT_FIGS = (
    [f"zscore_hierarchical/zscore_hierarchical_{b}" for b in BIOMARKERS]
    + ["zscore_hierarchical/fc_stage_mean_connectivity_246x246"]
    + [f"roi_stage_significance/pscore_{b}_roi_stage" for b in BIOMARKERS]
    + [f"roi_stage_significance/pscore_{b}_roi_stage_rawp" for b in BIOMARKERS]
    + ["overall_summary/overall_zscore_significance_heatmap",
       "overall_summary/combined_summary_panels"]
)
bad = []
for stem in EXPECT_FIGS:
    for ext, magic in ((".png", b"\x89PNG"), (".pdf", b"%PDF-")):
        p = os.path.join(OUT, stem + ext)
        if not os.path.exists(p):
            bad.append(f"{stem}{ext}: missing")
            continue
        size = os.path.getsize(p)
        with open(p, "rb") as f:
            head = f.read(len(magic))
        if head != magic or size < 5000:
            bad.append(f"{stem}{ext}: size={size} magic={head!r}")
check(f"all {len(EXPECT_FIGS)} figures exist as valid non-empty PNG and PDF",
      not bad, "; ".join(bad) if bad else f"{len(EXPECT_FIGS) * 2} files")

# ------------------------------------------------- originals untouched
BASELINE = os.path.getmtime(os.path.join(ROOT, "scripts", "audit",
                                         "zscore_hierarchical_heatmaps.py"))
touched = []
for dirpath, dirnames, filenames in os.walk(V2):
    if os.path.abspath(dirpath).startswith(os.path.abspath(OUT)):
        continue
    if in_sibling(dirpath):
        continue
    for fn in filenames:
        p = os.path.join(dirpath, fn)
        if os.path.getmtime(p) > BASELINE:
            touched.append(os.path.relpath(p, ROOT))
check("no pre-existing result file in v2_cohort116 was modified", not touched,
      "; ".join(touched[:5]) if touched else "checked all files outside the new folder")

parent = os.path.dirname(V2)
touched2 = [os.path.relpath(os.path.join(d, f), ROOT)
            for d, _, fs in os.walk(parent) for f in fs
            if not os.path.abspath(d).startswith(os.path.abspath(V2))
            and os.path.getmtime(os.path.join(d, f)) > BASELINE]
check("no earlier-cohort result outside v2_cohort116 was modified", not touched2,
      "; ".join(touched2[:5]) if touched2 else "earlier analysis preserved")

for name in ("HEATMAP_README.md", "HEATMAP_README.v1_uncorrected.md",
             "FINAL_STATISTICAL_AUDIT.md", "one_vs_rest_checkpoint.jsonl"):
    check(f"preserved: {name}", os.path.exists(os.path.join(V2, name)))

# ------------------------------------------------- summary
n_fail = sum(1 for _, ok, _ in results if not ok)
print("\n" + "=" * 70)
print(f"{len(results) - n_fail}/{len(results)} checks passed")
print("=" * 70)
if n_fail:
    for name, ok, detail in results:
        if not ok:
            print(f"  FAILED: {name}  {detail}")
    sys.exit(1)
print("All validation checks passed.")
