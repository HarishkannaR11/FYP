"""
Validation for the P-score hierarchical clustering outputs.

Every check recomputes from the source q-values or inspects the files on disk.
Nothing is read from a log. A failure is reported and the script exits non-zero.
"""
import csv
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")
V2 = os.path.join(ROOT, "derivatives", "INTRA_INTER_CLASS_HEATMAPS", "v2_cohort116")
BASE = os.path.join(V2, "zscore_and_pscore_heatmaps")
CSVD = os.path.join(BASE, "csv")
OUT = os.path.join(BASE, "pscore_hierarchical")

STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
BIOMARKERS = ["mALFF", "mReHo", "DC_z", "FC"]
N_ROI = 246
ALPHA = 0.05
THRESH = -np.log10(ALPHA)
FLOOR = 1e-12
PSCORE_VMIN, PSCORE_VMAX = 0.0, 1.40

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


def read_csv_matrix(path):
    rows = list(csv.DictReader(open(path)))
    ids = [int(r["ROI_ID"]) for r in rows]
    names = [r["ROI_NAME"] for r in rows]
    M = np.full((len(rows), len(STAGES)), np.nan)
    blanks = 0
    for i, r in enumerate(rows):
        for j, s in enumerate(STAGES):
            if r[s] in ("", "nan", "NaN", "NA"):
                blanks += 1
            else:
                M[i, j] = float(r[s])
    return M, ids, names, rows, blanks


# ------------------------------------------------------------- cohort / stages
y = np.load(os.path.join(SRC, "y.npy"))
subjects = np.load(os.path.join(SRC, "subjects.npy")).astype(str)
meta = json.load(open(os.path.join(SRC, "metadata.json")))
check("cohort is 116 unique subjects", len(subjects) == 116
      and len(set(subjects.tolist())) == 116, f"n = {len(subjects)}")
check("stage order is the established clinical order",
      meta["stage_order"] == STAGES, str(meta["stage_order"]))
check("threshold constant is correct for q = 0.05",
      abs(THRESH - 1.3010299956639813) < 1e-12, f"-log10(0.05) = {THRESH:.13f}")

# ------------------------------------------------------- P-score matrices
for bm in BIOMARKERS:
    P, ids, names, rows, blanks = read_csv_matrix(
        os.path.join(CSVD, f"pscore_matrix_{bm}.csv"))
    check(f"{bm}: P-score matrix is 246 x 6", P.shape == (N_ROI, len(STAGES)),
          str(P.shape))
    check(f"{bm}: ROI_IDs are 1..246 in atlas order",
          ids == list(range(1, N_ROI + 1)))
    hdr = list(csv.reader(open(os.path.join(CSVD, f"pscore_matrix_{bm}.csv"))))[0]
    check(f"{bm}: columns are the six stages in order", hdr[2:] == STAGES,
          str(hdr[2:]))

    Q, qids, _, _, qblanks = read_csv_matrix(
        os.path.join(CSVD, f"roi_stage_qvalues_{bm}.csv"))
    expect = -np.log10(np.where(np.isnan(Q), np.nan, np.maximum(Q, FLOOR)))
    expect[expect == 0] = 0.0
    check(f"{bm}: P-score equals -log10(q) from the source q-values",
          np.allclose(P, expect, atol=1e-12, equal_nan=True),
          f"max diff {np.nanmax(np.abs(P - expect)):.2e}")
    check(f"{bm}: missing values stay missing, never become 0",
          blanks == qblanks, f"{blanks} blank in P-score, {qblanks} blank in q")
    check(f"{bm}: no P-score is negative", bool(np.nanmin(P) >= 0.0),
          f"min = {np.nanmin(P):.6f}")
    check(f"{bm}: observed P-scores fall inside the fixed colour scale",
          bool(np.nanmax(P) <= PSCORE_VMAX),
          f"max = {np.nanmax(P):.4f} <= {PSCORE_VMAX}")
    n_sig_q = int(np.nansum(Q < ALPHA))
    n_sig_p = int(np.nansum(P > THRESH))
    check(f"{bm}: significance by q and by P-score agree",
          n_sig_q == n_sig_p, f"q < {ALPHA}: {n_sig_q}; P > {THRESH:.3f}: {n_sig_p}")
    check(f"{bm}: no ROI x stage cell claims q < 0.05", n_sig_q == 0,
          f"max P-score {np.nanmax(P):.4f} < threshold {THRESH:.4f}")

    # source q-values untouched by this task
    srcQ, _, _, _, _ = read_csv_matrix(os.path.join(V2, f"{bm}_roi_stage_qvalues.csv"))
    check(f"{bm}: source q-values still match the original analysis",
          np.allclose(Q, srcQ, atol=1e-12, equal_nan=True))

# ------------------------------------------------------- clustering order CSV
rows = list(csv.DictReader(open(os.path.join(CSVD, "pscore_clustering_order.csv"))))
check("clustering order CSV has 4 x 246 rows", len(rows) == 4 * N_ROI, f"{len(rows)}")
for bm in BIOMARKERS:
    sub = [r for r in rows if r["biomarker"] == bm]
    check(f"{bm}: clustering order covers all 246 ROIs", len(sub) == N_ROI)
    check(f"{bm}: order is a permutation of 0..245",
          sorted(int(r["original_index"]) for r in sub) == list(range(N_ROI)))
    check(f"{bm}: display rows are 0..245 contiguous",
          [int(r["display_row"]) for r in sub] == list(range(N_ROI)))
    check(f"{bm}: leaf index matches display row",
          all(r["dendrogram_leaf_index"] == r["display_row"] for r in sub))

# ------------------------------------------------------- figures
STEMS = ([f"pscore_hierarchical_{b}" for b in BIOMARKERS]
         + ["pscore_hierarchical_combined", "effectsize_hierarchical_alternative"])
bad = []
for stem in STEMS:
    for ext, magic in ((".png", b"\x89PNG"), (".pdf", b"%PDF-")):
        p = os.path.join(OUT, stem + ext)
        if not os.path.exists(p):
            bad.append(f"{stem}{ext}: missing")
            continue
        with open(p, "rb") as f:
            head = f.read(len(magic))
        if head != magic or os.path.getsize(p) < 5000:
            bad.append(f"{stem}{ext}: size={os.path.getsize(p)} magic={head!r}")
check(f"all {len(STEMS)} figures exist as valid non-empty PNG and PDF", not bad,
      "; ".join(bad) if bad else f"{len(STEMS) * 2} files")

newcsv = [f"pscore_matrix_{b}.csv" for b in BIOMARKERS] + ["pscore_clustering_order.csv"]
bad = [c for c in newcsv
       if not os.path.exists(os.path.join(CSVD, c))
       or os.path.getsize(os.path.join(CSVD, c)) < 200]
check(f"all {len(newcsv)} new CSVs exist and are non-empty", not bad, "; ".join(bad))
check("README exists",
      os.path.exists(os.path.join(OUT, "PSCORE_HIERARCHICAL_README.md")))

# ------------------------------------------------------- nothing else modified
NEW = set()
for d, _, fs in os.walk(OUT):
    NEW |= {os.path.join(d, f) for f in fs}
NEW |= {os.path.join(CSVD, c) for c in newcsv}
NEW.add(os.path.join(BASE, "logs", "pscore_hierarchical.log"))
NEW = {os.path.abspath(p) for p in NEW if os.path.exists(p)}

baseline = min(os.path.getmtime(p) for p in NEW)
touched = []
for d, _, fs in os.walk(V2):
    if in_sibling(d):
        continue
    for f in fs:
        p = os.path.abspath(os.path.join(d, f))
        if p in NEW:
            continue
        # User-created archives of these outputs are not results; bundling the
        # folder for sharing must not read as a modified statistical result.
        if f.lower().endswith((".zip", ".7z", ".tar", ".gz")):
            continue
        if os.path.getmtime(p) >= baseline:
            touched.append(os.path.relpath(p, ROOT))
check("no pre-existing file in v2_cohort116 was modified", not touched,
      "; ".join(touched[:5]) if touched else
      f"{len(NEW)} new files; everything else older")

parent = os.path.dirname(V2)
touched2 = [os.path.relpath(os.path.join(d, f), ROOT)
            for d, _, fs in os.walk(parent) for f in fs
            if not os.path.abspath(d).startswith(os.path.abspath(V2))
            and os.path.getmtime(os.path.join(d, f)) >= baseline]
check("earlier-cohort outputs outside v2_cohort116 untouched", not touched2,
      "; ".join(touched2[:5]) if touched2 else "preserved")

for name in ("HEATMAP_README.md", "HEATMAP_README.v1_uncorrected.md",
             "FINAL_STATISTICAL_AUDIT.md", "one_vs_rest_checkpoint.jsonl",
             "one_vs_rest_statistics.csv"):
    check(f"preserved: {name}", os.path.exists(os.path.join(V2, name)))

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
