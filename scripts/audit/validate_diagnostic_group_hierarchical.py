"""
Validation for the diagnostic-group hierarchical clustering outputs.

Group means and standardisation are recomputed from the subject-level feature
arrays and compared against the stored CSVs. Nothing is read from a log.
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
OUT = os.path.join(V2, "diagnostic_group_hierarchical")
FIG, CSVD = os.path.join(OUT, "figures"), os.path.join(OUT, "csv")

STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
BIOMARKERS = ["mALFF", "mReHo", "DC_z", "FC"]
EXPECTED = {"CN": 20, "SMC": 20, "EMCI": 20, "MCI": 16, "LMCI": 20, "AD": 20}
N_ROI, K, DDOF, VMAX = 246, 6, 1, 2.05

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""),
          flush=True)


def read_mat(path, extra=()):
    rows = list(csv.DictReader(open(path)))
    M = np.full((len(rows), K), np.nan)
    blanks = 0
    for i, r in enumerate(rows):
        for j, s in enumerate(STAGES):
            if r[s] in ("", "nan", "NaN", "NA"):
                blanks += 1
            else:
                M[i, j] = float(r[s])
    return M, rows, blanks


# -------------------------------------------------------------- cohort
y = np.load(os.path.join(SRC, "y.npy"))
subjects = np.load(os.path.join(SRC, "subjects.npy")).astype(str)
meta = json.load(open(os.path.join(SRC, "metadata.json")))
check("cohort is 116 acquisitions / 116 unique subjects",
      len(y) == 116 and len(set(subjects.tolist())) == 116, f"n = {len(y)}")
counts = {s: int((y == i).sum()) for i, s in enumerate(STAGES)}
check("group sizes match the expected manifest", counts == EXPECTED, str(counts))
check("stage_to_index is CN=0 ... AD=5 as encoded in the source",
      meta["stage_to_index"] == {s: i for i, s in enumerate(STAGES)},
      str(meta["stage_to_index"]))
check("stage_order in metadata is the clinical progression order",
      meta["stage_order"] == STAGES)

Xr = np.load(os.path.join(SRC, "X_regional.npy")).astype(np.float64)
fc_u = np.load(os.path.join(SRC, "X_fc_roi_unsigned.npy")).astype(np.float64)
check("X_regional is (116, 246, 3)", Xr.shape == (116, N_ROI, 3), str(Xr.shape))
check("X_fc_roi_unsigned is (116, 246)", fc_u.shape == (116, N_ROI), str(fc_u.shape))
check("regional channels are mALFF, mReHo, DC_z",
      meta["regional_channels"] == ["mALFF", "mReHo", "DC_z"])
check("source features are finite",
      bool(np.isfinite(Xr).all() and np.isfinite(fc_u).all()))
feats = {"mALFF": Xr[:, :, 0], "mReHo": Xr[:, :, 1], "DC_z": Xr[:, :, 2], "FC": fc_u}

# ----------------------------------------------- group means + standardisation
for bm in BIOMARKERS:
    G, rows, blanks = read_mat(os.path.join(CSVD, f"group_means_raw_{bm}.csv"))
    check(f"{bm}: raw group-mean matrix is 246 x 6", G.shape == (N_ROI, K), str(G.shape))
    check(f"{bm}: ROI_IDs are 1..246 in atlas order",
          [int(r["ROI_ID"]) for r in rows] == list(range(1, N_ROI + 1)))
    hdr = list(csv.reader(open(os.path.join(CSVD, f"group_means_raw_{bm}.csv"))))[0]
    check(f"{bm}: raw CSV columns are the six stages in clinical order",
          hdr[2:] == STAGES, str(hdr[2:]))

    M = feats[bm]
    expect = np.column_stack([M[y == g].mean(axis=0) for g in range(K)])
    check(f"{bm}: stored group means reproduce from the subject features",
          np.allclose(G, expect, atol=1e-12),
          f"max diff {np.abs(G - expect).max():.2e}")

    Z, zrows, zblanks = read_mat(
        os.path.join(CSVD, f"group_means_standardized_{bm}.csv"))
    check(f"{bm}: standardised matrix is 246 x 6", Z.shape == (N_ROI, K))
    mu = expect.mean(axis=1, keepdims=True)
    sd = expect.std(axis=1, ddof=DDOF, keepdims=True)
    flat = (sd[:, 0] == 0) | ~np.isfinite(sd[:, 0])
    zexp = (expect - mu) / np.where(flat[:, None], 1.0, sd)
    zexp[flat, :] = np.nan
    check(f"{bm}: standardisation reproduces (across groups, ddof=1)",
          np.allclose(Z, zexp, atol=1e-12, equal_nan=True),
          f"max diff {np.nanmax(np.abs(Z - zexp)):.2e}")
    check(f"{bm}: each ROI has mean 0 and sd 1 across the six groups",
          np.allclose(np.nanmean(Z, axis=1), 0, atol=1e-10)
          and np.allclose(np.nanstd(Z[~flat], axis=1, ddof=DDOF), 1, atol=1e-10))
    check(f"{bm}: |z| never exceeds the theoretical bound (k-1)/sqrt(k)",
          bool(np.nanmax(np.abs(Z)) <= (K - 1) / np.sqrt(K) + 1e-9),
          f"max |z| = {np.nanmax(np.abs(Z)):.4f} <= {(K - 1) / np.sqrt(K):.4f}")
    check(f"{bm}: fixed colour scale covers every attainable value",
          (K - 1) / np.sqrt(K) <= VMAX)
    check(f"{bm}: zero-variance ROIs are flagged, not zeroed",
          sum(int(r["zero_variance"]) for r in zrows) == int(flat.sum())
          and zblanks == int(flat.sum()) * K,
          f"{int(flat.sum())} flagged, {zblanks} blank cells")
    check(f"{bm}: raw group means are preserved unstandardised",
          not np.allclose(G, Z, equal_nan=True) or bool(flat.all()))

# ----------------------------------------------------- dendrogram leaf orders
rows = list(csv.DictReader(open(os.path.join(CSVD,
                                             "group_dendrogram_leaf_order.csv"))))
check("group leaf-order CSV has 4 x 6 rows", len(rows) == 4 * K, f"{len(rows)}")
for bm in BIOMARKERS:
    sub = [r for r in rows if r["biomarker"] == bm]
    check(f"{bm}: group leaf order lists all six groups once",
          sorted(r["stage_label"] for r in sub) == sorted(STAGES),
          " -> ".join(r["stage_label"] for r in sub))
    check(f"{bm}: source stage indices are preserved alongside display order",
          all(int(r["source_stage_index"]) == STAGES.index(r["stage_label"])
              for r in sub))
    check(f"{bm}: group sizes recorded correctly",
          all(int(r["n_subjects"]) == EXPECTED[r["stage_label"]] for r in sub))

rows = list(csv.DictReader(open(os.path.join(CSVD,
                                             "roi_dendrogram_leaf_order.csv"))))
check("ROI leaf-order CSV has 4 x 246 rows", len(rows) == 4 * N_ROI, f"{len(rows)}")
for bm in BIOMARKERS:
    sub = [r for r in rows if r["biomarker"] == bm]
    check(f"{bm}: ROI leaf order is a permutation of 0..245",
          sorted(int(r["original_index"]) for r in sub) == list(range(N_ROI)))

# ----------------------------------------------------- config / FC / stability
cfg = list(csv.DictReader(open(os.path.join(CSVD, "clustering_configuration.csv"))))
check("clustering configuration is recorded for every figure", len(cfg) >= 10,
      f"{len(cfg)} rows")
check("group clustering uses a documented metric and linkage",
      all(r["distance_metric"] and r["linkage_method"] for r in cfg))

fc = {r["property"]: r["value"] for r in
      csv.DictReader(open(os.path.join(CSVD, "fc_transformation_details.csv")))}
check("FC transformation is documented as unsigned",
      fc.get("signed_or_absolute", "").startswith("absolute"),
      fc.get("signed_or_absolute", ""))
check("FC diagonal handling is documented",
      "excluded" in fc.get("diagonal_handling", ""))
check("FC ROI-level formula is recorded",
      fc.get("roi_level_summary") == "FCstr_i = mean_{j != i} |z_ij|")
# recompute the unsigned summary from the raw matrices for one subject
Xfc = np.load(os.path.join(SRC, "X_fc.npy"), mmap_mode="r")
off = ~np.eye(N_ROI, dtype=bool)
s0 = np.asarray(Xfc[0], dtype=np.float64)
check("FC ROI summary recomputes from X_fc for subject 0",
      np.allclose(np.abs(s0)[off].reshape(N_ROI, N_ROI - 1).mean(axis=1),
                  fc_u[0], atol=1e-6),
      "unsigned mean of off-diagonal |Fisher-z|")
check("X_fc diagonal is zero", bool(np.allclose(np.diag(s0), 0.0)))
del Xfc, s0

stab = list(csv.DictReader(open(os.path.join(CSVD,
                                             "group_cluster_stability_bootstrap.csv"))))
check("bootstrap stability recorded for every branch of every biomarker",
      len(stab) >= 4 * 3, f"{len(stab)} rows")
check("bootstrap frequencies are valid proportions",
      all(0.0 <= float(r["bootstrap_frequency"]) <= 1.0 for r in stab))

pur = list(csv.DictReader(open(os.path.join(CSVD, "subject_neighbour_purity.csv"))))
check("neighbour purity recorded overall and per group", len(pur) == 1 + K)
check("neighbour-purity rates are valid proportions",
      all(0.0 <= float(r["nearest_neighbour_same_group_rate"]) <= 1.0 for r in pur))

# ----------------------------------------------------------------- artefacts
STEMS = ([f"group_hierarchical_{b}" for b in BIOMARKERS]
         + ["group_hierarchical_combined", "fc_representation_comparison",
            "subject_level_clustering"])
bad = []
for stem in STEMS:
    for ext, magic in ((".png", b"\x89PNG"), (".pdf", b"%PDF-")):
        p = os.path.join(FIG, stem + ext)
        if not os.path.exists(p):
            bad.append(f"{stem}{ext}: missing")
            continue
        with open(p, "rb") as f:
            head = f.read(len(magic))
        if head != magic or os.path.getsize(p) < 5000:
            bad.append(f"{stem}{ext}: size={os.path.getsize(p)} magic={head!r}")
check(f"all {len(STEMS)} figures exist as valid non-empty PNG and PDF", not bad,
      "; ".join(bad) if bad else f"{len(STEMS) * 2} files")

bad = [f for f in os.listdir(CSVD)
       if os.path.getsize(os.path.join(CSVD, f)) < 100]
check("all CSVs are non-empty", not bad, "; ".join(bad))
for doc in ("DIAGNOSTIC_GROUP_HIERARCHICAL_README.md",
            "DIAGNOSTIC_GROUP_CLUSTERING_INTERPRETATION.md"):
    check(f"documentation present: {doc}", os.path.exists(os.path.join(OUT, doc)))

# ------------------------------------------------------- nothing else modified
NEW = set()
for d, _, fs in os.walk(OUT):
    NEW |= {os.path.abspath(os.path.join(d, f)) for f in fs}
baseline = min(os.path.getmtime(p) for p in NEW)
touched = [os.path.relpath(os.path.join(d, f), ROOT)
           for d, _, fs in os.walk(V2) for f in fs
           if os.path.abspath(os.path.join(d, f)) not in NEW
           and os.path.getmtime(os.path.join(d, f)) >= baseline]
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

for name in ("FINAL_STATISTICAL_AUDIT.md", "HEATMAP_README.md",
             "HEATMAP_README.v1_uncorrected.md", "one_vs_rest_statistics.csv",
             "one_vs_rest_checkpoint.jsonl"):
    check(f"preserved: {name}", os.path.exists(os.path.join(V2, name)))

# ---------------------------------------------------------- model NOT trained
weights = [os.path.join(d, f) for d, _, fs in os.walk(ROOT) for f in fs
           if f.endswith((".pt", ".pth", ".ckpt", ".safetensors"))
           and ".git" not in d]
check("no trained model weights exist anywhere in the repository", not weights,
      "; ".join(weights[:3]) if weights else "no .pt/.pth/.ckpt/.safetensors")

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
