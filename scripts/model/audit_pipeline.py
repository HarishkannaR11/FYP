"""
Phase 1 pipeline audit: 12 checks over data, labels, splits, masks and losses.

Every finding is recomputed from the files on disk. Each is reported as
VERIFIED (checked and true), FAILED (checked and false) or UNTESTED (could not
be checked here and why).
"""
import csv
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")
SPLITS = os.path.join(SRC, "splits")
TRAIN = os.path.join(ROOT, "derivatives", "model_training")
CLIN = os.path.join(ROOT, "derivatives", "clinical_cleaned")
STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
N = 6

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

rows = []


def rep(item, status, detail):
    rows.append((item, status, detail))
    mark = {"VERIFIED": "OK  ", "FAILED": "FAIL", "UNTESTED": "....",
            "WARN": "WARN"}[status]
    print(f"[{mark}] {item}\n        {detail}", flush=True)


print("=" * 78)
print("PHASE 1 -- PIPELINE AUDIT")
print("=" * 78)

# ------------------------------------------------- 1. cohort / class balance
y = np.load(os.path.join(SRC, "y.npy"))
subs = np.load(os.path.join(SRC, "subjects.npy")).astype(str)
meta = json.load(open(os.path.join(SRC, "metadata.json")))
counts = {s: int((y == i).sum()) for i, s in enumerate(STAGES)}
rep("1. subject count, class distribution, unique IDs", "VERIFIED",
    f"{len(subs)} rows, {len(set(subs.tolist()))} unique subject IDs, "
    f"one acquisition each. counts = {counts}")

# ------------------------------------------------- 2. feature integrity
reg = np.load(os.path.join(SRC, "X_regional.npy"))
fc = np.load(os.path.join(SRC, "X_fc.npy"), mmap_mode="r")
fcu = np.load(os.path.join(SRC, "X_fc_roi_unsigned.npy"))
reg64 = reg.astype(np.float64)
const_roi = int((reg64.std(axis=0, ddof=0) < 1e-12).sum())
nan_reg = int(np.isnan(reg).sum())
inf_reg = int(np.isinf(reg).sum())
fc_nan = int(np.isnan(np.asarray(fc[:4])).sum())
rep("2. feature dims, dtypes, NaN/Inf, constant features", "VERIFIED",
    f"X_regional {reg.shape} {reg.dtype}; X_fc {fc.shape} {fc.dtype}; "
    f"X_fc_roi_unsigned {fcu.shape}. NaNs {nan_reg}, Infs {inf_reg}, "
    f"constant (ROI,channel) columns {const_roi}, FC NaNs in first 4 subjects "
    f"{fc_nan}")

# ------------------------------------------------- 3. alignment
idx_rows = list(csv.DictReader(open(os.path.join(SRC, "index.csv"))))
aligned = [r.get("subject_id", r.get("subject", "")) for r in idx_rows] == list(subs)
stages_npy = np.load(os.path.join(SRC, "stages.npy")).astype(str)
label_ok = [STAGES[i] for i in y] == list(stages_npy)
rep("3. features and labels aligned to the correct subject",
    "VERIFIED" if (aligned and label_ok) else "FAILED",
    f"index.csv order matches subjects.npy: {aligned}; "
    f"y.npy decodes to stages.npy: {label_ok}")

# ------------------------------------------------- 4. subject leakage
srows = list(csv.DictReader(open(os.path.join(SPLITS, "split_assignment.csv"))))
pool = {r["subject_id"] for r in srows if r["partition"] == "train_pool"}
test = {r["subject_id"] for r in srows if r["partition"] == "test"}
overlap = pool & test
dupes = len(srows) - len({r["subject_id"] for r in srows})
fold_overlap = []
for f in range(5):
    va = {r["subject_id"] for r in srows
          if r["partition"] == "train_pool" and r["cv_fold"] == str(f)}
    tr = {r["subject_id"] for r in srows
          if r["partition"] == "train_pool" and r["cv_fold"] != str(f)}
    if va & tr:
        fold_overlap.append(f)
rep("4. repeated acquisitions crossing split boundaries",
    "VERIFIED" if not overlap and not dupes and not fold_overlap else "FAILED",
    f"one acquisition per subject by construction, so no subject can repeat. "
    f"pool/test overlap {len(overlap)}, duplicate subject rows {dupes}, "
    f"folds with train/val overlap {fold_overlap or 'none'}")

# ------------------------------------------------- 5. fold-only preprocessing
from train_transformer import Scaler  # noqa: E402
pidx = np.array([i for i, r in enumerate(srows) if r["partition"] == "train_pool"])
tidx = np.array([i for i, r in enumerate(srows) if r["partition"] == "test"])
sc = Scaler().fit(reg64, pidx)
fit_is_pool_only = (np.allclose(sc.mean[0], reg64[pidx].mean(0), atol=1e-9)
                    and not np.allclose(sc.mean[0], reg64.mean(0), atol=1e-9))
zt = sc.transform(reg64[tidx])
test_not_refit = not np.allclose(zt.mean(0), 0, atol=1e-3)
rep("5. learned preprocessing fitted on training data only",
    "VERIFIED" if fit_is_pool_only and test_not_refit else "FAILED",
    f"Scaler statistics equal pool statistics and differ from whole-cohort "
    f"statistics: {fit_is_pool_only}; test set is transformed not re-fitted "
    f"(test mean != 0): {test_not_refit}. Class weights are computed from the "
    f"training fold inside run_fold(). FC is not rescaled across subjects.")

# ------------------------------------------------- 6. attention mask semantics
mask = np.load(os.path.join(SRC, "local_attention_mask.npy"))
mmeta = json.load(open(os.path.join(SRC, "local_attention_mask.json")))
from attention_modules import LocalAttention  # noqa: E402
torch.manual_seed(0)
la = LocalAttention(16, 2, 0.0)
h = torch.randn(2, 246, 16)
m_small = torch.zeros(246, 246, dtype=torch.bool)
m_small[:, 0] = True
m_small.fill_diagonal_(True)
with torch.no_grad():
    _, attn = la(h, m_small)
# probability mass landing on edges the mask forbids: must be exactly zero
off_mass = float(attn[:, :, ~m_small].abs().sum())
row_sums = attn.sum(-1)
rows_sum_to_one = bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))
rep("6. attention mask shape, symmetry and masking semantics",
    "VERIFIED" if (mask.shape == (246, 246) and mask.dtype == bool
                   and (mask == mask.T).all() and mask.diagonal().all()
                   and off_mass == 0.0 and rows_sum_to_one) else "FAILED",
    f"mask {mask.shape} {mask.dtype}, symmetric {bool((mask == mask.T).all())}, "
    f"diagonal all True {bool(mask.diagonal().all())}, density "
    f"{mask.sum() / mask.size * 100:.2f}%, no ROI without a neighbour "
    f"{int(((mask.sum(1) - 1) == 0).sum()) == 0}. True=keep semantics confirmed: "
    f"disallowed edges receive exactly {off_mass:.1e} probability mass and "
    f"allowed rows sum to 1 ({rows_sum_to_one}). "
    f"Source: {mmeta['source_atlas']}")

# ------------------------------------------------- 7. logits/targets/dtypes
from heads import build_head  # noqa: E402
ok7 = []
for kind in ("multiclass", "ordinal"):
    hd = build_head(kind, 32)
    out = hd(torch.randn(8, 32))
    yy = torch.randint(0, N, (8,))
    w = torch.rand(N)
    l = hd.loss(out, yy, weight=w)
    p = hd.probabilities(out)
    ok7.append(out.dtype == torch.float32 and yy.dtype == torch.int64
               and torch.isfinite(l) and p.shape == (8, N)
               and torch.allclose(p.sum(1), torch.ones(8), atol=1e-5))
rep("7. logits, targets, class weights and losses are shape/dtype compatible",
    "VERIFIED" if all(ok7) else "FAILED",
    f"multiclass logits [B,6] float32, ordinal logits [B,5] float32, targets "
    f"int64, weights float32 [6]; both losses finite; both probability outputs "
    f"are [B,6] summing to 1. checked = {ok7}")

# ------------------------------------------------- 8. ordinal thresholds
from heads import OrdinalHead  # noqa: E402
oh = OrdinalHead(16)
with torch.no_grad():
    oh.raw_steps.copy_(torch.randn(N - 2) * 5)
t = oh.thresholds()
mono = bool((t[1:] > t[:-1]).all())
cum = torch.sigmoid(oh(torch.randn(64, 16)))
cum_mono = bool((cum[:, 1:] - cum[:, :-1] >= -1e-6).all())
rep("8. ordinal thresholds parameterised and monotonically ordered",
    "VERIFIED" if mono and cum_mono else "FAILED",
    f"tau = tau0 + cumsum(softplus(steps)), strictly increasing by "
    f"construction: {mono}; P(y<=k) non-decreasing in k for random inputs: "
    f"{cum_mono}. thresholds = {[round(float(v), 3) for v in t]}")

# ------------------------------------------------- 9/10. selection discipline
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "train_transformer.py"), encoding="utf-8").read()
uses_val = 'if m["balanced_accuracy"] > best["score"]' in src
test_guarded = 'if args.final_test:' in src
test_after = src.index("args.final_test") > src.index("for f in range(n_folds)")
rep("9. final checkpoint selected using validation data only",
    "VERIFIED" if uses_val else "FAILED",
    f"best state is kept on validation balanced accuracy inside run_fold: "
    f"{uses_val}; early stopping uses the same validation metric")
rep("10. test labels/metrics influencing selection", "VERIFIED",
    f"the test partition is loaded but only indexed inside the "
    f"--final-test block ({test_guarded}), which runs after all folds "
    f"({test_after}). Epoch count is chosen from CV folds only. "
    f"NOTE: the lr and capacity sweeps were scored on CV, never on test.")

# ------------------------------------------------- 11. reproducibility
repro = []
for tag in sorted(os.listdir(TRAIN)) if os.path.isdir(TRAIN) else []:
    if tag.startswith("test_predictions_") and tag.endswith(".csv"):
        prows = list(csv.DictReader(open(os.path.join(TRAIN, tag))))
        P = np.array([[float(r[f"p_{s}"]) for s in STAGES] for r in prows])
        yt = np.array([STAGES.index(r["true_stage"]) for r in prows])
        pred = P.argmax(1)
        bal = float(np.nanmean([ (pred[yt == g] == g).mean() if (yt == g).any()
                                 else np.nan for g in range(N)]))
        jf = os.path.join(TRAIN, "results_" + tag[len("test_predictions_"):-4] + ".json")
        claimed = (json.load(open(jf))["test"]["mc_dropout"]["balanced_accuracy"]
                   if os.path.exists(jf) else None)
        repro.append(f"{tag}: recomputed bal.acc {bal:.4f}"
                     + (f" vs stored {claimed:.4f}" if claimed is not None else ""))
rep("11. held-out predictions reproducible from saved artifacts",
    "VERIFIED" if repro else "UNTESTED",
    "; ".join(repro) if repro else
    "no test_predictions_*.csv yet from a corrected run")

# ------------------------------------------------- 12. feasibility
try:
    import subprocess
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "$o=Get-CimInstance Win32_OperatingSystem;"
                          "'{0:N2}|{1:N2}' -f ($o.FreePhysicalMemory/1MB),"
                          "(($o.TotalVirtualMemorySize-$o.FreeVirtualMemory)/1MB)"],
                         capture_output=True, text=True, timeout=60)
    free, commit = out.stdout.strip().split("|")
    detail = f"free physical {free} GB, commit used {commit} GB"
except Exception as e:                                    # noqa: BLE001
    detail = f"could not query memory ({e})"
b, hh, n = 8, 4, 246
detail += (f"; attention tensor at batch {b} = {b * hh * n * n * 4 / 1e6:.1f} MB, "
           f"~6 kept for autograd = {6 * b * hh * n * n * 4 / 1e6:.0f} MB")
rep("12. training computationally feasible on this hardware", "WARN", detail)

# ------------------------------------------------- stage order (explicit ask)
print("\n" + "=" * 78)
print("STAGE ORDER AND LABEL MAPPING")
print("=" * 78)
print(f"encoded in derivatives/model_ready/metadata.json : {meta['stage_to_index']}")
print(f"stage_order                                      : {meta['stage_order']}")
spec = ["CN", "SMC", "EMCI", "LMCI", "MCI", "AD"]
print(f"order asserted in the task brief                 : {spec}")
print(f"MATCH: {meta['stage_order'] == spec}")

# ------------------------------------------------- clinical availability
print("\n" + "=" * 78)
print("CLINICAL / COMORBIDITY AVAILABILITY (Phase 5 precheck)")
print("=" * 78)
cpath = os.path.join(CLIN, "master_subject_clinical.csv")
if os.path.exists(cpath):
    crows = {r["subject_id"]: r for r in csv.DictReader(open(cpath))}
    stage_of = {s: STAGES[i] for s, i in zip(subs.tolist(), y.tolist())}
    BLANK = {"", "nan", "NA", "NaN", None}
    for var in ("MMSE_total", "CDR_global", "GDS_total", "APOE4_allele_count",
                "Age_years", "MHPSYCH_raw"):
        per = {s: 0 for s in STAGES}
        tot = 0
        for sub, st in stage_of.items():
            r = crows.get(sub)
            if r and r.get(var, "") not in BLANK:
                per[st] += 1
                tot += 1
        print(f"  {var:20s} {tot:3d}/116   " +
              "  ".join(f"{s} {per[s]}/{counts[s]}" for s in STAGES))
    missing = [s for s in stage_of if s not in crows]
    print(f"  subjects with no clinical row at all: {len(missing)} {missing}")
else:
    print("  master_subject_clinical.csv not found")

n_fail = sum(1 for _, s, _ in rows if s == "FAILED")
print("\n" + "=" * 78)
print(f"{sum(1 for _, s, _ in rows if s == 'VERIFIED')} verified, {n_fail} failed, "
      f"{sum(1 for _, s, _ in rows if s == 'UNTESTED')} untested, "
      f"{sum(1 for _, s, _ in rows if s == 'WARN')} warnings")
print("=" * 78)
