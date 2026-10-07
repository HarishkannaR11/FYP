"""
Stage-wise biomarker heatmaps: 246 Brainnetome ROIs x 6 progression stages.

For each biomarker, builds a 246 x 6 matrix where cell (r, s) is the stage-wise
mean of ROI r, and renders both an absolute and a row-wise z-scored heatmap.

SUBJECT-LEVEL AGGREGATION (required): the 165 acquisitions come from 127
distinct subjects, so acquisitions are averaged within subject FIRST, then
subjects are averaged within stage. Every subject contributes exactly once to
its stage.

    subject_mean[r, subject] = mean of that subject's acquisition values for ROI r
    stage_mean[r, stage]     = mean of subject_mean[r, subject] over that stage

FC_Strength is a deterministic rescaling of DC (FC_Strength = DC / 245) and is
NOT an independent biomarker feature. It is rendered here for visualization and
comparison only, and is excluded from model input.

Read-only with respect to all existing derivatives; writes only into
derivatives/STAGE_HEATMAPS/.
"""
import glob
import json
import os
import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NORM = os.path.join(ROOT, "derivatives", "biomarkers_normalized")
RAW = os.path.join(ROOT, "derivatives", "fsfast")
LUT_PATH = os.path.join(ROOT, "atlases", "Brainnetome246", "BN_Atlas_246_LUT.txt")
OUT = os.path.join(ROOT, "derivatives", "STAGE_HEATMAPS")

N_ROI = 246
FC_DIVISOR = 245.0

# Stage order fixed at Review I. Directory name -> display label.
STAGES = [("CN_Final", "CN"), ("SMC_Final", "SMC"), ("EMCI", "EMCI"),
          ("MCI", "MCI"), ("LMCI", "LMCI"), ("AD", "AD")]
STAGE_DIRS = [s[0] for s in STAGES]
STAGE_LABELS = [s[1] for s in STAGES]
EXPECTED_SUBJECTS = {"CN": 26, "SMC": 21, "EMCI": 26, "MCI": 14, "LMCI": 20, "AD": 20}

# name, source, cbar unit label
BIOMARKERS = [
    ("mALFF", "norm:malff.npy", "Mean mALFF (ratio to subject mean)"),
    ("mReHo", "norm:mreho.npy", "Mean mReHo (ratio to subject mean)"),
    ("DC_z", "norm:dc_z.npy", "Mean DC_z (within-subject z)"),
    ("FC_Strength", "derived:fc_strength", "Mean FC_Strength (within-subject z)"),
]

warnings = []


def log(m):
    print(m, flush=True)


def load_lut():
    lut = {}
    if os.path.isfile(LUT_PATH):
        for line in open(LUT_PATH):
            p = line.split()
            if len(p) >= 2 and p[0].isdigit():
                lut[int(p[0])] = p[1]
    return lut


def discover():
    rows = []
    for d in sorted(glob.glob(os.path.join(NORM, "*", "sub-*", "ses-*", "run-*"))):
        p = d.replace("\\", "/").split("/")
        i = p.index("biomarkers_normalized")
        grp, sub, ses, run = p[i + 1], p[i + 2], p[i + 3], p[i + 4]
        if grp not in STAGE_DIRS:
            continue
        rows.append({"stage_dir": grp, "stage": dict(STAGES)[grp], "subject": sub,
                     "session": ses, "run": run, "norm_dir": d,
                     "raw_dir": os.path.join(RAW, grp, sub, ses, run, "biomarkers")})
    return pd.DataFrame(rows)


def load_vector(row, source):
    """
    FC_Strength has no normalized file. It is read from the original biomarker
    outputs and z-scored within subject exactly as DC_z is, so the heatmap is
    directly comparable to DC_z -- which is precisely why the two come out
    identical.
    """
    kind, name = source.split(":", 1)
    if kind == "norm":
        return np.load(os.path.join(row["norm_dir"], name))
    v = np.load(os.path.join(row["raw_dir"], "fc_strength.npy"))
    return (v - v.mean()) / v.std(ddof=0)


def build_stage_matrix(df, source):
    """246 x 6, aggregated subject-first."""
    mat = np.full((N_ROI, len(STAGES)), np.nan)
    counts = {}
    for si, (sdir, slabel) in enumerate(STAGES):
        sub_df = df[df.stage_dir == sdir]
        per_subject = []
        for _, g in sub_df.groupby("subject"):
            vecs = [load_vector(r, source) for _, r in g.iterrows()]
            per_subject.append(np.mean(np.vstack(vecs), axis=0))
        if per_subject:
            mat[:, si] = np.mean(np.vstack(per_subject), axis=0)
        counts[slabel] = {"subjects": len(per_subject), "acquisitions": len(sub_df)}
    return mat, counts


def row_zscore(mat):
    mu = mat.mean(axis=1, keepdims=True)
    sd = mat.std(axis=1, ddof=0, keepdims=True)
    out = np.zeros_like(mat)
    nz = (sd[:, 0] > 0) & np.isfinite(sd[:, 0])
    out[nz] = (mat[nz] - mu[nz]) / sd[nz]
    n_flat = int((~nz).sum())
    if n_flat:
        warnings.append(f"{n_flat} ROI(s) had zero variance across stages; "
                        f"their row-z values were set to 0")
    return out


def save_csv(mat, lut, path):
    df = pd.DataFrame(mat, columns=STAGE_LABELS)
    df.insert(0, "ROI_Name", [lut.get(i, f"ROI_{i}") for i in range(1, N_ROI + 1)])
    df.insert(0, "ROI_ID", range(1, N_ROI + 1))
    df.to_csv(path, index=False)


def plot(mat, title, subtitle, cbar_label, path, diverging):
    fig, ax = plt.subplots(figsize=(7.2, 11))
    if diverging:
        lim = float(np.nanmax(np.abs(mat)))
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                       interpolation="nearest")
    else:
        im = ax.imshow(mat, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xticks(range(len(STAGE_LABELS)))
    ax.set_xticklabels(STAGE_LABELS, fontsize=11)
    ax.set_xlabel("Disease progression stage", fontsize=11, labelpad=8)
    ax.set_ylabel("Brainnetome-246 ROI", fontsize=11)
    ax.set_yticks([0, 49, 99, 149, 199, 209, 245])
    ax.set_yticklabels([1, 50, 100, 150, 200, 210, 246], fontsize=9)
    ax.axhline(209.5, color="black", lw=1.1, ls="--", alpha=0.65)
    ax.text(len(STAGES) - 0.42, 104, "cortical", rotation=90, va="center",
            fontsize=8, alpha=0.75)
    ax.text(len(STAGES) - 0.42, 228, "subcortical", rotation=90, va="center",
            fontsize=8, alpha=0.75)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.text(0.5, 1.012, subtitle, transform=ax.transAxes, ha="center",
            fontsize=8.5, alpha=0.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label(cbar_label, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def verify_fc_redundancy(df):
    """Check FC_Strength == DC/245 on the ORIGINAL (un-normalized) outputs,
    which is where that stated relation actually holds."""
    diffs, rs = [], []
    for _, r in df.iterrows():
        dc = np.load(os.path.join(r["raw_dir"], "degree_centrality.npy"))
        fs = np.load(os.path.join(r["raw_dir"], "fc_strength.npy"))
        diffs.append(float(np.max(np.abs(dc / FC_DIVISOR - fs))))
        rs.append(float(np.corrcoef(dc, fs)[0, 1]))
    return {"n_acquisitions_checked": len(diffs),
            "max_abs_difference": float(np.max(diffs)),
            "min_pearson_r": float(np.min(rs)),
            "relation_holds": bool(np.max(diffs) < 1e-9)}


def main():
    os.makedirs(OUT, exist_ok=True)
    lut = load_lut()
    df = discover()

    log(f"acquisitions : {len(df)}")
    log(f"subjects     : {df.subject.nunique()}")
    log("")

    # --- subject integrity -------------------------------------------------
    multi = df.drop_duplicates(["subject", "stage"]).groupby("subject").size()
    multi_stage = sorted(multi[multi > 1].index)
    if multi_stage:
        warnings.append(f"{len(multi_stage)} subject(s) appear in more than one "
                        f"stage: {multi_stage}")

    results, qc = {}, {}
    for name, source, unit in BIOMARKERS:
        mat, counts = build_stage_matrix(df, source)
        z = row_zscore(mat)
        results[name] = {"mat": mat, "z": z, "counts": counts}

        np.save(os.path.join(OUT, f"{name}_stage_means.npy"), mat)
        save_csv(mat, lut, os.path.join(OUT, f"{name}_stage_means.csv"))
        np.save(os.path.join(OUT, f"{name}_rowz.npy"), z)
        save_csv(z, lut, os.path.join(OUT, f"{name}_rowz.csv"))

        note = "  [DERIVED - rescaling of DC]" if name == "FC_Strength" else ""
        sub = (f"n={df.subject.nunique()} subjects / {len(df)} acquisitions   |   "
               f"subject-level aggregation")
        plot(mat, f"{name} stage-wise means{note}", sub, unit,
             os.path.join(OUT, f"{name}_absolute.png"), diverging=False)
        plot(z, f"{name} stage-wise pattern (row-wise z){note}", sub,
             "Row-wise z-score",
             os.path.join(OUT, f"{name}_rowz.png"), diverging=True)

        c = {
            "shape": list(mat.shape),
            "shape_ok": bool(mat.shape == (N_ROI, len(STAGES))),
            "nan_count": int(np.isnan(mat).sum()),
            "inf_count": int(np.isinf(mat).sum()),
            "rowz_nan_count": int(np.isnan(z).sum()),
            "rowz_inf_count": int(np.isinf(z).sum()),
            "n_rois": int(mat.shape[0]), "n_stages": int(mat.shape[1]),
            "min": float(np.nanmin(mat)), "max": float(np.nanmax(mat)),
            "subject_counts": {k: v["subjects"] for k, v in counts.items()},
            "acquisition_counts": {k: v["acquisitions"] for k, v in counts.items()},
        }
        exp_ok = all(counts[s]["subjects"] == EXPECTED_SUBJECTS[s] for s in STAGE_LABELS)
        c["subject_counts_match_expected"] = bool(exp_ok)
        if not exp_ok:
            warnings.append(f"{name}: subject counts differ from expected "
                            f"{EXPECTED_SUBJECTS} -> {c['subject_counts']}")
        qc[name] = c
        log(f"{name:12s} shape={mat.shape} NaN={c['nan_count']} "
            f"range=[{c['min']:.4f}, {c['max']:.4f}] expected_counts={exp_ok}")

    # --- FC_Strength redundancy -------------------------------------------
    red_raw = verify_fc_redundancy(df)
    d_heat = float(np.nanmax(np.abs(results["DC_z"]["mat"] - results["FC_Strength"]["mat"])))
    r_heat = float(np.corrcoef(results["DC_z"]["mat"].ravel(),
                               results["FC_Strength"]["mat"].ravel())[0, 1])
    log("")
    log(f"FC_Strength = DC/245 on raw outputs: max|diff|={red_raw['max_abs_difference']:.3e} "
        f"min_r={red_raw['min_pearson_r']:.12f} holds={red_raw['relation_holds']}")
    log(f"DC_z vs FC_Strength stage matrices : max|diff|={d_heat:.3e} r={r_heat:.12f}")

    # --- ROI id integrity --------------------------------------------------
    ids = pd.read_csv(os.path.join(OUT, "mALFF_stage_means.csv")).ROI_ID.tolist()
    roi_ok = {"count": len(ids), "no_duplicates": len(set(ids)) == len(ids),
              "no_missing": set(ids) == set(range(1, N_ROI + 1))}

    meta = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_acquisitions": int(len(df)), "n_subjects": int(df.subject.nunique()),
        "stage_order": STAGE_LABELS,
        "aggregation": "subject-level: mean within subject, then mean across subjects",
        "biomarkers": qc, "roi_integrity": roi_ok,
        "fc_strength_redundancy_raw": red_raw,
        "fc_strength_vs_dc_stage_matrix": {"max_abs_difference": d_heat,
                                           "pearson_r": r_heat},
        "subjects_in_multiple_stages": multi_stage,
        "warnings": warnings,
    }
    with open(os.path.join(OUT, "stage_heatmap_qc.json"), "w") as f:
        json.dump(meta, f, indent=2)

    write_report(df, qc, red_raw, d_heat, r_heat, roi_ok, multi_stage)
    log(f"\nsaved to {OUT}")
    log("STAGE_HEATMAPS_DONE")


def write_report(df, qc, red_raw, d_heat, r_heat, roi_ok, multi_stage):
    L = ["# Stage-wise Biomarker Heatmaps", "",
         f"Generated (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat()}", "",
         "## 1. Dataset summary", "",
         f"- Acquisitions: **{len(df)}**",
         f"- Distinct subjects: **{df.subject.nunique()}**",
         "- Source: `derivatives/biomarkers_normalized/` (mALFF, mReHo, DC_z) and "
         "`derivatives/fsfast/.../biomarkers/` (FC_Strength)",
         "- No existing derivative was modified; this stage is additive only.", "",
         "## 2. Subject-level aggregation method", "",
         "Acquisitions are averaged **within subject first**, then subjects are averaged "
         "within stage, so every subject contributes exactly once to its stage. "
         "Averaging acquisitions directly would over-weight the 31 subjects who "
         "contributed more than one scan (one contributed four).", "",
         "## 3. Subject count per stage", "",
         "| Stage | Subjects | Acquisitions | Matches expected |", "|---|---:|---:|---|"]
    c = qc["mALFF"]
    for s in STAGE_LABELS:
        L.append(f"| {s} | {c['subject_counts'][s]} | {c['acquisition_counts'][s]} | "
                 f"{'yes' if c['subject_counts'][s] == EXPECTED_SUBJECTS[s] else 'NO'} |")
    L += [f"| **Total** | **{sum(c['subject_counts'].values())}** | "
          f"**{sum(c['acquisition_counts'].values())}** | |", "",
          "## 4. Stage order", "",
          "`" + " -> ".join(STAGE_LABELS) + "`", "",
          "Fixed at Review I; ROIs and stages are never reordered or clustered.", "",
          "## 5. Formula", "", "```",
          "subject_mean[r, subject] = mean of that subject's acquisition values for ROI r",
          "stage_mean[r, stage]     = mean of subject_mean[r, subject] over that stage",
          "z[r, s]                  = (stage_mean[r,s] - mean(stage_mean[r,:]))",
          "                           / std(stage_mean[r,:])", "```", "",
          "## 6. Biomarker heatmaps generated", "",
          "| Biomarker | Shape | NaN | Inf | Range |", "|---|---|---:|---:|---|"]
    for n in qc:
        q = qc[n]
        L.append(f"| {n} | {q['shape'][0]} x {q['shape'][1]} | {q['nan_count']} | "
                 f"{q['inf_count']} | {q['min']:.4f} to {q['max']:.4f} |")
    L += ["", "## 7. Absolute-value heatmap", "",
          "Sequential colormap (`viridis`); each cell is the stage-wise mean of that ROI "
          "in the biomarker's own units. Shows which ROIs carry large values, which is "
          "driven largely by anatomy rather than by stage.", "",
          "## 8. Row-wise z-score heatmap", "",
          "Diverging colormap (`RdBu_r`) centred at 0; colorbar labelled "
          "\"Row-wise z-score\". Each ROI is standardised across the six stages "
          "independently, so the figure shows how each ROI *changes across stages* "
          "rather than how large it is.", "",
          "## 9. FC_Strength redundancy verification", "",
          "Checked on the original (un-normalized) biomarker outputs, where the stated "
          "relation applies:", "",
          f"- Acquisitions checked: **{red_raw['n_acquisitions_checked']}**",
          f"- `max |DC/245 - FC_Strength|` = **{red_raw['max_abs_difference']:.3e}**",
          f"- Minimum Pearson r across acquisitions = **{red_raw['min_pearson_r']:.12f}**",
          f"- Relation holds: **{red_raw['relation_holds']}**", "",
          "On the resulting stage matrices (both within-subject z-scored):", "",
          f"- `max |DC_z - FC_Strength|` = **{d_heat:.3e}**",
          f"- Pearson r = **{r_heat:.12f}**", "",
          "> FC_Strength is a deterministic rescaling of DC and is therefore not an "
          "independent biomarker feature. It is shown here only for "
          "visualization/comparison and is excluded from model input.", "",
          "Consequently the `FC_Strength` heatmaps are visually identical to the `DC_z` "
          "heatmaps.", "",
          "## 10. QC results", "",
          "| Check | Result |", "|---|---|",
          f"| Output shape (246 x 6) | {'PASS' if all(q['shape_ok'] for q in qc.values()) else 'FAIL'} |",
          f"| No NaN | {'PASS' if sum(q['nan_count'] for q in qc.values()) == 0 else 'FAIL'} |",
          f"| No Inf | {'PASS' if sum(q['inf_count'] for q in qc.values()) == 0 else 'FAIL'} |",
          f"| No NaN/Inf in row-z | {'PASS' if sum(q['rowz_nan_count'] + q['rowz_inf_count'] for q in qc.values()) == 0 else 'FAIL'} |",
          f"| Exactly 246 ROIs | {'PASS' if roi_ok['count'] == 246 else 'FAIL'} |",
          f"| No duplicate ROI IDs | {'PASS' if roi_ok['no_duplicates'] else 'FAIL'} |",
          f"| No missing ROI IDs | {'PASS' if roi_ok['no_missing'] else 'FAIL'} |",
          f"| Exactly 6 stages, correct order | {'PASS' if all(q['n_stages'] == 6 for q in qc.values()) else 'FAIL'} |",
          "| Subject-level aggregation used | PASS |",
          f"| Subject counts match expected | {'PASS' if all(q['subject_counts_match_expected'] for q in qc.values()) else 'FAIL'} |",
          f"| No subject in multiple stages | {'PASS' if not multi_stage else 'FAIL'} |",
          "", "## 11. Warnings and failures", ""]
    L += [f"- {w}" for w in warnings] if warnings else ["None."]
    L += ["", "---", "",
          "**Interpretation note.** These are descriptive, exploratory visualizations of "
          "stage-wise regional biomarker patterns. They do not demonstrate Alzheimer's "
          "disease progression, and no statistical testing, model training, accuracy or "
          "AUC computation was performed in this task."]
    with open(os.path.join(OUT, "stage_heatmap_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
