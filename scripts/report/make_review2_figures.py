"""Generate the data-driven figures and summary numbers for the Review-II report.

Reads only existing derivatives (never writes inside derivatives/). Outputs go to
reports/review2/figures/. Run from the repository root:

    python scripts/report/make_review2_figures.py
"""
import glob
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FS = os.path.join(ROOT, "derivatives", "fsfast")
NORM = os.path.join(ROOT, "derivatives", "biomarkers_normalized")
OUT = os.path.join(ROOT, "reports", "review2", "figures")
os.makedirs(OUT, exist_ok=True)

ORDER = ["CN_Final", "SMC_Final", "EMCI", "MCI", "LMCI", "AD"]
LABEL = {"CN_Final": "CN", "SMC_Final": "SMC", "EMCI": "EMCI",
         "MCI": "MCI", "LMCI": "LMCI", "AD": "AD"}
# Brainnetome-246 lobe boundaries (first ROI of each lobe, 1-based)
LOBES = [("Frontal", 1), ("Temporal", 69), ("Parietal", 125), ("Insula", 163),
         ("Limbic", 175), ("Occipital", 189), ("Subcort.", 211)]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})


def acquisitions():
    rows = []
    for p in sorted(glob.glob(os.path.join(NORM, "*", "sub-*", "ses-*", "run-*"))):
        g, s, ses, r = p.split(os.sep)[-4:]
        rows.append(dict(group=g, subject=s, session=ses, run=r, norm=p,
                         raw=os.path.join(FS, g, s, ses, r, "biomarkers")))
    return pd.DataFrame(rows)


def fig_qc(qc):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    data_fd = [qc.loc[qc.group == g, "mean_FD_mm"].values for g in ORDER]
    data_ts = [qc.loc[qc.group == g, "mean_tSNR"].values for g in ORDER]
    data_snr = [qc.loc[qc.group == g, "SNR"].values for g in ORDER]
    for ax, data, title, yl in [
            (axes[0], data_fd, "Mean framewise displacement", "FD (mm)"),
            (axes[1], data_ts, "Mean tSNR (outlier clipped)", "tSNR"),
            (axes[2], data_snr, "Native-space SNR", "SNR")]:
        ax.boxplot(data, tick_labels=[LABEL[g] for g in ORDER], showfliers=True,
                   flierprops=dict(markersize=3))
        ax.set_title(title)
        ax.set_ylabel(yl)
    axes[0].axhline(0.5, ls="--", lw=0.8, color="grey")
    axes[1].set_ylim(0, 300)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "qc_by_group.pdf"))
    plt.close(fig)


def fig_alff_scaling(acq):
    raw_alff, raw_reho, grp = [], [], []
    for _, a in acq.iterrows():
        raw_alff.append(np.load(os.path.join(a.raw, "alff.npy")).mean())
        raw_reho.append(np.load(os.path.join(a.raw, "reho.npy")).mean())
        grp.append(a.group)
    df = pd.DataFrame(dict(group=grp, alff=raw_alff, reho=raw_reho))
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    for i, g in enumerate(ORDER):
        v = df[df.group == g]
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(v))
        axes[0].scatter(np.full(len(v), i) + jitter, v.alff, s=8)
        axes[1].scatter(np.full(len(v), i) + jitter, v.reho, s=8)
    for ax in axes:
        ax.set_xticks(range(len(ORDER)), [LABEL[g] for g in ORDER])
    axes[0].set_yscale("log")
    axes[0].set_title("Per-acquisition mean raw ALFF (log scale)")
    axes[0].set_ylabel("FFT amplitude")
    axes[1].set_title("Per-acquisition mean raw ReHo")
    axes[1].set_ylabel("Kendall's W")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "alff_reho_scaling.pdf"))
    plt.close(fig)
    return df


def fig_dc_strength(acq):
    dc, st = [], []
    for _, a in acq.iterrows():
        dc.append(np.load(os.path.join(a.raw, "degree_centrality.npy")))
        st.append(np.load(os.path.join(a.raw, "fc_strength.npy")))
    dc, st = np.concatenate(dc), np.concatenate(st)
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.scatter(dc, st, s=1, alpha=0.3, rasterized=True)
    x = np.array([dc.min(), dc.max()])
    ax.plot(x, x / 245.0, color="k", lw=0.8, ls="--", label="DC / 245")
    ax.set_xlabel("Degree centrality (DC)")
    ax.set_ylabel("FC strength")
    ax.legend(frameon=False)
    ax.set_title(f"{len(dc):,} region values, 165 scans")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dc_vs_fcstrength.pdf"), dpi=200)
    plt.close(fig)
    return float(np.corrcoef(dc, st)[0, 1]), float(np.abs(st - dc / 245.0).max())


def fig_group_fc(acq):
    means, stats = {}, []
    for g in ORDER:
        sub_means = []
        for s, rows in acq[acq.group == g].groupby("subject"):
            mats = [np.load(os.path.join(p, "fc_fisher_z.npy")) for p in rows.norm]
            sub_means.append(np.mean(mats, axis=0))  # subject first: no session inflation
        m = np.mean(sub_means, axis=0)
        means[g] = m
        off = m[~np.eye(246, dtype=bool)]
        stats.append(dict(group=g, subjects=len(sub_means), mean_z=off.mean(),
                          mean_abs_z=np.abs(off).mean()))
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.6))
    for ax, g in zip(axes.ravel(), ORDER):
        im = ax.imshow(means[g], cmap="RdBu_r", vmin=-0.6, vmax=0.6)
        for _, start in LOBES[1:]:
            ax.axhline(start - 1.5, color="k", lw=0.3)
            ax.axvline(start - 1.5, color="k", lw=0.3)
        ax.set_title(f"{LABEL[g]} (n = {len(acq[acq.group == g].subject.unique())} subjects)")
        bounds = [s - 1 for _, s in LOBES] + [246]
        centres = [(a + b) / 2 for a, b in zip(bounds[:-1], bounds[1:])]
        ax.set_xticks(centres, [n for n, _ in LOBES], rotation=90, fontsize=6)
        ax.set_yticks(centres, [n for n, _ in LOBES], fontsize=6)
        ax.tick_params(length=0)
    fig.colorbar(im, ax=axes, shrink=0.6, label="mean Fisher-z")
    fig.savefig(os.path.join(OUT, "group_mean_fc.pdf"), dpi=200)
    plt.close(fig)
    return pd.DataFrame(stats)


def copy_pilot_qc():
    src = os.path.join(FS, "AD", "sub-019S4549", "ses-01", "run-01", "qc")
    for f in ["registration_qc.png", "fd_plot.png", "tsnr_map.png"]:
        shutil.copy(os.path.join(src, f), os.path.join(OUT, "pilot_" + f))


def main():
    acq = acquisitions()
    qc = pd.read_csv(os.path.join(FS, "FINAL_6GROUP_AUDIT", "dataset_qc_summary.csv"))
    fig_qc(qc)
    alff = fig_alff_scaling(acq)
    r, dev = fig_dc_strength(acq)
    fcstats = fig_group_fc(acq)
    copy_pilot_qc()
    summary = dict(
        acquisitions=len(acq), subjects=int(acq.subject.nunique()),
        scans_per_subject=acq.groupby("subject").size().value_counts().sort_index().to_dict(),
        raw_alff_mean_range=[float(alff.alff.min()), float(alff.alff.max())],
        dc_fcstrength_r=r, dc_fcstrength_max_abs_dev=dev,
        group_fc=fcstats.round(4).to_dict(orient="records"),
        group_reho_mean=alff.groupby("group").reho.mean().round(4).to_dict(),
    )
    with open(os.path.join(OUT, "figure_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
