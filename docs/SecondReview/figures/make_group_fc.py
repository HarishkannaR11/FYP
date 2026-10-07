"""Stage-wise group-mean functional connectivity (Fisher-z, 246 x 246), one panel
per stage, with a single shared colour bar.

Acquisitions are averaged within subject first, then subjects within stage, so each
subject counts once. Reads derivatives/biomarkers_normalized/<GROUP>/sub-*/ses-*/run-*/
fc_fisher_z.npy; writes figures/group_fc_by_stage.pdf.
Run from docs/SecondReview/:  python figures/make_group_fc.py
"""
import collections
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..", "derivatives", "biomarkers_normalized")
ORDER = [("CN_Final", "CN"), ("SMC_Final", "SMC"), ("EMCI", "EMCI"),
         ("MCI", "MCI"), ("LMCI", "LMCI"), ("AD", "AD")]
# Brainnetome-246 lobe starts (checked against the atlas look-up table)
LOBES = [("Frontal", 1), ("Temporal", 69), ("Parietal", 125), ("Insular", 163),
         ("Limbic", 175), ("Occipital", 189), ("Subcort.", 211)]

means, counts = {}, {}
for grp, label in ORDER:
    by_subj = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(ROOT, grp, "sub-*", "ses-*", "run-*", "fc_fisher_z.npy"))):
        by_subj[f.split(os.sep)[-4]].append(np.load(f))
    subj_means = [np.mean(v, axis=0) for v in by_subj.values()]
    means[label] = np.mean(subj_means, axis=0)
    counts[label] = len(subj_means)
print(counts)
assert sum(counts.values()) == 127

plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 3, figsize=(6.3, 5.4))
bounds = [s - 0.5 for _, s in LOBES] + [246.5]
centres = [(a + b) / 2 for a, b in zip(bounds[:-1], bounds[1:])]
vmax = 0.6
for ax, (_, label) in zip(axes.ravel(), ORDER):
    im = ax.imshow(means[label], cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest",
                   extent=(0.5, 246.5, 246.5, 0.5))
    for b in bounds[1:-1]:
        ax.axhline(b, color="k", lw=0.3)
        ax.axvline(b, color="k", lw=0.3)
    ax.set_title(f"{label} (n = {counts[label]})", fontsize=9, fontweight="bold")
    ax.set_xticks(centres, [n for n, _ in LOBES], rotation=90, fontsize=6)
    ax.set_yticks(centres, [n for n, _ in LOBES], fontsize=6)
    ax.tick_params(length=0)
fig.subplots_adjust(left=0.1, right=0.98, top=0.94, bottom=0.19, wspace=0.35, hspace=0.45)
cax = fig.add_axes([0.25, 0.075, 0.5, 0.02])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("Group-mean Fisher-z functional connectivity", fontsize=8)
cb.ax.tick_params(labelsize=7)
fig.savefig(os.path.join(HERE, "group_fc_by_stage.pdf"))
print("written")
