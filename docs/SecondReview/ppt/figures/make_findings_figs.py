"""Slide-sized figures for Findings 3 and 4 and the "Why 246 regions?" backup.

Same data and same definitions as scripts/audit/review2_finding_figures.py and
scripts/audit/atlas_granularity_figure.py (derivatives/REVIEW2_FINDINGS/*.png),
redrawn at the size they are shown on a 16:9 slide (5.75 in = the full text
width, so the fonts are shown at 1:1) with the scope of every number stated.
Read-only; no stage label is used anywhere, and nothing here says that any
region or biomarker predicts disease.

One difference from the originals, on purpose: the left-right homotopic
control uses all 165 acquisitions and all 123 homologous pairs against all
other pairs. The original used the first 60 acquisitions (27 CN, 31 SMC,
2 EMCI) and 200 sampled random pairs per scan.

Writes into this folder:
    findings_complementarity.pdf   Finding 3
    findings_reliability.pdf       Finding 4
    findings_granularity.pdf       "Why 246 regions?" (backup)

Run from anywhere:  python3 docs/SecondReview/ppt/figures/make_findings_figs.py
"""
import glob
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
NRM = os.path.join(ROOT, "derivatives", "biomarkers_normalized")
DRV = os.path.join(ROOT, "derivatives", "fsfast")
ORDER = ["CN_Final", "SMC_Final", "EMCI", "MCI", "LMCI", "AD"]
UBLUE, ABLUE, LBLUE, DGRAY = "#103E6E", "#1E88E5", "#E8F2FC", "#374151"
PALE, RED, GREEN = "#9FB6CE", "#C0392B", "#2E7D32"

plt.rcParams.update({"font.size": 7.5, "pdf.fonttype": 42, "axes.linewidth": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "xtick.major.pad": 2, "ytick.major.pad": 2})
TITLE = dict(fontsize=8, fontweight="bold", color=UBLUE, pad=4)


def style(ax, grid="both"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, alpha=0.22, lw=0.5)
    ax.set_axisbelow(True)


def minus(text):
    return text.replace("-", "−")


# ------------------------------------------------------------------ data
rows = []
for g in ORDER:
    for d in sorted(glob.glob(os.path.join(NRM, g, "sub-*", "ses-*", "run-*"))):
        p = d.split(os.sep)
        rows.append(dict(subject=p[-3], ses=p[-2], norm=d,
                         raw=os.path.join(DRV, g, p[-3], p[-2], p[-1], "biomarkers")))
df = pd.DataFrame(rows)
N = len(df)
F = {k: np.vstack([np.load(os.path.join(p, f)) for p in df.norm])
     for k, f in [("mALFF", "malff.npy"), ("mReHo", "mreho.npy"), ("DC_z", "dc_z.npy")]}
LAB = {"mALFF": "mALFF", "mReHo": "mReHo", "DC_z": "DC$_z$"}
print("%d acquisitions, %d subjects" % (N, df.subject.nunique()))

rep = df.groupby("subject").size()
rep = rep[rep >= 2]
idx = {s: np.where(df.subject.values == s)[0] for s in rep.index}
sel = np.concatenate([idx[s] for s in rep.index])
own = np.concatenate([[s] * len(idx[s]) for s in rep.index])
CHANCE = 100 / len(rep)                      # one in 31 people, as in the originals


def ident(M):
    """Percent of scans whose most correlated other scan is the same person's."""
    C = np.corrcoef(M)
    np.fill_diagonal(C, -np.inf)
    return 100 * sum(own[C[i].argmax()] == own[i] for i in range(len(sel))) / len(sel)


# ===================================================== FINDING 3
fcs_dc = np.array([stats.pearsonr(np.load(os.path.join(p, "degree_centrality.npy")),
                                  np.load(os.path.join(p, "fc_strength.npy")))[0]
                   for p in df.raw])
pairs = [("FC_Strength–DC", fcs_dc, True)]
for a, b in itertools.combinations(F, 2):
    r = np.array([stats.pearsonr(F[a][i], F[b][i])[0] for i in range(N)])
    pairs.append(("%s–%s" % (LAB[a], LAB[b]), r, False))
for nm, r, _ in pairs:
    print("%-16s mean r %+.3f | shared variance %.1f%% | scans with r<0: %d/%d"
          % (nm, r.mean(), 100 * np.mean(r ** 2), (r < 0).sum(), N))

fig = plt.figure(figsize=(5.75, 1.7))
axA = fig.add_axes([0.20, 0.20, 0.43, 0.64])
axB = fig.add_axes([0.675, 0.20, 0.30, 0.64], sharey=axA)
rng = np.random.default_rng(0)
ys = list(range(len(pairs) - 1, -1, -1))
for (nm, r, red), y in zip(pairs, ys):
    c = RED if red else UBLUE
    axA.scatter(r, rng.normal(y, 0.07, len(r)), s=5, color=c, alpha=0.4, edgecolors="none", zorder=3)
    axA.scatter([r.mean()], [y], s=90, marker="|", color=c, zorder=4, lw=2)
    axA.text(1.08, y, minus("mean r = %+.2f" % r.mean()), va="center", fontsize=7, color=c,
             fontweight="bold")
axA.axvline(0, color=DGRAY, lw=0.8, ls=":")
axA.text(0.9, ys[0], "removed (Finding 1)", color=RED, fontsize=6.5, ha="right", va="center",
         fontweight="bold")
axA.set_yticks(ys)
axA.set_yticklabels([p[0] for p in pairs], fontsize=7.5)
axA.set_ylim(-0.6, len(pairs) - 0.4)
axA.set_xlim(-0.75, 1.9)
axA.set_xticks([-0.5, 0, 0.5, 1.0])
axA.set_xticklabels(["−0.5", "0", "0.5", "1.0"])
axA.set_xlabel("correlation across the 246 regions (1 dot = 1 scan)")
axA.set_title("(a) per-scan correlation", **TITLE)
style(axA, grid=None)
for (nm, r, red), y in zip(pairs, ys):
    v = 100 * np.mean(r ** 2)
    axB.barh(y, v, height=0.55, color=RED if red else ABLUE)
    axB.barh(y, 100 - v, left=v, height=0.55, color=LBLUE)
    axB.text(v + 2.5 if v < 60 else v - 3, y, "%.1f%%" % v if v < 99.95 else "100%",
             va="center", ha="left" if v < 60 else "right", fontsize=7.5, fontweight="bold",
             color=UBLUE if v < 60 else "white")
axB.set_xlim(0, 100)
axB.tick_params(labelleft=False)
axB.set_xlabel("shared variance, mean r² (%)")
axB.set_title("(b) how much is shared", **TITLE)
style(axB, grid=None)
fig.savefig(os.path.join(HERE, "findings_complementarity.pdf"))
plt.close(fig)

# ===================================================== FINDING 4
pr = [(i, j) for s in rep.index for i, j in itertools.combinations(idx[s], 2)]
n_diff = sum(df.ses[i] != df.ses[j] for i, j in pr)
print("repeat subjects %d, scans %d, within-person pairs %d (%d from different sessions)"
      % (len(rep), len(sel), len(pr), n_diff))
rng = np.random.default_rng(0)
subs = list(rep.index)
stat = {}
for n, X in F.items():
    within = [stats.pearsonr(X[i], X[j])[0] for i, j in pr]
    between = []
    for _ in range(3000):
        a, b = rng.choice(len(subs), 2, replace=False)
        between.append(stats.pearsonr(X[rng.choice(idx[subs[a]])],
                                      X[rng.choice(idx[subs[b]])])[0])
    C = np.corrcoef(X[sel])
    np.fill_diagonal(C, -np.inf)
    hit = sum(own[C[i].argmax()] == own[i] for i in range(len(sel)))
    stat[n] = (np.mean(within), np.mean(between), hit)
    print("%s: same person %.2f, different people %.2f, identification %d/%d = %.1f%%"
          % (n, stat[n][0], stat[n][1], hit, len(sel), 100 * hit / len(sel)))

iu = np.triu_indices(246, 1)
is_hom = (iu[0] % 2 == 0) & (iu[1] == iu[0] + 1)       # IDs 2k+1 / 2k+2 = left / right
hom, oth = [], []
for p in df.norm:
    v = np.load(os.path.join(p, "fc_fisher_z.npy"))[iu]
    hom.append(v[is_hom].mean())
    oth.append(v[~is_hom].mean())
hom, oth = np.array(hom), np.array(oth)
lacking = np.where(hom <= oth)[0]
print("homotopic pairs %d; mean Fisher-z %.3f vs %.3f over %d scans; homotopic > other in %d/%d scans; "
      "exceptions: %s" % (is_hom.sum(), hom.mean(), oth.mean(), N, (hom > oth).sum(), N,
                          [(df.subject[i], df.ses[i]) for i in lacking]))

fig = plt.figure(figsize=(5.75, 1.78))
names = list(F)
x = np.arange(len(names))
w = 0.36
ax = fig.add_axes([0.08, 0.20, 0.265, 0.62])
ax.bar(x - w / 2, [stat[n][0] for n in names], w, color=UBLUE, label="same person")
ax.bar(x + w / 2, [stat[n][1] for n in names], w, color=PALE, label="different people")
for i, n in enumerate(names):
    ax.text(i - w / 2, stat[n][0] + 0.02, "%.2f" % stat[n][0], ha="center", fontsize=7,
            fontweight="bold", color=UBLUE)
    ax.text(i + w / 2, stat[n][1] + 0.02, "%.2f" % stat[n][1], ha="center", fontsize=6.3,
            color=DGRAY)
ax.set_xticks(x)
ax.set_xticklabels([LAB[n] for n in names])
ax.set_ylim(0, 1.32)
ax.set_yticks([0, 0.5, 1.0])
ax.set_ylabel("correlation (246 regions)")
ax.legend(fontsize=6.5, frameon=False, loc="upper right", handlelength=1, borderaxespad=0.1)
ax.set_title("(a) repeat scans agree", **TITLE)
style(ax, grid="y")

ax = fig.add_axes([0.44, 0.20, 0.25, 0.62])
acc = [100 * stat[n][2] / len(sel) for n in names]
b = ax.bar([LAB[n] for n in names], acc, color=[ABLUE, ABLUE, PALE], width=0.55)
ln = ax.axhline(CHANCE, color=RED, ls="--", lw=1.1)
ax.legend([ln], ["chance ≈ %d%%" % round(CHANCE)], loc="upper right", frameon=False,
          fontsize=6.8, handlelength=1.6, borderaxespad=0.1)
for bb, v, n in zip(b, acc, names):
    ax.text(bb.get_x() + bb.get_width() / 2, v + 2, "%.0f%%\n(%d/%d)" % (v, stat[n][2], len(sel)),
            ha="center", fontsize=6.8, fontweight="bold", color=UBLUE, linespacing=1.1)
ax.set_ylim(0, 122)
ax.set_yticks([0, 50, 100])
ax.set_ylabel("identification rate (%)")
ax.set_title("(b) identification rate", **TITLE)
style(ax, grid="y")

ax = fig.add_axes([0.775, 0.20, 0.205, 0.62])
bp = ax.boxplot([hom, oth], positions=[0, 1], widths=0.5, patch_artist=True, showfliers=False,
                medianprops=dict(color="black", lw=1.1), whiskerprops=dict(lw=0.7, color=DGRAY),
                capprops=dict(lw=0.7, color=DGRAY))
for bx, c in zip(bp["boxes"], [UBLUE, PALE]):
    bx.set(facecolor=c, alpha=0.8, edgecolor=c)
jit = np.random.default_rng(2)
ax.scatter(jit.normal(0, 0.06, N), hom, s=2.5, color=DGRAY, alpha=0.35, zorder=3, edgecolors="none")
ax.scatter(jit.normal(1, 0.06, N), oth, s=2.5, color=DGRAY, alpha=0.35, zorder=3, edgecolors="none")
if len(lacking):
    ax.scatter(np.zeros(len(lacking)), hom[lacking], s=9, color=RED, zorder=5)
    ax.annotate("%d scan" % len(lacking), (0, hom[lacking][0]), (0.42, 0.2), fontsize=6.5,
                color=RED, fontweight="bold", arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
ax.axhline(0, color=DGRAY, lw=0.7, ls=":")
ax.text(0, 1.1, "mean %.2f" % hom.mean(), ha="center", fontsize=6.8, fontweight="bold", color=UBLUE)
ax.text(1, 0.1, "mean %.2f" % oth.mean(), ha="center", fontsize=6.8, fontweight="bold", color=DGRAY)
ax.set_ylim(-0.08, 1.25)
ax.set_xticks([0, 1])
ax.set_xticklabels(["left–right\nhomologues", "all other\npairs"], fontsize=6.8)
ax.set_ylabel("mean FC (Fisher z)")
ax.set_title("(c) positive control", **TITLE)
style(ax, grid="y")
fig.savefig(os.path.join(HERE, "findings_reliability.pdf"))
plt.close(fig)

# ===================================================== WHY 246 REGIONS
X = F["mALFF"]
rng = np.random.default_rng(0)
ns = [5, 10, 25, 50, 75, 100, 150, 200, 246]
mu, sd = [], []
for n in ns:
    a = [ident(X[sel][:, rng.choice(246, n, replace=False)]) for _ in range(1 if n == 246 else 40)]
    mu.append(np.mean(a))
    sd.append(np.std(a))
print("random-subset curve:", ["%d:%.1f" % (n, m) for n, m in zip(ns, mu)])
mg = []
for k in [1, 2, 3, 6, 41]:
    n = 246 // k
    mg.append((n, ident(X[sel][:, :n * k].reshape(len(sel), n, k).mean(axis=2))))
print("merge curve:", ["%d:%.1f" % t for t in mg])
ev = np.linalg.eigvalsh(np.corrcoef(X.T))[::-1]
ev = ev[ev > 1e-10]
cum = 100 * np.cumsum(ev) / ev.sum()
d80, d95 = int(np.searchsorted(cum, 80)) + 1, int(np.searchsorted(cum, 95)) + 1
c = np.corrcoef(X.T)
print("components for 80%% / 95%% of the variance: %d / %d (rank %d); mean pairwise r^2 between "
      "regions %.1f%%" % (d80, d95, len(ev), 100 * np.mean(c[iu] ** 2)))

fig = plt.figure(figsize=(5.75, 1.78))
ax = fig.add_axes([0.08, 0.20, 0.26, 0.62])
ax.errorbar(ns, mu, yerr=sd, color=UBLUE, lw=1.2, marker="o", ms=2.8, capsize=1.8, zorder=3)
ax.axhline(CHANCE, color=RED, ls="--", lw=1)
ax.text(246, CHANCE + 4, "chance ≈ %d%%" % round(CHANCE), color=RED, fontsize=6.5, ha="right",
        fontweight="bold")
ax.scatter([246], [mu[-1]], s=22, color=RED, zorder=5)
ax.annotate("all 246:\n%.1f%%" % mu[-1], (246, mu[-1]), (118, 52), fontsize=7, fontweight="bold",
            color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
ax.set_ylim(0, 100)
ax.set_xlabel("regions used (random subset)")
ax.set_ylabel("identification rate (%)")
ax.set_title("(a) random subsets", **TITLE)
style(ax)

ax = fig.add_axes([0.44, 0.20, 0.25, 0.62])
nn, aa = zip(*mg)
ax.plot(nn, aa, color=ABLUE, lw=1.2, marker="s", ms=3, zorder=3)
OFFSET = {6: (7, -4.5, "left"), 82: (7, -7.5, "left")}      # (dx, dy, align); default: above
for n_, a_ in mg:
    dx, dy, ha = OFFSET.get(n_, (0, 3.5, "center"))
    ax.text(n_ + dx, a_ + dy, "%.1f" % a_, ha=ha, fontsize=6.8, fontweight="bold", color=UBLUE)
ax.scatter([246], [aa[0]], s=22, color=RED, zorder=5)
ax.set_ylim(40, 105)
ax.set_xlim(-30, 262)
ax.set_xticks([6, 41, 82, 123, 246])
ax.set_xticklabels(["6", "41", "82", "123", "246"], fontsize=6.8)
ax.set_xlabel("regions after merging")
ax.set_ylabel("identification rate (%)")
ax.set_title("(b) merged labels", **TITLE)
style(ax)

ax = fig.add_axes([0.785, 0.20, 0.195, 0.62])
ax.plot(range(1, len(cum) + 1), cum, color=UBLUE, lw=1.2)
ax.fill_between(range(1, len(cum) + 1), 0, cum, color=LBLUE)
for d, lab, col, yy in [(d80, "80%", ABLUE, 0.36), (d95, "95%", RED, 0.20)]:
    ax.axvline(d, color=col, ls="--", lw=0.9)
    ax.text(0.97, yy, "%d → %s" % (d, lab), transform=ax.transAxes, ha="right", color=col,
            fontsize=7, fontweight="bold")
ax.set_xlim(0, 246)
ax.set_ylim(0, 101)
ax.set_xlabel("components (of 246)")
ax.set_ylabel("variance explained (%)")
ax.set_title("(c) many dimensions", **TITLE)
style(ax)
fig.savefig(os.path.join(HERE, "findings_granularity.pdf"))
plt.close(fig)
print("done")
