"""
RECOVERED from session history (see verify_brainnetome_atlas.py header note).

TASK 5: Biomarker QC -- NaN/Inf, dimensions, empty ROIs, coverage,
per-biomarker validity, FC symmetry/diagonal, distributions. QC plots.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/mnt/c/Users/krish/FYP/test/biomarkers/sub-019S4549"

lines = []


def log(msg=""):
    print(msg)
    lines.append(str(msg))


def load_csv_col(path, col_idx, skip_header=True):
    vals = []
    with open(path) as f:
        r = csv.reader(f)
        if skip_header:
            next(r)
        for row in r:
            vals.append(float(row[col_idx]))
    return np.array(vals)


def main():
    log("=== TASK 5: Biomarker QC ===\n")

    roi_ts = np.load(f"{OUT_DIR}/roi_timeseries.npy")
    alff = load_csv_col(f"{OUT_DIR}/ALFF.csv", 2)
    reho = load_csv_col(f"{OUT_DIR}/ReHo.csv", 2)
    dc = load_csv_col(f"{OUT_DIR}/degree_centrality.csv", 2)
    fc = np.load(f"{OUT_DIR}/functional_connectivity.npy")
    coverage = load_csv_col(f"{OUT_DIR}/roi_coverage/roi_coverage.csv", 4)

    log(f"ROI time series shape: {roi_ts.shape} PASS={roi_ts.shape == (140, 246)}")
    log(f"Coverage: min={coverage.min():.1f}% mean={coverage.mean():.1f}%")

    alff_ok = alff.shape == (246,) and not np.isnan(alff).any() and alff.min() >= 0 and alff.max() <= 1
    reho_ok = reho.shape == (246,) and not np.isnan(reho).any() and reho.min() >= 0 and reho.max() <= 1
    dc_ok = dc.shape == (246,) and not np.isnan(dc).any()
    symmetric = np.allclose(fc, fc.T, atol=1e-10)
    diag_ok = np.allclose(np.diag(fc), 1.0)
    fc_ok = fc.shape == (246, 246) and symmetric and diag_ok and not np.isnan(fc).any()

    log(f"ALFF validity: {'PASS' if alff_ok else 'FAIL'}")
    log(f"ReHo validity: {'PASS' if reho_ok else 'FAIL'}")
    log(f"DC validity: {'PASS' if dc_ok else 'FAIL'}")
    log(f"FC validity: {'PASS' if fc_ok else 'FAIL'} (symmetric={symmetric}, diag_ok={diag_ok})")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].hist(alff, bins=30, color="steelblue")
    axes[0, 0].set_title("ALFF distribution (246 ROIs)")
    axes[0, 1].hist(reho, bins=30, color="darkorange")
    axes[0, 1].set_title("ReHo distribution (246 ROIs)")
    axes[1, 0].hist(dc, bins=30, color="seagreen")
    axes[1, 0].set_title("Degree Centrality distribution (246 ROIs)")
    im = axes[1, 1].imshow(fc, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1, 1].set_title("Functional Connectivity (246x246)")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/biomarker_qc_plots.png", dpi=150)

    overall = alff_ok and reho_ok and dc_ok and fc_ok
    log(f"\n=== TASK 5 OVERALL: {'PASS' if overall else 'FAIL'} ===")

    with open(f"{OUT_DIR}/task5_qc_report.txt", "w") as f:
        f.write("\n".join(lines))
    print(f"OVERALL={overall}")


if __name__ == "__main__":
    main()
