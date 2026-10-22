"""
Driver for the outer-only TabPFN configurations of experiment v3 (P_site*, P_glob*, L_frac*).

run_tabpfn_configs.py crashed after saving the first P_site1 record: its progress message read the configuration
flag "inner": False as if it were inner-CV results (TypeError in a print statement, after the record was written).
That script stays unchanged so its pinned hash still identifies the code that produced every record; this driver
runs the same configurations through its unchanged run_split() and only prints a correct progress message.

Run:  python scripts/experiments/biomarker_ablation_v3/run_tabpfn_outer_only.py --configs P_site1 ... L_frac75
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_tabpfn_configs as T  # noqa: E402  (sets thread env vars first)

import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True, choices=[c for c, v in T.CONFIGS.items() if not v["inner"]])
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    T.manifest()                                     # refuses if run_tabpfn_configs.py has changed
    lock = T.acquire_lock("tabpfn")
    try:
        d = T.S.load_cohort()
        y = d[3]
        plan = T.S.cv_plan(y, d[4], 5)
        for cname in args.configs:
            t_cfg = time.time()
            for seed in T.H.SEEDS:
                fp, folds = plan[seed]
                for f, (tr, te) in enumerate(folds):
                    path = os.path.join(T.EXP, "grid", cname, f"cv5_seed{seed}_fold{f}.json")
                    if os.path.exists(path):
                        continue
                    t0 = time.time()
                    rec = T.run_split(d, y, cname, seed, f, fp, tr, te)
                    rec["seconds"] = round(time.time() - t0, 1)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    json.dump(rec, open(path + ".tmp", "w"))
                    os.replace(path + ".tmp", path)
                    print(f"{cname} seed {seed} fold {f}: {rec['n_features']} features, {rec['n_train']} training "
                          f"subjects, outer only, {rec['seconds']:.0f}s", flush=True)
            print(f"{cname} done in {(time.time() - t_cfg) / 60:.1f} min", flush=True)
    finally:
        os.remove(lock)


if __name__ == "__main__":
    main()
