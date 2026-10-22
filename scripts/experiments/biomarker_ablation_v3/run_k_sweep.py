"""
Addendum 1 to experiment v3 (PLAN.md): SVM-RFE feature count k in {10, 20, 30, 75} on all four biomarkers with
TabPFN v2, outer folds + 3-fold inner CV; k = 50 is A_all4 (the baseline, already run). NESTED_K picks k per outer
split by inner CV (analyze_v3.py).

Uses run_tabpfn_configs' functions unchanged, so that script's pinned hash stays valid. Refuses to start while the
main v3 job (TabPFN or transformer) is still running. Outputs grid/K_all4_k{k}/cv5_seed{s}_fold{f}.json, written once.

Run:  python scripts/experiments/biomarker_ablation_v3/run_k_sweep.py
"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_tabpfn_configs as T  # noqa: E402  (sets thread env vars first)

import psutil  # noqa: E402
import torch  # noqa: E402

K_CONFIGS = {f"K_all4_k{k}": {"pool": "all4", "k": k, "inner": True} for k in (10, 20, 30, 75)}
NESTED_K_ORDER = ["A_all4", "K_all4_k10", "K_all4_k20", "K_all4_k30", "K_all4_k75"]   # ties: baseline, then smaller k


def main():
    for name in ("tabpfn", "transformer"):
        p = os.path.join(T.EXP, f"{name}.lock")
        if os.path.exists(p) and psutil.pid_exists(int(open(p).read().strip() or 0)):
            raise SystemExit(f"the main v3 {name} run is still active; the sweep starts only after it finishes")
    torch.set_num_threads(4)
    T.CONFIGS.update(K_CONFIGS)
    own = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
    mpath = os.path.join(T.EXP, "manifest_k_sweep.json")
    if os.path.exists(mpath) and json.load(open(mpath))["script_sha256"] != own:
        raise SystemExit("run_k_sweep.py changed since the sweep started; refusing to mix versions")
    json.dump({"script_sha256": own, "uses": {"run_tabpfn_configs.py": T.script_hash()}, "configs": K_CONFIGS,
               "nested_k_order": NESTED_K_ORDER, "created": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(mpath, "w"), indent=2)
    lock = T.acquire_lock("ksweep")
    try:
        d = T.S.load_cohort()
        y = d[3]
        plan = T.S.cv_plan(y, d[4], 5)
        for cname in K_CONFIGS:
            t_cfg = time.time()
            for seed in T.H.SEEDS:
                fp, folds = plan[seed]
                for f, (tr, te) in enumerate(folds):
                    path = os.path.join(T.EXP, "grid", cname, f"cv5_seed{seed}_fold{f}.json")
                    if os.path.exists(path):
                        continue
                    t0 = time.time()
                    rec = T.run_split(d, y, cname, seed, f, fp, tr, te)
                    rec["sweep_script_sha256"] = own
                    rec["seconds"] = round(time.time() - t0, 1)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    json.dump(rec, open(path + ".tmp", "w"))
                    os.replace(path + ".tmp", path)
                    print(f"{cname} seed {seed} fold {f}: inner {rec['inner']['score']:.3f}, {rec['seconds']:.0f}s", flush=True)
            print(f"{cname} done in {(time.time() - t_cfg) / 60:.1f} min", flush=True)
    finally:
        os.remove(lock)


if __name__ == "__main__":
    main()
