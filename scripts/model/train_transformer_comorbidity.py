"""
Comorbidity-aware transformer (Block 6): SVM-RFE rs-fMRI tokens + comorbidity,
fused by a per-patient gate (fusion_heads.py). 116-subject cohort.

Clinical block (--features, scaled on training subjects only)
-------------------------------------------------------------
  comorbidity23  age, sex, APOE4 count, GDS total, 19 medical-history flags
                 -> derivatives/model_training/transformer_comorbidity/
  block5         age, GDS total, APOE4 count, MMSE total, CDR sum-of-boxes
                 (the project document's Block 5 list, chosen by the user)
                 -> derivatives/model_training/transformer_block5/
  comorbidity8   model A: age, sex, APOE4 count, GDS total, and the cardiovascular
                 (hypertension proxy), endocrine-metabolic (diabetes proxy),
                 psychiatric and smoking flags
                 -> derivatives/model_training/transformer_comorbidity8/
  clinical10     model B: comorbidity8 + MMSE total + CDR sum-of-boxes
                 -> derivatives/model_training/transformer_clinical10/
  sources: derivatives/comorbidity_features/demographics_116.csv (age, sex)
           derivatives/comorbidity_features/model_clinical_116.csv (the rest)
Missing values are filled with the TRAINING subjects' median; nothing from a
test subject enters the fill or the scaling.

MMSE and CDR are part of ADNI's diagnostic criteria for the stage labels. In
comorbidity23 they are not inputs and appear only as `ceiling_cdr_mmse`. In
block5 and clinical10 they are inputs by the user's decision, and results must be reported
next to the imaging-only and MMSE+CDR-only rows so the share that comes from
the label-defining scores is visible.

Designs
-------
  split70 : the project's 70/30 split (derivatives/model_ready/splits,
            fingerprint 6f478fbcf56364e2): 81 train / 35 test, evaluated once.
            Its 35 test subjects were never in the feature-selection study.
  cv      : 5 folds x 5 seeds on all 116, the SAME folds as
            train_transformer_svmrfe.py, so every record pairs with the
            imaging-only records already saved there.

Everything else (scaling, SVM-RFE 50 inside each training set, tokens,
architecture, 4 inner-fold early-stopped models averaged) is identical to
train_transformer_svmrfe.py and is imported from it.

Candidates
----------
  transformer_svmrfe50_clin            primary: transformer + comorbidity fusion
  logreg_clin                          comorbidity only
  logreg_svmrfe50_clin                 SVM-RFE 50 + comorbidity, logistic regression
  ceiling_cdr_mmse                     MMSE + CDR global + CDR-SB only (circular ceiling)
  ceiling_mmse_cdrsb                   MMSE + CDR-SB only (circular ceiling for clinical10)
  transformer_svmrfe50_clin_SHUFFLED   training labels permuted (cv only)
  transformer_svmrfe50, logreg_svmrfe50  imaging only (split70 only; their cv
                                       records are reused from transformer_svmrfe/)

Run:
  python scripts/model/train_transformer_comorbidity.py --smoke
  python scripts/model/train_transformer_comorbidity.py --eval both
"""
import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_transformer_svmrfe as S  # noqa: E402  (sets thread env vars first)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from fusion_heads import FusionPredictor  # noqa: E402
from heads import build_head  # noqa: E402
from multi_attention_transformer import build_model  # noqa: E402
from train_transformer import Scaler, evaluate, load_all  # noqa: E402

H, M = S.H, S.M
ROOT = S.ROOT
CF = os.path.join(ROOT, "derivatives", "comorbidity_features")
N_STAGES = S.N_STAGES
SPLIT70_SEED = 20261015                  # make_splits.SEED, the 70/30 split's seed
N_PERM_REPEATS = 20

MH = ["MH_psychiatric", "MH_neurologic_non_AD", "MH_head_eyes_ears_nose_throat",
      "MH_cardiovascular", "MH_respiratory", "MH_hepatic", "MH_dermatologic_connective_tissue",
      "MH_musculoskeletal", "MH_endocrine_metabolic", "MH_gastrointestinal",
      "MH_hematopoietic_lymphatic", "MH_renal_genitourinary",
      "MH_allergies_drug_sensitivities", "MH_alcohol_abuse", "MH_drug_abuse", "MH_smoking",
      "MH_malignancy", "MH_major_surgical_procedures", "MH_other"]
FEATURE_SETS = {"comorbidity23": ["age", "sex_male", "APOE4_count", "GDS_total"] + MH,
                "block5": ["age", "GDS_total", "APOE4_count", "MMSE_total", "CDR_SOB"],
                # model A: the 8 comorbidities chosen from the project's research question (age, sex,
                # APOE4, depression, hypertension and diabetes proxies, psychiatric history, smoking)
                "comorbidity8": ["age", "sex_male", "APOE4_count", "GDS_total", "MH_cardiovascular",
                                 "MH_endocrine_metabolic", "MH_psychiatric", "MH_smoking"]}
FEATURE_SETS["clinical10"] = FEATURE_SETS["comorbidity8"] + ["MMSE_total", "CDR_SOB"]      # model B
# user's request: MMSE, CDR (sum of boxes), GDS, APOE4 and the cardiovascular flag as one model type
FEATURE_SETS["custom5"] = ["MMSE_total", "CDR_SOB", "GDS_total", "APOE4_count", "MH_cardiovascular"]
OUT_DIRS = {"comorbidity23": "transformer_comorbidity", "block5": "transformer_block5",
            "comorbidity8": "transformer_comorbidity8", "clinical10": "transformer_clinical10",
            "custom5": "transformer_custom5"}
CLIN_COLS = FEATURE_SETS["comorbidity23"]
CEILING_COLS = ["MMSE_total", "CDR_global", "CDR_SOB"]

PRIMARY = "transformer_svmrfe50_clin"
SHUF = PRIMARY + "_SHUFFLED"
IMAGING_ONLY = ["transformer_svmrfe50", "logreg_svmrfe50"]
CANDIDATES = [PRIMARY, "logreg_clin", "logreg_svmrfe50_clin", "ceiling_cdr_mmse", "ceiling_mmse_cdrsb",
              SHUF] + IMAGING_ONLY

_lines = []


def log(m=""):
    print(m, flush=True)
    _lines.append(str(m))


# ------------------------------------------------------------------ data
def load_clinical(y, subjects, cols=CLIN_COLS):
    clin = pd.read_csv(os.path.join(CF, "model_clinical_116.csv"))
    demo = pd.read_csv(os.path.join(CF, "demographics_116.csv"))
    for t in (clin, demo):
        if t["bids_id"].tolist() != list(subjects):
            raise SystemExit("clinical table not in subjects.npy order")
    if clin["stage_index"].tolist() != y.tolist():
        raise SystemExit("clinical stage labels do not match y.npy")
    C = pd.DataFrame({"age": demo["age_at_scan"].astype(float),
                      "sex_male": (demo["sex"] == "M").astype(float)})
    for c in ["APOE4_count", "GDS_total"] + MH + CEILING_COLS:
        C[c] = clin[c].astype(float)
    return C[cols].to_numpy(np.float64), clin[CEILING_COLS].to_numpy(np.float64)


class ClinPrep:
    """Median fill + standardisation, fitted on training subjects only."""

    def fit(self, C, idx):
        self.med = np.nanmedian(C[idx], axis=0)
        Ci = np.where(np.isnan(C), self.med, C)
        self.mu = Ci[idx].mean(0)
        sd = Ci[idx].std(0)
        self.sd = np.where(sd < 1e-8, 1.0, sd)
        return self

    def transform(self, C):
        return ((np.where(np.isnan(C), self.med, C) - self.mu) / self.sd).astype(np.float32)


def logreg_pipe():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()), ("clf", M.classifier())])


# ------------------------------------------------------------- training
def new_model(n_roi, n_feat, n_clin, seed):
    torch.manual_seed(seed)
    enc = build_model(n_roi=n_roi, n_features=n_feat, **S.ARCH)
    head = build_head("multiclass", S.ARCH["d_model"], dropout=S.ARCH["dropout"])
    return FusionPredictor(enc, head, n_clin, S.ARCH["d_model"], S.ARCH["dropout"])


@torch.no_grad()
def probs_of(model, X, F, mask_t, C):
    model.eval()
    out = [model.head.probabilities(model(X[s:s + S.CHUNK], F[s:s + S.CHUNK], mask_t, C[s:s + S.CHUNK]))
           for s in range(0, X.shape[0], S.CHUNK)]
    return torch.cat(out).numpy()


def bag_probs(models, X, F, mask_t, C):
    return np.mean([probs_of(m, X, F, mask_t, C) for m in models], axis=0)


def train_model(X, F, mask_t, C, y_used, itr, iva, seed, max_epochs):
    """As train_transformer_svmrfe.train_model, with the comorbidity input."""
    model = new_model(X.shape[1], X.shape[2], C.shape[1], seed)
    opt = torch.optim.AdamW(model.parameters(), lr=S.TRAIN["lr"], weight_decay=S.TRAIN["weight_decay"])
    w = S.class_weights(y_used[itr])
    Xtr, Ftr, Ctr = X[itr], F[itr], C[itr]
    ytr = torch.from_numpy(y_used[itr])
    g = torch.Generator().manual_seed(seed)
    best = {"score": -np.inf, "epoch": -1, "state": None}
    bs = S.TRAIN["batch_size"]
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(itr), generator=g)
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            opt.zero_grad()
            loss = model.head.loss(model(Xtr[b], Ftr[b], mask_t, Ctr[b]), ytr[b], weight=w)
            if not torch.isfinite(loss):
                raise SystemExit("non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), S.TRAIN["grad_clip"])
            opt.step()
        pv = probs_of(model, X[iva], F[iva], mask_t, C[iva])
        score = H.evaluate(y_used[iva], pv.argmax(1))["balanced_accuracy"]
        if score > best["score"]:
            best = {"score": score, "epoch": ep,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        if ep - best["epoch"] >= S.TRAIN["patience"]:
            break
    model.load_state_dict(best["state"])
    model.eval()
    return model, best["epoch"]


def fit_bag(X, F, mask_t, C, y_used, tr, seed, max_epochs):
    """Same inner folds and seeds as train_transformer_svmrfe.fit_bag."""
    fo = H.stratified_folds(y_used, tr, S.N_INNER, np.random.default_rng([seed, 7]))
    models, epochs = [], []
    for j in range(S.N_INNER):
        itr = np.array([i for i in tr if fo[int(i)] != j])
        iva = np.array([i for i in tr if fo[int(i)] == j])
        m, ep = train_model(X, F, mask_t, C, y_used, itr, iva, seed * 10 + j, max_epochs)
        models.append(m)
        epochs.append(ep)
    return models, epochs


@torch.no_grad()
def explain(models, Xte, Fte, mask_t, Cte, y_te, p_te, seed, cols):
    """Gate, MC-dropout entropy, modality ablation, per-variable permutation importance."""
    rows = np.arange(len(y_te))
    gate = np.mean([m.forward_full(Xte, Fte, mask_t, Cte)[1].squeeze(1).numpy() for m in models], axis=0)
    pm = torch.stack([m.predict_mc(Xte, Fte, mask_t, Cte, S.MC_SAMPLES)[0] for m in models]).mean(0)
    pm = pm / pm.sum(1, keepdim=True)
    entropy = -(pm * torch.log(pm.clamp_min(1e-12))).sum(1).numpy()
    p_true = p_te[rows, y_te]
    # remove one modality: comorbidity -> training mean (0 after scaling);
    # brain -> blank tokens and no connectivity
    p_no_clin = bag_probs(models, Xte, Fte, mask_t, torch.zeros_like(Cte))[rows, y_te]
    p_no_brain = bag_probs(models, torch.zeros_like(Xte), torch.zeros_like(Fte), mask_t, Cte)[rows, y_te]
    # permutation importance: shuffle one variable across the test subjects
    rng = np.random.default_rng(seed)
    imp = {}
    for j, name in enumerate(cols):
        drops = []
        for _ in range(N_PERM_REPEATS):
            Cp = Cte.clone()
            Cp[:, j] = Cp[rng.permutation(len(y_te)), j]
            drops.append(float((p_true - bag_probs(models, Xte, Fte, mask_t, Cp)[rows, y_te]).mean()))
        imp[name] = float(np.mean(drops))
    return {"gate": gate, "entropy": entropy, "p_true": p_true, "p_true_no_clin": p_no_clin,
            "p_true_no_brain": p_no_brain}, imp


# ------------------------------------------------------------ one split
def run_split(cand, data, clin, ceil, cols, tr, te, seed, fold, fp, max_epochs, smoke=False,
              explain_outputs=True):
    X4, flat, fc, y, subjects, mask = data
    assert not set(tr.tolist()) & set(te.tolist())
    shuffled = cand.endswith("_SHUFFLED")
    y_used = y.copy()
    if shuffled:
        ytr = y[tr].copy()
        np.random.default_rng(seed * 100 + fold).shuffle(ytr)
        y_used[tr] = ytr
    t0 = time.time()
    rec = {"candidate": cand, "seed": seed, "fold": fold, "fold_fingerprint": fp,
           "n_train": int(tr.size), "n_test": int(te.size), "shuffled_labels": shuffled,
           "test_index": [int(i) for i in te], "subjects": [str(subjects[i]) for i in te],
           "true": [int(v) for v in y[te]]}

    if cand in ("logreg_clin", "ceiling_cdr_mmse", "ceiling_mmse_cdrsb", "logreg_svmrfe50_clin"):
        if cand == "logreg_clin":
            Xm = clin
        elif cand == "ceiling_cdr_mmse":
            Xm = ceil
        elif cand == "ceiling_mmse_cdrsb":
            Xm = ceil[:, [CEILING_COLS.index("MMSE_total"), CEILING_COLS.index("CDR_SOB")]]
        else:
            _, support, _ = S.fit_selection(flat, y_used, tr, seed + fold)
            Xm = np.hstack([flat[:, support], clin])
            rec["support"] = support.tolist()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe = logreg_pipe().fit(Xm[tr], y_used[tr])
        p_te = pipe.predict_proba(Xm[te])
        pred = pipe.predict(Xm[te]).astype(int)
        rec["train_balanced_accuracy"] = H.evaluate(y_used[tr], pipe.predict(Xm[tr]).astype(int))["balanced_accuracy"]
        if cand == "logreg_clin":
            coef = np.abs(pipe.named_steps["clf"].coef_).mean(0)
            rec["coef_abs_mean"] = dict(zip(cols, S.r6(coef)))
    else:
        sc = Scaler().fit(X4, tr)                                   # TRAINING SUBJECTS ONLY
        Xs = sc.transform(X4).astype(np.float32)
        _, support, chmask = S.fit_selection(flat, y_used, tr, seed + fold)
        tok, rois = S.make_tokens(Xs, chmask)
        X = torch.from_numpy(np.ascontiguousarray(tok))
        Ft = torch.from_numpy(np.ascontiguousarray(fc[:, rois][:, :, rois], dtype=np.float32))
        mask_t = torch.from_numpy(np.ascontiguousarray(mask[np.ix_(rois, rois)]))
        Ct = torch.from_numpy(ClinPrep().fit(clin, tr).transform(clin))   # TRAINING SUBJECTS ONLY
        assert torch.isfinite(Ct).all()
        if smoke:
            log(f"    tokens {tuple(X.shape)}  fc {tuple(Ft.shape)}  comorbidity {tuple(Ct.shape)}  K={rois.size}")
        models, epochs = fit_bag(X, Ft, mask_t, Ct, y_used, tr, seed * 100 + fold, max_epochs)
        p_te = bag_probs(models, X[te], Ft[te], mask_t, Ct[te])
        pred = p_te.argmax(1)
        p_tr = bag_probs(models, X[tr], Ft[tr], mask_t, Ct[tr])
        rec["train_balanced_accuracy"] = H.evaluate(y_used[tr], p_tr.argmax(1))["balanced_accuracy"]
        rec.update({"support": support.tolist(), "rois": rois.tolist(), "inner_best_epochs": epochs,
                    "n_parameters": int(sum(q.numel() for q in models[0].parameters()))})
        if cand == PRIMARY and explain_outputs:
            ex, imp = explain(models, X[te], Ft[te], mask_t, Ct[te], y[te], p_te, seed * 100 + fold, cols)
            for k, v in ex.items():
                rec[k] = S.r6(v)
            rec["perm_importance"] = imp
            if smoke:
                log(f"    explain: gate {ex['gate'].shape}  importance {len(imp)} variables")

    if not np.isfinite(p_te).all():
        raise SystemExit("non-finite probabilities")
    m = evaluate(np.asarray(p_te), y[te], pred=pred)
    rec.update({k: m[k] for k in S.METRIC_KEYS})
    rec["pred"] = [int(v) for v in pred]
    rec["probs"] = S.r6(p_te)
    rec["seconds"] = round(time.time() - t0, 1)
    log(f"    seed {seed} fold {fold}  test bal.acc {m['balanced_accuracy']:.3f}  "
        f"train {rec['train_balanced_accuracy']:.3f}  MAE {m['mae_stages']:.2f}  QWK {m['qwk']:+.3f}  "
        f"cls {m['n_predicted_classes']}  epochs {rec.get('inner_best_epochs', '-')}  {rec['seconds']:.0f}s")
    return rec


def save(out_dir, design, cand, records, extra):
    stem = os.path.join(out_dir, f"{design}_{cand}")
    json.dump({"extra": extra, "records": records}, open(stem + ".json", "w"))
    open(stem + ".log", "w", encoding="utf-8").write("\n".join(_lines) + "\n")
    log(f"  wrote {os.path.relpath(stem, ROOT)}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", choices=["split70", "cv", "both"], default="both")
    ap.add_argument("--candidates", nargs="+", default=CANDIDATES, choices=CANDIDATES)
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="comorbidity23")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--folds", type=int, default=5,
                    help="CV folds per seed; 5 -> cv116 (imaging-only records reused), "
                         "other k -> cv116k<k> (imaging-only rerun on the same folds)")
    ap.add_argument("--no-explain", action="store_true",
                    help="skip gate / importance / uncertainty outputs (accuracy-only runs)")
    ap.add_argument("--smoke", action="store_true", help="one CV fold of the primary, 3 epochs, nothing saved")
    args = ap.parse_args()
    expl = not args.no_explain
    torch.set_num_threads(args.threads)
    cols = FEATURE_SETS[args.features]
    out_dir = os.path.join(ROOT, "derivatives", "model_training", OUT_DIRS[args.features])
    os.makedirs(out_dir, exist_ok=True)

    data = S.load_cohort()
    y, subjects = data[3], data[4]
    clin, ceil = load_clinical(y, subjects, cols)
    _, _, y2, _, _, pool, test, _, smeta = load_all()
    assert np.array_equal(y2, y) and smeta["split_fingerprint"] == H.EXPECTED_SPLIT_FINGERPRINT
    plan = S.cv_plan(y, subjects, args.folds)
    common = (f"cohort {y.size}; clinical block '{args.features}', {len(cols)} features {cols}; "
              f"missing values {int(np.isnan(clin).sum())} (training-median fill)\n"
              f"architecture {S.ARCH}; training {S.TRAIN}; inner folds {S.N_INNER} (bagged)")

    if args.smoke:
        log("SMOKE: primary, CV seed 1 fold 0, 3 epochs, nothing saved")
        fp, folds = plan[H.SEEDS[0]]
        tr, te = folds[0]
        run_split(PRIMARY, data, clin, ceil, cols, tr, te, H.SEEDS[0], 0, fp, 3, smoke=True)
        log("SMOKE OK")
        return

    designs = ["split70", "cv"] if args.eval == "both" else [args.eval]
    for design in designs:
        for cand in args.candidates:
            if design == "split70" and cand == SHUF:
                continue
            if design == "cv" and cand in IMAGING_ONLY and args.folds == 5:
                print(f"{cand}: cv records are reused from derivatives/model_training/transformer_svmrfe/")
                continue
            dname = "split70" if design == "split70" else ("cv116" if args.folds == 5 else f"cv116k{args.folds}")
            if os.path.exists(os.path.join(out_dir, f"{dname}_{cand}.json")):
                print(f"COLLISION: {dname}_{cand}.json exists; skipped, not overwritten")
                continue
            _lines.clear()
            log("=" * 100)
            log(f"COMORBIDITY-AWARE TRANSFORMER -- design {dname}, candidate {cand}")
            log("=" * 100)
            log(common)
            t0 = time.time()
            records = []
            if design == "split70":
                log(f"split: derivatives/model_ready/splits, fingerprint {smeta['split_fingerprint']}, "
                    f"train {pool.size} / test {test.size}, test counts {smeta['test_class_counts']}")
                if cand in IMAGING_ONLY:
                    records.append(S.run_split(cand, data, pool, test, SPLIT70_SEED, 0,
                                               smeta["split_fingerprint"], S.TRAIN["max_epochs"], None,
                                               explain_outputs=expl))
                    r = records[-1]
                    log(f"    seed {SPLIT70_SEED} fold 0  test bal.acc {r['balanced_accuracy']:.3f}  "
                        f"train {r['train_balanced_accuracy']:.3f}")
                else:
                    records.append(run_split(cand, data, clin, ceil, cols, pool, test, SPLIT70_SEED, 0,
                                             smeta["split_fingerprint"], S.TRAIN["max_epochs"],
                                             explain_outputs=expl))
                extra = {"design": dname, "split_fingerprint": smeta["split_fingerprint"],
                         "test_class_counts": smeta["test_class_counts"]}
            else:
                log(f"{args.folds} folds x {len(H.SEEDS)} seeds; fold fingerprints "
                    + json.dumps({s: plan[s][0] for s in H.SEEDS}))
                for seed in H.SEEDS:
                    fp, folds = plan[seed]
                    for f, (tr, te) in enumerate(folds):
                        if cand in IMAGING_ONLY:
                            records.append(S.run_split(cand, data, tr, te, seed, f, fp, S.TRAIN["max_epochs"],
                                                       None, explain_outputs=expl))
                        else:
                            records.append(run_split(cand, data, clin, ceil, cols, tr, te, seed, f, fp,
                                                     S.TRAIN["max_epochs"], explain_outputs=expl))
                    sub = [r for r in records if r["seed"] == seed]
                    log(f"  seed {seed}: bal.acc {np.mean([r['balanced_accuracy'] for r in sub]):.3f}")
                s = H.summarise(records)
                log(f"  {cand}: seed-level bal.acc {s['balanced_accuracy'][0]:.3f} +/- "
                    f"{s['balanced_accuracy'][1]:.3f}  macroF1 {s['macro_f1'][0]:.3f}  "
                    f"MAE {s['mae_stages'][0]:.3f}  QWK {s['qwk'][0]:+.3f}  "
                    f"train {np.mean([r['train_balanced_accuracy'] for r in records]):.3f}")
                extra = {"design": dname, "n_folds": args.folds,
                         "fold_fingerprints": {str(s_): plan[s_][0] for s_ in H.SEEDS}}
            extra.update({"candidate": cand, "features": args.features, "clinical_columns": cols, "arch": S.ARCH,
                          "train": S.TRAIN, "n_inner": S.N_INNER, "seconds": round(time.time() - t0, 1)})
            log(f"  elapsed {time.time() - t0:.0f}s")
            save(out_dir, dname, cand, records, extra)


if __name__ == "__main__":
    main()
