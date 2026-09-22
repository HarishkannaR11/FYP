import os, sys, csv, time
sys.path.insert(0, "/mnt/c/Users/krish/FYP/pipeline")
from fsfast_production_4mm6mm import process_one_acquisition, GROUP_MANIFEST, MANIFEST_DIR, DERIV_ROOT, SCRATCH_ROOT, TPL_NAME
import templateflow.api as tflow

group = "AD"
manifest_rel, expected_n = GROUP_MANIFEST[group]
rows = list(csv.DictReader(open(os.path.join(MANIFEST_DIR, manifest_rel), newline="", encoding="utf-8")))
r = rows[0]
sub, ses, run = r["subject_id"], r["session_id"], r["run_id"]
bold_path = r["input_path"]
print(f"SMOKE TEST: {sub}/{ses}/{run}")

out_dir = os.path.join(DERIV_ROOT, group, sub, ses, run)
work_dir = os.path.join(SCRATCH_ROOT, "smoketest", f"{sub}_{ses}_{run}")

tpl_brain = str(tflow.get(TPL_NAME, resolution=2, desc="brain", suffix="T1w", extension="nii.gz"))
tpl_mask_path = str(tflow.get(TPL_NAME, resolution=2, desc="brain", suffix="mask", extension="nii.gz"))
wm_src = str(tflow.get("MNI152NLin2009cAsym", resolution=2, label="WM", suffix="probseg", extension="nii.gz"))
csf_src = str(tflow.get("MNI152NLin2009cAsym", resolution=2, label="CSF", suffix="probseg", extension="nii.gz"))


def log(msg):
    print(msg, flush=True)


t0 = time.time()
row = process_one_acquisition(group, sub, ses, run, bold_path, work_dir, out_dir,
                              tpl_brain, tpl_mask_path, wm_src, csf_src, log)
print("RESULT:", row)
print(f"SMOKE_TEST_ELAPSED: {time.time()-t0:.1f}s")
print("SMOKE_TEST_DONE")
