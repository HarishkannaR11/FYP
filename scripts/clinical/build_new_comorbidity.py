"""Build Comorbidity/new_comorbidity: one comorbidity (MEDHIST = medical history) CSV per stage,
restricted to the first MAX_SUBJECTS unique subjects of that stage (in sheet order) in
selected_acquisitions.csv (fewer if the stage has fewer, e.g. MCI = 14). ZIPs are only read."""
import csv
import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "Comorbidity"
OUT = SRC / "new_comorbidity"
INVENTORY = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "selected_acquisitions.csv"
ZIPS = {"AD": "Tables_01Sep2026 _AD.zip", "CN": "Tables_01Sep2026 _CN.zip", "EMCI": "Tables_01Sep2026 _EMCI.zip",
        "LMCI": "Tables_12Oct2026_LMCI.zip", "MCI": "Tables_01Sep2026_MCI.zip", "SMC": "Tables_01Sep2026_SMC.zip"}

MAX_SUBJECTS = 20
inv = {}
with open(INVENTORY, newline="", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        ids = inv.setdefault(r["stage"], [])
        pid = r["subject"][4:7] + "_S_" + r["subject"][-4:]
        if pid not in ids and len(ids) < MAX_SUBJECTS:
            ids.append(pid)
OUT.mkdir(exist_ok=True)
for stage, z in ZIPS.items():
    with zipfile.ZipFile(SRC / z) as zf:
        n = next(x for x in zf.namelist() if "_MEDHIST_" in x)
        rd = csv.reader(io.TextIOWrapper(zf.open(n), "utf-8-sig", newline=""))
        header = next(rd)
        rows = [r for r in rd if r[header.index("PTID")] in inv[stage]]
    with open(OUT / f"{stage}_comorbidity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    found = {r[1] for r in rows}
    print(stage, "subjects", len(found), "/", len(inv[stage]), "rows", len(rows), "missing", sorted(set(inv[stage]) - found))
