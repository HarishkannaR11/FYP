# FreeSurfer 7.4.1 + FS-FAST + Nipype — WSL Setup Guide

This guide installs everything needed for BOLD-only, FS-FAST + Nipype rs-fMRI
preprocessing on WSL2 Ubuntu. It's written from a real install on this project
that hit several real problems — each "gotcha" below actually happened, so
don't skip them.

---

## 0. Prerequisites

- Windows 10/11 with WSL2 and an Ubuntu distro already installed
  (`wsl --install -d Ubuntu` from PowerShell if you don't have one yet)
- Your normal user account inside that Ubuntu distro, with `sudo` access
- At least **10–15 GB free disk space** inside the WSL filesystem
- A **stable** internet connection (the archive is ~9.3 GB — see the gotcha
  below about why a flaky connection matters more than you'd think)

Open your WSL Ubuntu terminal for everything below.

---

## 1. Check how much RAM your WSL VM actually has

```bash
free -m
nproc
```

⚠️ **Gotcha:** WSL2 caps the VM's memory, typically to about 50% of your
Windows host's physical RAM, unless overridden. Don't assume you have as much
RAM as your machine actually has — check it. If you need more, create/edit
`C:\Users\<you>\.wslconfig` on the Windows side:

```ini
[wsl2]
memory=6GB
processors=4
```

Then run `wsl --shutdown` in PowerShell and reopen your terminal.

---

## 2. Install system dependencies

```bash
sudo apt update
sudo apt install -y tcsh wget build-essential python3-venv python3-pip
```

⚠️ **Gotcha — this one will silently break motion correction if skipped:**
`recon-all`, `preproc-sess`, and **`mc-afni2`** (the actual motion-correction
command this pipeline uses) are `csh`/`tcsh` **scripts**, not compiled
binaries. Without `tcsh` installed, they fail with:
```
bash: /path/to/mc-afni2: cannot execute: required file not found
```
This is easy to miss because `mri_fwhm`, `mri_glmfit`, and `mri_convert` *are*
compiled binaries and work fine without `tcsh` — so a partial verification can
look successful when it isn't. Install `tcsh` up front.

Verify it actually installed (don't just trust the install command ran):
```bash
apt-cache policy tcsh
# should show "Installed: <version>", NOT "Installed: (none)"
```

---

## 3. Download FreeSurfer 7.4.1 — ONE download, verified

```bash
cd ~
wget -O freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz \
  "https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz"
```

This is a **large file (~9.3 GB)** and can take a long time on a slow
connection. Let it run to completion in one sitting if you can.

⚠️ **Gotcha:** if this download gets interrupted (closed terminal, lost
connection, laptop sleep) and you restart it as a *second* process while the
first is still running in the background, **both processes write to the same
file at once and corrupt it silently** — `wget` won't necessarily error, but
the resulting archive fails to extract cleanly, sometimes only partway
through, wasting a lot of time. Before restarting a download, always check
nothing is already running:
```bash
ps aux | grep wget
```
If you do need to resume an interrupted download, use `wget -c` (continue),
never a second plain `wget` on the same output file.

**Verify integrity before extracting — don't skip this:**
```bash
gzip -t freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz && echo "GZIP OK" || echo "GZIP BAD"
tar -tzf freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz | wc -l
# should print roughly 80,000+ entries with no errors
```
There's no official published checksum for this file, so this two-step check
(gzip stream integrity + full tar listing) is the practical substitute. If
either step fails or hangs, the archive is incomplete — delete it and
re-download rather than trying to extract a bad file.

---

## 4. Extract

```bash
tar -xzf freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz -C ~/
```

This takes several minutes. When done, confirm `bin/` actually has content
(a truncated/interrupted extraction leaves everything *except* `bin/`, since
tar writes in archive order and `bin/` isn't first):

```bash
ls ~/freesurfer/bin | wc -l
# should be 800+
```

---

## 5. Get a license file

FreeSurfer requires a free license. Register here:
https://surfer.nmr.mgh.harvard.edu/registration.html

You'll get a `license.txt` (usually emailed, or shown directly on the
registration page as 5 lines of text — an email address, an id number, and
3 key lines). **Copy it exactly as given** — don't retype it by hand from a
screenshot; character-transcription mistakes (like `1` vs `l`) are easy to
make and will produce a license that silently fails to validate.

Save it as plain text with **Linux line endings** (not Windows CRLF):

```bash
cat > ~/freesurfer/license.txt << 'EOF'
<paste your exact license text here>
EOF
```

Confirm there's no `\r` contamination:
```bash
cat -A ~/freesurfer/license.txt | head -1
# each line should end in just "$", not "^M$"
```

---

## 6. Configure the environment

```bash
cat >> ~/.bashrc << 'EOF'

# FreeSurfer Setup
export FREESURFER_HOME=$HOME/freesurfer
export SUBJECTS_DIR=$FREESURFER_HOME/subjects
source $FREESURFER_HOME/SetUpFreeSurfer.sh > /dev/null 2>&1
EOF
```

⚠️ **Gotcha:** `~/.bashrc` starts with a standard guard —
```bash
case $- in
    *i*) ;;
      *) return;;
esac
```
— which returns immediately for **non-interactive** shells. That means
`FREESURFER_HOME` etc. only get set when you open a normal interactive
terminal. If you ever run FreeSurfer commands from a script, a cron job, or
via `bash -c "..."` non-interactively (including from tools like Nipype if it
spawns a plain subprocess), they **won't** see these variables unless you set
them explicitly in that context too. This is normal `.bashrc` behavior, not a
bug — just don't be surprised if a script can't find `recon-all` even though
your terminal can.

⚠️ **Include `SUBJECTS_DIR`, even with no T1w/anatomical subjects.**
`mri_glmfit` (and several other FS-FAST tools) check for `SUBJECTS_DIR` being
set at all, even for operations that don't use a subject — without it you'll
get:
```
ERROR: SUBJECTS_DIR not defined in environment
```
Pointing it at FreeSurfer's own default `$FREESURFER_HOME/subjects` (which
ships with the install, e.g. `fsaverage`) satisfies this without needing any
of your own subject data in it.

Reload your shell:
```bash
source ~/.bashrc
```

---

## 7. Verify — in an actual interactive shell

Open a **new terminal window** (don't just re-source in a script), then:

```bash
echo $FREESURFER_HOME
which recon-all
which preproc-sess
which mc-afni2
which mri_fwhm
which mri_glmfit
which mri_convert
```

All six should resolve to real paths under `~/freesurfer/`. If any say
"not found," go back to step 2 (tcsh) or step 6 (.bashrc) depending on which
ones fail — compiled tools (`mri_fwhm`, `mri_glmfit`, `mri_convert`) failing
too means it's a PATH/`.bashrc` issue; only the `tcsh`-dependent ones
(`recon-all`, `preproc-sess`, `mc-afni2`) failing means step 2 didn't take.

---

## 8. Python environment for Nipype orchestration

```bash
cd ~
python3 -m venv fyp-neuro-env
source fyp-neuro-env/bin/activate
pip install --upgrade pip
pip install nipype nibabel nilearn pybids pydicom numpy matplotlib
```

Verify:
```bash
python3 -c "
import nipype, nibabel, nilearn, bids, pydicom
print('nipype', nipype.__version__)
print('nibabel', nibabel.__version__)
print('nilearn', nilearn.__version__)
print('pybids', bids.__version__)
print('pydicom', pydicom.__version__)
"
```

---

## 9. End-to-end sanity check

Confirm FreeSurfer commands actually *run* (not just resolve on PATH), and
that a Nipype `CommandLine` node can call them — since `mc-afni2`/`mri_fwhm`
have no dedicated Nipype interface, they must be wrapped manually:

```bash
mri_fwhm --help | head -5
mc-afni2 --help | head -5
```

Both should print usage text (not "command not found" or a licensing error).
If `mri_fwhm`/`mri_glmfit` complain about a missing or invalid license, go
back to step 5 and double check the license file for `\r` contamination or a
transcription mistake.

---

## Summary checklist

- [ ] `tcsh` installed and verified (`apt-cache policy tcsh`)
- [ ] FreeSurfer archive downloaded **once**, no overlapping `wget` processes
- [ ] Archive integrity verified (`gzip -t` + `tar -tzf`) before extracting
- [ ] `bin/` has 800+ files after extraction
- [ ] `license.txt` in place, no CRLF contamination
- [ ] `.bashrc` has `FREESURFER_HOME`, `SUBJECTS_DIR`, and sources
      `SetUpFreeSurfer.sh`
- [ ] All 6 commands resolve in a **new interactive terminal**
- [ ] Python venv has nipype/nibabel/nilearn/pybids/pydicom installed
- [ ] `mri_fwhm --help` and `mc-afni2 --help` both print usage text with no
      license error
