#!/bin/bash

# Configuration
FS_URL="https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz"
FS_ARCHIVE="/home/harish/freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz"
FS_DIR="/home/harish/freesurfer"
AUDIT_TXT="/mnt/c/Users/krish/FYP/audit/fsfast_installation_verification.txt"
LICENSE_FILE="$FS_DIR/license.txt"

echo "=== FREESURFER INSTALLATION SCRIPT ==="

# 1. Download
if [ ! -f "$FS_ARCHIVE" ]; then
    echo "Downloading FreeSurfer 7.4.1 (This may take 10-20 minutes)..."
    wget -qO "$FS_ARCHIVE" "$FS_URL"
else
    echo "FreeSurfer archive already downloaded."
fi

# 2. Extract
if [ ! -d "$FS_DIR" ]; then
    echo "Extracting FreeSurfer (This may take 5-10 minutes)..."
    tar -xzf "$FS_ARCHIVE" -C /home/harish/
else
    echo "FreeSurfer directory already exists."
fi

# 3. Configure .bashrc
if ! grep -q "FREESURFER_HOME" /home/harish/.bashrc; then
    echo "" >> /home/harish/.bashrc
    echo "# FreeSurfer Setup" >> /home/harish/.bashrc
    echo "export FREESURFER_HOME=$FS_DIR" >> /home/harish/.bashrc
    echo "source \$FREESURFER_HOME/SetUpFreeSurfer.sh > /dev/null 2>&1" >> /home/harish/.bashrc
fi

# 4. Source for verification
export FREESURFER_HOME=$FS_DIR
source $FREESURFER_HOME/SetUpFreeSurfer.sh > /dev/null 2>&1 || true

# 5. Check License
LICENSE_STATUS="MISSING. A license.txt file is strictly required to run FreeSurfer commands."
if [ -f "$LICENSE_FILE" ]; then
    LICENSE_STATUS="FOUND"
fi

# 6. Verify Commands
cmd_recon=$(which recon-all || echo "Not Found")
cmd_preproc=$(which preproc-sess || echo "Not Found")
cmd_mcafni=$(which mc-afni2 || echo "Not Found")
cmd_mrifwhm=$(which mri_fwhm || echo "Not Found")
cmd_mriglm=$(which mri_glmfit || echo "Not Found")
cmd_mriconv=$(which mri_convert || echo "Not Found")

# 7. Check Python & Nipype
source /home/harish/fyp-neuro-env/bin/activate
py_ver=$(python3 --version 2>&1)
nipype_ver=$(python3 -c "import nipype; print(nipype.__version__)" 2>/dev/null || echo "Nipype Missing")

# 8. Write Verification Report
cat <<EOF > "$AUDIT_TXT"
=== FS-FAST INSTALLATION VERIFICATION ===

-- FreeSurfer Information --
FreeSurfer Version: 7.4.1
FREESURFER_HOME: $FREESURFER_HOME
Installation Path: $FS_DIR
License Status: $LICENSE_STATUS

-- FS-FAST / FreeSurfer Commands Found --
recon-all: $cmd_recon
preproc-sess: $cmd_preproc
mc-afni2: $cmd_mcafni
mri_fwhm: $cmd_mrifwhm
mri_glmfit: $cmd_mriglm
mri_convert: $cmd_mriconv

-- Python Environment --
Python Version: $py_ver
Nipype Version: $nipype_ver
Virtual Environment: /home/harish/fyp-neuro-env

-- Installation Warnings/Errors --
If License Status is MISSING, all commands above will fail to execute when run (they will throw a license error). 
You must obtain a free license from https://surfer.nmr.mgh.harvard.edu/registration.html and save it to $LICENSE_FILE
EOF

echo "Installation complete. Verification saved to $AUDIT_TXT"
