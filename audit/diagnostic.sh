#!/bin/bash

echo "=== 1. Check directories ==="
ls -ld /home/harish/freesurfer 2>/dev/null || echo "MISSING /home/harish/freesurfer"
ls -ld /usr/local/freesurfer 2>/dev/null || echo "MISSING /usr/local/freesurfer"

echo "=== 2. Check binaries ==="
ls -l /home/harish/freesurfer/bin/recon-all 2>/dev/null || echo "MISSING /home/harish/recon-all"
ls -l /usr/local/freesurfer/bin/recon-all 2>/dev/null || echo "MISSING /usr/local/recon-all"

ls -l /home/harish/freesurfer/bin/mc-afni2 2>/dev/null || echo "MISSING /home/harish/mc-afni2"
ls -l /usr/local/freesurfer/bin/mc-afni2 2>/dev/null || echo "MISSING /usr/local/mc-afni2"

ls -l /home/harish/freesurfer/fsfast/bin/preproc-sess 2>/dev/null || echo "MISSING /home/harish/preproc-sess"
ls -l /usr/local/freesurfer/fsfast/bin/preproc-sess 2>/dev/null || echo "MISSING /usr/local/preproc-sess"

echo "=== 3. Search for installation ==="
find /home/harish /usr/local -maxdepth 4 \( -name recon-all -o -name preproc-sess -o -name mc-afni2 \) -type f 2>/dev/null

echo "=== 4. Check license ==="
ls -l /home/harish/freesurfer/license.txt 2>/dev/null || echo "MISSING /home/harish/freesurfer/license.txt"
ls -l /usr/local/freesurfer/license.txt 2>/dev/null || echo "MISSING /usr/local/freesurfer/license.txt"

echo "=== 5. Check build stamp ==="
cat /home/harish/freesurfer/build-stamp.txt 2>/dev/null || echo "MISSING /home/harish/freesurfer/build-stamp.txt"
cat /usr/local/freesurfer/build-stamp.txt 2>/dev/null || echo "MISSING /usr/local/freesurfer/build-stamp.txt"

echo "=== 7. Test sourcing ==="
if [ -f "/home/harish/freesurfer/SetUpFreeSurfer.sh" ]; then
    echo "--- Sourcing /home/harish/freesurfer ---"
    export FREESURFER_HOME=/home/harish/freesurfer
    source /home/harish/freesurfer/SetUpFreeSurfer.sh
    echo "FREESURFER_HOME: $FREESURFER_HOME"
    which recon-all || echo "recon-all MISSING"
    which preproc-sess || echo "preproc-sess MISSING"
    which mc-afni2 || echo "mc-afni2 MISSING"
    which mri_fwhm || echo "mri_fwhm MISSING"
    which mri_glmfit || echo "mri_glmfit MISSING"
    which mri_convert || echo "mri_convert MISSING"
fi

if [ -f "/usr/local/freesurfer/SetUpFreeSurfer.sh" ]; then
    echo "--- Sourcing /usr/local/freesurfer ---"
    export FREESURFER_HOME=/usr/local/freesurfer
    source /usr/local/freesurfer/SetUpFreeSurfer.sh
    echo "FREESURFER_HOME: $FREESURFER_HOME"
    which recon-all || echo "recon-all MISSING"
    which preproc-sess || echo "preproc-sess MISSING"
    which mc-afni2 || echo "mc-afni2 MISSING"
    which mri_fwhm || echo "mri_fwhm MISSING"
    which mri_glmfit || echo "mri_glmfit MISSING"
    which mri_convert || echo "mri_convert MISSING"
fi

echo "=== 8. Check Nipype ==="
/home/harish/fyp-neuro-env/bin/python3 -c "
import nipype
import nibabel
import nilearn
from nipype.interfaces.freesurfer import Info
print('Nipype version:', nipype.__version__)
print('NiBabel version:', nibabel.__version__)
print('Nilearn version:', nilearn.__version__)
print('FreeSurfer version detected by Nipype:', Info.version())
"
