#!/bin/bash
source ~/.bashrc 2>/dev/null
echo "=== FREESURFER_HOME ==="
echo $FREESURFER_HOME
echo "=== BUILD STAMP ==="
cat $FREESURFER_HOME/build-stamp.txt 2>/dev/null
echo "=== WHICH COMMANDS ==="
for cmd in recon-all preproc-sess mc-afni2 mri_fwhm mri_glmfit mri_convert; do
    echo -n "$cmd: "
    which $cmd || echo "MISSING"
done
echo "=== LICENSE ==="
if [ -f "$FREESURFER_HOME/license.txt" ]; then
    echo "AVAILABLE (license.txt)"
elif [ -f "$FREESURFER_HOME/.license" ]; then
    echo "AVAILABLE (.license)"
else
    echo "MISSING"
fi
echo "=== MC-AFNI2 HELP ==="
mc-afni2 --help 2>&1 | head -n 30
