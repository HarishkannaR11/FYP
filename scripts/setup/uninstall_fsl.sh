#!/bin/bash
set -e

echo "=== UNINSTALLING FSL ==="
if [ -d "/home/harish/fsl" ]; then
    rm -rf /home/harish/fsl
    echo "Removed /home/harish/fsl directory"
else
    echo "FSL directory not found."
fi

echo "=== CLEANING .BASHRC ==="
sed -i '/# FSL Setup/d' /home/harish/.bashrc
sed -i '/FSLDIR=\/home\/harish\/fsl/d' /home/harish/.bashrc
sed -i '/PATH=${FSLDIR}\/share\/fsl\/bin:${PATH}/d' /home/harish/.bashrc
sed -i '/export FSLDIR PATH/d' /home/harish/.bashrc
sed -i '/\. ${FSLDIR}\/etc\/fslconf\/fsl.sh/d' /home/harish/.bashrc
echo "Removed FSL references from /home/harish/.bashrc"

echo "=== VERIFYING UNINSTALL ==="
fsl_check=$(which fsl 2>/dev/null || echo "Missing")
flirt_check=$(which flirt 2>/dev/null || echo "Missing")
mcflirt_check=$(which mcflirt 2>/dev/null || echo "Missing")
fslmaths_check=$(which fslmaths 2>/dev/null || echo "Missing")

echo "fsl: $fsl_check"
echo "flirt: $flirt_check"
echo "mcflirt: $mcflirt_check"
echo "fslmaths: $fslmaths_check"

echo "=== VERIFYING PYTHON ENV ==="
source /home/harish/fyp-neuro-env/bin/activate
pip_check=$(pip list | grep nipype || echo "Missing")

cat <<EOF > /home/harish/fsl_uninstall_verification.txt
=== FSL UNINSTALL VERIFICATION ===

Removed:
- /home/harish/fsl directory (if it existed)
- FSLDIR and PATH exports from /home/harish/.bashrc

FSL Executables Check:
fsl: $fsl_check
flirt: $flirt_check
mcflirt: $mcflirt_check
fslmaths: $fslmaths_check

Retained:
- /home/harish/fyp-neuro-env
- Python neuroimaging packages (e.g. Nipype installed: YES)
- BIDS dataset at /mnt/c/Users/krish/FYP/BIDS
EOF

echo "Verification written to /home/harish/fsl_uninstall_verification.txt"
