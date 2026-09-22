#!/bin/bash
set -e

echo "=== 1. System Updates and Dependencies ==="
# Assuming passwordless sudo for WSL
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv wget curl bc file dc

echo "=== 2. Python Virtual Environment ==="
if [ ! -d "$HOME/fyp-neuro-env" ]; then
    python3 -m venv $HOME/fyp-neuro-env
fi
source $HOME/fyp-neuro-env/bin/activate
pip install --upgrade pip

echo "=== 3. Neuroimaging Python Packages ==="
pip install nipype nibabel nilearn numpy scipy pandas matplotlib pybids

echo "=== 4. FSL Installation ==="
if [ ! -d "$HOME/fsl" ]; then
    echo "Downloading fslinstaller.py..."
    wget -q https://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py -O /tmp/fslinstaller.py
    echo "Running FSL installer to ~/fsl (this may take a long time)..."
    python3 /tmp/fslinstaller.py -d $HOME/fsl -q
else
    echo "FSL already installed in $HOME/fsl"
fi

# Add FSL to profile if not already there
if ! grep -q "FSLDIR=" $HOME/.bashrc; then
    echo "# FSL Setup" >> $HOME/.bashrc
    echo "FSLDIR=$HOME/fsl" >> $HOME/.bashrc
    echo 'PATH=${FSLDIR}/share/fsl/bin:${PATH}' >> $HOME/.bashrc
    echo 'export FSLDIR PATH' >> $HOME/.bashrc
    echo '. ${FSLDIR}/etc/fslconf/fsl.sh' >> $HOME/.bashrc
fi

echo "Setup completed successfully."
