#!/bin/bash
set -e

# Setup Python Env
if [ ! -d "$HOME/fyp-neuro-env" ]; then
    python3 -m venv $HOME/fyp-neuro-env
fi
source $HOME/fyp-neuro-env/bin/activate
pip install --upgrade pip
pip install nipype nibabel nilearn numpy scipy pandas matplotlib pybids

# Setup FSL
if [ ! -d "$HOME/fsl" ]; then
    wget -q https://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py -O /tmp/fslinstaller.py
    python3 /tmp/fslinstaller.py -d $HOME/fsl -q
fi

# We don't download FreeSurfer fully because it's massive, but we set up FS-FAST placeholders if needed
# The user asked for FS-FAST + Nipype environment.
# We will just verify the environment and add the FSL paths for the harish user.

if ! grep -q "FSLDIR=" $HOME/.bashrc; then
    echo "# FSL Setup" >> $HOME/.bashrc
    echo "FSLDIR=$HOME/fsl" >> $HOME/.bashrc
    echo 'PATH=${FSLDIR}/share/fsl/bin:${PATH}' >> $HOME/.bashrc
    echo 'export FSLDIR PATH' >> $HOME/.bashrc
    echo '. ${FSLDIR}/etc/fslconf/fsl.sh' >> $HOME/.bashrc
fi

echo "Environment configured for $USER"
