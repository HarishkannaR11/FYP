#!/bin/bash
echo "Python version:"
python3 --version || echo "python3 not found"
echo "Pip version:"
pip3 --version || echo "pip3 not found"

echo "FSL check:"
which fsl || echo "fsl not found"
which flirt || echo "flirt not found"
which mcflirt || echo "mcflirt not found"
which fslmaths || echo "fslmaths not found"

echo "FreeSurfer check:"
which freesurfer || echo "freesurfer not found"

echo "Python packages check:"
python3 -c "
try:
    import nipype
    print('Nipype:', nipype.__version__)
except ImportError:
    print('Nipype: Missing')

try:
    import nibabel
    print('NiBabel:', nibabel.__version__)
except ImportError:
    print('NiBabel: Missing')

try:
    import nilearn
    print('Nilearn:', nilearn.__version__)
except ImportError:
    print('Nilearn: Missing')

try:
    import numpy
    print('NumPy:', numpy.__version__)
except ImportError:
    print('NumPy: Missing')

try:
    import scipy
    print('SciPy:', scipy.__version__)
except ImportError:
    print('SciPy: Missing')
"
