import nipype
import nibabel
import nilearn
from nipype.interfaces.freesurfer import Info

print("Nipype:", nipype.__version__)
print("NiBabel:", nibabel.__version__)
print("Nilearn:", nilearn.__version__)
print("FS Info:", Info.version())
