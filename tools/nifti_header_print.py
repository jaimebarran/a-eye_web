import sys

import nibabel as nib

# use with : python tools/nifti_header_print.py data/test-webplatform/input/dcm_nii_niigz/2022160100003.nii

img = nib.load(sys.argv[1])
print(img.header)
