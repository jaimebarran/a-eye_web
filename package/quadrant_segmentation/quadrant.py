"""Image handler for the quadrant segmentation.

This module contains functions to perform the quadrant-based segmentation:
cropping each eye's quadrant before segmentation, then uncropping and merging
the two segmented quadrants back into a single image.

The cropping slices the data array directly, so it only finds an eye when array
axis 0 runs left-right and axis 1 runs posterior-anterior. That is true of a
NIfTI stored in RAS order but not of one stored any other way -- a PIL-ordered
image (as produced by some MPRAGE conversions) would have the same slice take an
inferior/posterior quadrant containing no eye at all, and the segmentation would
silently come back empty. Every image is therefore reoriented to canonical RAS
with `to_canonical` before it is cropped.
"""

# Adapted from (quadrant_segmentation.ipynb, jaimebarran, accessed 26.05.2026)
# URL: https://github.com/jaimebarran/a-eye_preprocessing/blob/main/Code/quadrant_segmentation.ipynb
from pathlib import Path

import nibabel as nib
import numpy as np


def to_canonical(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """To reorient a NIfTI to canonical RAS (axis 0 = R, 1 = A, 2 = S).

    This is a lossless permutation and flip of the data array with a matching
    change to the affine, so the image occupies exactly the same world space
    afterwards. Applying it to an already-canonical image is a no-op, which
    makes it safe to call more than once.

    Args:
        img (nib.Nifti1Image): the image to reorient

    Returns:
        nib.Nifti1Image: the same image in canonical RAS orientation
    """
    return nib.as_closest_canonical(img)


def crop_quadrant(img_path: Path, left_side: bool) -> nib.Nifti1Image:
    """To crop the NIfTI to the left/right upper quadrant, where the eye lies.

    The image is reoriented to canonical RAS first, so the crop finds the eye
    whatever orientation the uploaded file was stored in.

    Args:
        img_path (Path): path to the NIfTI file
        left_side (bool): True for the left eye quadrant, False for the right

    Returns:
        nib.Nifti1Image: the cropped NIfTI, in canonical RAS orientation
    """
    img = to_canonical(nib.load(img_path))
    data = img.get_fdata()
    # Crop to upper quadrant cube (RAS orientation)
    # Right: reduce left dimension (axis 0)
    # Upper: reduce inferior dimension (axis 2)
    # Anterior: reduce posterior dimension (axis 1)
    mid_x = data.shape[0] // 2
    mid_y = data.shape[1] // 2

    if left_side:
        cropped = data[:mid_x, mid_y:, :]
        offset = (0, mid_y, 0)
    else:
        cropped = data[mid_x:, mid_y:, :]
        offset = (mid_x, mid_y, 0)

    # Shift the origin to the corner the crop actually starts at, so the cropped
    # image still reports the correct world position. uncrop_quadrant reassembles
    # by index and does not depend on this, but a crop carrying the uncropped
    # origin misplaces the eye for anything that reads its affine.
    affine = img.affine.copy()
    affine[:3, 3] = img.affine[:3, :3] @ np.array(offset, dtype=float) + img.affine[:3, 3]

    return nib.Nifti1Image(cropped, affine, img.header)


def uncrop_quadrant(
    cropped_img: nib.Nifti1Image, original_shape: tuple[int, int, int], left_side: bool
) -> nib.Nifti1Image:
    """To reverse a cropped file back into the original image space

    Args:
        cropped_img (nib.Nifti1Image): segmentation output of the cropped quadrant
        original_shape (tuple[int, int, int]): shape of the original full image
        left_side (bool): True for the left eye quadrant, False for the right

    Returns:
        nib.Nifti1Image: the segmentation NIfTI in the original image space
    """
    cropped_data = np.asarray(cropped_img.dataobj).astype(np.uint8)
    full_data = np.zeros(original_shape, dtype=np.uint8)

    mid_x = original_shape[0] // 2
    mid_y = original_shape[1] // 2

    if left_side:
        full_data[:mid_x, mid_y:, :] = cropped_data
    else:
        full_data[mid_x:, mid_y:, :] = cropped_data

    new_header = cropped_img.header.copy()
    new_header.set_data_dtype(np.uint8)
    return nib.Nifti1Image(full_data, cropped_img.affine, new_header)


def merge_quadrants(
    left_img: nib.Nifti1Image, right_img: nib.Nifti1Image
) -> nib.Nifti1Image:
    """To merge the left and right uncropped quadrants

    Args:
        left_img (nib.Nifti1Image): uncropped left eye segmentation
        right_img (nib.Nifti1Image): uncropped right eye segmentation

    Returns:
        nib.Nifti1Image: merged NIfTI
    """
    left_data = np.asarray(left_img.dataobj).astype(np.uint8)
    right_data = np.asarray(right_img.dataobj).astype(np.uint8)
    # Merge: combine both segmentations (right overwrites where non-zero)
    merged_data = left_data.copy()
    merged_data[right_data > 0] = right_data[right_data > 0]

    return nib.Nifti1Image(merged_data, left_img.affine, left_img.header)
