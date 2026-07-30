"""Image handler for the quadrant segmentation.

This module contains functions to perform the quadrant-based segmentation:
cropping each eye's quadrant before segmentation, then uncropping and merging
the two segmented quadrants back into a single image.
"""

# Adapted from (quadrant_segmentation.ipynb, jaimebarran, accessed 26.05.2026)
# URL: https://github.com/jaimebarran/a-eye_preprocessing/blob/main/Code/quadrant_segmentation.ipynb
from pathlib import Path

import nibabel as nib
import numpy as np


def crop_quadrant(img_path: Path, left_side: bool) -> nib.Nifti1Image:
    """To crop the NIfTI to the left/right upper quadrant, where the eye lies.

    Args:
        img_path (Path): path to the NIfTI file
        left_side (bool): True for the left eye quadrant, False for the right

    Returns:
        nib.Nifti1Image: the cropped NIfTI
    """
    img = nib.load(img_path)
    data = img.get_fdata()
    # Crop to upper quadrant cube (RAS orientation)
    # Right: reduce left dimension (axis 0)
    # Upper: reduce inferior dimension (axis 2)
    # Anterior: reduce posterior dimension (axis 1)
    mid_x = data.shape[0] // 2
    mid_y = data.shape[1] // 2

    if left_side:
        cropped = data[:mid_x, mid_y:, :]
    else:
        cropped = data[mid_x:, mid_y:, :]

    return nib.Nifti1Image(cropped, img.affine, img.header)


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
