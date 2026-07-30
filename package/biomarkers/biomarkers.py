"""Biomarker extraction from a segmented eye NIfTI image.

This module contains both biomarkers (volumetry and axial length) extraction
functions.
"""

# Adapted from (extract_biometrics.ipynb, jaimebarran, accessed 26.05.2026)
# URL: https://github.com/jaimebarran/a-eye_segmentation/blob/main/deep_learning/3D_multilabel/extract_biometrics.ipynb
import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import convolve, find_peaks
from skimage.measure import regionprops

from .al_data import ALData

LABELS: dict[int, str] = {
    1: "vol_lens",
    2: "vol_globe",
    3: "vol_nerve",
    4: "vol_int_fat",
    5: "vol_ext_fat",
    6: "vol_lat_mus",
    7: "vol_med_mus",
    8: "vol_inf_mus",
    9: "vol_sup_mus",
}

EXTRA_ANT_MEDIAN = 2.72


def _centroid_of_mask(mask: np.ndarray) -> np.ndarray:
    """Return centroid (x,y,z) in voxel coordinates as float array"""
    region_properties = regionprops(mask.astype(int))

    if not region_properties:
        raise ValueError("Empty mask passed to _centroid_of_mask.")

    return np.asarray(region_properties[0].centroid, dtype=float)


def _sample_ray(start: np.ndarray, end: np.ndarray, n_samples: int) -> np.ndarray:
    """Return n_samples coordinates evenly spaced from start to end along the ray"""
    return np.linspace(start, end, n_samples)


def _sample_intensities(image: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Sample image intensities at ray coordinates using trilinear interpolation"""
    return map_coordinates(image, coords.T, order=1, mode="nearest")


def _nearest_labels(mask: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Sample mask labels at each ray coordinate using nearest-neighbour rounding"""
    idx = np.round(coords).astype(int)
    idx[:, 0] = np.clip(idx[:, 0], 0, mask.shape[0] - 1)
    idx[:, 1] = np.clip(idx[:, 1], 0, mask.shape[1] - 1)
    idx[:, 2] = np.clip(idx[:, 2], 0, mask.shape[2] - 1)
    return mask[idx[:, 0], idx[:, 1], idx[:, 2]]


def _line_box_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Compute entry/exit points of a line through a 3D volume bounding box

    Using slab algorithm, reference : https://en.wikipedia.org/wiki/Slab_method

    Args:
        origin (np.ndarray): ray's starting point in voxel coordinates
        direction (np.ndarray): ray's unit direction vector
        shape (tuple[int, ...]): 3D volume bounding box's shape

    Returns:
        tuple[np.ndarray, np.ndarray] | tuple[None, None]: entry and exit points in voxel coordinates
    """
    eps = 1e-12
    t_close_list = []
    t_far_list = []

    # for each axis, compute the t values where the ray hits the low and high bounding planes
    for axis in range(3):
        direction_component = float(direction[axis])
        origin_component = float(origin[axis])
        axis_low = 0.0
        axis_high = float(shape[axis] - 1)
        if abs(direction_component) < eps:
            # ray is parallel to this axis — valid only if origin is already inside the slab
            if not (axis_low <= origin_component <= axis_high):
                return None, None
            continue
        # t at which the ray crosses each bounding plane along this axis
        t_entry_candidate = (axis_low - origin_component) / direction_component
        t_exit_candidate = (axis_high - origin_component) / direction_component
        # t_close is the entry into this slab and t_far the exit
        t_close_list.append(min(t_entry_candidate, t_exit_candidate))
        t_far_list.append(max(t_entry_candidate, t_exit_candidate))
    if not t_close_list:
        return None, None

    # the ray is inside the box only in the overlap of all three slabs
    t_close = max(t_close_list)
    t_far = min(t_far_list)
    if t_far < t_close:
        # slabs don't overlap — ray misses the box
        return None, None
    return origin + t_close * direction, origin + t_far * direction


def _detect_cornea(
    int_profile: np.ndarray,
    limit_idx: int,
    kernel: np.ndarray | None = None,
    thr_list: list[float] | None = None,
    min_drop: float = 0.03,
    peak_prominence: float = 0.05,
) -> int | None:
    """Detect cornea as the first gradient peak after the first major gradient drop.

    Args:
        int_profile (np.ndarray): MRI intensity profile sampled along the ray
        limit_idx (int): Limit the search region
        kernel (np.ndarray | None, optional): kernel used to compute the gradient
        thr_list (list[float] | None, optional): gradient thresholds normalised, for the search of the first peak
        min_drop (float, optional): minimum gradient normalised decrease to be taken as a drop
        peak_prominence (float, optional): minimum peak prominence, for the cornea peak

    Returns:
        int | None: index of the detected cornea along the ray if found
    """
    if kernel is None:
        kernel = np.array([1, 0, -1])

    # Calculate the gradient (intensity change's magnitude) of tissue boundaries
    gradient = np.abs(convolve(int_profile, kernel, mode="same"))
    if gradient.size == 0 or gradient.max() == 0:
        return None

    # normalise to apply general threshold then
    normalized_gradient = gradient / gradient.max()

    if thr_list is None:
        thr_list = [0.45, 0.35, 0.25, 0.15, 0.1, 0.05]

    # restrict the detection until the lens boundary
    candidate_mask = np.arange(len(normalized_gradient)) < limit_idx
    first_drop_idx = None
    # Using custom thresholds, from strictest to loosest. We prefer to find a clear demarcation
    for threshold in thr_list:
        # finding all potential starting peak
        candidate_indices = np.where(
            (normalized_gradient > threshold) & candidate_mask
        )[0]
        if candidate_indices.size == 0:
            continue
        # scanning for the bottom of the drop after a peak
        idx = int(candidate_indices[0])
        while idx < limit_idx - 1:
            if normalized_gradient[idx + 1] < normalized_gradient[idx] - min_drop:
                # going down to the bottom of the drop
                while (
                    idx < limit_idx - 1
                    and normalized_gradient[idx + 1] < normalized_gradient[idx]
                ):
                    idx += 1
                first_drop_idx = idx
                break
            idx += 1
        if first_drop_idx is not None:
            break

    # if no significant drop found : cornea can't be located
    if first_drop_idx is None or first_drop_idx >= limit_idx - 2:
        return None

    # the cornea should be a smaller secondary peak right after the drop
    search_region = normalized_gradient[first_drop_idx + 1 : limit_idx]
    if search_region.size < 3:
        return None

    # searching the cornea peak
    for prominence in [peak_prominence, 0.05, 0.03, 0.01]:
        peaks, _ = find_peaks(search_region, prominence=prominence)
        if len(peaks):
            return int(first_drop_idx + 1 + peaks[0])

    # Fallback: when no clear peak, take the region's maximum (after the drop) as the cornea position
    return int(first_drop_idx + 1 + np.argmax(search_region))


def _find_first_intfat_index(
    globe_ext_idx: int, intfat_labels_along: np.ndarray
) -> int | None:
    """Find index of the first intraconal fat sample beyond the posterior globe boundary"""
    search = np.where(intfat_labels_along[globe_ext_idx + 1 :] > 0)[0]
    return int(globe_ext_idx + 1 + search[0]) if search.size else None


def compute_axial_length_data(
    segmented_image: nib.Nifti1Image, raw_image: nib.Nifti1Image
) -> ALData | None:
    """Compute all ray-based coordinates and measurements for axial length.

    Args:
        segmented_image (nib.Nifti1Image): cropped quadrant segmented image
        raw_image (nib.Nifti1Image): cropped quadrant original image

    Returns:
        ALData | None: populated ALData, or None if no lens or globe after segmentation
    """
    voxel_dim = np.array(segmented_image.header.get_zooms()[:3])
    segmented_arr = segmented_image.get_fdata()
    raw_image_array = raw_image.get_fdata()

    # extract binary masks for each structure from the segmentation labels
    lens_mask = (segmented_arr == 1).astype(int)
    globe_mask = (segmented_arr == 2).astype(int)
    optic_nerve_mask = (segmented_arr == 3).astype(int)
    intfat_mask = (segmented_arr == 4).astype(int)

    # ensure that lens and globe are present
    if np.count_nonzero(lens_mask) == 0 or np.count_nonzero(globe_mask) == 0:
        return None

    # find the geometric centroids
    try:
        lens_centroid = _centroid_of_mask(lens_mask)
        globe_centroid = _centroid_of_mask(globe_mask)
    except ValueError:
        return None

    # get the unit vector defining the ray axis from lens to globe
    lens_to_globe_vector = globe_centroid - lens_centroid
    norm = float(np.linalg.norm(lens_to_globe_vector))
    lens_to_globe_unit_vector = lens_to_globe_vector / norm

    # extend the ray to the full volume bounding box to capture cornea and intraconal_fat
    start, end = _line_box_intersection(
        lens_centroid, lens_to_globe_unit_vector, raw_image_array.shape
    )
    if start is None or end is None:
        return None

    # sample densely enough to not miss any sub-voxel structure boundary
    approx_mm_length = float(np.linalg.norm((end - start) * voxel_dim))
    min_vox_mm = float(voxel_dim.min())
    samples = max(800, int(np.ceil(approx_mm_length / (min_vox_mm / 4.0))))
    ray_coords = _sample_ray(start, end, samples)

    # sample MRI intensities and structure labels at each point along the ray
    intensities = _sample_intensities(raw_image_array, ray_coords)
    lens_labels_along = _nearest_labels(lens_mask, ray_coords)
    globe_labels_along = _nearest_labels(globe_mask, ray_coords)
    intfat_labels_along = _nearest_labels(intfat_mask, ray_coords)

    # find the anterior lens edge (first lens hit) and posterior globe edge (last globe hit)
    lens_idxs = np.where(lens_labels_along > 0)[0]
    globe_idxs = np.where(globe_labels_along > 0)[0]
    lens_ant_idx = int(lens_idxs.min())
    lens_ext_idx = int(lens_idxs.max())
    globe_ext_idx = int(globe_idxs.max())
    lens_anterior_boundary = ray_coords[lens_ant_idx]
    globe_posterior_boundary = ray_coords[globe_ext_idx]

    # detect cornea: first try a wide search (before posterior lens edge), then validate
    # that the result is truly anterior to the lens; fall back to a narrow search if not
    cornea_idx = _detect_cornea(intensities, lens_ext_idx)
    if cornea_idx is None or cornea_idx >= lens_ant_idx:
        cornea_idx = _detect_cornea(intensities, lens_ant_idx)
    if cornea_idx is not None and cornea_idx < lens_ant_idx:
        cornea_boundary = ray_coords[cornea_idx]
    else:
        cornea_boundary = None

    # find first intraconal fat sample beyond the posterior globe boundary
    intfat_idx = _find_first_intfat_index(globe_ext_idx, intfat_labels_along)
    intfat_boundary = ray_coords[intfat_idx] if intfat_idx is not None else None

    # compute distances in mm
    lens_to_globe_vec_mm = (
        lens_anterior_boundary - globe_posterior_boundary
    ) * voxel_dim
    axial_length_mm = float(np.round(np.linalg.norm(lens_to_globe_vec_mm), 2))

    if cornea_boundary is not None:
        cornea_to_lens_vec_mm = (cornea_boundary - lens_anterior_boundary) * voxel_dim
        extra_ant_mm = float(np.round(np.linalg.norm(cornea_to_lens_vec_mm), 2))
    else:
        # cornea is undetected, use the median fallback
        extra_ant_mm = EXTRA_ANT_MEDIAN
    axial_length_cornea_mm = float(np.round(extra_ant_mm + axial_length_mm, 2))

    return ALData(
        raw_image_array=raw_image_array,
        voxel_dim=voxel_dim,
        lens_mask=lens_mask,
        globe_mask=globe_mask,
        optic_nerve_mask=optic_nerve_mask,
        intfat_mask=intfat_mask,
        lens_centroid=lens_centroid,
        globe_centroid=globe_centroid,
        lens_to_globe_unit_vector=lens_to_globe_unit_vector,
        lens_anterior_boundary=lens_anterior_boundary,
        globe_posterior_boundary=globe_posterior_boundary,
        cornea_boundary=cornea_boundary,
        intfat_boundary=intfat_boundary,
        axial_length_mm=axial_length_mm,
        extra_ant_mm=extra_ant_mm,
        axial_length_cornea_mm=axial_length_cornea_mm,
    )


def extract_axial_length_measurements(ray_data: ALData | None) -> dict[str, float]:
    """Extract axial length measurements from computed ray data.

    Args:
        ray_data (ALData | None): computed data for a single eye's axial length biomarker

    Returns:
        dict[str, float]: axial_length, axial_length_cornea and extra_ant measurements in mm, NaN if not available

    """
    nan = float("nan")
    if ray_data is None:
        return {"axial_length": nan, "axial_length_cornea": nan, "extra_ant": nan}

    return {
        "axial_length": ray_data.axial_length_mm,
        "axial_length_cornea": ray_data.axial_length_cornea_mm,
        "extra_ant": ray_data.extra_ant_mm,
    }


def compute_volumetry(image: nib.Nifti1Image) -> dict[str, float]:
    """Compute volumes for each class of the ROI in mm3.

    Args:
        image (nib.Nifti1Image): cropped segmented NIfTI image

    Returns:
        dict[str, float]: class name -> volume in mm3

    """
    dx, dy, dz = image.header.get_zooms()[:3]
    voxel_volume = float(dx * dy * dz)

    label_array = image.get_fdata()
    volumes: dict[str, float] = {}

    for i, name in LABELS.items():
        volumes[name] = float(np.count_nonzero(label_array == i)) * voxel_volume

    volumes["vol_total"] = float(np.count_nonzero(label_array)) * voxel_volume

    return volumes
