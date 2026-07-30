"""Dataclass for the computed data during an axial length extraction of one eye.

This module contains the ALData dataclass, with all values computed while
extracting the axial length biomarker. These values are computed once and
shared between the extraction and plotting functions.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ALData:
    """All computed data for a single eye's axial length biomarker

    Attributes:
        raw_image_array:            Array form of the raw MRI (X, Y, Z)
        voxel_dim:                  Voxel dimensions in mm (dx, dy, dz)
        lens_mask:                  Mask for the lens
        globe_mask:                 Mask for the globe
        optic_nerve_mask:           Mask for the optic nerve
        intfat_mask:                Mask for the intraconal fat
        lens_centroid:              Centroid of the lens in voxel coordinates
        globe_centroid:             Centroid of the globe in voxel coordinates
        lens_to_globe_unit_vector:  Unit vector from lens centroid to globe centroid
        lens_anterior_boundary:     Most anterior lens voxel along the ray
        globe_posterior_boundary:   Most posterior globe voxel along the ray
        cornea_boundary:            First cornea voxel along the ray; None if not found
        intfat_boundary:            First intraconal fat voxel along the ray; None if not found
        axial_length_mm:            Distance from lens to globe, in mm
        extra_ant_mm:               Distance from cornea to lens, in mm; if no cornea found, median fallback value
        axial_length_cornea_mm:     Distance from cornea to globe, in mm; if no cornea found, median fallback value
    """

    raw_image_array: np.ndarray
    voxel_dim: np.ndarray
    lens_mask: np.ndarray
    globe_mask: np.ndarray
    optic_nerve_mask: np.ndarray
    intfat_mask: np.ndarray
    lens_centroid: np.ndarray
    globe_centroid: np.ndarray
    lens_to_globe_unit_vector: np.ndarray
    lens_anterior_boundary: np.ndarray
    globe_posterior_boundary: np.ndarray
    cornea_boundary: np.ndarray | None
    intfat_boundary: np.ndarray | None
    axial_length_mm: float
    extra_ant_mm: float
    axial_length_cornea_mm: float
