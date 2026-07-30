"""Visualisation maker for the biomarkers extraction part.

This module contains functions to make the axial length plot.
"""

# Adapted from (extract_biometrics.ipynb, jaimebarran, accessed 26.05.2026)
# URL: https://github.com/jaimebarran/a-eye_segmentation/blob/main/deep_learning/3D_multilabel/extract_biometrics.ipynb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .al_data import ALData


def _project_to_axial(
    coord: np.ndarray,
    voxel_dim: tuple[float, float],
) -> tuple[float, float]:
    """Project the 3D X,Y voxel coordinates to physical mm in the axial plane."""
    projected_coord: tuple = coord[0] * voxel_dim[0], coord[1] * voxel_dim[1]
    return projected_coord


def plot_axial_length(ray_data: ALData | None, case_name: str) -> Figure | None:
    """Plot of axial image (globe centroid slice)

    Args:
        ray_data (ALData | None): dataclass of all computed data for a single eye's axial length biomarker
        case_name (str): name of the case plotted

    Returns:
        Figure | None : the plot or None if not ray_data
    """
    if ray_data is None:
        return None

    # get axial slice through the globe centroid z coordinate
    axial_slice: int = int(round(ray_data.globe_centroid[2]))

    # extract the axial slice
    image_slice: np.ndarray = ray_data.raw_image_array[:, :, axial_slice].T

    # project 3D voxel coord -> physical mm in the axial plane
    voxel_xy_size: tuple = (
        float(ray_data.voxel_dim[0]),
        float(ray_data.voxel_dim[1]),
    )

    lens_xy: tuple = _project_to_axial(ray_data.lens_anterior_boundary, voxel_xy_size)
    globe_xy: tuple = _project_to_axial(
        ray_data.globe_posterior_boundary, voxel_xy_size
    )
    lens_centroid_xy: tuple = _project_to_axial(ray_data.lens_centroid, voxel_xy_size)
    globe_centroid_xy: tuple = _project_to_axial(ray_data.globe_centroid, voxel_xy_size)
    if ray_data.cornea_boundary is not None:
        cornea_xy: tuple = _project_to_axial(ray_data.cornea_boundary, voxel_xy_size)
    else:
        cornea_xy = None
    if ray_data.intfat_boundary is not None:
        intfat_xy: tuple = _project_to_axial(ray_data.intfat_boundary, voxel_xy_size)
    else:
        intfat_xy = None

    image_height: int
    image_width: int
    image_height, image_width = image_slice.shape  # rows=height, cols=width

    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor("white")

    ax.set_title(
        f"Axial image (globe centroid slice)\n"
        f"Case: {case_name} | z = {axial_slice}\n"
        f"Axial length: {ray_data.axial_length_mm} mm"
        f" | Extra ant: {ray_data.extra_ant_mm} mm "
        f" | AL cornea: {ray_data.axial_length_cornea_mm} mm",
        fontsize=10,
    )

    ax.imshow(
        image_slice,
        origin="lower",
        cmap="gray",
        interpolation="none",
        extent=[0, image_width * voxel_xy_size[0], 0, image_height * voxel_xy_size[1]],
    )
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x - (globe_centroid_xy[0] - pad):.0f}")
    )
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: f"{y - (globe_centroid_xy[1] - pad):.0f}")
    )
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")

    # line from lens to globe
    ax.plot([lens_xy[0], globe_xy[0]], [lens_xy[1], globe_xy[1]], "-y", lw=1.5)
    # line from cornea to lens
    if cornea_xy is not None:
        ax.plot([cornea_xy[0], lens_xy[0]], [cornea_xy[1], lens_xy[1]], "-c", lw=1.5)

    # points of interest
    ax.plot(*lens_xy, "+y", markersize=8, markeredgewidth=1.5)
    ax.plot(*globe_xy, "+y", markersize=8, markeredgewidth=1.5)
    ax.plot(*lens_centroid_xy, "+b", markersize=8, markeredgewidth=1.5)
    ax.plot(*globe_centroid_xy, "+b", markersize=8, markeredgewidth=1.5)
    if cornea_xy is not None:
        ax.plot(*cornea_xy, "+c", markersize=8, markeredgewidth=1.5)
    if intfat_xy is not None:
        ax.plot(*intfat_xy, "+c", markersize=8, markeredgewidth=1.5)

    # Zoom to region of interest
    pad: float = 25
    ax.set_xlim(globe_centroid_xy[0] - pad, globe_centroid_xy[0] + pad)
    ax.set_ylim(globe_centroid_xy[1] - pad, globe_centroid_xy[1] + pad)

    ax.legend(
        handles=[
            Line2D([0], [0], color="y", lw=2, label="Axial line"),
            Line2D(
                [],
                [],
                color="y",
                marker="+",
                markersize=8,
                linestyle="None",
                label="Extreme points",
            ),
            Line2D(
                [],
                [],
                color="b",
                marker="+",
                markersize=8,
                linestyle="None",
                label="Lens/Globe centroids",
            ),
            Line2D(
                [],
                [],
                color="c",
                marker="+",
                markersize=8,
                linestyle="None",
                label="Extra boundaries",
            ),
        ],
        loc="lower right",
        fontsize=6,
    )

    fig.tight_layout()
    return fig
