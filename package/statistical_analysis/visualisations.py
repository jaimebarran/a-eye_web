"""Visualisation maker for case's vs reference population comparison.

This module contains functions to make both plot comparing the case's extracted
biomarker value and the reference dataset ones.
"""

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


# Adapted from (extract_biometrics.ipynb, jaimebarran, accessed 26.05.2026)
# URL: https://github.com/jaimebarran/a-eye_segmentation/blob/main/deep_learning/3D_multilabel/extract_biometrics.ipynb
def plot_volumetry_violin(
    ref_df: pd.DataFrame, case_vol_data: dict[str, float]
) -> Figure:
    """Plot the volumetry in violin plot, distinguishing sex and region

    Args:
        ref_df (pd.DataFrame): Dataframe of the reference values
        case_vol_data (dict[str, float]): Dict of the extracted values of current case

    Returns:
        Figure: A nine violin plot grid, with case's data highlighted
    """
    # High-resolution, readable figure settings
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16,
    })

    # Create a list of the structures to loop through
    structures = [
        ("LENS", ref_df, "vol_lens"),
        ("GLOBE", ref_df, "vol_globe"),
        ("OPTIC NERVE", ref_df, "vol_nerve"),
        ("INTRACONAL FAT", ref_df, "vol_int_fat"),
        ("EXTRACONAL FAT", ref_df, "vol_ext_fat"),
        ("LATERAL RECTUS MUSCLE", ref_df, "vol_lat_mus"),
        ("MEDIAL RECTUS MUSCLE", ref_df, "vol_med_mus"),
        ("INFERIOR RECTUS MUSCLE", ref_df, "vol_inf_mus"),
        ("SUPERIOR RECTUS MUSCLE", ref_df, "vol_sup_mus"),
    ]

    ref_df["Sex"] = ref_df["Sex"].map({"M": "Male", "F": "Female"})

    # Subplots configuration
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    fig.suptitle(
        f"Volumetry per structure in mm³ per sex (N={ref_df.shape[0]})", fontsize=22
    )
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=3)

    # Color palette
    palette = sns.color_palette("Blues", 2)

    # Plot each structure in a separate subplot
    for ax, (title, ref_df, vol_col) in zip(axes.flatten(), structures):
        # Plot the violin plot
        if "method" not in ref_df.columns:
            ref_df["method"] = "Reference dataset"
        sns.violinplot(
            data=ref_df,
            x="method",
            y=vol_col,
            ax=ax,
            hue="Sex",
            split=True,
            inner="quart",
            palette=palette,
        )
        ax.set_title(title)
        ax.set_ylabel("Volume (mm³)")
        ax.set_xlabel("")

        # Add median values as text
        median_val = ref_df[ref_df["Sex"] == "Male"][vol_col].median()
        ax.text(
            -0.1,
            median_val,
            f"{int(median_val)}",
            ha="left",
            va="bottom",
            fontsize=10,
            color="black",
        )
        median_val = ref_df[ref_df["Sex"] == "Female"][vol_col].median()
        ax.text(
            0.1,
            median_val,
            f"{int(median_val)}",
            ha="right",
            va="bottom",
            fontsize=10,
            color="black",
        )

        # add case value
        ax.axhline(case_vol_data[vol_col], color="red", linewidth=1, label="Case value")
        ax.legend(loc="upper right", fontsize=8)

    return fig


# Adapted from (extract_biometrics.ipynb, jaimebarran, accessed 26.05.2026)
# URL: https://github.com/jaimebarran/a-eye_segmentation/blob/main/deep_learning/3D_multilabel/extract_biometrics.ipynb
def plot_axial_length_violin(
    ref_df: pd.DataFrame, case_al_data: dict[str, float]
) -> Figure:
    """Plot the axial length in violin plot, distinguishing sex

    Args:
        ref_df (pd.DataFrame): Dataframe of the reference values
        case_al_data (dict[str, float]): Dict of the extracted values of current case

    Returns:
        Figure: A violin plot, with case's data highlighted
    """
    # High-resolution, readable figure settings
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16,
    })

    ref_df["Sex"] = ref_df["Sex"].map({"M": "Male", "F": "Female"})
    al_col = "axial_length_cornea"

    # Figure configuration
    fig, ax = plt.subplots(figsize=(20 / 3, 20 / 3))
    fig.suptitle(f"Axial length per sex (N={ref_df.shape[0]})", fontsize=22)
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=3)

    # Color palette
    palette = sns.color_palette("Blues", 2)

    # Plot the violin plot
    if "method" not in ref_df.columns:
        ref_df["method"] = "Reference dataset"
    sns.violinplot(
        data=ref_df,
        x="method",
        y=al_col,
        ax=ax,
        hue="Sex",
        split=True,
        inner="quart",
        palette=palette,
    )
    ax.set_title("AXIAL LENGTH (globe post. to cornea post.)")
    ax.set_ylabel("Axial length (mm)")
    ax.set_xlabel("")

    # Add median values as text
    median_val = ref_df[ref_df["Sex"] == "Male"][al_col].median()
    ax.text(
        -0.1,
        median_val,
        f"{median_val:.2f}",
        ha="left",
        va="bottom",
        fontsize=10,
        color="black",
    )
    median_val = ref_df[ref_df["Sex"] == "Female"][al_col].median()
    ax.text(
        0.1,
        median_val,
        f"{median_val:.2f}",
        ha="right",
        va="bottom",
        fontsize=10,
        color="black",
    )

    # add case value
    ax.axhline(case_al_data[al_col], color="red", linewidth=1, label="Case value")
    ax.legend(loc="upper right", fontsize=8)

    return fig
