"""
feature_extraction.py

Extracts single-cell morphological properties and multiplexed biomarker
intensities from whole-cell and nuclear segmentation masks.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

logger = logging.getLogger(__name__)

def extract_single_cell_features(
    whole_cell_mask: np.ndarray,
    nuclear_mask: np.ndarray,
    multichannel_image: np.ndarray,
    marker_names: Optional[List[str]] = None,
    pixel_size_um: float = 0.5,
    tile_offset_x: int = 0,
    tile_offset_y: int = 0
) -> pd.DataFrame:
    """
    Measures morphological and biomarker intensity metrics for every segmented cell.

    Args:
        whole_cell_mask: 2D labeled mask of whole-cell boundaries (0 = background).
        nuclear_mask: 2D labeled mask of nuclear boundaries.
        multichannel_image: 3D array (H, W, C) of multiplexed marker fluorescence.
        marker_names: Optional list of channel/marker labels (e.g. ['DAPI', 'CD3', 'CD8']).
        pixel_size_um: Microns per pixel calibration factor.
        tile_offset_x: Global X coordinate offset of the tile within the whole slide.
        tile_offset_y: Global Y coordinate offset of the tile within the whole slide.

    Returns:
        pd.DataFrame containing single-cell morphological and intensity measurements.
    """
    if whole_cell_mask.max() == 0:
        logger.warning("Segmentation mask contains no cells. Returning empty DataFrame.")
        return pd.DataFrame()

    num_channels = multichannel_image.shape[-1] if multichannel_image.ndim == 3 else 1
    if marker_names is None or len(marker_names) < num_channels:
        channel_names = [f"Channel_{i}" for i in range(num_channels)]
        if marker_names:
            for i, name in enumerate(marker_names):
                channel_names[i] = name
    else:
        channel_names = marker_names[:num_channels]

    # Compute cytoplasm mask: whole-cell mask minus nuclear mask
    cytoplasm_mask = np.where((whole_cell_mask > 0) & (nuclear_mask == 0), whole_cell_mask, 0)

    # 1. Whole-cell morphological measurements
    cell_props = regionprops_table(
        whole_cell_mask,
        properties=(
            "label",
            "centroid",
            "area",
            "perimeter",
            "eccentricity",
            "major_axis_length",
            "minor_axis_length"
        )
    )

    df_cells = pd.DataFrame(cell_props)
    df_cells.rename(
        columns={
            "label": "CellID",
            "centroid-1": "Centroid_X_Local",
            "centroid-0": "Centroid_Y_Local",
            "area": "Cell_Area_Pixels",
            "perimeter": "Cell_Perimeter",
            "eccentricity": "Eccentricity",
            "major_axis_length": "MajorAxisLength",
            "minor_axis_length": "MinorAxisLength"
        },
        inplace=True
    )

    # Compute global spatial coordinates (for Whole Slide Image reconstruction)
    df_cells["XPos"] = df_cells["Centroid_X_Local"] + tile_offset_x
    df_cells["YPos"] = df_cells["Centroid_Y_Local"] + tile_offset_y
    df_cells["ZPos"] = 0  # 2D default

    # Area in physical units (square microns)
    area_factor = pixel_size_um ** 2
    df_cells["Cell_Area_um2"] = df_cells["Cell_Area_Pixels"] * area_factor

    # 2. Nuclear morphology measurements
    if nuclear_mask is not None and nuclear_mask.max() > 0:
        nuc_props = regionprops_table(nuclear_mask, properties=("label", "area"))
        df_nuc = pd.DataFrame(nuc_props).rename(
            columns={"label": "CellID", "area": "Nucleus_Area_Pixels"}
        )
        df_nuc["Nucleus_Area_um2"] = df_nuc["Nucleus_Area_Pixels"] * area_factor
        df_cells = df_cells.merge(df_nuc, on="CellID", how="left").fillna(0)
        df_cells["Nuclear_to_Cell_Ratio"] = (
            df_cells["Nucleus_Area_Pixels"] / df_cells["Cell_Area_Pixels"].replace(0, np.nan)
        ).fillna(0)
    else:
        df_cells["Nucleus_Area_Pixels"] = 0
        df_cells["Nucleus_Area_um2"] = 0
        df_cells["Nuclear_to_Cell_Ratio"] = 0

    # 3. Multiplexed Marker Intensity Extraction
    # Map label locations for efficient intensity aggregation
    labels = df_cells["CellID"].values
    cell_idx_map = {lbl: idx for idx, lbl in enumerate(labels)}

    for ch_idx, ch_name in enumerate(channel_names):
        channel_img = (
            multichannel_image[..., ch_idx]
            if multichannel_image.ndim == 3
            else multichannel_image
        )

        # Whole-cell mean intensity
        ch_props = regionprops_table(
            whole_cell_mask,
            intensity_image=channel_img,
            properties=("label", "intensity_mean", "intensity_max", "intensity_min")
        )
        df_ch = pd.DataFrame(ch_props).rename(
            columns={
                "intensity_mean": f"Mean_{ch_name}",
                "intensity_max": f"Max_{ch_name}",
                "intensity_min": f"Min_{ch_name}"
            }
        )
        df_cells = df_cells.merge(df_ch, on="label", how="left")

        # Nuclear mean intensity
        if nuclear_mask is not None and nuclear_mask.max() > 0:
            nuc_ch_props = regionprops_table(
                nuclear_mask,
                intensity_image=channel_img,
                properties=("label", "intensity_mean")
            )
            df_nuc_ch = pd.DataFrame(nuc_ch_props).rename(
                columns={"intensity_mean": f"Nuc_Mean_{ch_name}"}
            )
            df_cells = df_cells.merge(df_nuc_ch, on="label", how="left").fillna(0)

        # Cytoplasmic mean intensity
        if cytoplasm_mask.max() > 0:
            cyto_ch_props = regionprops_table(
                cytoplasm_mask,
                intensity_image=channel_img,
                properties=("label", "intensity_mean")
            )
            df_cyto_ch = pd.DataFrame(cyto_ch_props).rename(
                columns={"intensity_mean": f"Cyto_Mean_{ch_name}"}
            )
            df_cells = df_cells.merge(df_cyto_ch, on="label", how="left").fillna(0)

    # Clean up merged identifier column if redundant
    if "label" in df_cells.columns:
        df_cells.drop(columns=["label"], inplace=True)

    return df_cells

