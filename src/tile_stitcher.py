"""
tile_stitcher.py

Whole Slide Image (WSI) Tile Stitcher & Boundary Deduplication Engine.
Resolves overlapping tile boundaries via Spatial Non-Maximum Suppression (Spatial NMS),
eliminating fractured cell detections and double-counted edge segmentations.
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


class WSITileStitcher:
    """
    Stitches multi-tile single-cell quantification datasets, eliminating
    duplicate detections in overlap regions using spatial distance pruning.
    """

    def __init__(
        self,
        nms_distance_threshold_um: float = 4.0,
        pixel_size_um: float = 0.5
    ):
        """
        Args:
            nms_distance_threshold_um: Distance below which two cells from adjacent
                                      tiles are considered identical duplicate detections.
            pixel_size_um: Spatial resolution calibration.
        """
        self.distance_thresh_pixels = nms_distance_threshold_um / pixel_size_um
        self.pixel_size_um = pixel_size_um

    def deduplicate_cells_across_tiles(
        self,
        df: pd.DataFrame,
        tile_source_col: str = "Tile_Source"
    ) -> pd.DataFrame:
        """
        Applies Spatial Non-Maximum Suppression to remove boundary cell duplicates.
        When two candidate cells from different tiles are within the distance threshold,
        retains the detection with larger cell area or further from tile boundary.

        Args:
            df: Concatenated DataFrame of cells across all tiles.
            tile_source_col: Column indicating which tile originated the detection.

        Returns:
            Deduplicated, globally indexed pd.DataFrame.
        """
        if df.empty or len(df) < 2:
            return df

        if tile_source_col not in df.columns:
            logger.warning(f"Column '{tile_source_col}' not found. Skipping tile deduplication.")
            return df

        initial_count = len(df)
        coords = df[["XPos", "YPos"]].values
        tree = KDTree(coords)

        # Query all pairs within spatial conflict distance
        pairs = tree.query_pairs(r=self.distance_thresh_pixels)

        if not pairs:
            logger.info("No boundary cell duplicates detected.")
            return df

        tiles = df[tile_source_col].values
        areas = df["Cell_Area_Pixels"].values if "Cell_Area_Pixels" in df.columns else np.ones(len(df))

        to_remove = set()
        for idx1, idx2 in pairs:
            # Only prune if detections originated from different tiles
            if tiles[idx1] != tiles[idx2]:
                # Retain the detection with higher area (less truncation at border)
                if areas[idx1] < areas[idx2]:
                    to_remove.add(idx1)
                else:
                    to_remove.add(idx2)

        dedup_df = df.drop(index=list(to_remove)).reset_index(drop=True)
        dedup_df["CellID"] = np.arange(1, len(dedup_df) + 1)

        removed_count = initial_count - len(dedup_df)
        logger.info(
            f"Spatial NMS complete: removed {removed_count} boundary duplicate cells "
            f"({len(dedup_df)} unique cells retained across WSI)."
        )

        return dedup_df

