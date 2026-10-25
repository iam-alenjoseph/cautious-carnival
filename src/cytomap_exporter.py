"""
cytomap_exporter.py

Exports single-cell quantification and spatial coordinates into formats
directly compatible with CytoMap (MATLAB / standalone spatial biology platform).
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class CytoMapExporter:
    """
    Formats, validates, and exports single-cell spatial measurements for
    cellular neighborhood mapping in CytoMap.
    """

    def __init__(
        self,
        x_col: str = "XPos",
        y_col: str = "YPos",
        z_col: str = "ZPos",
        cell_id_col: str = "CellID"
    ):
        self.x_col = x_col
        self.y_col = y_col
        self.z_col = z_col
        self.cell_id_col = cell_id_col

    def prepare_table(
        self,
        df: pd.DataFrame,
        sample_name: str = "Sample_01",
        apply_arcsinh: bool = False,
        arcsinh_cofactor: float = 5.0
    ) -> pd.DataFrame:
        """
        Validates columns, standardizes spatial coordinates, and optionally
        applies arcsinh normalization for marker intensity channels.

        Args:
            df: Single-cell DataFrame from feature extraction.
            sample_name: Identifier for the sample/slide.
            apply_arcsinh: If True, computes arcsinh(x / cofactor) on marker columns.
            arcsinh_cofactor: Denominator scale factor for arcsinh transformation.

        Returns:
            CytoMap-compliant pd.DataFrame.
        """
        if df.empty:
            logger.warning("Input DataFrame is empty.")
            return df

        export_df = df.copy()

        # Ensure unique global cell identifier
        if self.cell_id_col not in export_df.columns:
            export_df[self.cell_id_col] = np.arange(1, len(export_df) + 1)

        # Ensure spatial centroid coordinates exist
        if self.x_col not in export_df.columns or self.y_col not in export_df.columns:
            raise KeyError(f"Missing required coordinate columns: '{self.x_col}' or '{self.y_col}'")

        if self.z_col not in export_df.columns:
            export_df[self.z_col] = 0

        # Sample / Slide identifier column
        export_df["SampleName"] = sample_name

        # Arcsinh transformation for fluorescence intensity markers
        if apply_arcsinh:
            intensity_cols = [c for c in export_df.columns if c.startswith("Mean_") or c.startswith("Nuc_Mean_")]
            for col in intensity_cols:
                export_df[col] = np.arcsinh(export_df[col].clip(lower=0) / arcsinh_cofactor)
            logger.info(f"Applied arcsinh transformation (cofactor={arcsinh_cofactor}) to {len(intensity_cols)} columns.")

        # Reorder key columns to the front for easy inspection
        leading_cols = [self.cell_id_col, "SampleName", self.x_col, self.y_col, self.z_col]
        remaining_cols = [c for c in export_df.columns if c not in leading_cols]
        export_df = export_df[leading_cols + remaining_cols]

        return export_df

    def export(
        self,
        df: pd.DataFrame,
        output_filepath: str,
        sample_name: str = "Sample_01",
        apply_arcsinh: bool = False
    ) -> str:
        """
        Writes the CytoMap-ready table to CSV and generates a metadata descriptor.

        Args:
            df: Single-cell DataFrame.
            output_filepath: Destination path (e.g. /path/to/cytomap_input.csv).
            sample_name: Sample label.
            apply_arcsinh: Whether to transform intensities.

        Returns:
            Path to the written file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        prepared_df = self.prepare_table(df, sample_name=sample_name, apply_arcsinh=apply_arcsinh)

        prepared_df.to_csv(output_filepath, index=False)
        logger.info(f"Successfully exported CytoMap table ({len(prepared_df)} cells) to: {output_filepath}")

        # Write CytoMap dataset metadata sidecar
        meta_filepath = os.path.splitext(output_filepath)[0] + "_cytomap_meta.json"
        metadata = {
            "dataset_name": sample_name,
            "total_cells": len(prepared_df),
            "spatial_columns": {
                "x": self.x_col,
                "y": self.y_col,
                "z": self.z_col
            },
            "marker_columns": [c for c in prepared_df.columns if c.startswith("Mean_")],
            "morphology_columns": [
                c for c in prepared_df.columns
                if any(k in c.lower() for k in ["area", "perimeter", "eccentricity", "axis", "ratio"])
            ]
        }
        with open(meta_filepath, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Generated CytoMap metadata: {meta_filepath}")
        return output_filepath

