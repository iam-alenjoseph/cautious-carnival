"""
phenotyping.py

Single-Cell Phenotyping & Cellular Compartment Classification Engine.
Implements automated Gaussian Mixture Model (GMM) biomarker gating,
hierarchical lineage classification (Tumor, Immune, Stroma), and
Tumor-Immune Infiltration Index quantification.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AutomatedCellPhenotyper:
    """
    Performs probabilistic marker thresholding and classifies segmented cells
    into canonical tissue lineages and functional activation states.
    """

    def __init__(
        self,
        gating_percentile_fallback: float = 85.0
    ):
        self.fallback_pct = gating_percentile_fallback

    def fit_marker_threshold_gmm(
        self,
        values: np.ndarray
    ) -> float:
        """
        Determines the positive vs negative threshold using a 2-component Gaussian Mixture Model,
        falling back to Otsu/percentile thresholding if scikit-learn is absent or convergence fails.
        """
        clean_vals = values[np.isfinite(values)]
        if len(clean_vals) < 20:
            return float(np.median(clean_vals)) if len(clean_vals) > 0 else 0.0

        try:
            from sklearn.mixture import GaussianMixture
            X = clean_vals.reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, random_state=42, max_iter=100)
            gmm.fit(X)

            means = gmm.means_.flatten()
            neg_idx = np.argmin(means)
            pos_idx = np.argmax(means)

            # Intersection point or mean of means
            threshold = (means[neg_idx] + means[pos_idx]) / 2.0
            return float(threshold)
        except Exception:
            # Fallback to percentile thresholding
            return float(np.percentile(clean_vals, self.fallback_pct))

    def classify_lineages(
        self,
        df: pd.DataFrame,
        tumor_marker: str = "Mean_PanCK",
        immune_marker: str = "Mean_CD45",
        prolif_marker: str = "Mean_Ki67"
    ) -> pd.DataFrame:
        """
        Classifies cells into canonical immuno-oncology lineages:
          - Epithelial / Tumor (PanCK+, CD45-)
          - Proliferating Tumor (PanCK+, Ki67+)
          - Leukocyte / Immune (CD45+, PanCK-)
          - Proliferating Immune (CD45+, Ki67+)
          - Double Positive / Ambiguous (PanCK+, CD45+)
          - Stromal / Other (PanCK-, CD45-)
        """
        if df.empty:
            return df

        result_df = df.copy()

        # Check marker availability
        has_tumor = tumor_marker in result_df.columns
        has_immune = immune_marker in result_df.columns
        has_prolif = prolif_marker in result_df.columns

        # Compute gating thresholds
        tumor_thresh = self.fit_marker_threshold_gmm(result_df[tumor_marker].values) if has_tumor else 0.0
        immune_thresh = self.fit_marker_threshold_gmm(result_df[immune_marker].values) if has_immune else 0.0
        prolif_thresh = self.fit_marker_threshold_gmm(result_df[prolif_marker].values) if has_prolif else 0.0

        result_df["Is_Tumor_Positive"] = (result_df[tumor_marker] > tumor_thresh) if has_tumor else False
        result_df["Is_Immune_Positive"] = (result_df[immune_marker] > immune_thresh) if has_immune else False
        result_df["Is_Proliferating"] = (result_df[prolif_marker] > prolif_thresh) if has_prolif else False

        # Assign biological phenotypes
        phenotypes = []
        for _, row in result_df.iterrows():
            t = row["Is_Tumor_Positive"]
            i = row["Is_Immune_Positive"]
            p = row["Is_Proliferating"]

            if t and not i:
                phenotypes.append("Proliferating_Tumor" if p else "Epithelial_Tumor")
            elif i and not t:
                phenotypes.append("Proliferating_Immune" if p else "Leukocyte_Immune")
            elif t and i:
                phenotypes.append("Double_Positive")
            else:
                phenotypes.append("Stroma_Unlabeled")

        result_df["Cell_Phenotype"] = phenotypes
        logger.info(f"Classified {len(result_df)} cells into {len(set(phenotypes))} biological phenotypes.")

        return result_df

    def compute_infiltration_metrics(
        self,
        df: pd.DataFrame,
        coords: np.ndarray,
        radius_um: float = 40.0,
        pixel_size_um: float = 0.5
    ) -> Dict[str, float]:
        """
        Quantifies Tumor-Immune Infiltration score and proximity index:
        Proportion of tumor cells with at least one immune neighbor within radius_um.
        """
        if "Cell_Phenotype" not in df.columns or len(df) < 5:
            return {"infiltration_score": 0.0, "immune_to_tumor_ratio": 0.0}

        from scipy.spatial import KDTree

        tumor_indices = df.index[df["Cell_Phenotype"].str.contains("Tumor")].values
        immune_indices = df.index[df["Cell_Phenotype"].str.contains("Immune")].values

        n_tumor = len(tumor_indices)
        n_immune = len(immune_indices)

        if n_tumor == 0 or n_immune == 0:
            return {
                "infiltration_score": 0.0,
                "immune_to_tumor_ratio": float(n_immune / max(n_tumor, 1)),
                "total_tumor_cells": int(n_tumor),
                "total_immune_cells": int(n_immune)
            }

        immune_tree = KDTree(coords[immune_indices])
        tumor_coords = coords[tumor_indices]

        # Query immune neighbors for tumor cells
        radius_pixels = radius_um / pixel_size_um
        infiltrated_tumor_count = sum(
            len(immune_tree.query_ball_point(pt, r=radius_pixels)) > 0
            for pt in tumor_coords
        )

        infiltration_score = infiltrated_tumor_count / n_tumor

        return {
            "infiltration_score": float(infiltration_score),
            "immune_to_tumor_ratio": float(n_immune / n_tumor),
            "total_tumor_cells": int(n_tumor),
            "total_immune_cells": int(n_immune)
        }

