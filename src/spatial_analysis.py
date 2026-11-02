"""
spatial_analysis.py

Spatial biology and cellular microenvironment analysis engine.
Constructs spatial adjacency graphs (k-NN / Delaunay), computes cellular
neighborhood (CN) motifs, and quantifies spatial cell-cell interaction scores.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import KDTree, Delaunay

logger = logging.getLogger(__name__)


class SpatialNeighborhoodAnalyzer:
    """
    Quantifies spatial cellular architectures, microenvironment compositions,
    and cell-cell interaction graphs from multiplexed single-cell data.
    """

    def __init__(
        self,
        k_neighbors: int = 20,
        radius_um: float = 50.0,
        pixel_size_um: float = 0.5
    ):
        """
        Args:
            k_neighbors: Number of nearest neighbors to construct the spatial graph.
            radius_um: Microns radius for localized microenvironment pooling.
            pixel_size_um: Image resolution factor.
        """
        self.k_neighbors = k_neighbors
        self.radius_um = radius_um
        self.pixel_size_um = pixel_size_um
        self.radius_pixels = radius_um / pixel_size_um

    def build_spatial_kdtree(self, df: pd.DataFrame) -> Tuple[KDTree, np.ndarray]:
        """Builds a 2D KD-Tree from cell centroids."""
        coords = df[["XPos", "YPos"]].values
        tree = KDTree(coords)
        return tree, coords

    def compute_cellular_neighborhoods(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        n_clusters: int = 5
    ) -> pd.DataFrame:
        """
        Clusters local microenvironments into distinct Cellular Neighborhood (CN) motifs
        (similar to Schürch et al., Cell 2020 and CytoMAP spatial motifs).

        For each cell, aggregates the average phenotypic composition of its
        k-nearest spatial neighbors, then groups cells into neighborhood clusters.
        """
        if df.empty or len(df) < self.k_neighbors:
            logger.warning("Insufficient cells to compute cellular neighborhoods.")
            return df

        result_df = df.copy()
        tree, coords = self.build_spatial_kdtree(result_df)

        # Query k-nearest neighbors for every cell
        distances, indices = tree.query(coords, k=min(self.k_neighbors + 1, len(df)))
        # Exclude self-index (column 0)
        neighbor_indices = indices[:, 1:]

        # Select marker intensity features to characterize local microenvironment
        if feature_cols is None:
            feature_cols = [c for c in result_df.columns if c.startswith("Mean_")]

        if not feature_cols:
            logger.warning("No marker expression features found for neighborhood pooling.")
            return result_df

        marker_matrix = result_df[feature_cols].values

        # Aggregate neighbor feature vectors: Shape (N_cells, N_features)
        neighbor_profiles = np.zeros_like(marker_matrix)
        for i in range(len(result_df)):
            neighbor_profiles[i] = marker_matrix[neighbor_indices[i]].mean(axis=0)

        # Standardize neighborhood vectors
        norm_profiles = (neighbor_profiles - neighbor_profiles.mean(axis=0)) / (
            neighbor_profiles.std(axis=0) + 1e-8
        )

        # MiniBatch K-Means / Spectral-style clustering heuristic
        # If scikit-learn is available, use KMeans; otherwise use fast vector quantization
        try:
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(
                n_clusters=min(n_clusters, len(df)),
                random_state=42,
                batch_size=256
            )
            result_df["Cellular_Neighborhood_ID"] = kmeans.fit_predict(norm_profiles)
        except ImportError:
            # Fallback simple spatial quantile partition
            logger.info("scikit-learn not detected. Using spatial sector partitioning.")
            angles = np.arctan2(coords[:, 1] - coords[:, 1].mean(), coords[:, 0] - coords[:, 0].mean())
            result_df["Cellular_Neighborhood_ID"] = pd.qcut(angles, q=n_clusters, labels=False, duplicates="drop")

        # Compute Local Cell Density (cells within radius_um)
        counts_in_radius = [len(tree.query_ball_point(pt, r=self.radius_pixels)) for pt in coords]
        result_df["Local_Cell_Density_per_100um2"] = np.array(counts_in_radius) / (
            np.pi * (self.radius_um ** 2) / 10000.0
        )

        logger.info(
            f"Successfully quantified spatial microenvironments for {len(result_df)} cells "
            f"(k={self.k_neighbors}, radius={self.radius_um} um)."
        )
        return result_df

    def compute_spatial_interaction_matrix(
        self,
        df: pd.DataFrame,
        cell_type_col: str = "Cellular_Neighborhood_ID"
    ) -> pd.DataFrame:
        """
        Quantifies spatial contact frequencies between distinct cellular motifs.
        Returns an adjacency interaction matrix.
        """
        if cell_type_col not in df.columns:
            return pd.DataFrame()

        tree, coords = self.build_spatial_kdtree(df)
        distances, indices = tree.query(coords, k=min(self.k_neighbors + 1, len(df)))
        neighbor_indices = indices[:, 1:]

        labels = df[cell_type_col].values
        unique_labels = np.unique(labels)
        interaction_matrix = pd.DataFrame(
            0, index=unique_labels, columns=unique_labels, dtype=int
        )

        for i, source_label in enumerate(labels):
            for target_idx in neighbor_indices[i]:
                target_label = labels[target_idx]
                interaction_matrix.loc[source_label, target_label] += 1

        # Normalize rows to represent interaction probabilities
        interaction_prob = interaction_matrix.div(interaction_matrix.sum(axis=1), axis=0).fillna(0)
        return interaction_prob

