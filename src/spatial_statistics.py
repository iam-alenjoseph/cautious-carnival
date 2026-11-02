"""
spatial_statistics.py

Advanced Spatial Statistics & Point-Pattern Analytics for Spatial Biology.
Implements Moran's I spatial autocorrelation, Ripley's K / Besag's L spatial clustering
functions, Delaunay contact networks, and Shannon/Simpson microenvironment diversity.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import KDTree, Delaunay

logger = logging.getLogger(__name__)


def compute_morans_i(
    values: np.ndarray,
    coords: np.ndarray,
    distance_threshold: float = 50.0,
    permutations: int = 199
) -> Dict[str, float]:
    """
    Computes Global Moran's I Spatial Autocorrelation coefficient:
        I = (N / W) * (sum_i sum_j w_ij (x_i - x_mean)(x_j - x_mean)) / sum_i (x_i - x_mean)^2

    Tests whether biomarker expression clusters spatially (I > 0),
    is dispersed (I < 0), or is randomly distributed (I ~ 0).

    Args:
        values: 1D array of single-cell expression or morphological values.
        coords: 2D array of spatial centroid coordinates (N, 2).
        distance_threshold: Maximum spatial distance for edge weighting.
        permutations: Number of Monte Carlo randomizations for p-value estimation.

    Returns:
        Dict with keys: 'morans_i', 'expected_i', 'p_value', 'z_score'
    """
    n = len(values)
    if n < 4:
        return {"morans_i": 0.0, "expected_i": 0.0, "p_value": 1.0, "z_score": 0.0}

    tree = KDTree(coords)
    # Binary / inverse-distance spatial weight matrix via KDTree pairs
    pairs = tree.query_pairs(r=distance_threshold)

    if not pairs:
        return {"morans_i": 0.0, "expected_i": -1.0 / (n - 1), "p_value": 1.0, "z_score": 0.0}

    # Mean center
    z = values - np.mean(values)
    s0 = np.sum(z ** 2)

    if s0 == 0:
        return {"morans_i": 0.0, "expected_i": -1.0 / (n - 1), "p_value": 1.0, "z_score": 0.0}

    # Symmetric spatial weights
    w_sum = 2.0 * len(pairs)
    numerator = 0.0
    for i, j in pairs:
        numerator += 2.0 * z[i] * z[j]

    morans_i = (n / w_sum) * (numerator / s0)
    expected_i = -1.0 / (n - 1)

    # Monte Carlo permutation test for statistical significance
    perm_i = []
    z_copy = z.copy()
    for _ in range(permutations):
        np.random.shuffle(z_copy)
        num_perm = 0.0
        for i, j in pairs:
            num_perm += 2.0 * z_copy[i] * z_copy[j]
        perm_i.append((n / w_sum) * (num_perm / s0))

    perm_i = np.array(perm_i)
    p_value = (np.sum(np.abs(perm_i) >= np.abs(morans_i)) + 1.0) / (permutations + 1.0)
    std_perm = np.std(perm_i)
    z_score = (morans_i - expected_i) / (std_perm + 1e-9)

    return {
        "morans_i": float(morans_i),
        "expected_i": float(expected_i),
        "p_value": float(p_value),
        "z_score": float(z_score)
    }


def compute_ripleys_l(
    coords: np.ndarray,
    radii: Optional[np.ndarray] = None,
    study_area: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Besag's transformation of Ripley's K-function:
        L(r) = sqrt(K(r) / pi) - r

    L(r) > 0 indicates spatial aggregation/clustering at scale r.
    L(r) = 0 indicates complete spatial randomness (Poisson process).
    L(r) < 0 indicates spatial regularity / dispersion.

    Args:
        coords: (N, 2) centroid coordinates in physical units (microns).
        radii: Evaluation distances. Default: 10 equidistant radii from 5 to 100 um.
        study_area: Total area of tissue window in um^2.

    Returns:
        Tuple of (radii, L_values)
    """
    n = len(coords)
    if radii is None:
        radii = np.linspace(5.0, 100.0, 15)

    if n < 5:
        return radii, np.zeros_like(radii)

    # Estimate bounding box area if not provided
    if study_area is None:
        min_x, max_x = np.min(coords[:, 0]), np.max(coords[:, 0])
        min_y, max_y = np.min(coords[:, 1]), np.max(coords[:, 1])
        study_area = max((max_x - min_x) * (max_y - min_y), 1.0)

    tree = KDTree(coords)
    lambda_density = n / study_area

    l_values = []
    for r in radii:
        # Count all pairs within distance r
        pairs_count = len(tree.query_pairs(r=r)) * 2
        # K(r) estimator without edge correction
        k_r = pairs_count / (lambda_density * n)
        # Besag's L(r)
        l_r = np.sqrt(k_r / np.pi) - r
        l_values.append(l_r)

    return radii, np.array(l_values)


def extract_delaunay_contact_network(
    coords: np.ndarray,
    max_edge_length_um: float = 35.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs a Delaunay triangulation cell-cell contact network, pruning
    unrealistically long edges to reflect genuine biological cell-cell interfaces.

    Args:
        coords: (N, 2) spatial coordinates.
        max_edge_length_um: Maximum inter-centroid distance for physical contact.

    Returns:
        Tuple of (edges_array of shape (E, 2), edge_lengths of shape (E,))
    """
    if len(coords) < 4:
        return np.empty((0, 2), dtype=int), np.empty(0, dtype=float)

    tri = Delaunay(coords)
    edges = set()

    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                u, v = simplex[i], simplex[j]
                edge = (min(u, v), max(u, v))
                edges.add(edge)

    valid_edges = []
    valid_lengths = []

    for u, v in edges:
        dist = np.hypot(coords[u, 0] - coords[v, 0], coords[u, 1] - coords[v, 1])
        if dist <= max_edge_length_um:
            valid_edges.append((u, v))
            valid_lengths.append(dist)

    return np.array(valid_edges, dtype=int), np.array(valid_lengths, dtype=float)


def compute_microenvironment_diversity(
    neighborhood_labels: np.ndarray
) -> Dict[str, float]:
    """
    Computes Shannon Entropy and Gini-Simpson Index of cellular diversity:
        Shannon H = - sum(p_i * ln(p_i))
        Simpson D = 1 - sum(p_i^2)
    """
    _, counts = np.unique(neighborhood_labels, return_counts=True)
    probs = counts / np.sum(counts)

    shannon = -np.sum(probs * np.log(probs + 1e-12))
    simpson = 1.0 - np.sum(probs ** 2)

    return {
        "shannon_entropy": float(shannon),
        "simpson_diversity": float(simpson)
    }

