"""
evaluation_metrics.py

Production-grade quantitative segmentation evaluation suite for bioimaging.
Implements the international digital pathology gold standards:
  - Panoptic Quality (PQ), Segmentation Quality (SQ), Recognition Quality (RQ)
    (Kirillov et al., CVPR 2019 / CoNSeP benchmark standard)
  - Aggregated Jaccard Index (AJI) (Kumar et al., IEEE TMI 2017)
  - Multi-threshold Dice / F1 Score (IoU @ 0.5, 0.75)
"""

import logging
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


def compute_iou_matrix(
    ground_truth_mask: np.ndarray,
    predicted_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes pairwise Intersection-over-Union (IoU) matrix between all
    unique ground truth and predicted instance labels.

    Args:
        ground_truth_mask: 2D integer labeled mask (0 = background).
        predicted_mask: 2D integer labeled mask (0 = background).

    Returns:
        Tuple of (iou_matrix of shape (N_gt, N_pred), gt_labels, pred_labels)
    """
    gt_labels = np.unique(ground_truth_mask)[1:]  # Exclude background 0
    pred_labels = np.unique(predicted_mask)[1:]

    n_gt = len(gt_labels)
    n_pred = len(pred_labels)

    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred), dtype=np.float32), gt_labels, pred_labels

    # Fast label-to-index mapping
    gt_map = {lbl: i for i, lbl in enumerate(gt_labels)}
    pred_map = {lbl: j for j, lbl in enumerate(pred_labels)}

    # Compute intersection area using flattened pixel pairs
    flat_gt = ground_truth_mask.ravel()
    flat_pred = predicted_mask.ravel()

    # Mask foreground overlap
    fg_mask = (flat_gt > 0) & (flat_pred > 0)
    fg_gt = flat_gt[fg_mask]
    fg_pred = flat_pred[fg_mask]

    intersections = np.zeros((n_gt, n_pred), dtype=np.int32)
    for g, p in zip(fg_gt, fg_pred):
        intersections[gt_map[g], pred_map[p]] += 1

    # Compute individual object areas
    gt_areas = np.bincount(flat_gt)[gt_labels]
    pred_areas = np.bincount(flat_pred)[pred_labels]

    # IoU = Intersection / (Area_gt + Area_pred - Intersection)
    unions = gt_areas[:, None] + pred_areas[None, :] - intersections
    iou_matrix = np.where(unions > 0, intersections / unions, 0.0).astype(np.float32)

    return iou_matrix, gt_labels, pred_labels


def compute_panoptic_quality(
    ground_truth_mask: np.ndarray,
    predicted_mask: np.ndarray,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ).
        PQ = SQ * RQ
        SQ = sum(IoU of matched pairs) / |Matched Pairs|
        RQ = |Matched Pairs| / (|Matched Pairs| + 0.5 * |False Positives| + 0.5 * |False Negatives|)

    Args:
        ground_truth_mask: 2D integer labeled ground truth mask.
        predicted_mask: 2D integer labeled predicted instance mask.
        iou_threshold: Minimum IoU to count as a true positive match (typically 0.5).

    Returns:
        Dict containing PQ, SQ, RQ, TP, FP, and FN.
    """
    iou_matrix, gt_labels, pred_labels = compute_iou_matrix(ground_truth_mask, predicted_mask)
    n_gt = len(gt_labels)
    n_pred = len(pred_labels)

    if n_gt == 0 and n_pred == 0:
        return {"pq": 1.0, "sq": 1.0, "rq": 1.0, "tp": 0, "fp": 0, "fn": 0}
    if n_gt == 0 or n_pred == 0:
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "fp": n_pred, "fn": n_gt}

    # Bipartite matching via Hungarian algorithm (linear sum assignment on -IoU)
    gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)

    matched_ious = []
    matched_gt = set()
    matched_pred = set()

    for g_idx, p_idx in zip(gt_indices, pred_indices):
        iou = iou_matrix[g_idx, p_idx]
        if iou > iou_threshold:
            matched_ious.append(iou)
            matched_gt.add(g_idx)
            matched_pred.add(p_idx)

    tp = len(matched_ious)
    fp = n_pred - tp
    fn = n_gt - tp

    sq = float(np.sum(matched_ious) / tp) if tp > 0 else 0.0
    rq = float(tp / (tp + 0.5 * fp + 0.5 * fn)) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0.0
    pq = float(sq * rq)

    return {
        "pq": pq,
        "sq": sq,
        "rq": rq,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "f1_score": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    }


def compute_aggregated_jaccard_index(
    ground_truth_mask: np.ndarray,
    predicted_mask: np.ndarray
) -> float:
    """
    Computes Aggregated Jaccard Index (AJI) (Kumar et al., IEEE TMI 2017).
    Standard metric for nuclear instance segmentation:
        AJI = sum(C_i \cap P_j*) / (sum(C_i \cup P_j*) + sum(P_k for unmatched pred))
    """
    iou_matrix, gt_labels, pred_labels = compute_iou_matrix(ground_truth_mask, predicted_mask)
    n_gt = len(gt_labels)
    n_pred = len(pred_labels)

    if n_gt == 0 and n_pred == 0:
        return 1.0
    if n_gt == 0 or n_pred == 0:
        return 0.0

    flat_gt = ground_truth_mask.ravel()
    flat_pred = predicted_mask.ravel()

    gt_areas = np.bincount(flat_gt)[gt_labels]
    pred_areas = np.bincount(flat_pred)[pred_labels]

    total_intersection = 0
    total_union = 0
    used_pred_indices = set()

    # For each ground truth cell, find predicted cell with maximal Jaccard index
    for g_idx in range(n_gt):
        best_p_idx = int(np.argmax(iou_matrix[g_idx]))
        best_iou = iou_matrix[g_idx, best_p_idx]

        if best_iou > 0.0:
            # Recompute exact intersection and union
            g_lbl = gt_labels[g_idx]
            p_lbl = pred_labels[best_p_idx]

            inter = np.sum((flat_gt == g_lbl) & (flat_pred == p_lbl))
            union = gt_areas[g_idx] + pred_areas[best_p_idx] - inter

            total_intersection += inter
            total_union += union
            used_pred_indices.add(best_p_idx)
        else:
            total_union += gt_areas[g_idx]

    # Add remaining unmatched predicted cells to the denominator union
    for p_idx in range(n_pred):
        if p_idx not in used_pred_indices:
            total_union += pred_areas[p_idx]

    aji = float(total_intersection / total_union) if total_union > 0 else 0.0
    return aji

