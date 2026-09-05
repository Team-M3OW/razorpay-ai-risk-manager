"""Turns a model's raw score into an operating threshold using an explicit,
configurable false-positive-cost model - the actual "honest metrics including
false-positive cost" deliverable the track asks for, not just a PR curve.

Two cost terms, both named and defaulted rather than hidden in code:
  - FN_COST: cost of a missed fraud/illicit case that goes unflagged.
  - FP_COST: cost of a legitimate case wrongly flagged (review/friction cost).

Both accept either a flat constant (the same cost for every case - the
honest default when there's no real per-case cost signal, which is true for
Yelp/Amazon/Elliptic here: review-fraud has no currency amount at all, and
Elliptic's features don't include a trustworthy raw transaction value) or a
per-example array (e.g. transaction amount) so FN cost scales with what's
actually at stake in each case, which is what UPI's synthetic amount data
lets us demonstrate - missing a small transaction and missing a large one
aren't the same cost, and pretending they are is itself a form of dishonesty
in an "including false-positive cost" deliverable.

For ring-level flagging specifically, a false positive means every innocent
member of a wrongly-flagged group gets swept in - so FP cost scales with
group size, not per-flag. That asymmetry (one wrong ring flag can cost many
times more than one wrong node flag) is the actual point of doing this at
the ring level rather than the node level.
"""

from __future__ import annotations

import numpy as np


def sweep_node_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    fn_cost,
    fp_cost,
    n_points: int = 200,
):
    """Per-node cost sweep. fn_cost/fp_cost are each either a flat float
    (same cost for every case) or a per-example np.ndarray the same length
    as probs/labels (e.g. transaction amount for FN cost)."""
    fn_cost_arr = np.full_like(probs, fn_cost, dtype=float) if np.isscalar(fn_cost) else np.asarray(fn_cost, dtype=float)
    fp_cost_arr = np.full_like(probs, fp_cost, dtype=float) if np.isscalar(fp_cost) else np.asarray(fp_cost, dtype=float)

    thresholds = np.linspace(0.0, 1.0, n_points)
    results = []
    for t in thresholds:
        preds = probs >= t
        fn_mask = (~preds) & (labels == 1)
        fp_mask = preds & (labels == 0)
        fp = int(np.sum(fp_mask))
        fn = int(np.sum(fn_mask))
        tp = int(np.sum(preds & (labels == 1)))
        tn = int(np.sum((~preds) & (labels == 0)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        cost = float(fn_cost_arr[fn_mask].sum() + fp_cost_arr[fp_mask].sum())
        results.append(
            {"threshold": float(t), "precision": precision, "recall": recall, "fp": fp, "fn": fn, "tp": tp, "tn": tn, "cost": cost}
        )
    best = min(results, key=lambda r: r["cost"])
    return results, best


def sweep_ring_threshold(
    group_probs: np.ndarray,
    group_labels: np.ndarray,
    group_sizes: np.ndarray,
    fn_cost_per_case: float,
    fp_cost_per_member: float,
    n_points: int = 200,
):
    """Ring-level cost sweep. A flagged ring's FP cost is
    fp_cost_per_member * group_size (every innocent member gets swept in);
    an unflagged illicit ring's FN cost is fn_cost_per_case (a missed
    laundering/abuse network, counted once per ring, not per member -
    catching the ring at all is what matters operationally)."""
    thresholds = np.linspace(0.0, 1.0, n_points)
    results = []
    for t in thresholds:
        flagged = group_probs >= t
        fp_mask = flagged & (group_labels == 0)
        fn_mask = (~flagged) & (group_labels == 1)
        tp = int(np.sum(flagged & (group_labels == 1)))
        fp_rings = int(np.sum(fp_mask))
        fn_rings = int(np.sum(fn_mask))
        fp_members_swept = int(np.sum(group_sizes[fp_mask])) if fp_rings else 0
        precision = tp / (tp + fp_rings) if (tp + fp_rings) > 0 else 0.0
        recall = tp / (tp + fn_rings) if (tp + fn_rings) > 0 else 0.0
        cost = fn_rings * fn_cost_per_case + fp_members_swept * fp_cost_per_member
        results.append(
            {
                "threshold": float(t),
                "precision": precision,
                "recall": recall,
                "flagged_rings": int(np.sum(flagged)),
                "fp_rings": fp_rings,
                "fn_rings": fn_rings,
                "fp_members_swept": fp_members_swept,
                "cost": cost,
            }
        )
    best = min(results, key=lambda r: r["cost"])
    return results, best
