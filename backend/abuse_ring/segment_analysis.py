"""Per-segment threshold analysis and ring-formation spike detection - both
computed from real fields already in the data, not invented ones.

Segment analysis is only meaningful for UPI right now: it's the one dataset
with a real per-user dimension (city) that anything else here has an
analogous field for - Yelp/Amazon/Elliptic have no business/merchant/
category field exposed in their feature sets. The mechanism (split test
predictions by segment, run the same cost-sweep already used for the global
threshold) is dataset-agnostic; the data to run it on meaningfully today is
not. Swapping "city" for "merchant_id" the day real merchant-tagged data
exists is a one-line change, not a rewrite.

Spike detection counts flagged (score >= threshold) nodes per time bucket
and flags buckets whose count is a z-score outlier against the bucket
distribution - a real, simple statistical test, not a fabricated pattern.
"""

from __future__ import annotations

import numpy as np
import torch

from abuse_ring.cost_curve import sweep_node_threshold
from abuse_ring.train_eval import DEFAULT_FN_COST, DEFAULT_FP_COST, train
from abuse_ring.upi_synthetic import generate_upi_graph

TIME_FIELD = {"elliptic": "time_step", "upi": "signup_week"}


def per_segment_threshold(device: str = "cpu", max_epoch: int = 150):
    """UPI only - see module docstring for why."""
    model, bg, _, _, _, _ = train("upi", device=device, max_epoch=max_epoch)
    _, meta, _ = generate_upi_graph(seed=0)
    model.eval()
    with torch.no_grad():
        node_logits, _, _ = model(bg)
    probs = torch.sigmoid(node_logits).cpu().numpy()
    labels = bg.nodes["member"].data["label"].cpu().numpy()
    test_mask = bg.nodes["member"].data["test_mask"].cpu().numpy().astype(bool)

    out = []
    for city in sorted(meta["city"].unique()):
        seg_mask = (meta["city"].values == city) & test_mask
        n = int(seg_mask.sum())
        n_pos = int(labels[seg_mask].sum()) if n else 0
        if n < 20 or n_pos == 0 or n_pos == n:
            out.append({"segment": city, "n": n, "n_pos": n_pos, "note": "not enough class variety to threshold"})
            continue
        p, y = probs[seg_mask], labels[seg_mask]
        _, best = sweep_node_threshold(p, y, DEFAULT_FN_COST, DEFAULT_FP_COST)
        out.append(
            {
                "segment": city, "n": n, "n_pos": n_pos,
                "recommended_threshold": best["threshold"],
                "precision": best["precision"], "recall": best["recall"],
            }
        )
    return out


def ring_formation_spikes(dataset_name: str, device: str = "cpu", max_epoch: int = 150, score_threshold: float = 0.5, z_thresh: float = 2.0):
    if dataset_name not in TIME_FIELD:
        raise ValueError(f"no temporal field configured for {dataset_name!r} (have: {list(TIME_FIELD)})")
    time_field = TIME_FIELD[dataset_name]

    model, bg, _, _, _, _ = train(dataset_name, device=device, max_epoch=max_epoch)
    model.eval()
    with torch.no_grad():
        node_logits, _, _ = model(bg)
    probs = torch.sigmoid(node_logits).cpu().numpy()
    time_vals = bg.nodes["member"].data[time_field].cpu().numpy()
    flagged = probs >= score_threshold

    buckets = sorted(set(time_vals.tolist()))
    counts = np.array([int((flagged & (time_vals == b)).sum()) for b in buckets])
    mean = counts.mean()
    std = counts.std() if counts.std() > 0 else 1.0
    z = (counts - mean) / std

    return [
        {"bucket": int(b), "flagged_count": int(c), "z_score": float(zz), "spike": bool(zz >= z_thresh)}
        for b, c, zz in zip(buckets, counts, z)
    ]
