"""Trains CA-HGAT and evaluates it two ways:

1. Node-level fraud classification on the dataset's fixed test mask.
   Metric definitions (recall at argmax threshold, ROC-AUC on the raw
   probability) match dmlc/dgl/examples/pytorch/caregnn/main.py, so
   results are comparable to the published CARE-GNN numbers.
2. Ring-level (group) classification, with no published baseline to
   compare against.

A group's label used in the training loss is built only from that
group's train-split members. A group's label used in validation or test
evaluation is built only from that group's validation-split or
test-split members, respectively. Groups with no majority split are
excluded from both training and evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import numpy as np

from abuse_ring.cost_curve import sweep_node_threshold
from abuse_ring.data import load_dataset
from abuse_ring.elliptic_data import TEST_MAX_STEP, TRAIN_MAX_STEP, VAL_MAX_STEP, load_elliptic
from abuse_ring.model import CAHGAT
from abuse_ring.ring_extraction import build_bipartite_graph, extract_hyperedges
from abuse_ring.upi_synthetic import generate_upi_graph

# Illustrative, configurable cost constants (not a real institution's actual
# figures - documented as assumptions). FN_COST: cost of one missed
# fraud/illicit case going unflagged. FP_COST: cost of one legitimate case
# wrongly flagged (analyst review + customer friction).
DEFAULT_FN_COST = 500.0
DEFAULT_FP_COST = 15.0

SEED = 717

# CARE-GNN benchmark (Yelp/Amazon) reference numbers - see model.py docstring
# and dmlc/dgl/examples/pytorch/caregnn/README.md for provenance.
PUBLISHED_REFERENCE = {
    "amazon": {
        "paper_auc": 0.897,
        "paper_recall": 0.885,
        "dgl_auc": 0.892,
        "dgl_recall": 0.854,
    },
    "yelp": {
        "paper_auc": 0.757,
        "paper_recall": 0.719,
        "dgl_auc": 0.687,
        "dgl_recall": 0.662,
    },
}

# Weber et al. (KDD'19 workshop) Table 1 - illicit-class Precision/Recall/F1
# on their 70:30 temporal split (train steps 1-34, test 35-49). Pulled
# directly from the paper (arXiv:1908.02591), not from memory.
ELLIPTIC_REFERENCE = {
    "Logistic Regression (AF)": {"precision": 0.404, "recall": 0.593, "f1": 0.481},
    "Random Forest (AF)": {"precision": 0.956, "recall": 0.670, "f1": 0.788},
    "Random Forest (AF+NE)": {"precision": 0.971, "recall": 0.675, "f1": 0.796},
    "MLP (AF)": {"precision": 0.694, "recall": 0.617, "f1": 0.653},
    "GCN": {"precision": 0.812, "recall": 0.512, "f1": 0.628},
    "Skip-GCN": {"precision": 0.812, "recall": 0.623, "f1": 0.705},
}

# Sparse-vs-dense heuristic in ring_extraction.extract_hyperedges (based on
# average degree) is wrong for Elliptic: the flow graph has low average
# degree overall (~2.3) yet each of the 49 time steps still collapses into
# ONE giant connected component (verified empirically - a low-degree chain
# structure can still span thousands of nodes). Forcing the dense/Louvain
# branch (threshold=0) is what actually finds meaningful sub-clusters.
DATASET_CONFIG = {
    "yelp": {"loader": lambda: load_dataset("yelp"), "sparse_degree_threshold": 10.0, "max_group_size": 60},
    "amazon": {"loader": lambda: load_dataset("amazon"), "sparse_degree_threshold": 10.0, "max_group_size": 60},
    "elliptic": {"loader": load_elliptic, "sparse_degree_threshold": 0.0, "max_group_size": 100},
    # Synthetic UPI demo dataset (Razorpay-shaped, NOT a rigor/benchmark
    # claim - see upi_synthetic.py docstring). No published reference exists.
    "upi": {"loader": lambda: generate_upi_graph(seed=0)[0], "sparse_degree_threshold": 10.0, "max_group_size": 60},
}


def _group_split_labels(bg, gtype: str, groups: list[list[int]]):
    """For each group, returns (bucket, label) where bucket in
    {"train", "val", "test", None} - the split holding a strict majority of
    that group's members - and label is the majority-fraud vote computed
    ONLY from members in that same bucket (never mixing splits)."""
    label = bg.nodes["member"].data["label"]
    train_mask = bg.nodes["member"].data["train_mask"].bool()
    val_mask = bg.nodes["member"].data["val_mask"].bool()
    test_mask = bg.nodes["member"].data["test_mask"].bool()

    buckets: list[str | None] = []
    labels: list[float] = []
    for members in groups:
        idx = torch.tensor(members, dtype=torch.long)
        n = len(members)
        frac_train = train_mask[idx].float().mean().item()
        frac_val = val_mask[idx].float().mean().item()
        frac_test = test_mask[idx].float().mean().item()

        if frac_train >= 0.5:
            bucket, mask = "train", train_mask
        elif frac_val >= 0.5:
            bucket, mask = "val", val_mask
        elif frac_test >= 0.5:
            bucket, mask = "test", test_mask
        else:
            buckets.append(None)
            labels.append(float("nan"))
            continue

        sel = idx[mask[idx]]
        y = label[sel].float().mean().item() > 0.5
        buckets.append(bucket)
        labels.append(float(y))

    return buckets, torch.tensor(labels)


def _group_split_labels_by_time(bg, groups: list[list[int]]):
    """Elliptic-specific version of _group_split_labels: since flow edges
    never cross time steps, every reconstructed group lies entirely within
    one time step (verified empirically), so the bucket is exact rather than
    a majority-vote approximation. The label is the majority vote among
    KNOWN-labeled members only (Elliptic is ~77% unknown-labeled, so
    skipping unknowns - rather than counting them as a third outcome - is
    what keeps most groups usable)."""
    label = bg.nodes["member"].data["label"]
    time_step = bg.nodes["member"].data["time_step"]

    buckets: list[str | None] = []
    labels: list[float] = []
    for members in groups:
        idx = torch.tensor(members, dtype=torch.long)
        ts = time_step[idx[0]].item()
        if ts <= TRAIN_MAX_STEP:
            bucket = "train"
        elif ts <= VAL_MAX_STEP:
            bucket = "val"
        elif ts <= TEST_MAX_STEP:
            bucket = "test"
        else:
            bucket = None

        known = label[idx] >= 0
        if bucket is None or known.sum().item() == 0:
            buckets.append(None)
            labels.append(float("nan"))
            continue
        y = label[idx][known].float().mean().item() > 0.5
        buckets.append(bucket)
        labels.append(float(y))

    return buckets, torch.tensor(labels)


def _bootstrap_ci(probs: np.ndarray, labels: np.ndarray, n_boot: int = 500, seed: int = 0):
    """95% bootstrap CI for AUC, resampled from already-computed predictions
    - no retraining needed. Resamples that happen to be single-class are
    skipped (AUC undefined), which is itself informative when N is small: a
    low `n_boot_valid` means the point estimate is on thin ice."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    if n == 0 or len(np.unique(labels)) < 2:
        return None
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y, p = labels[idx], probs[idx]
        if len(np.unique(y)) < 2:
            continue
        aucs.append(roc_auc_score(y, p))
    if not aucs:
        return None
    return {
        "mean": float(np.mean(aucs)), "ci_low": float(np.percentile(aucs, 2.5)),
        "ci_high": float(np.percentile(aucs, 97.5)), "n_boot_valid": len(aucs),
    }


def relation_importance(model, bg, val_mask, label, relations: list[str]) -> dict:
    """Per-relation contribution to VALIDATION node AUC, measured two ways
    from an already-trained model's forward pass - no retraining needed:

    - leave-one-out: exclude just this relation, keep the rest. A small drop
      here has TWO possible causes that look identical from this test alone:
      the relation is genuinely uninformative, OR it's informative but
      redundant with another relation that covers the same signal.
    - solo: exclude every relation EXCEPT this one. This is what
      disambiguates the two cases - a relation with a high solo AUC but a
      small leave-one-out drop is redundant (useful alone, substitutable);
      a relation with both low solo AUC and low leave-one-out drop is
      genuinely uninformative.

    This whole function replaces an earlier attempt to gate relations by
    their Louvain modularity score, which was tried and falsified
    empirically (see ring_extraction.py docstring) - modularity is a
    graph-structure property, not a measure of whether a relation predicts
    the label. This measures the actual thing: does removing it, or
    isolating it, change validation performance."""
    model.eval()
    with torch.no_grad():
        full_logits, _, _ = model(bg)
        full_auc = _node_metrics(full_logits[val_mask], label[val_mask])["auc"]

        # "no relations" baseline: member features + residual only, every
        # group relation excluded. Needed to interpret a solo-AUC number -
        # if this baseline is already high, a relation's high "solo" AUC
        # might just be riding the node features, not the relation itself.
        no_rel_logits, _, _ = model(bg, exclude_relations=set(relations))
        no_rel_auc = _node_metrics(no_rel_logits[val_mask], label[val_mask])["auc"]

        importance = {}
        for rel in relations:
            others = set(relations) - {rel}
            logits_without, _, _ = model(bg, exclude_relations={rel})
            auc_without = _node_metrics(logits_without[val_mask], label[val_mask])["auc"]
            logits_solo, _, _ = model(bg, exclude_relations=others)
            auc_solo = _node_metrics(logits_solo[val_mask], label[val_mask])["auc"]

            drop = full_auc - auc_without
            solo_lift = auc_solo - no_rel_auc
            if drop > 0.005 :
                verdict = "load-bearing"
            elif solo_lift > 0.01 and drop <= 0.005:
                verdict = "redundant (useful alone, substitutable)"
            else:
                verdict = "uninformative"

            importance[rel] = {
                "val_auc_full": full_auc,
                "val_auc_without": auc_without,
                "auc_drop": drop,
                "val_auc_solo": auc_solo,
                "val_auc_no_relations": no_rel_auc,
                "solo_lift": solo_lift,
                "verdict": verdict,
            }
    return importance


def _standardize_features(g):
    """Z-score node features in place, using TRAIN-split statistics only (no
    leakage from val/test). Yelp/Amazon/Elliptic ship pre-normalized
    features already, so this is a no-op in spirit for them; the synthetic
    UPI dataset's raw features (rupee amounts, day counts, week numbers) are
    on wildly different scales and blow up training without this."""
    feat = g.ndata["feature"]
    train_mask = g.ndata["train_mask"].bool()
    mean = feat[train_mask].mean(dim=0, keepdim=True)
    std = feat[train_mask].std(dim=0, keepdim=True).clamp(min=1e-6)
    g.ndata["feature"] = (feat - mean) / std


class WeightedBCE(nn.Module):
    """Matches the reference recipe's inverse-class-frequency weighting
    (there it's CrossEntropyLoss(weight=1/class_count); here expressed as a
    pos_weight on binary cross-entropy for the same effect)."""

    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits, targets):
        return nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


def _node_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    preds = (probs > 0.5).astype(int)
    return {
        "auc": roc_auc_score(y, probs),
        "pr_auc": average_precision_score(y, probs),
        "recall": recall_score(y, preds, zero_division=0),
        "precision": precision_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "flag_rate": float(preds.mean()),
    }


def train(dataset_name: str, device: str = "cpu", max_epoch: int = 150, patience: int = 30, loader_override=None):
    """loader_override: if given, replaces config['loader']() - used by
    camouflage_sweep.py to train on the same pipeline with a parametrized
    dataset variant, without duplicating this whole function."""
    torch.manual_seed(SEED)

    config = DATASET_CONFIG[dataset_name]
    g = loader_override() if loader_override is not None else config["loader"]()
    _standardize_features(g)
    hyperedges, ring_diagnostics = extract_hyperedges(
        g,
        max_size=config["max_group_size"],
        sparse_degree_threshold=config["sparse_degree_threshold"],
    )
    print(f"\n=== {dataset_name} ring extraction diagnostics (modularity gate + multi-resolution) ===")
    for key, d in ring_diagnostics.items():
        status = "GATED OUT" if d["gated_out"] else f"{d['n_groups']} groups"
        mod_str = f"modularity={d['modularity']:.3f}" if d["modularity"] is not None else "n/a"
        print(f"  {key:24s} [{d['method']:11s}] {mod_str:20s} -> {status}")
    bg, group_meta = build_bipartite_graph(g, hyperedges)

    member_in_dim = bg.nodes["member"].data["feature"].shape[1]
    group_in_dims = {gt: bg.nodes[gt].data["feature"].shape[1] for gt in bg.ntypes if gt != "member"}

    # Compute group bucket/label assignment on CPU (indices built here are
    # plain CPU tensors) before moving the graph to the training device.
    group_buckets: dict[str, list] = {}
    group_labels: dict[str, torch.Tensor] = {}
    for rel_name, groups in group_meta.items():
        gtype = f"group_{rel_name}"
        if dataset_name == "elliptic":
            buckets, labels = _group_split_labels_by_time(bg, groups)
        else:
            buckets, labels = _group_split_labels(bg, gtype, groups)
        group_buckets[gtype] = buckets
        group_labels[gtype] = labels.to(device)

    bg = bg.to(device)

    model = CAHGAT(member_in_dim, group_in_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)

    label = bg.nodes["member"].data["label"].float()
    train_mask = bg.nodes["member"].data["train_mask"].bool()
    val_mask = bg.nodes["member"].data["val_mask"].bool()
    test_mask = bg.nodes["member"].data["test_mask"].bool()

    n_pos = label[train_mask].sum().clamp(min=1)
    n_neg = (train_mask.sum() - n_pos).clamp(min=1)
    node_loss_fn = WeightedBCE(pos_weight=(n_neg / n_pos)).to(device)

    group_loss_fns = {}
    for gtype, buckets in group_buckets.items():
        train_idx = [i for i, b in enumerate(buckets) if b == "train"]
        if not train_idx:
            continue
        y_train = group_labels[gtype][train_idx]
        gp = y_train.sum().clamp(min=1)
        gn = (len(y_train) - gp).clamp(min=1)
        group_loss_fns[gtype] = WeightedBCE(pos_weight=(gn / gp)).to(device)

    best_val_auc = -1.0
    best_state = None
    patience_ctr = 0
    history = []

    t0 = time.time()
    for epoch in range(max_epoch):
        model.train()
        node_logits, group_logits, _ = model(bg)

        loss = node_loss_fn(node_logits[train_mask], label[train_mask])
        for gtype, loss_fn in group_loss_fns.items():
            buckets = group_buckets[gtype]
            train_idx = [i for i, b in enumerate(buckets) if b == "train"]
            gl = group_logits[gtype][train_idx]
            gy = group_labels[gtype][train_idx]
            loss = loss + 0.5 * loss_fn(gl, gy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            node_logits, group_logits, _ = model(bg)
            val_metrics = _node_metrics(node_logits[val_mask], label[val_mask])
            train_metrics = _node_metrics(node_logits[train_mask], label[train_mask])

        history.append(
            {"epoch": epoch, "loss": loss.item(), "train_auc": train_metrics["auc"], "val_auc": val_metrics["auc"]}
        )
        print(
            f"[{dataset_name}] epoch {epoch:3d} loss {loss.item():.4f} "
            f"train_auc {train_metrics['auc']:.4f} val_auc {val_metrics['auc']:.4f} "
            f"val_recall {val_metrics['recall']:.4f} val_precision {val_metrics['precision']:.4f} "
            f"val_flag_rate {val_metrics['flag_rate']:.4f}"
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[{dataset_name}] early stopping at epoch {epoch}")
                break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        node_logits, group_logits, _ = model(bg)
        test_metrics = _node_metrics(node_logits[test_mask], label[test_mask])

        # Cost-minimizing threshold is chosen on VAL only, then applied
        # as-is to TEST - picking it on TEST would be tuning on the test set.
        val_probs = torch.sigmoid(node_logits[val_mask]).cpu().numpy()
        val_labels = label[val_mask].cpu().numpy()

        # UPI carries a real (synthetic but numeric) transaction-amount
        # field, so FN cost there scales with what's actually at stake per
        # case instead of one flat constant - missing a ring member who
        # moved lakhs isn't the same cost as missing one who moved a few
        # hundred rupees. Yelp/Amazon/Elliptic have no trustworthy per-case
        # currency amount, so they keep the flat constant rather than fake one.
        if "amount_proxy" in bg.nodes["member"].data:
            amount = bg.nodes["member"].data["amount_proxy"]
            val_fn_cost = amount[val_mask].cpu().numpy().clip(min=1.0)
        else:
            val_fn_cost = DEFAULT_FN_COST

        _, best_val_point = sweep_node_threshold(val_probs, val_labels, val_fn_cost, DEFAULT_FP_COST)
        chosen_threshold = best_val_point["threshold"]

        test_probs = torch.sigmoid(node_logits[test_mask]).cpu().numpy()
        test_labels_np = label[test_mask].cpu().numpy()

        # Concept-drift check (Elliptic only, since it's the one dataset with
        # a real timeline): report F1 per individual test time step instead
        # of one aggregate number, the way the original paper's Figure 2
        # does - an aggregate test score can hide a real collapse around a
        # specific event (their paper documents one: a dark-market shutdown
        # at step 43 that degrades every method afterward).
        drift_by_step = None
        if dataset_name == "elliptic":
            time_step = bg.nodes["member"].data["time_step"]
            drift_by_step = {}
            for step in sorted(set(time_step[test_mask].tolist())):
                step_mask = test_mask & (time_step == step)
                if step_mask.sum().item() == 0:
                    continue
                y = label[step_mask].cpu().numpy()
                if len(set(y.tolist())) < 2:
                    drift_by_step[int(step)] = {"n": int(step_mask.sum().item()), "f1": None}
                    continue
                p = torch.sigmoid(node_logits[step_mask]).cpu().numpy()
                drift_by_step[int(step)] = {
                    "n": int(step_mask.sum().item()),
                    "f1": float(f1_score(y, (p > 0.5).astype(int), zero_division=0)),
                }
        test_preds_at_cost = test_probs >= chosen_threshold
        fn_mask_test = (~test_preds_at_cost) & (test_labels_np == 1)
        fp_mask_test = test_preds_at_cost & (test_labels_np == 0)
        tp = int(np.sum(test_preds_at_cost & (test_labels_np == 1)))
        fp = int(np.sum(fp_mask_test))
        fn = int(np.sum(fn_mask_test))
        if "amount_proxy" in bg.nodes["member"].data:
            test_amount = bg.nodes["member"].data["amount_proxy"][test_mask].cpu().numpy().clip(min=1.0)
            realized_cost = float(test_amount[fn_mask_test].sum() + fp * DEFAULT_FP_COST)
        else:
            realized_cost = fn * DEFAULT_FN_COST + fp * DEFAULT_FP_COST
        test_metrics_at_cost = {
            "threshold": chosen_threshold,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "flag_rate": float(test_preds_at_cost.mean()),
            "cost": realized_cost,
        }

    rel_names = [gt[len("group_"):] for gt in group_in_dims]
    rel_importance = relation_importance(model, bg, val_mask, label, rel_names)
    print(f"\n=== {dataset_name} relation importance (leave-one-out drop + solo lift, no-relation baseline={list(rel_importance.values())[0]['val_auc_no_relations']:.4f}) ===")
    for rel, imp in sorted(rel_importance.items(), key=lambda kv: -kv[1]["auc_drop"]):
        print(
            f"  {rel:22s} drop={imp['auc_drop']:+.4f}  solo={imp['val_auc_solo']:.4f} "
            f"(lift={imp['solo_lift']:+.4f})  -> {imp['verdict']}"
        )

    # Node-level test AUC bootstrap CI, from the predictions already computed above.
    node_test_auc_ci = _bootstrap_ci(test_probs, test_labels_np)

    group_test_metrics = {}
    for gtype, buckets in group_buckets.items():
        test_idx = [i for i, b in enumerate(buckets) if b == "test"]
        n_test = len(test_idx)
        n_pos = int(group_labels[gtype][test_idx].sum().item()) if n_test else 0
        # AUC is mathematically undefined with only one class present - that's
        # a hard requirement, not a judgment call, so this can't be relaxed.
        # But N and n_pos are ALWAYS reported, even when there's no metric -
        # "not enough test groups" used to be a silent cutoff; now the reader
        # sees exactly how thin the evidence is instead of a blanket omission.
        if n_test == 0 or n_pos == 0 or n_pos == n_test:
            group_test_metrics[gtype] = {"n": n_test, "n_pos": n_pos, "metrics": None}
            continue
        gl = group_logits[gtype][test_idx]
        gy = group_labels[gtype][test_idx]
        m = _node_metrics(gl, gy)
        probs_np = torch.sigmoid(gl).detach().cpu().numpy()
        labels_np = gy.detach().cpu().numpy()
        auc_ci = _bootstrap_ci(probs_np, labels_np)
        group_test_metrics[gtype] = {"n": n_test, "n_pos": n_pos, "metrics": m, "auc_ci95": auc_ci}

    print(f"\n=== {dataset_name} comparison (node-level test metrics) ===")
    if dataset_name == "elliptic":
        for name, m in ELLIPTIC_REFERENCE.items():
            print(f"{name:26s}: Precision {m['precision']:.3f} / Recall {m['recall']:.3f} / F1 {m['f1']:.3f}")
        print(
            f"{'ours (CA-HGAT)':26s}: Precision {test_metrics['precision']:.3f} / "
            f"Recall {test_metrics['recall']:.3f} / F1 {test_metrics['f1']:.3f} / AUC {test_metrics['auc']:.3f}"
        )
        ref = ELLIPTIC_REFERENCE
    elif dataset_name == "upi":
        print("(synthetic demo dataset - no published baseline exists; not a rigor claim, see upi_synthetic.py)")
        print(
            f"ours (CA-HGAT) : AUC {test_metrics['auc']:.3f} / Recall {test_metrics['recall']:.3f} "
            f"/ Precision {test_metrics['precision']:.3f} / F1 {test_metrics['f1']:.3f} "
            f"/ flag_rate {test_metrics['flag_rate']:.3f} (test)"
        )
        ref = None
    else:
        ref = PUBLISHED_REFERENCE[dataset_name]
        print(f"paper reported : AUC {ref['paper_auc']:.3f} / Recall {ref['paper_recall']:.3f} (val)")
        print(f"DGL reference  : AUC {ref['dgl_auc']:.3f} / Recall {ref['dgl_recall']:.3f} (test)")
        print(
            f"ours (CA-HGAT) : AUC {test_metrics['auc']:.3f} / Recall {test_metrics['recall']:.3f} "
            f"/ Precision {test_metrics['precision']:.3f} / flag_rate {test_metrics['flag_rate']:.3f} (test)"
        )
    if drift_by_step:
        print(f"\n=== {dataset_name} concept drift: F1 per test time step ===")
        for step, d in drift_by_step.items():
            f1_str = f"{d['f1']:.3f}" if d["f1"] is not None else "n/a (single class)"
            marker = "  <-- dark market shutdown (paper)" if step == 43 else ""
            print(f"  step {step:3d} (n={d['n']:4d}): F1 {f1_str}{marker}")
    if node_test_auc_ci:
        print(
            f"AUC 95% bootstrap CI: {node_test_auc_ci['ci_low']:.3f}-{node_test_auc_ci['ci_high']:.3f} "
            f"(point estimate {test_metrics['auc']:.3f} is a single seed, not a distribution)"
        )
    print(
        f"\n=== {dataset_name} cost-minimizing threshold (chosen on val, applied to test; "
        f"FN_COST={DEFAULT_FN_COST:.0f}, FP_COST={DEFAULT_FP_COST:.0f}) ==="
    )
    print(
        f"threshold {test_metrics_at_cost['threshold']:.3f} : Precision {test_metrics_at_cost['precision']:.3f} / "
        f"Recall {test_metrics_at_cost['recall']:.3f} / flag_rate {test_metrics_at_cost['flag_rate']:.3f} "
        f"(vs threshold 0.5: Precision {test_metrics['precision']:.3f} / Recall {test_metrics['recall']:.3f})"
    )
    print(f"\n=== {dataset_name} ring/group-level test metrics (no published baseline) ===")
    for gtype, entry in group_test_metrics.items():
        n, n_pos, m = entry["n"], entry["n_pos"], entry["metrics"]
        if m is None:
            print(f"{gtype}: N={n} test groups, {n_pos} positive - insufficient class variety to compute AUC")
        else:
            ci = entry.get("auc_ci95")
            ci_str = f" (95% CI {ci['ci_low']:.3f}-{ci['ci_high']:.3f}, n_boot={ci['n_boot_valid']})" if ci else ""
            print(
                f"{gtype}: N={n} ({n_pos} positive) AUC {m['auc']:.3f}{ci_str} / Recall {m['recall']:.3f} "
                f"/ Precision {m['precision']:.3f} / PR-AUC {m['pr_auc']:.3f}"
            )

    result = {
        "dataset": dataset_name,
        "elapsed_sec": elapsed,
        "epochs_trained": len(history),
        "history": history,
        "node_test_metrics": test_metrics,
        "node_test_auc_ci95": node_test_auc_ci,
        "drift_by_step": drift_by_step,
        "node_test_metrics_at_cost_threshold": test_metrics_at_cost,
        # Raw test predictions, so a UI can recompute precision/recall/cost
        # at any threshold live (an interactive slider) without needing the
        # model reloaded or the sweep re-run server-side.
        "test_probs": test_probs.tolist(),
        "test_labels": test_labels_np.tolist(),
        "cost_constants": {"fn_cost": DEFAULT_FN_COST, "fp_cost": DEFAULT_FP_COST},
        "group_test_metrics": group_test_metrics,
        "published_reference": ref,
        "group_meta_sizes": {gt: len(g) for gt, g in group_meta.items()},
        "ring_diagnostics": ring_diagnostics,
        "relation_importance": rel_importance,
    }
    return model, bg, group_meta, group_buckets, group_labels, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["yelp", "amazon", "elliptic", "upi", "all"], default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-epoch", type=int, default=150)
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "data" / "processed"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["yelp", "amazon", "elliptic", "upi"] if args.dataset == "all" else [args.dataset]
    for name in datasets:
        _, _, _, _, _, result = train(name, device=args.device, max_epoch=args.max_epoch)
        with open(out_dir / f"metrics_{name}.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {out_dir / f'metrics_{name}.json'}")


if __name__ == "__main__":
    main()
