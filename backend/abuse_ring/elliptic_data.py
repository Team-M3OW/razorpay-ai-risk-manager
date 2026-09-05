"""Loads the Elliptic Bitcoin AML dataset (Weber et al., KDD'19 workshop) -
a REAL Bitcoin transaction graph (nodes = transactions, edges = literal BTC
flow) with real illicit/licit labels derived from law-enforcement-linked
entity tracing, not a reconstructed/approximate graph.

Source: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
(a regular Kaggle dataset, no competition gating, fetched via kagglehub).

Because edges only ever connect transactions within the same ~2-week time
step (49 steps total, no cross-step edges), every connected component lies
entirely inside a single time step - so the temporal train/val/test split
used here never splits a candidate ring across two buckets. That's a much
cleaner leakage story than the Yelp/Amazon relation graphs, where a
reconstructed group's members could span splits and needed a majority-vote
rule (see train_eval._group_split_labels).

Split (matches the paper's 70:30 temporal split for the test range, so our
node-level test metrics are directly comparable to their reported numbers):
  train: time steps 1-29   (paper uses 1-34; we hold out 30-34 for validation
                             instead of the paper's fixed-epoch-no-early-stop
                             recipe)
  val:   time steps 30-34
  test:  time steps 35-49  (identical to the paper)

Unknown-labeled transactions (class == "unknown") are kept in the graph for
message passing (their features are real) but excluded from every mask -
never trained on, never evaluated on.
"""

from __future__ import annotations

import kagglehub
import dgl
import pandas as pd
import torch

TRAIN_MAX_STEP = 29
VAL_MAX_STEP = 34
TEST_MAX_STEP = 49


def load_elliptic():
    root = kagglehub.dataset_download("ellipticco/elliptic-data-set")
    base = f"{root}/elliptic_bitcoin_dataset"

    features = pd.read_csv(f"{base}/elliptic_txs_features.csv", header=None)
    features.columns = ["txId", "time_step"] + [f"f{i}" for i in range(165)]
    classes = pd.read_csv(f"{base}/elliptic_txs_classes.csv")
    edges = pd.read_csv(f"{base}/elliptic_txs_edgelist.csv")

    merged = features.merge(classes, on="txId", how="left")
    tx_id_to_idx = {tx: i for i, tx in enumerate(merged["txId"])}

    feat_cols = ["time_step"] + [f"f{i}" for i in range(165)]
    feat_tensor = torch.tensor(merged[feat_cols].values, dtype=torch.float32)

    label = torch.full((len(merged),), -1, dtype=torch.long)
    label[merged["class"] == "1"] = 1  # illicit
    label[merged["class"] == "2"] = 0  # licit

    time_step = torch.tensor(merged["time_step"].values, dtype=torch.long)

    src = edges["txId1"].map(tx_id_to_idx)
    dst = edges["txId2"].map(tx_id_to_idx)
    valid = src.notna() & dst.notna()
    src_t = torch.tensor(src[valid].values, dtype=torch.long)
    dst_t = torch.tensor(dst[valid].values, dtype=torch.long)

    # Directed flow edges are kept as-is: ring_extraction.py's connected-
    # component step treats the relation as undirected regardless (it builds
    # a plain networkx.Graph from the edge list), so direction only matters
    # for the model's own message-passing graph (the member<->group
    # bipartite graph built downstream), not for this raw transaction graph.
    g = dgl.heterograph({("tx", "flow", "tx"): (src_t, dst_t)}, num_nodes_dict={"tx": len(merged)})
    g.ndata["feature"] = feat_tensor
    g.ndata["label"] = label
    g.ndata["time_step"] = time_step

    known = label >= 0
    g.ndata["train_mask"] = known & (time_step <= TRAIN_MAX_STEP)
    g.ndata["val_mask"] = known & (time_step > TRAIN_MAX_STEP) & (time_step <= VAL_MAX_STEP)
    g.ndata["test_mask"] = known & (time_step > VAL_MAX_STEP) & (time_step <= TEST_MAX_STEP)

    return g
