"""Ablation baseline: a plain heterogeneous GAT over the ORIGINAL relation
graphs (no reconstructed group/hyperedge nodes at all). Same training recipe
(weighted BCE, same optimizer/epoch budget/early stopping) as CA-HGAT, so any
gap between this and CA-HGAT's node-level score isolates what the ring/group
mechanism itself contributes, rather than "a modern GNN beats a 2020 one."
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn

from abuse_ring.train_eval import (
    DATASET_CONFIG,
    ELLIPTIC_REFERENCE,
    PUBLISHED_REFERENCE,
    WeightedBCE,
    _node_metrics,
    _standardize_features,
)

SEED = 717


class PlainHeteroGAT(nn.Module):
    def __init__(self, in_dim, relations, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        head_dim = hidden_dim // num_heads
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                dglnn.HeteroGraphConv(
                    {
                        rel: dglnn.GATConv(
                            hidden_dim, head_dim, num_heads,
                            feat_drop=dropout, attn_drop=dropout,
                            allow_zero_in_degree=True,
                        )
                        for rel in relations
                    },
                    aggregate="sum",
                )
                for _ in range(num_layers)
            ]
        )
        self.relations = relations
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, g, ntype):
        h = torch.relu(self.encoder(g.nodes[ntype].data["feature"]))
        h0 = h
        for layer in self.layers:
            out = layer(g, {ntype: h})
            h = torch.relu(out[ntype].flatten(1)) + h0
        return self.head(h).squeeze(-1)


def train_baseline(dataset_name: str, device: str = "cpu", max_epoch: int = 150, patience: int = 30):
    torch.manual_seed(SEED)
    g = DATASET_CONFIG[dataset_name]["loader"]()
    _standardize_features(g)
    g = g.to(device)
    ntype = g.ntypes[0]
    relations = [et[1] for et in g.canonical_etypes]

    label = g.ndata["label"].float()
    train_mask = g.ndata["train_mask"].bool()
    val_mask = g.ndata["val_mask"].bool()
    test_mask = g.ndata["test_mask"].bool()

    model = PlainHeteroGAT(g.ndata["feature"].shape[1], relations).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)

    n_pos = label[train_mask].sum().clamp(min=1)
    n_neg = (train_mask.sum() - n_pos).clamp(min=1)
    loss_fn = WeightedBCE(pos_weight=(n_neg / n_pos)).to(device)

    best_val_auc = -1.0
    best_state = None
    patience_ctr = 0
    t0 = time.time()
    epoch = 0
    for epoch in range(max_epoch):
        model.train()
        logits = model(g, ntype)
        loss = loss_fn(logits[train_mask], label[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(g, ntype)
            val_metrics = _node_metrics(logits[val_mask], label[val_mask])
        print(f"[{dataset_name}-baseline] epoch {epoch:3d} loss {loss.item():.4f} val_auc {val_metrics['auc']:.4f}")

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[{dataset_name}-baseline] early stopping at epoch {epoch}")
                break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(g, ntype)
        test_metrics = _node_metrics(logits[test_mask], label[test_mask])

    print(f"\n=== {dataset_name} baseline (plain hetero-GAT, no ring mechanism) ===")
    if dataset_name == "elliptic":
        ref = ELLIPTIC_REFERENCE
        for name, m in ref.items():
            print(f"{name:26s}: Precision {m['precision']:.3f} / Recall {m['recall']:.3f} / F1 {m['f1']:.3f}")
        print(
            f"{'plain hetero-GAT':26s}: Precision {test_metrics['precision']:.3f} / "
            f"Recall {test_metrics['recall']:.3f} / F1 {test_metrics['f1']:.3f} / AUC {test_metrics['auc']:.3f}"
        )
    elif dataset_name == "upi":
        ref = None
        print("(synthetic demo dataset - no published baseline)")
        print(
            f"plain hetero-GAT       : AUC {test_metrics['auc']:.3f} / Recall {test_metrics['recall']:.3f} "
            f"/ Precision {test_metrics['precision']:.3f} / F1 {test_metrics['f1']:.3f} (test)"
        )
    else:
        ref = PUBLISHED_REFERENCE[dataset_name]
        print(f"paper reported        : AUC {ref['paper_auc']:.3f} / Recall {ref['paper_recall']:.3f} (val)")
        print(f"DGL reference         : AUC {ref['dgl_auc']:.3f} / Recall {ref['dgl_recall']:.3f} (test)")
        print(
            f"plain hetero-GAT       : AUC {test_metrics['auc']:.3f} / Recall {test_metrics['recall']:.3f} "
            f"/ Precision {test_metrics['precision']:.3f} (test)"
        )

    return {
        "dataset": dataset_name,
        "elapsed_sec": elapsed,
        "epochs_trained": epoch + 1,
        "node_test_metrics": test_metrics,
        "published_reference": ref,
    }


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
        result = train_baseline(name, device=args.device, max_epoch=args.max_epoch)
        with open(out_dir / f"baseline_{name}.json", "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
