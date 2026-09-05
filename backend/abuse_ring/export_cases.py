"""Exports ranked ring cases from a trained model into a JSON file the API
serves without reloading the model or graph per request. Each case is a
detected group (relation and group index) with member details, an
evidence block, and a score.

Each case is exported as a star: members connected to a central hub
representing the shared relation or entity, matching CA-HGAT's bipartite
member-to-group structure rather than a reconstructed pairwise graph.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from abuse_ring.train_eval import train
from abuse_ring.upi_synthetic import generate_upi_graph

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
TOP_K_PER_DATASET = 300
MAX_MEMBERS_PER_CASE = 60  # payload-size cap for very large groups


def _is_nan(x: float) -> bool:
    return x != x  # NaN != NaN, true for numpy/python floats alike


def export_dataset_cases(dataset_name: str, device: str = "cpu", max_epoch: int = 150):
    model, bg, group_meta, group_buckets, group_labels, result = train(
        dataset_name, device=device, max_epoch=max_epoch
    )
    model.eval()
    with torch.no_grad():
        node_logits, group_logits, _ = model(bg)
    node_scores = torch.sigmoid(node_logits).cpu().numpy()
    label = bg.nodes["member"].data["label"].cpu().numpy()

    upi_meta = None
    if dataset_name == "upi":
        _, upi_meta, _ = generate_upi_graph(seed=0)

    cases = []
    for rel_name, groups in group_meta.items():
        gtype = f"group_{rel_name}"
        scores = torch.sigmoid(group_logits[gtype]).cpu().numpy()
        buckets = group_buckets[gtype]
        labels = group_labels[gtype].cpu().numpy()

        for gi, members in enumerate(groups):
            evidence = {"relation": rel_name}
            if upi_meta is not None:
                sub = upi_meta.iloc[members]
                for col in ("device_id", "bank_account_id", "city"):
                    vals = sub[col].unique().tolist()
                    if len(vals) == 1:
                        evidence[col] = vals[0]

            member_rows = []
            for m in members[:MAX_MEMBERS_PER_CASE]:
                row = {
                    "id": int(m),
                    "score": float(node_scores[m]),
                    "label": None if label[m] < 0 else int(label[m]),
                }
                if upi_meta is not None:
                    row["vpa"] = upi_meta.iloc[m]["vpa"]
                    row["city"] = upi_meta.iloc[m]["city"]
                member_rows.append(row)

            true_label = None if _is_nan(float(labels[gi])) else int(labels[gi])
            cases.append(
                {
                    "dataset": dataset_name,
                    "relation": rel_name,
                    "group_index": gi,
                    "size": len(members),
                    "truncated": len(members) > MAX_MEMBERS_PER_CASE,
                    "score": float(scores[gi]),
                    "bucket": buckets[gi],
                    "true_label": true_label,
                    "evidence": evidence,
                    "members": member_rows,
                }
            )

    cases.sort(key=lambda c: -c["score"])
    cases = cases[:TOP_K_PER_DATASET]

    out_path = OUT_DIR / f"cases_{dataset_name}.json"
    out_path.write_text(json.dumps(cases, indent=2))
    print(f"wrote {len(cases)} cases to {out_path}")
    return cases


if __name__ == "__main__":
    import argparse

    import torch as _torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["yelp", "amazon", "elliptic", "upi", "all"], default="all")
    parser.add_argument("--max-epoch", type=int, default=150)
    parser.add_argument("--device", default="cuda" if _torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    datasets = ["yelp", "amazon", "elliptic", "upi"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        export_dataset_cases(ds, device=args.device, max_epoch=args.max_epoch)
