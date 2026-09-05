"""Checks whether scoring one node requires only its local neighborhood
rather than the full graph in memory.

For a sampled test node, this walks outward through the bipartite
member-to-group structure for as many rounds as the model uses
(num_rounds=2), extracts the induced subgraph with dgl.node_subgraph,
runs the model on that subgraph alone, and compares the resulting score
for the target node to the full-graph score.
"""

from __future__ import annotations

import dgl
import numpy as np
import torch

from abuse_ring.train_eval import train


def _extract_ego_node_set(bg, target_idx: int, relations: list[str], num_rounds: int):
    frontier_members = {target_idx}
    all_members = {target_idx}
    all_groups: dict[str, set] = {}

    device = bg.device
    for _ in range(num_rounds):
        hit_groups: dict[str, set] = {}
        frontier_t = torch.tensor(sorted(frontier_members), dtype=torch.long, device=device)
        for rel in relations:
            gtype = f"group_{rel}"
            etype = ("member", f"in_{rel}", gtype)
            if etype not in bg.canonical_etypes:
                continue
            src, dst = bg.edges(etype=etype)
            mask = torch.isin(src, frontier_t)
            if mask.any():
                hit_groups.setdefault(gtype, set()).update(dst[mask].tolist())

        new_members = set()
        for gtype, gs in hit_groups.items():
            all_groups.setdefault(gtype, set()).update(gs)
            rel = gtype[len("group_"):]
            etype = (gtype, f"has_{rel}", "member")
            src, dst = bg.edges(etype=etype)
            mask = torch.isin(src, torch.tensor(sorted(gs), dtype=torch.long, device=device))
            new_members.update(dst[mask].tolist())

        all_members.update(new_members)
        frontier_members = new_members

    return all_members, all_groups


def inductive_score_check(model, bg, relations: list[str], sample_indices, num_rounds: int = 2):
    model.eval()
    with torch.no_grad():
        full_logits, _, _ = model(bg)

    rows = []
    for idx in sample_indices:
        idx = int(idx)
        members, groups = _extract_ego_node_set(bg, idx, relations, num_rounds)
        node_dict = {"member": sorted(members)}
        for ntype in bg.ntypes:
            if ntype == "member":
                continue
            node_dict[ntype] = sorted(groups.get(ntype, set()))

        sub = dgl.node_subgraph(bg, node_dict)
        orig_ids = sub.nodes["member"].data[dgl.NID]
        local_idx = (orig_ids == idx).nonzero(as_tuple=True)[0].item()

        with torch.no_grad():
            sub_logits, _, _ = model(sub)

        full_score = torch.sigmoid(full_logits[idx]).item()
        sub_score = torch.sigmoid(sub_logits[local_idx]).item()
        rows.append(
            {
                "member": idx, "full_score": full_score, "subgraph_score": sub_score,
                "abs_diff": abs(full_score - sub_score), "subgraph_n_members": len(members),
            }
        )
    return rows


def run_demo(dataset_name: str = "upi", n_samples: int = 40, max_epoch: int = 100, device: str = "cpu"):
    model, bg, group_meta, _, _, result = train(dataset_name, device=device, max_epoch=max_epoch)
    relations = list(group_meta.keys())

    test_mask = bg.nodes["member"].data["test_mask"].bool()
    test_idx_set = set(torch.nonzero(test_mask, as_tuple=True)[0].tolist())

    # Most nodes belong to zero groups at all (e.g. UPI's background traffic
    # users share no device/bank with anyone) - for those, the ego-subgraph
    # trivially IS just the node itself and the check is vacuous. Sample
    # preferentially from nodes that actually have at least one group
    # membership, so multi-hop message passing is actually exercised.
    grouped_test_members = set()
    for groups in group_meta.values():
        for members in groups:
            grouped_test_members.update(m for m in members if m in test_idx_set)

    pool = sorted(grouped_test_members) if grouped_test_members else sorted(test_idx_set)
    rng = np.random.default_rng(0)
    sample = rng.choice(pool, size=min(n_samples, len(pool)), replace=False)

    rows = inductive_score_check(model, bg, relations, sample)
    diffs = [r["abs_diff"] for r in rows]
    sizes = [r["subgraph_n_members"] for r in rows]
    print(f"\n=== {dataset_name} inductive-scoring check (N={len(rows)} sampled test nodes) ===")
    print(f"mean |full - subgraph| score diff: {np.mean(diffs):.6f}  (max: {np.max(diffs):.6f})")
    print(f"mean ego-subgraph size: {np.mean(sizes):.1f} members  (full graph: {bg.num_nodes('member')} members)")
    print("Near-zero diff means a node's score only actually depends on its local")
    print("neighborhood, not the full graph - the concrete precondition for online,")
    print("per-node scoring instead of full-graph batch retraining.")
    return rows


if __name__ == "__main__":
    import argparse

    import torch as _torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="upi")
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--device", default="cuda" if _torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_demo(args.dataset, args.n_samples, args.max_epoch, args.device)
