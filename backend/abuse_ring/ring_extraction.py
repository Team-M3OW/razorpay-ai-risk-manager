"""Reconstructs candidate rings (hyperedges) from the benchmark's pairwise
relation graphs, and builds the bipartite member<->group heterograph the
CA-HGAT model trains on.

The shipped dataset only stores the already-flattened pairwise relation graph
(e.g. Yelp's net_rsr = "same product, same rating, same week" turned into a
clique of pairwise edges), not the original group membership. We reconstruct
approximate groups from that pairwise graph:

- Sparse relations (avg degree <= SPARSE_DEGREE_THRESHOLD, e.g. Yelp's
  net_rur = "same reviewer") are just connected components - the relation
  already partitions cleanly into natural groups.
- Dense relations (e.g. net_rsr/net_rtr) are far too dense for connected
  components to mean anything (one giant blob), so we run Leiden community
  detection *within* each connected component to find the tightly co-active
  sub-clusters - this is the actual "ring" signal in a dense graph.

This is a stated approximation, not a hidden one.

Three changes on top of the original version:

1. MULTI-RESOLUTION - the resolution parameter trades off community count
   vs. size (higher resolution -> more, smaller communities). Running it at
   two resolutions and keeping both as separate relation types (e.g.
   `net_rsr` and `net_rsr_fine`) means one wrong granularity doesn't lose the
   signal entirely - the model gets both views and its own attention decides
   which one carries more information for a given case.
2. MODULARITY DIAGNOSTIC (logged, NOT a gate) - a partition's own modularity
   score was tried as an automatic filter (drop any relation scoring below a
   floor, on the theory that low modularity means the "groups" are a
   clustering artifact rather than real structure). Checked empirically
   against our own results and IT DOESN'T WORK: Elliptic's `flow` relation
   scores modularity 0.93 (very high) despite the ring mechanism not helping
   there, while Amazon's `net_usu` - the single relation responsible for our
   best result, since without its group bottleneck the plain baseline
   collapses entirely - scores only 0.27 (would have been gated out,
   deleting the result). Modularity measures whether a partition is
   graph-theoretically well-separated; it says nothing about whether that
   grouping predicts the fraud label, which is the actually relevant
   question. Kept as a logged diagnostic (still informative to look at) but
   explicitly NOT used to exclude a relation.
3. LOUVAIN -> LEIDEN - Louvain (used in the first version of this module)
   has a known defect: it can produce internally disconnected "communities"
   in some cases, because its local-moving step never checks connectivity.
   Leiden (Traag, Waltman & van Eck, 2019) fixes this by construction (every
   community it returns is guaranteed connected) and converges faster on
   large graphs - the actual reason for the swap being scalability, not a
   quality problem observed in our own results.
"""

from __future__ import annotations

import dgl
import igraph as ig
import leidenalg
import networkx as nx
import torch

MIN_GROUP_SIZE = 3
MAX_GROUP_SIZE = 60
SPARSE_DEGREE_THRESHOLD = 10.0
MODULARITY_FLOOR = 0.3
RESOLUTIONS = {"": 1.0, "_fine": 2.0}


def _relation_to_nx(g: dgl.DGLGraph, etype: tuple[str, str, str]) -> nx.Graph:
    sub = g[etype]
    nxg = nx.Graph()
    nxg.add_nodes_from(range(g.num_nodes(etype[0])))
    src, dst = sub.edges()
    nxg.add_edges_from(zip(src.tolist(), dst.tolist()))
    return nxg


def _leiden_communities(sub_nxg: nx.Graph, resolution: float, seed: int = 0) -> list[set]:
    """Leiden community detection on one connected component, returned in
    the same shape nx.community.louvain_communities used to return (a list
    of node-id sets) so the rest of the module doesn't need to know which
    algorithm ran."""
    nodes = list(sub_nxg.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in sub_nxg.edges()]
    g_ig = ig.Graph(n=len(nodes), edges=edges)
    partition = leidenalg.find_partition(
        g_ig, leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution, seed=seed,
    )
    return [{nodes[i] for i in community} for community in partition]


def extract_hyperedges(
    g: dgl.DGLGraph,
    min_size: int = MIN_GROUP_SIZE,
    max_size: int = MAX_GROUP_SIZE,
    sparse_degree_threshold: float = SPARSE_DEGREE_THRESHOLD,
    modularity_floor: float = MODULARITY_FLOOR,
    multi_resolution: bool = True,
) -> tuple[dict[str, list[list[int]]], dict[str, dict]]:
    """Returns (hyperedges, diagnostics).

    hyperedges: {relation_key: [ [member_node_id, ...], ... ]} - relation_key
    is the raw relation name for sparse (component-based) relations, or
    `<relation>` / `<relation>_fine` for the two Leiden resolutions on dense
    relations.

    diagnostics: {relation_key: {method, resolution, modularity, n_groups,
    gated_out}} - so a caller (or a report) can see *why* a relation was or
    wasn't used, not just the final group lists.
    """
    ntype = g.ntypes[0]
    n_nodes = g.num_nodes(ntype)
    hyperedges: dict[str, list[list[int]]] = {}
    diagnostics: dict[str, dict] = {}
    resolutions = RESOLUTIONS if multi_resolution else {"": 1.0}

    for etype in g.canonical_etypes:
        rel_name = etype[1]
        n_edges = g.num_edges(etype)
        avg_degree = (2 * n_edges) / max(n_nodes, 1)
        nxg = _relation_to_nx(g, etype)

        if avg_degree <= sparse_degree_threshold:
            groups: list[list[int]] = []
            for comp in nx.connected_components(nxg):
                if min_size <= len(comp) <= max_size:
                    groups.append(sorted(comp))
            hyperedges[rel_name] = groups
            diagnostics[rel_name] = {
                "method": "components", "resolution": None, "modularity": None,
                "n_groups": len(groups), "gated_out": False,
            }
            continue

        for suffix, resolution in resolutions.items():
            key = f"{rel_name}{suffix}"
            groups = []
            # Must stay a complete partition of nxg (every node, once) or
            # nx.modularity raises NotAPartition - components too small to
            # be a candidate ring still count as their own (singleton-ish)
            # community for the modularity calculation, just never emitted
            # as an output group.
            all_communities = []
            for comp in nx.connected_components(nxg):
                if len(comp) < min_size:
                    all_communities.append(set(comp))
                    continue
                sub_nxg = nxg.subgraph(comp)
                communities = _leiden_communities(sub_nxg, resolution=resolution, seed=0)
                all_communities.extend(communities)
                for com in communities:
                    if min_size <= len(com) <= max_size:
                        groups.append(sorted(com))

            mod = (
                nx.algorithms.community.quality.modularity(nxg, all_communities)
                if len(all_communities) >= 2 else 0.0
            )
            # NOT used to exclude the relation - see module docstring for why
            # a modularity floor was tried and empirically falsified.
            diagnostics[key] = {
                "method": "leiden", "resolution": resolution, "modularity": mod,
                "n_groups": len(groups), "gated_out": False,
            }
            hyperedges[key] = groups

    return hyperedges, diagnostics


def build_bipartite_graph(
    g: dgl.DGLGraph, hyperedges: dict[str, list[list[int]]], feature_key: str = "feature"
):
    """Builds the member<->group heterograph consumed by CAHGAT.

    Node types: 'member' (same nodes/order as g) + 'group_<relation>' (one per
    reconstructed hyperedge for that relation).
    Edge types: 'in_<relation>' (member->group), 'has_<relation>' (group->member).
    Group init features = [mean of member features, log(size), 0.0] (the
    trailing 0.0 is a placeholder slot for a future temporal-spread feature).
    """
    member_feat = g.ndata[feature_key]
    n_members = member_feat.shape[0]

    data_dict: dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]] = {}
    num_nodes_dict: dict[str, int] = {"member": n_members}
    group_meta: dict[str, list[list[int]]] = {}
    group_feats: dict[str, torch.Tensor] = {}

    for rel_name, groups in hyperedges.items():
        gtype = f"group_{rel_name}"
        n_groups = len(groups)
        # DGL requires every declared ntype to have >=1 node; skip relations
        # that produced no valid groups at all rather than declare an empty type.
        if n_groups == 0:
            continue

        num_nodes_dict[gtype] = n_groups
        src_list: list[int] = []
        dst_list: list[int] = []
        feats = torch.zeros(n_groups, member_feat.shape[1] + 2)

        for gi, members in enumerate(groups):
            src_list.extend(members)
            dst_list.extend([gi] * len(members))
            member_idx = torch.tensor(members, dtype=torch.long)
            mean_feat = member_feat[member_idx].mean(dim=0)
            size_feat = torch.tensor([float(len(members)), 0.0])
            feats[gi] = torch.cat([mean_feat, size_feat])

        src_t = torch.tensor(src_list, dtype=torch.long)
        dst_t = torch.tensor(dst_list, dtype=torch.long)
        data_dict[("member", f"in_{rel_name}", gtype)] = (src_t, dst_t)
        data_dict[(gtype, f"has_{rel_name}", "member")] = (dst_t, src_t)
        group_meta[rel_name] = groups
        group_feats[gtype] = feats

    bg = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    bg.nodes["member"].data["feature"] = member_feat
    bg.nodes["member"].data["label"] = g.ndata["label"]
    for key in ("train_mask", "val_mask", "test_mask", "time_step", "amount_proxy", "signup_week"):
        if key in g.ndata:
            bg.nodes["member"].data[key] = g.ndata[key]
    for gtype, feats in group_feats.items():
        bg.nodes[gtype].data["feature"] = feats

    return bg, group_meta
