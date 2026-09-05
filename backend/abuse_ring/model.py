"""CA-HGAT: Camouflage-Aware Hypergraph Attention Network.

A heterogeneous, bipartite, multi-relational Graph Attention Network over a
member<->group incidence graph (see ring_extraction.py for how groups/
hyperedges are built). Each round alternates:

  member -> group : GAT attention pools each group's members into a group
                     embedding. This is the camouflage-resistance mechanism -
                     a decoy member that doesn't help predict the group's
                     label gets down-weighted by the learned attention score,
                     not averaged in uniformly.
  group -> member : GAT attention broadcasts group context back to members.

Two heads read off the final representations:
  - node_head:  per-member fraud probability (comparable to the published
                CARE-GNN node-classification numbers on this benchmark).
  - group_head: per-group ring probability (the platform's actual
                contribution - no published model on this benchmark scores
                the group/ring itself).

Both heads are trained jointly through one shared trunk, so gradients from
the ring objective reshape the member embeddings too - not a frozen node
encoder with a classifier stacked on top afterwards.
"""

from __future__ import annotations

import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn


class CAHGAT(nn.Module):
    def __init__(
        self,
        member_in_dim: int,
        group_in_dims: dict[str, int],
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_rounds: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_rounds = num_rounds
        self.relations = [gtype[len("group_"):] for gtype in group_in_dims]

        self.member_encoder = nn.Linear(member_in_dim, hidden_dim)
        self.group_encoders = nn.ModuleDict(
            {gtype: nn.Linear(dim, hidden_dim) for gtype, dim in group_in_dims.items()}
        )

        head_dim = hidden_dim // num_heads
        self.m2g_layers = nn.ModuleList()
        self.g2m_layers = nn.ModuleList()
        for _ in range(num_rounds):
            self.m2g_layers.append(
                dglnn.HeteroGraphConv(
                    {
                        f"in_{rel}": dglnn.GATConv(
                            hidden_dim, head_dim, num_heads,
                            feat_drop=dropout, attn_drop=dropout,
                            allow_zero_in_degree=True,
                        )
                        for rel in self.relations
                    },
                    aggregate="sum",
                )
            )
            self.g2m_layers.append(
                dglnn.HeteroGraphConv(
                    {
                        f"has_{rel}": dglnn.GATConv(
                            hidden_dim, head_dim, num_heads,
                            feat_drop=dropout, attn_drop=dropout,
                            allow_zero_in_degree=True,
                        )
                        for rel in self.relations
                    },
                    aggregate="sum",
                )
            )

        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.group_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, bg, exclude_relations: set[str] | None = None):
        """exclude_relations: relation names to leave out of message passing
        entirely (used for post-hoc relation-importance analysis - see
        relation_importance() in train_eval.py). Excluded groups' own
        group_logits are meaningless (never updated), only node_logits should
        be read when this is set."""
        exclude_relations = exclude_relations or set()
        active = [r for r in self.relations if r not in exclude_relations]

        # HeteroGraphConv iterates over *every* canonical etype in the graph
        # it's given (only skipping by source-type presence), so the m2g and
        # g2m passes each need a view containing only their own edge
        # direction - otherwise m2g's HeteroGraphConv (which only registers
        # 'in_*' modules) chokes on the graph's 'has_*' edges too. Restricting
        # to `active` here is also what makes relation exclusion work: an
        # excluded relation's edges simply aren't in this view, so its
        # GATConv submodule never gets called.
        # dgl.edge_type_subgraph([]) errors on an empty list (used by the
        # relation-importance "no relations at all" baseline) - skip message
        # passing entirely in that case rather than special-casing an empty
        # graph view.
        bg_m2g = bg.edge_type_subgraph([f"in_{rel}" for rel in active]) if active else None
        bg_g2m = bg.edge_type_subgraph([f"has_{rel}" for rel in active]) if active else None

        h = {"member": torch.relu(self.member_encoder(bg.nodes["member"].data["feature"]))}
        for gtype, enc in self.group_encoders.items():
            h[gtype] = torch.relu(enc(bg.nodes[gtype].data["feature"]))

        member_h0 = h["member"]

        if active:
            for r in range(self.num_rounds):
                group_out = self.m2g_layers[r](bg_m2g, h)
                for gtype in self.group_encoders:
                    if gtype in group_out:
                        h[gtype] = torch.relu(group_out[gtype].flatten(1))

                member_out = self.g2m_layers[r](bg_g2m, h)
                h["member"] = torch.relu(member_out["member"].flatten(1)) + member_h0

        node_logits = self.node_head(h["member"]).squeeze(-1)
        group_logits = {
            gtype: self.group_head(h[gtype]).squeeze(-1) for gtype in self.group_encoders
        }
        return node_logits, group_logits, h
