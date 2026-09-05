"""CSV log upload pipeline.

Builds a shared-identifier graph over uploaded rows using union-find
(device_id, ip_address, phone_hash, bank_account_id, or vpa columns, as
present) and scores connected components with a structural heuristic
based on component size and the number of distinct identifier types
shared. This is not the trained CA-HGAT model; it has no ML dependencies,
so it runs in the lightweight serving tier.
"""

from __future__ import annotations

import csv
import io
import math
from collections import defaultdict

# canonical relation -> accepted header aliases (case-insensitive)
COLUMN_ALIASES = {
    "device_id": ["device_id", "device", "deviceid"],
    "ip_address": ["ip_address", "ip", "ipaddr", "ip_addr"],
    "phone_hash": ["phone_hash", "phone", "phone_number", "mobile", "mobile_number"],
    "bank_account_id": ["bank_account_id", "bank_account", "account_id", "account_number"],
    "vpa": ["vpa", "upi_id", "upi_handle", "handle"],
}
MERCHANT_ALIASES = ["merchant", "merchant_id", "merchant_name"]
ID_ALIASES = ["id", "row_id", "account_id", "user_id"]
MAX_ROWS = 5000

# Minimum component size to flag as a ring.
MIN_RING_SIZE = 3
FLAG_SCORE_THRESHOLD = 0.4


class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _resolve_columns(fieldnames: list[str]) -> tuple[dict[str, str], str | None, str | None]:
    lower_map = {f.lower().strip(): f for f in fieldnames}
    resolved = {}
    for canon, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in lower_map:
                resolved[canon] = lower_map[a]
                break
    merchant_col = next((lower_map[a] for a in MERCHANT_ALIASES if a in lower_map), None)
    id_col = next((lower_map[a] for a in ID_ALIASES if a in lower_map), None)
    return resolved, merchant_col, id_col


def _score(size: int, n_id_types: int) -> float:
    """Score a cluster from its size and the number of distinct identifier
    types it shares. Both terms saturate."""
    size_term = min(1.0, math.log2(size + 1) / 5.0)
    type_term = min(1.0, n_id_types / 3.0)
    return round(0.5 * size_term + 0.5 * type_term, 4)


def run_pipeline(raw_csv: bytes) -> dict:
    text = raw_csv.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV has no header row")
    rows = []
    for i, row in enumerate(reader):
        if i >= MAX_ROWS:
            break
        rows.append(row)
    if not rows:
        raise ValueError("CSV has no data rows")

    id_cols, merchant_col, id_col = _resolve_columns(reader.fieldnames)
    if not id_cols:
        raise ValueError(
            "no recognizable identifier columns found - expected at least one of: "
            + ", ".join(sorted({a for aliases in COLUMN_ALIASES.values() for a in aliases}))
        )

    n = len(rows)
    dsu = DSU(n)
    # value -> list of row indices, per identifier column
    hubs: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        for canon, col in id_cols.items():
            val = (row.get(col) or "").strip()
            if val:
                hubs[(canon, val)].append(idx)

    for (canon, val), members in hubs.items():
        if len(members) < 2:
            continue
        for i in range(1, len(members)):
            dsu.union(members[0], members[i])

    components: dict[int, list[int]] = defaultdict(list)
    for idx in range(n):
        components[dsu.find(idx)].append(idx)

    def row_label(idx: int) -> str:
        row = rows[idx]
        if id_col:
            return row.get(id_col, str(idx))
        for canon, col in id_cols.items():
            v = row.get(col)
            if v:
                return v
        return f"row_{idx}"

    rings = []
    clean_count = 0
    graph_nodes: dict[str, dict] = {}
    graph_edges = []
    ring_seq = 0
    for root, members in components.items():
        if len(members) < 2:
            clean_count += 1
            continue
        # which identifier types actually connect this specific component
        types_here = set()
        hubs_here = []
        for (canon, val), hub_members in hubs.items():
            overlap = [m for m in hub_members if dsu.find(m) == root]
            if len(overlap) >= 2:
                types_here.add(canon)
                hubs_here.append((canon, val, overlap))
        if len(members) < MIN_RING_SIZE:
            clean_count += len(members)
            continue

        score = _score(len(members), len(types_here))
        flagged = score >= FLAG_SCORE_THRESHOLD
        merchants = sorted({rows[m].get(merchant_col) for m in members if merchant_col and rows[m].get(merchant_col)}) if merchant_col else []

        rings.append(
            {
                "ring_id": ring_seq,
                "size": len(members),
                "score": score,
                "flagged": flagged,
                "shared_via": sorted(types_here),
                "members": [{"row_index": m, "label": row_label(m), "raw": rows[m]} for m in members[:60]],
                "merchants": merchants,
            }
        )

        for canon, val, overlap in hubs_here:
            hub_id = f"hub:{canon}:{val}"
            graph_nodes[hub_id] = {"id": hub_id, "kind": "hub", "relation": canon, "score": score, "size": len(overlap), "label_text": f"{canon}={val}"}
            for m in overlap:
                mid = f"m:{m}"
                if mid not in graph_nodes:
                    graph_nodes[mid] = {
                        "id": mid, "kind": "member", "raw_id": m, "score": score,
                        "label": 1 if flagged else None, "vpa": row_label(m),
                        "merchant": rows[m].get(merchant_col) if merchant_col else None,
                    }
                graph_edges.append({"source": mid, "target": hub_id})
        ring_seq += 1

    rings.sort(key=lambda r: -r["score"])

    return {
        "row_count": n,
        "columns_used": sorted(id_cols.keys()),
        "merchant_column_found": merchant_col is not None,
        "rings": rings,
        "flagged_count": sum(1 for r in rings if r["flagged"]),
        "clean_count": clean_count,
        "graph": {"nodes": list(graph_nodes.values()), "edges": graph_edges},
    }


def sample_csv() -> str:
    """Return a small synthetic sample log file shaped like UPI
    transaction logs, for use with the upload demo."""
    rows = [
        ["id", "vpa", "device_id", "ip_address", "bank_account_id", "phone_hash", "merchant", "amount", "city"],
    ]
    # ring 1: shares device + bank account, 6 members, spends at one merchant
    for i in range(6):
        rows.append([f"u{100+i}", f"u{100+i}@okhdfcbank", "dev_ring_a", f"ip_{200+i}", "bank_ring_a", f"ph_{300+i}", "merchant_electronics_7", "4999", "Jaipur"])
    # ring 2: shares ip only, 5 members, spread across two merchants
    for i in range(5):
        rows.append([f"u{200+i}", f"u{200+i}@ybl", f"dev_{400+i}", "ip_ring_b", f"bank_{500+i}", f"ph_{600+i}", "merchant_giftcards_3" if i % 2 else "merchant_giftcards_3b", "1999", "Pune"])
    # clean rows: unique everything, spread across many merchants
    merchants = ["merchant_grocery_1", "merchant_fuel_2", "merchant_pharmacy_9", "merchant_apparel_4", "merchant_food_5"]
    for i in range(14):
        rows.append([f"u{900+i}", f"u{900+i}@oksbi", f"dev_{700+i}", f"ip_{800+i}", f"bank_{900+i}", f"ph_{999+i}", merchants[i % len(merchants)], str(200 + i * 37), "Mumbai"])

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerows(rows)
    return buf.getvalue()
