"""Razorpay webhook ingestion demo: real payment-webhook shape, a real HMAC
signature scheme, and shared-identifier ring detection over the incoming
event buffer - the same union-find mechanism `csv_pipeline.py` uses for an
uploaded CSV, applied here to live "transactions" instead.

Real vs. simulated traffic are meant to be indistinguishable to the
pipeline: a simulated event is signed with the same webhook secret used to
verify a real inbound Razorpay webhook, and both land through the exact
same processing function in main.py. If a real Razorpay test-mode webhook
is ever pointed at this app, it enters the identical code path - there is
no separate "fake mode" branch. Nothing here calls any Razorpay API
(inbound webhooks only) - no payment is created, moved, or held for real.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import random
import time
import uuid
from collections import defaultdict

from .csv_pipeline import DSU

# In real use this is the secret configured in the Razorpay dashboard's
# Webhooks page, mirrored here via an env var. The fallback lets the
# simulated-traffic demo work with zero configuration - see module
# docstring: this is the value used to self-sign synthetic events too, so
# nothing about verification is bypassed for the no-account demo path.
RAZORPAY_WEBHOOK_SECRET = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "demo_secret_for_local_testing")

CITIES = ["Mumbai", "Pune", "Jaipur", "Kolkata", "Ahmedabad", "Hyderabad"]
BANKS = ["okhdfcbank", "oksbi", "okicici", "okaxis", "ybl"]
ID_FIELDS = ("vpa", "email", "contact")


def sign_payload(raw_body: bytes, secret: str = RAZORPAY_WEBHOOK_SECRET) -> str:
    """Razorpay's documented webhook signature scheme: HMAC-SHA256 of the
    raw request body, hex-encoded."""
    return hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()


def verify_signature(raw_body: bytes, signature: str | None, secret: str = RAZORPAY_WEBHOOK_SECRET) -> bool:
    if not signature:
        return False
    expected = sign_payload(raw_body, secret)
    return hmac.compare_digest(expected, signature)


def sample_event(event_type: str, *, vpa: str, contact: str, email: str, amount: int, notes: dict | None = None) -> dict:
    """One Razorpay-shaped webhook payload - real field names/nesting for
    payment.captured / payment.failed, matching Razorpay's documented
    webhook schema. The payment id and timestamps are synthetic; the shape
    is real."""
    payment_id = f"pay_{uuid.uuid4().hex[:14]}"
    now = int(time.time())
    return {
        "entity": "event",
        "account_id": "acc_demo",
        "event": event_type,
        "contains": ["payment"],
        "payload": {
            "payment": {
                "entity": {
                    "id": payment_id,
                    "entity": "payment",
                    "amount": amount,
                    "currency": "INR",
                    "status": "captured" if event_type == "payment.captured" else "failed",
                    "method": "upi",
                    "vpa": vpa,
                    "email": email,
                    "contact": contact,
                    "notes": notes or {},
                    "created_at": now,
                }
            }
        },
        "created_at": now,
    }


def generate_traffic_batch(n_normal: int = 20, ring_size: int = 6, seed: int | None = None) -> list[dict]:
    """A background of distinct-identifier normal payments plus one planted
    cluster sharing a contact number (and bank) across ring_size events -
    fully synthetic, disclosed as such. Same honesty register as
    upi_synthetic.py and csv_pipeline.sample_csv(): this proves the
    detection mechanism, it is not a claim about real traffic."""
    rng = random.Random(seed)
    events = []
    for _ in range(n_normal):
        uid = rng.randint(10000, 99999)
        events.append(
            sample_event(
                "payment.captured",
                vpa=f"u{uid}@{rng.choice(BANKS)}",
                contact=f"9{rng.randint(100000000, 999999999)}",
                email=f"user{uid}@example.com",
                amount=rng.randint(199, 4999) * 100,
            )
        )
    ring_contact = f"9{rng.randint(100000000, 999999999)}"
    ring_bank = rng.choice(BANKS)
    for _ in range(ring_size):
        uid = rng.randint(10000, 99999)
        events.append(
            sample_event(
                "payment.captured",
                vpa=f"u{uid}@{ring_bank}",
                contact=ring_contact,
                email=f"user{uid}@example.com",
                amount=rng.choice([4999, 9999]) * 100,
                notes={"simulated_ring": "true"},
            )
        )
    rng.shuffle(events)
    return events


def extract_identifiers(event: dict) -> dict:
    """Pulls the fields ring-detection cares about out of a Razorpay
    payment-webhook payload. Returns {} for anything not shaped like one -
    the caller decides whether that's an error."""
    try:
        entity = event["payload"]["payment"]["entity"]
    except (KeyError, TypeError):
        return {}
    return {
        "event_id": entity.get("id") or f"evt_{uuid.uuid4().hex[:10]}",
        "event_type": event.get("event", "unknown"),
        "vpa": entity.get("vpa"),
        "email": entity.get("email"),
        "contact": entity.get("contact"),
        "amount": entity.get("amount"),
    }


def detect_rings(events: list[dict]) -> dict:
    """Real union-find over shared vpa/email/contact across the event
    buffer - identical mechanism to csv_pipeline.py's DSU-based matching,
    applied to live events instead of CSV rows. `events` must be a list of
    dicts with vpa/email/contact keys (case_store.list_razorpay_events'
    shape), ordered oldest-first so indices stay stable across polls."""
    n = len(events)
    dsu = DSU(n)
    hubs: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, ev in enumerate(events):
        for field in ID_FIELDS:
            val = ev.get(field)
            if val:
                hubs[(field, val)].append(i)

    for (_field, _val), members in hubs.items():
        if len(members) < 2:
            continue
        for j in range(1, len(members)):
            dsu.union(members[0], members[j])

    components: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        components[dsu.find(i)].append(i)

    rings = []
    graph_nodes: dict[str, dict] = {}
    graph_edges = []
    ring_seq = 0
    for root, members in components.items():
        if len(members) < 3:
            continue
        types_here = set()
        hubs_here = []
        for (field, val), hub_members in hubs.items():
            overlap = [m for m in hub_members if dsu.find(m) == root]
            if len(overlap) >= 2:
                types_here.add(field)
                hubs_here.append((field, val, overlap))

        rings.append(
            {
                "ring_id": ring_seq,
                "member_indices": sorted(members),
                "size": len(members),
                "shared_via": sorted(types_here),
                "events": [events[m]["event_id"] for m in members],
            }
        )
        for field, val, overlap in hubs_here:
            hub_id = f"hub:{field}:{val}"
            graph_nodes[hub_id] = {"id": hub_id, "kind": "hub", "relation": field, "score": 0.9, "size": len(overlap), "label_text": f"{field}={val}"}
            for m in overlap:
                mid = f"m:{events[m]['event_id']}"
                if mid not in graph_nodes:
                    graph_nodes[mid] = {"id": mid, "kind": "member", "raw_id": events[m]["event_id"], "score": 0.9, "label": 1, "vpa": events[m].get("vpa")}
                graph_edges.append({"source": mid, "target": hub_id})
        ring_seq += 1

    return {"rings": rings, "graph": {"nodes": list(graph_nodes.values()), "edges": graph_edges}}


def sign_and_process_all(events: list[dict], process_fn, delay_seconds: float = 0.35) -> None:
    """Signs each event with the real webhook secret and pushes it through
    `process_fn(raw_body: bytes, signature: str, is_simulated: bool)` -
    the exact same function the real `/webhooks/razorpay` route calls after
    verifying a real Razorpay signature. Paced with a delay so a polling
    frontend can show traffic arriving over a few seconds rather than all
    at once."""
    for ev in events:
        body = json.dumps(ev).encode()
        sig = sign_payload(body)
        process_fn(body, sig, True)
        if delay_seconds:
            time.sleep(delay_seconds)
