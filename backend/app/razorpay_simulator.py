"""Razorpay webhook ingestion.

Verifies Razorpay's HMAC-SHA256 webhook signature scheme and applies
shared-identifier ring detection (union-find, matching csv_pipeline.py's
approach) to incoming payment events. Also generates signed synthetic
traffic through the same processing path, for testing without a live
Razorpay account. This module only receives webhooks; it never calls the
Razorpay API.
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

# Mirrors the secret configured in the Razorpay dashboard's Webhooks page.
# The fallback allows simulated traffic to work without configuration.
RAZORPAY_WEBHOOK_SECRET = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "demo_secret_for_local_testing")

CITIES = ["Mumbai", "Pune", "Jaipur", "Kolkata", "Ahmedabad", "Hyderabad"]
BANKS = ["okhdfcbank", "oksbi", "okicici", "okaxis", "ybl"]
ID_FIELDS = ("vpa", "email", "contact")


def sign_payload(raw_body: bytes, secret: str = RAZORPAY_WEBHOOK_SECRET) -> str:
    """Compute Razorpay's webhook signature: HMAC-SHA256 of the raw
    request body, hex-encoded."""
    return hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()


def verify_signature(raw_body: bytes, signature: str | None, secret: str = RAZORPAY_WEBHOOK_SECRET) -> bool:
    if not signature:
        return False
    expected = sign_payload(raw_body, secret)
    return hmac.compare_digest(expected, signature)


def sample_event(event_type: str, *, vpa: str, contact: str, email: str, amount: int, notes: dict | None = None) -> dict:
    """Build one webhook payload matching Razorpay's payment.captured and
    payment.failed schema. The payment id and timestamps are synthetic."""
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
    """Generate synthetic traffic: n_normal payments with distinct
    identifiers, plus one planted cluster of ring_size payments sharing a
    contact number and bank."""
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
    """Extract the fields used for ring detection from a Razorpay
    payment-webhook payload. Returns {} if the payload does not match the
    expected shape."""
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
    """Run union-find over shared vpa, email, and contact fields across
    the event buffer. Expects a list of dicts with those keys, ordered
    oldest-first (see case_store.list_razorpay_events)."""
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
    """Sign each event and pass it to process_fn(raw_body, signature,
    is_simulated), the same function the /webhooks/razorpay route calls.
    Sleeps delay_seconds between events so a polling frontend can show
    traffic arriving over time."""
    for ev in events:
        body = json.dumps(ev).encode()
        sig = sign_payload(body)
        process_fn(body, sig, True)
        if delay_seconds:
            time.sleep(delay_seconds)
