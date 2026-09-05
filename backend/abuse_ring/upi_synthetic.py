"""Synthetic UPI (Unified Payments Interface) transaction network.

This is synthetic data: no public India-specific financial fraud graph
dataset with real labels was found. It demonstrates the model on a
Razorpay-relevant data shape; the published-benchmark comparison lives in
elliptic_data.py.

Simulated pattern: a money-mule ring.
  1. A small set of collector VPAs are registered together in a tight
     window, sharing 1-3 device IDs and 1-2 bank accounts.
  2. Ordinary accounts each send one small payment to a randomly chosen
     collector within a short window (fan-in). These senders are not
     labeled fraud.
  3. Collectors forward pooled funds to 1-2 cash-out VPAs within hours
     (fan-out). Only collectors and cash-out VPAs are labeled fraud.

Hard negatives: separate legitimate shared-device or shared-bank clusters
(for example, a household sharing one phone for UPI) with no fan-in or
fan-out pattern, to test that shared identifiers alone do not trigger a
flag.

Split: by signup week, not random, so test-split rings are unseen during
training.
"""

from __future__ import annotations

import random

import dgl
import numpy as np
import pandas as pd
import torch

N_LEGIT_USERS = 18000
N_HARD_NEGATIVE_CLUSTERS = 150
HARD_NEGATIVE_CLUSTER_SIZE = (2, 12)
N_RINGS = 45
RING_COLLECTOR_SIZE = (5, 16)
RING_CASHOUT_SIZE = (1, 2)
RING_SOURCE_SENDERS = (15, 60)
N_WEEKS = 12
TRAIN_MAX_WEEK = 7
VAL_MAX_WEEK = 9
BANKS = ["oksbi", "okhdfcbank", "okicici", "okaxis", "ybl", "paytm"]
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Hyderabad", "Chennai", "Ahmedabad", "Jaipur", "Lucknow", "Kolkata"]


def _vpa(uid: int, rng: random.Random) -> str:
    return f"u{uid}{rng.randint(100,999)}@{rng.choice(BANKS)}"


def generate_upi_population(seed: int = 0, share_fraction_override: float | None = None):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    users = []  # each: dict of attributes
    uid_counter = 0

    def new_user(signup_week, device_id=None, bank_account_id=None, city=None, ip_address=None, phone_hash=None):
        nonlocal uid_counter
        uid = uid_counter
        uid_counter += 1
        users.append(
            {
                "uid": uid,
                "vpa": _vpa(uid, rng),
                "signup_week": signup_week,
                "device_id": device_id if device_id is not None else f"dev{uid}",
                "bank_account_id": bank_account_id if bank_account_id is not None else f"bank{uid}",
                "ip_address": ip_address if ip_address is not None else f"ip{uid}",
                "phone_hash": phone_hash if phone_hash is not None else f"ph{uid}",
                "city": city or rng.choice(CITIES),
                "is_ring": False,
            }
        )
        return uid

    # --- ordinary legitimate users, independent devices/banks ---
    for _ in range(N_LEGIT_USERS):
        week = rng.randint(0, N_WEEKS - 1)
        new_user(week)

    # --- legitimate hard-negative shared-device/bank/ip/phone clusters (families/shops) ---
    hard_negative_uids = []
    for _ in range(N_HARD_NEGATIVE_CLUSTERS):
        size = rng.randint(*HARD_NEGATIVE_CLUSTER_SIZE)
        week = rng.randint(0, N_WEEKS - 1)
        shared_device = f"shdev{uid_counter}"
        shared_bank = f"shbank{uid_counter}" if rng.random() < 0.5 else None
        shared_ip = f"ship{uid_counter}"  # a household/shop's router - genuinely shared
        shared_phone = f"shph{uid_counter}" if rng.random() < 0.3 else None  # e.g. a shop's registered OTP line
        city = rng.choice(CITIES)
        cluster_uids = []
        for _ in range(size):
            u = new_user(
                week + rng.randint(0, 1),
                device_id=shared_device,
                bank_account_id=shared_bank if shared_bank else f"bank{uid_counter}",
                ip_address=shared_ip,
                phone_hash=shared_phone if shared_phone else f"ph{uid_counter}",
                city=city,
            )
            cluster_uids.append(u)
        hard_negative_uids.append(cluster_uids)

    # --- mule rings (camouflaged) ---
    rings = []
    for _ in range(N_RINGS):
        ring_week = rng.randint(2, N_WEEKS - 1)  # rings appear over time, incl. late (test-only) ones
        n_collectors = rng.randint(*RING_COLLECTOR_SIZE)
        n_cashout = rng.randint(*RING_CASHOUT_SIZE)
        n_devices = rng.randint(1, 3)
        n_banks = rng.randint(1, 2)
        n_ips = rng.randint(1, 2)  # a ring operating from a small number of locations/VPNs
        devices = [f"ringdev{uid_counter}_{i}" for i in range(n_devices)]
        banks_ = [f"ringbank{uid_counter}_{i}" for i in range(n_banks)]
        ips = [f"ringip{uid_counter}_{i}" for i in range(n_ips)]
        phones = [f"ringph{uid_counter}_{i}" for i in range(rng.randint(1, 2))]
        city = rng.choice(CITIES)

        # Camouflage 1: only a fraction of collectors actually share any
        # given identifier with each other - the rest use unique values, so
        # 1-hop connectivity on any single relation alone can't cleanly
        # separate the whole ring (real rings don't put every member on one
        # traceable device/IP/phone). share_fraction_override fixes all four
        # to one exact value for the camouflage-frontier sweep in
        # camouflage_sweep.py - "how low can this go before detection breaks."
        if share_fraction_override is not None:
            device_share_fraction = bank_share_fraction = ip_share_fraction = phone_share_fraction = share_fraction_override
        else:
            device_share_fraction = rng.uniform(0.35, 0.75)
            bank_share_fraction = rng.uniform(0.35, 0.75)
            ip_share_fraction = rng.uniform(0.35, 0.75)
            phone_share_fraction = rng.uniform(0.20, 0.55)  # phone sharing is rarer even within real rings

        collector_uids = []
        for _ in range(n_collectors):
            u = new_user(
                ring_week + rng.randint(0, 1),
                device_id=rng.choice(devices) if rng.random() < device_share_fraction else None,
                bank_account_id=rng.choice(banks_) if rng.random() < bank_share_fraction else None,
                ip_address=rng.choice(ips) if rng.random() < ip_share_fraction else None,
                phone_hash=rng.choice(phones) if rng.random() < phone_share_fraction else None,
                city=city,
            )
            users[u]["is_ring"] = True
            collector_uids.append(u)

        # Cash-out nodes are the actual operator infrastructure - always on
        # a shared identifier (someone has to actually receive the money).
        cashout_uids = []
        for _ in range(n_cashout):
            u = new_user(
                ring_week + rng.randint(0, 2),
                device_id=rng.choice(devices),
                bank_account_id=rng.choice(banks_),
                ip_address=rng.choice(ips),
                phone_hash=rng.choice(phones),
                city=city,
            )
            users[u]["is_ring"] = True
            cashout_uids.append(u)

        # Camouflage 2: innocent bystanders coincidentally registered on the
        # same shared device/IP (e.g. a kiosk/agent who onboarded several
        # real customers on one device on one connection) - real hard
        # negatives sitting INSIDE a mostly-fraudulent cluster.
        for _ in range(rng.randint(1, 3)):
            new_user(ring_week + rng.randint(-2, 3), device_id=rng.choice(devices), ip_address=rng.choice(ips), city=city)

        n_sources = rng.randint(*RING_SOURCE_SENDERS)
        source_pool = [u["uid"] for u in users if u["signup_week"] <= ring_week]
        sources = rng.sample(source_pool, min(n_sources, len(source_pool))) if source_pool else []

        rings.append(
            {
                "week": ring_week,
                "collectors": collector_uids,
                "cashout": cashout_uids,
                "sources": sources,
            }
        )

    return users, hard_negative_uids, rings, rng, np_rng


def _amount(rng: random.Random, kind: str) -> float:
    if kind == "normal":
        # Occasional large legitimate payment (rent, purchase, etc.) so
        # transaction amount alone isn't a trivial giveaway for fan_out -
        # the model has to actually use structure, not just "big number".
        if rng.random() < 0.08:
            return float(rng.choice([6000, 10000, 20000, 35000, 50000]) * (0.8 + 0.4 * rng.random()))
        return float(rng.choice([10, 50, 100, 200, 500, 1000, 1500, 2000, 5000]) * (0.8 + 0.4 * rng.random()))
    if kind == "fan_in":
        return float(rng.choice([500, 750, 1000, 1500, 2000, 3000]) * (0.8 + 0.4 * rng.random()))
    if kind == "fan_out":
        return float(rng.choice([8000, 15000, 25000, 40000, 60000]) * (0.8 + 0.4 * rng.random()))
    raise ValueError(kind)


def generate_transactions(users, hard_negative_uids, rings, rng: random.Random):
    """Returns a list of (payer_uid, payee_uid, amount, day, is_ring_txn)."""
    txns = []
    n_users = len(users)

    # Ordinary background traffic: every user makes a handful of normal P2P
    # payments to roughly-nearby (same city) or random other users.
    by_city = {}
    for u in users:
        by_city.setdefault(u["city"], []).append(u["uid"])

    for u in users:
        n_txn = rng.randint(1, 8)
        week_day0 = u["signup_week"] * 7
        candidates = by_city.get(u["city"], [u["uid"]])
        for _ in range(n_txn):
            payee = rng.choice(candidates) if rng.random() < 0.7 else rng.randint(0, n_users - 1)
            if payee == u["uid"]:
                continue
            day = week_day0 + rng.randint(0, 30)
            txns.append((u["uid"], payee, _amount(rng, "normal"), day, False))

    # Hard-negative clusters: normal intra-cluster + external traffic, no burst pattern.
    for cluster in hard_negative_uids:
        for uid in cluster:
            week_day0 = users[uid]["signup_week"] * 7
            for _ in range(rng.randint(2, 6)):
                other = rng.choice(cluster) if rng.random() < 0.4 else rng.randint(0, n_users - 1)
                if other == uid:
                    continue
                day = week_day0 + rng.randint(0, 30)
                txns.append((uid, other, _amount(rng, "normal"), day, False))

    # Ring fan-in / fan-out bursts.
    for ring in rings:
        burst_day0 = ring["week"] * 7 + rng.randint(0, 3)
        for src in ring["sources"]:
            collector = rng.choice(ring["collectors"])
            day = burst_day0 + rng.randint(0, 2)  # fan-in burst window: ~2 days
            txns.append((src, collector, _amount(rng, "fan_in"), day, True))
        for collector in ring["collectors"]:
            cashout = rng.choice(ring["cashout"])
            day = burst_day0 + rng.randint(1, 3)  # fan-out shortly after fan-in
            txns.append((collector, cashout, _amount(rng, "fan_out"), day, True))

        # Camouflage 3: ring members also do a handful of ordinary-looking
        # transactions unrelated to the ring, so their aggregate behavioral
        # features (avg amount, txn count) don't stand out as pure-burst
        # outliers - a real operator tries to make the account look used.
        ring_members = ring["collectors"] + ring["cashout"]
        for uid in ring_members:
            for _ in range(rng.randint(1, 4)):
                other = rng.randint(0, n_users - 1)
                if other == uid:
                    continue
                day = users[uid]["signup_week"] * 7 + rng.randint(-5, 20)
                txns.append((uid, other, _amount(rng, "normal"), day, False))

    return txns


def build_features(users, txns_by_uid, n_users):
    # Deliberately EXCLUDES device/bank sharing counts - those would just
    # restate the shares_device/shares_bank graph relations as a scalar,
    # making the task trivially solvable without any graph reasoning at all
    # and defeating the point of the demo. Only genuine per-account
    # behavioral signals go in here; the ring signal has to come from
    # structure (the relations themselves), not a leaked feature.
    feats = np.zeros((n_users, 7), dtype=np.float32)
    for u in users:
        uid = u["uid"]
        sent = txns_by_uid["sent"].get(uid, [])
        recv = txns_by_uid["recv"].get(uid, [])
        all_amt = [a for a, _, _ in sent] + [a for a, _, _ in recv]
        days = [d for _, d, _ in sent] + [d for _, d, _ in recv]
        counterparties = {c for _, _, c in sent} | {c for _, _, c in recv}
        feats[uid] = [
            float(len(sent)),
            float(len(recv)),
            float(np.mean(all_amt)) if all_amt else 0.0,
            float(np.max(all_amt)) if all_amt else 0.0,
            float(np.std(days)) if len(days) > 1 else 0.0,
            float(len(counterparties)),
            float(u["signup_week"]),
        ]
    return feats


def generate_upi_graph(seed: int = 0, share_fraction_override: float | None = None):
    users, hard_negative_uids, rings, rng, _ = generate_upi_population(seed, share_fraction_override)
    n_users = len(users)
    txns = generate_transactions(users, hard_negative_uids, rings, rng)

    src = np.array([t[0] for t in txns], dtype=np.int64)
    dst = np.array([t[1] for t in txns], dtype=np.int64)
    amounts = np.array([t[2] for t in txns], dtype=np.float32)
    days = np.array([t[3] for t in txns], dtype=np.int64)

    txns_by_uid = {"sent": {}, "recv": {}}
    for p, q, amt, day, _ in txns:
        txns_by_uid["sent"].setdefault(p, []).append((amt, day, q))
        txns_by_uid["recv"].setdefault(q, []).append((amt, day, p))

    feats = build_features(users, txns_by_uid, n_users)

    device_ids = [u["device_id"] for u in users]
    bank_ids = [u["bank_account_id"] for u in users]
    ip_ids = [u["ip_address"] for u in users]
    phone_ids = [u["phone_hash"] for u in users]

    def shared_edges(keys):
        groups: dict[str, list[int]] = {}
        for uid, k in enumerate(keys):
            groups.setdefault(k, []).append(uid)
        s, d = [], []
        for members in groups.values():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    s.append(members[i])
                    d.append(members[j])
                    s.append(members[j])
                    d.append(members[i])
        return np.array(s, dtype=np.int64), np.array(d, dtype=np.int64)

    dev_s, dev_d = shared_edges(device_ids)
    bank_s, bank_d = shared_edges(bank_ids)
    ip_s, ip_d = shared_edges(ip_ids)
    phone_s, phone_d = shared_edges(phone_ids)

    g = dgl.heterograph(
        {
            ("user", "sent_to", "user"): (torch.tensor(src), torch.tensor(dst)),
            ("user", "shares_device", "user"): (torch.tensor(dev_s), torch.tensor(dev_d)),
            ("user", "shares_bank", "user"): (torch.tensor(bank_s), torch.tensor(bank_d)),
            ("user", "shares_ip", "user"): (torch.tensor(ip_s), torch.tensor(ip_d)),
            ("user", "shares_phone", "user"): (torch.tensor(phone_s), torch.tensor(phone_d)),
        },
        num_nodes_dict={"user": n_users},
    )

    label = torch.tensor([1 if u["is_ring"] else 0 for u in users], dtype=torch.long)
    signup_week = torch.tensor([u["signup_week"] for u in users], dtype=torch.long)

    g.ndata["feature"] = torch.tensor(feats)
    g.ndata["label"] = label
    g.ndata["signup_week"] = signup_week
    # Raw (pre-standardization) max transaction amount in rupees - kept as a
    # separate field so amount-scaled false-negative cost has a real number
    # to work with even after `feature` gets z-scored for the model input.
    g.ndata["amount_proxy"] = torch.tensor(feats[:, 3].copy())
    g.ndata["train_mask"] = signup_week <= TRAIN_MAX_WEEK
    g.ndata["val_mask"] = (signup_week > TRAIN_MAX_WEEK) & (signup_week <= VAL_MAX_WEEK)
    g.ndata["test_mask"] = signup_week > VAL_MAX_WEEK

    meta = pd.DataFrame(users)
    meta["day0"] = meta["signup_week"] * 7

    return g, meta, txns
