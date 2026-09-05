"""Persistence layer for the case-queue product: analyst dispositions (the
real ring-level labels this project has been honest about not having yet)
and a watchlist of confirmed-bad shared identifiers, both backing the
FastAPI app in app/main.py.

The disposition table addresses a limitation stated throughout this
project: every ring-level label so far is a majority-vote proxy over real
node labels, because no real "this cluster was confirmed as a ring" label
exists anywhere. That label can only come from an analyst reviewing an
actual flagged case. This table is where those accumulate as a byproduct of
normal review, not a retrofit.

The watchlist table is the mechanism behind the "if a confirmed-bad
identifier resurfaces, flag it immediately" product feature: when a case is
confirmed, its shared entity value (e.g. a specific device_id or
bank_account_id) goes on the watchlist, and any future case sharing that
same value gets flagged regardless of what the model itself scores it -
useful specifically because a low-volume new instance of a known-bad
identifier may not yet have enough transaction history for the graph model
to catch on its own.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "case_store.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS ring_case_status (
    dataset TEXT NOT NULL,
    relation TEXT NOT NULL,
    group_index INTEGER NOT NULL,
    member_ids TEXT NOT NULL,          -- JSON list of member node indices, for audit
    model_score REAL,                  -- group_head probability at time of review
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | confirmed_ring | dismissed
    reviewer TEXT,
    note TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (dataset, relation, group_index)
);

CREATE TABLE IF NOT EXISTS watchlist (
    dataset TEXT NOT NULL,
    entity_type TEXT NOT NULL,   -- e.g. 'device_id', 'bank_account_id', 'ip_address'
    entity_value TEXT NOT NULL,
    source_relation TEXT,
    source_group_index INTEGER,
    reviewer TEXT,
    created_at REAL NOT NULL,
    PRIMARY KEY (dataset, entity_type, entity_value)
);

CREATE TABLE IF NOT EXISTS razorpay_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    vpa TEXT,
    email TEXT,
    contact TEXT,
    amount INTEGER,
    is_simulated INTEGER NOT NULL DEFAULT 1,
    raw_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


def connect(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def upsert_case(
    conn: sqlite3.Connection,
    dataset: str,
    relation: str,
    group_index: int,
    member_ids: list[int],
    model_score: float | None = None,
):
    """Create a case row if it doesn't exist yet (status='pending'). Called
    once per flagged group when a case queue is populated - never overwrites
    an existing disposition."""
    now = time.time()
    conn.execute(
        """INSERT INTO ring_case_status
           (dataset, relation, group_index, member_ids, model_score, status, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
           ON CONFLICT(dataset, relation, group_index) DO NOTHING""",
        (dataset, relation, group_index, str(member_ids), model_score, now, now),
    )
    conn.commit()


def set_disposition(
    conn: sqlite3.Connection,
    dataset: str,
    relation: str,
    group_index: int,
    status: str,
    reviewer: str,
    note: str = "",
):
    """An analyst's actual verdict on a case - this is the real label this
    whole module exists to eventually collect."""
    if status not in ("confirmed_ring", "dismissed", "pending"):
        raise ValueError(f"unknown status {status!r}")
    conn.execute(
        """UPDATE ring_case_status SET status=?, reviewer=?, note=?, updated_at=?
           WHERE dataset=? AND relation=? AND group_index=?""",
        (status, reviewer, note, time.time(), dataset, relation, group_index),
    )
    conn.commit()


def set_disposition_bulk(conn: sqlite3.Connection, dataset: str, cases: list[tuple[str, int]], status: str, reviewer: str, note: str = ""):
    """Apply one disposition to many cases at once - e.g. confirm every case
    in a filtered view in one action instead of one at a time."""
    now = time.time()
    conn.executemany(
        """UPDATE ring_case_status SET status=?, reviewer=?, note=?, updated_at=?
           WHERE dataset=? AND relation=? AND group_index=?""",
        [(status, reviewer, note, now, dataset, relation, group_index) for relation, group_index in cases],
    )
    conn.commit()


def add_to_watchlist(
    conn: sqlite3.Connection,
    dataset: str,
    entity_type: str,
    entity_value: str,
    source_relation: str,
    source_group_index: int,
    reviewer: str,
):
    conn.execute(
        """INSERT INTO watchlist (dataset, entity_type, entity_value, source_relation, source_group_index, reviewer, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(dataset, entity_type, entity_value) DO NOTHING""",
        (dataset, entity_type, entity_value, source_relation, source_group_index, reviewer, time.time()),
    )
    conn.commit()


def list_watchlist(conn: sqlite3.Connection, dataset: str) -> list[dict]:
    rows = conn.execute(
        """SELECT entity_type, entity_value, source_relation, source_group_index, reviewer, created_at
           FROM watchlist WHERE dataset=? ORDER BY created_at DESC""",
        (dataset,),
    ).fetchall()
    return [
        {
            "entity_type": r[0], "entity_value": r[1], "source_relation": r[2],
            "source_group_index": r[3], "reviewer": r[4], "created_at": r[5],
        }
        for r in rows
    ]


def check_watchlist(conn: sqlite3.Connection, dataset: str, entity_type: str, entity_value: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM watchlist WHERE dataset=? AND entity_type=? AND entity_value=?",
        (dataset, entity_type, str(entity_value)),
    ).fetchone()
    return row is not None


def add_razorpay_event(
    conn: sqlite3.Connection,
    event_id: str,
    event_type: str,
    vpa: str | None,
    email: str | None,
    contact: str | None,
    amount: int | None,
    raw_json: str,
    is_simulated: bool = True,
):
    """One ingested Razorpay-shaped webhook payload (real inbound webhook or
    self-signed simulated traffic - both land here identically). Idempotent
    on event_id so a webhook retry doesn't double-count."""
    conn.execute(
        """INSERT INTO razorpay_events
           (event_id, event_type, vpa, email, contact, amount, is_simulated, raw_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(event_id) DO NOTHING""",
        (event_id, event_type, vpa, email, contact, amount, int(is_simulated), raw_json, time.time()),
    )
    conn.commit()


def list_razorpay_events(conn: sqlite3.Connection, limit: int = 300) -> list[dict]:
    """Oldest-first, so member indices derived from this list stay stable
    across polls within one demo run as new events are appended."""
    rows = conn.execute(
        """SELECT event_id, event_type, vpa, email, contact, amount, is_simulated, created_at
           FROM razorpay_events ORDER BY created_at ASC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [
        {
            "event_id": r[0], "event_type": r[1], "vpa": r[2], "email": r[3],
            "contact": r[4], "amount": r[5], "is_simulated": bool(r[6]), "created_at": r[7],
        }
        for r in rows
    ]


def clear_razorpay_events(conn: sqlite3.Connection):
    conn.execute("DELETE FROM razorpay_events")
    conn.commit()


def export_labels(conn: sqlite3.Connection, dataset: str) -> list[dict]:
    """Real disposition-derived labels, ready to replace the majority-vote
    proxy the moment enough of them exist. Returns only reviewed cases -
    pending ones aren't labels yet."""
    rows = conn.execute(
        """SELECT relation, group_index, member_ids, status, reviewer, updated_at
           FROM ring_case_status WHERE dataset=? AND status != 'pending'""",
        (dataset,),
    ).fetchall()
    return [
        {
            "relation": r[0], "group_index": r[1], "member_ids": r[2],
            "label": 1 if r[3] == "confirmed_ring" else 0,
            "reviewer": r[4], "updated_at": r[5],
        }
        for r in rows
    ]
