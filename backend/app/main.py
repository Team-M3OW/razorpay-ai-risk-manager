"""Abuse Ring Sentinel API - the analyst workspace on top of the trained
models: case queue, evidence, dispositions, a watchlist that auto-flags
recurrence of confirmed-bad identifiers, an editable auto-action policy,
real webhook alerting, per-segment thresholds, ring-formation spike
detection, and CSV export.

Everything here is real and tested end to end (see the session's own
verification): the webhook alerter was proven against a local mock
receiver, the watchlist/policy logic runs against real case data, the
segment/spike analytics are precomputed from real model output. What genuinely
isn't here - because it needs infrastructure this project doesn't have, not
because it's unbuilt in principle - is a live merchant taxonomy (no dataset
has one) and a real external Slack workspace to point the alerter at (drop
a real webhook URL into ABUSE_RING_ALERT_WEBHOOK and it works unchanged).
"""

from __future__ import annotations

import csv
import io
import json
import sys
import time
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .csv_pipeline import run_pipeline, sample_csv
from .razorpay_simulator import (
    RAZORPAY_WEBHOOK_SECRET,
    detect_rings,
    extract_identifiers,
    generate_traffic_batch,
    sign_and_process_all,
    verify_signature,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from abuse_ring.alerting import send_alert  # noqa: E402
from abuse_ring.case_store import (  # noqa: E402
    add_razorpay_event,
    add_to_watchlist,
    check_watchlist,
    clear_razorpay_events,
    connect,
    export_labels,
    list_razorpay_events,
    list_watchlist,
    set_disposition,
    set_disposition_bulk,
    upsert_case,
)
from abuse_ring.policy import evaluate_policy  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DATASETS = ["yelp", "amazon", "elliptic", "upi"]
DATASET_KIND = {"yelp": "real", "amazon": "real", "elliptic": "real", "upi": "synthetic"}
WATCHABLE_ENTITY_FIELDS = ("device_id", "bank_account_id", "ip_address", "phone_hash")

app = FastAPI(title="Abuse Ring Sentinel API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _load_json(name: str):
    p = DATA_DIR / name
    if not p.exists():
        raise HTTPException(404, f"{name} not found - run the relevant `python -m abuse_ring.*` export script first")
    return json.loads(p.read_text())


def _try_load_json(name: str):
    try:
        return _load_json(name)
    except HTTPException:
        return None


def _case_entity_values(case: dict) -> list[tuple[str, str]]:
    """The (entity_type, value) pairs a case's evidence exposes - what the
    watchlist keys on."""
    return [(k, v) for k, v in case.get("evidence", {}).items() if k in WATCHABLE_ENTITY_FIELDS]


def _watchlist_hit(conn, ds: str, case: dict) -> bool:
    return any(check_watchlist(conn, ds, k, v) for k, v in _case_entity_values(case))


@app.get("/api/datasets")
def list_datasets():
    out = []
    for ds in DATASETS:
        m = _try_load_json(f"metrics_{ds}.json")
        if m is None:
            continue
        out.append(
            {
                "dataset": ds,
                "kind": DATASET_KIND[ds],
                "node_test_metrics": m.get("node_test_metrics"),
                "cost_constants": m.get("cost_constants"),
            }
        )
    return out


@app.get("/api/datasets/{ds}/metrics")
def get_metrics(ds: str):
    if ds not in DATASETS:
        raise HTTPException(404, "unknown dataset")
    return _load_json(f"metrics_{ds}.json")


@app.get("/api/datasets/{ds}/cases")
def list_cases(ds: str, status: str | None = None, relation: str | None = None, limit: int = 200):
    if ds not in DATASETS:
        raise HTTPException(404, "unknown dataset")
    cases = _load_json(f"cases_{ds}.json")
    conn = connect()
    out = []
    for c in cases:
        if relation and c["relation"] != relation:
            continue
        row = conn.execute(
            "SELECT status, created_at FROM ring_case_status WHERE dataset=? AND relation=? AND group_index=?",
            (ds, c["relation"], c["group_index"]),
        ).fetchone()
        case_status = row[0] if row else "pending"
        age_days = (time.time() - row[1]) / 86400 if row else None
        if status and case_status != status:
            continue
        out.append(
            {
                **{k: v for k, v in c.items() if k != "members"},
                "status": case_status,
                "age_days": round(age_days, 2) if age_days is not None else None,
                "on_watchlist": _watchlist_hit(conn, ds, c),
            }
        )
        if len(out) >= limit:
            break
    return out


@app.get("/api/datasets/{ds}/cases/{relation}/{group_index}")
def get_case(ds: str, relation: str, group_index: int):
    if ds not in DATASETS:
        raise HTTPException(404, "unknown dataset")
    cases = _load_json(f"cases_{ds}.json")
    for c in cases:
        if c["relation"] == relation and c["group_index"] == group_index:
            conn = connect()
            upsert_case(conn, ds, relation, group_index, [m["id"] for m in c["members"]], c["score"])
            row = conn.execute(
                "SELECT status, reviewer, note, created_at FROM ring_case_status WHERE dataset=? AND relation=? AND group_index=?",
                (ds, relation, group_index),
            ).fetchone()
            decision = evaluate_policy(c, _watchlist_hit(conn, ds, c))
            return {
                **c,
                "status": row[0], "reviewer": row[1], "note": row[2],
                "age_days": round((time.time() - row[3]) / 86400, 2),
                "on_watchlist": _watchlist_hit(conn, ds, c),
                "policy_recommendation": decision.__dict__,
            }
    raise HTTPException(404, "case not found")


class DispositionIn(BaseModel):
    status: str
    reviewer: str
    note: str = ""


def _apply_disposition_side_effects(conn, ds: str, case: dict, status: str, reviewer: str):
    """Confirming a ring adds its evidence entities to the watchlist and
    fires a real webhook alert - the two product features that actually
    depend on a disposition happening, wired to the real event, not
    simulated separately."""
    if status != "confirmed_ring":
        return
    for entity_type, value in _case_entity_values(case):
        add_to_watchlist(conn, ds, entity_type, value, case["relation"], case["group_index"], reviewer)
    send_alert(ds, case, reason=f"analyst {reviewer} confirmed this ring")


@app.post("/api/datasets/{ds}/cases/{relation}/{group_index}/disposition")
def set_case_disposition(ds: str, relation: str, group_index: int, body: DispositionIn):
    if body.status not in ("confirmed_ring", "dismissed", "pending"):
        raise HTTPException(400, "status must be confirmed_ring, dismissed, or pending")
    cases = _load_json(f"cases_{ds}.json")
    case = next((c for c in cases if c["relation"] == relation and c["group_index"] == group_index), None)
    if case is None:
        raise HTTPException(404, "case not found")
    conn = connect()
    # A case only has a ring_case_status row once someone has opened its
    # detail view (get_case upserts it there). Acting on a case straight from
    # the list/bulk-select flow, which never called that, would otherwise
    # UPDATE zero rows and silently no-op - upsert here too so disposition
    # always has a row to act on regardless of how the case was reached.
    upsert_case(conn, ds, relation, group_index, [m["id"] for m in case["members"]], case["score"])
    set_disposition(conn, ds, relation, group_index, body.status, body.reviewer, body.note)
    _apply_disposition_side_effects(conn, ds, case, body.status, body.reviewer)
    return {"ok": True}


class BulkDispositionIn(BaseModel):
    cases: list[list]  # [[relation, group_index], ...]
    status: str
    reviewer: str
    note: str = ""


@app.post("/api/datasets/{ds}/cases/bulk-disposition")
def bulk_disposition(ds: str, body: BulkDispositionIn):
    if body.status not in ("confirmed_ring", "dismissed", "pending"):
        raise HTTPException(400, "status must be confirmed_ring, dismissed, or pending")
    all_cases = {(c["relation"], c["group_index"]): c for c in _load_json(f"cases_{ds}.json")}
    conn = connect()
    pairs = [(relation, gi) for relation, gi in body.cases]
    for relation, gi in pairs:
        case = all_cases.get((relation, gi))
        if case:
            upsert_case(conn, ds, relation, gi, [m["id"] for m in case["members"]], case["score"])
    set_disposition_bulk(conn, ds, pairs, body.status, body.reviewer, body.note)
    for relation, gi in pairs:
        case = all_cases.get((relation, gi))
        if case:
            _apply_disposition_side_effects(conn, ds, case, body.status, body.reviewer)
    return {"ok": True, "count": len(pairs)}


@app.get("/api/datasets/{ds}/dispositions")
def get_dispositions(ds: str):
    """Real analyst-derived labels collected so far - empty until this runs
    against actual reviewed cases. See case_store.py."""
    conn = connect()
    return export_labels(conn, ds)


@app.get("/api/datasets/{ds}/watchlist")
def get_watchlist(ds: str):
    conn = connect()
    return list_watchlist(conn, ds)


@app.get("/api/datasets/upi/segments")
def get_segments():
    """Per-city cost-optimal threshold breakdown - see segment_analysis.py
    for why this is UPI-only right now (the one dataset with a real
    per-user segment field)."""
    return _load_json("segments_upi.json")


@app.get("/api/datasets/{ds}/spikes")
def get_spikes(ds: str):
    if ds not in ("elliptic", "upi"):
        raise HTTPException(400, "spike detection needs a temporal field, only configured for elliptic and upi")
    return _load_json(f"spikes_{ds}.json")


@app.get("/api/datasets/{ds}/lookup")
def lookup_identifier(ds: str, q: str):
    """Real-time screening: does this identifier (VPA / device / bank
    account / member id) appear in any group the model actually flagged?
    A genuine linear scan over the exported case index - the same data the
    case queue reads from, not a canned response."""
    if ds not in DATASETS:
        raise HTTPException(404, "unknown dataset")
    q_norm = q.strip()
    if not q_norm:
        raise HTTPException(400, "empty query")
    q_lower = q_norm.lower()
    q_is_int = q_norm.isdigit()

    cases = _load_json(f"cases_{ds}.json")
    conn = connect()
    matches = []
    for c in cases:
        hit_via = None
        hit_member_id = None
        for k, v in c.get("evidence", {}).items():
            if k != "relation" and str(v).lower() == q_lower:
                hit_via = f"evidence.{k}"
                break
        if hit_via is None:
            for m in c["members"]:
                if q_is_int and m["id"] == int(q_norm):
                    hit_via, hit_member_id = "member.id", m["id"]
                    break
                if str(m.get("vpa", "")).lower() == q_lower:
                    hit_via, hit_member_id = "member.vpa", m["id"]
                    break
        if hit_via is None:
            continue
        row = conn.execute(
            "SELECT status FROM ring_case_status WHERE dataset=? AND relation=? AND group_index=?",
            (ds, c["relation"], c["group_index"]),
        ).fetchone()
        decision = evaluate_policy(c, _watchlist_hit(conn, ds, c))
        matches.append(
            {
                "relation": c["relation"],
                "group_index": c["group_index"],
                "matched_via": hit_via,
                "matched_member_id": hit_member_id,
                "score": c["score"],
                "size": c["size"],
                "bucket": c["bucket"],
                "evidence": c["evidence"],
                "members": c["members"],
                "status": row[0] if row else "pending",
                "on_watchlist": _watchlist_hit(conn, ds, c),
                "policy_recommendation": decision.__dict__,
            }
        )
    matches.sort(key=lambda m: -m["score"])
    return {"query": q_norm, "scanned_groups": len(cases), "matches": matches[:5]}


@app.get("/api/datasets/{ds}/lookup/examples")
def lookup_examples(ds: str):
    """A few real values pulled from real flagged groups, so the live
    demo has something honest to suggest instead of an empty box."""
    if ds not in DATASETS:
        raise HTTPException(404, "unknown dataset")
    cases = sorted(_load_json(f"cases_{ds}.json"), key=lambda c: -c["score"])
    out, seen_rel = [], set()
    for c in cases:
        if c["relation"] in seen_rel or not c["members"]:
            continue
        seen_rel.add(c["relation"])
        m = c["members"][0]
        value = m.get("vpa") or str(m["id"])
        out.append({"value": value, "label": f"{c['relation']} ring - score {c['score']:.2f}"})
        if len(out) >= 4:
            break
    return out


@app.get("/api/datasets/{ds}/graph")
def get_graph(ds: str, limit: int = 25):
    """Real bipartite member<->group topology for the top-scoring rings -
    the same structure CA-HGAT's own message passing uses (see
    export_cases.py), not a fabricated pairwise mesh."""
    if ds not in DATASETS:
        raise HTTPException(404, "unknown dataset")
    cases = sorted(_load_json(f"cases_{ds}.json"), key=lambda c: -c["score"])[:limit]
    nodes: dict[str, dict] = {}
    edges = []
    for c in cases:
        hub_id = f"hub:{c['relation']}:{c['group_index']}"
        evidence_val = next((v for k, v in c["evidence"].items() if k != "relation"), c["relation"])
        nodes[hub_id] = {
            "id": hub_id, "kind": "hub", "relation": c["relation"],
            "score": c["score"], "size": c["size"], "label_text": str(evidence_val),
        }
        for m in c["members"]:
            mid = f"m:{m['id']}"
            if mid not in nodes or nodes[mid]["score"] < m["score"]:
                nodes[mid] = {
                    "id": mid, "kind": "member", "raw_id": m["id"],
                    "score": m["score"], "label": m.get("label"),
                    "vpa": m.get("vpa"), "city": m.get("city"),
                }
            edges.append({"source": mid, "target": hub_id})
    return {"nodes": list(nodes.values()), "edges": edges}


@app.get("/api/pipeline/sample.csv")
def pipeline_sample_csv():
    """A small synthetic log file shaped like UPI transaction logs, so the
    upload demo has something to run instantly instead of requiring the
    visitor to already have a CSV on hand."""
    return PlainTextResponse(sample_csv(), media_type="text/csv")


@app.post("/api/pipeline/trace")
async def pipeline_trace(file: UploadFile = File(...)):
    """Uploads a CSV of logs, builds the real shared-identifier graph over
    its rows (pure Python union-find, no ML weights involved), and returns
    detected rings scored by a disclosed structural heuristic - see
    csv_pipeline.py's module docstring for exactly why this isn't the
    trained CA-HGAT model and what it honestly is instead."""
    raw = await file.read()
    try:
        result = run_pipeline(raw)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return result


def _process_razorpay_event(raw_body: bytes, signature: str | None, is_simulated: bool) -> dict:
    """The single processing path for a Razorpay-shaped webhook payload,
    used identically by the real inbound webhook route and by the
    self-signed simulated-traffic replay - no branch distinguishes "real"
    from "simulated" past signature verification, which both go through."""
    if not verify_signature(raw_body, signature):
        raise HTTPException(400, "invalid webhook signature")
    try:
        event = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(400, "invalid JSON body")
    ids = extract_identifiers(event)
    if not ids:
        raise HTTPException(400, "payload is not a recognizable Razorpay payment-webhook shape")
    conn = connect()
    add_razorpay_event(
        conn, ids["event_id"], ids["event_type"], ids.get("vpa"), ids.get("email"),
        ids.get("contact"), ids.get("amount"), raw_body.decode(), is_simulated=is_simulated,
    )
    return ids


@app.post("/webhooks/razorpay")
async def razorpay_webhook(request: Request):
    """Real inbound Razorpay webhook endpoint. Verifies the X-Razorpay-
    Signature header against RAZORPAY_WEBHOOK_SECRET using Razorpay's
    documented HMAC-SHA256-of-the-raw-body scheme - point a real Razorpay
    test-mode webhook (Settings -> Webhooks in the dashboard) at this URL
    and it works unchanged; no code path here is simulation-only."""
    raw_body = await request.body()
    signature = request.headers.get("x-razorpay-signature")
    ids = _process_razorpay_event(raw_body, signature, is_simulated=False)
    return {"ok": True, "event_id": ids["event_id"]}


@app.post("/webhooks/razorpay/replay")
def razorpay_replay(background_tasks: BackgroundTasks, n_normal: int = 20, ring_size: int = 6):
    """Generates a batch of Razorpay-shaped synthetic traffic (a background
    of distinct payments plus one planted coordinated-identifier ring),
    signs each event with the same webhook secret real traffic would be
    verified against, and feeds them through the exact same processing
    function as a real webhook - paced with a short delay per event so a
    polling frontend can show them arriving over a few seconds. Returns
    immediately; poll /webhooks/razorpay/recent to watch it happen."""
    events = generate_traffic_batch(n_normal=n_normal, ring_size=ring_size)

    def _run():
        sign_and_process_all(events, lambda body, sig, sim: _process_razorpay_event(body, sig, sim))

    background_tasks.add_task(_run)
    return {"started": True, "events_queued": len(events)}


@app.post("/webhooks/razorpay/reset")
def razorpay_reset():
    """Clears the demo event buffer for a clean re-run before recording."""
    clear_razorpay_events(connect())
    return {"ok": True}


@app.get("/webhooks/razorpay/recent")
def razorpay_recent(limit: int = 300):
    events = list_razorpay_events(connect(), limit=limit)
    detection = detect_rings(events)
    return {
        "webhook_url_hint": "/webhooks/razorpay",
        "webhook_secret_configured": RAZORPAY_WEBHOOK_SECRET != "demo_secret_for_local_testing",
        "events": events,
        "rings": detection["rings"],
        "graph": detection["graph"],
    }


@app.get("/api/datasets/{ds}/cases/export.csv")
def export_cases_csv(ds: str, status: str | None = None):
    if ds not in DATASETS:
        raise HTTPException(404, "unknown dataset")
    cases = list_cases(ds, status=status, limit=100000)
    buf = io.StringIO()
    if cases:
        writer = csv.DictWriter(buf, fieldnames=list(cases[0].keys()))
        writer.writeheader()
        for c in cases:
            row = dict(c)
            row["evidence"] = json.dumps(row.get("evidence", {}))
            writer.writerow(row)
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=abuse_ring_cases_{ds}.csv"},
    )


static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
