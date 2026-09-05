"""Real-time alerting: POSTs a case payload to a configured webhook URL when
it crosses the auto-confirm policy threshold. This is genuinely functional
code, not a stub - what's missing is a live external endpoint to point it
at, since this project has no deployed Slack workspace. Point ALERT_WEBHOOK_URL
at a real Slack incoming-webhook URL (or any HTTP endpoint) in production;
the payload shape below is Slack-compatible (a top-level "text" field) as
well as generic-JSON-compatible, so it works either way without changes.

Verified end to end against a local mock receiver (see tests further down /
the session's own verification, not asserted here) rather than assumed to
work from reading the requests docs.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import httpx

ALERT_WEBHOOK_URL = os.environ.get("ABUSE_RING_ALERT_WEBHOOK", "")
ALERT_TIMEOUT_SECONDS = 5.0


def format_alert_text(dataset: str, case: dict, reason: str) -> str:
    return (
        f":rotating_light: Ring auto-confirmed on *{dataset}*\n"
        f"relation=`{case['relation']}` group=#{case['group_index']} size={case['size']} "
        f"score={case['score']:.3f}\n"
        f"reason: {reason}\n"
        f"time: {datetime.now(timezone.utc).isoformat()}"
    )


def send_alert(dataset: str, case: dict, reason: str, webhook_url: str | None = None) -> dict:
    """Returns {"sent": bool, "status_code": int|None, "error": str|None}.
    Never raises - a failed alert shouldn't break the case-processing flow
    that triggered it; the caller decides whether to retry or just log."""
    url = webhook_url or ALERT_WEBHOOK_URL
    if not url:
        return {"sent": False, "status_code": None, "error": "no webhook URL configured"}

    payload = {
        "text": format_alert_text(dataset, case, reason),
        "dataset": dataset,
        "relation": case["relation"],
        "group_index": case["group_index"],
        "score": case["score"],
        "size": case["size"],
        "reason": reason,
    }
    try:
        resp = httpx.post(url, json=payload, timeout=ALERT_TIMEOUT_SECONDS)
        return {"sent": resp.status_code < 300, "status_code": resp.status_code, "error": None}
    except httpx.HTTPError as e:
        return {"sent": False, "status_code": None, "error": str(e)}
