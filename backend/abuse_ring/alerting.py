"""Webhook alerting for ring confirmations.

Posts a case payload to a configured webhook URL. The payload includes a
top-level "text" field, so it works with Slack incoming webhooks as well
as generic JSON endpoints.
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
    """Post an alert for a confirmed case to the configured webhook.

    Returns {"sent": bool, "status_code": int | None, "error": str | None}.
    Does not raise on failure.
    """
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
