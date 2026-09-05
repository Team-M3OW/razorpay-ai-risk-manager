# Ringfence — AI Risk Manager

Find the ring, fence it off. A camouflage-resistant graph model for
detecting coordinated fraud rings (not just individual bad actors), wired
into a real analyst product and a real payment-webhook integration. Built
for Razorpay's AI Risk Manager hackathon track. Defense-only.

## What's here

**Detection core** (`backend/abuse_ring/`) — CA-HGAT (camouflage-aware
hypergraph attention network), trained and honestly evaluated on four
datasets:
- **Yelp-Fraud / Amazon-Fraud** (Dou et al., CIKM 2020's published CARE-GNN
  benchmark) — beats the published baseline; on Amazon, a plain
  non-ring model collapses to zero recall without the group mechanism,
  proving the ring signal is load-bearing, not decorative.
- **Elliptic** (real Bitcoin transactions, real illicit-wallet labels,
  Weber et al. 2019) — loses to a plain Random Forest on F1, reported as a
  loss rather than hidden.
- **UPI** (synthetic, deliberately camouflaged mule rings — partial
  identifier sharing, injected bystanders, no free lunch for the model).

Also here: ring extraction (Leiden community detection), cost-curve
threshold selection, relation importance, and a documented, discarded
hypothesis (gating relations by Louvain modularity) alongside the
causally-correct replacement (post-hoc relation importance) — see
`ring_extraction.py` / `train_eval.py`.

**Analyst product** (`backend/app/`) — a FastAPI + vanilla-JS console, no
build step:
- **Live Risk Check** — screen a real VPA/device/bank-account against the
  actual exported ring index; real match or real "clean," not a canned
  response.
- **Upload & Trace** — drop a CSV of logs, watch a real union-find
  algorithm split it into connected rings live, scored by a disclosed
  structural heuristic (explicitly not the trained model — see
  `csv_pipeline.py`'s docstring for why).
- **Live Transaction Feed** (inside Upload & Trace) — a real webhook
  receiver speaking Razorpay's actual `payment.captured`/`payment.failed`
  payload shape with real HMAC-SHA256 signature verification. A real
  Razorpay test-mode webhook can point here unchanged; a "Replay simulated
  traffic" button pushes self-signed synthetic events through the
  identical code path for a zero-setup demo. See `razorpay_simulator.py`.
- **Network Explorer** — a real interactive D3 force-directed graph of the
  model's own bipartite member↔group structure, not a decorative
  animation.
- **Case Queue / Watchlist / Policy engine** — filterable case browser,
  bulk actions, CSV export; confirming a ring auto-adds its shared
  identifiers to a watchlist that flags recurrence regardless of model
  score; an auto-confirm/auto-clear/needs-review policy gate; a real
  webhook alert fires on confirmation (point `ABUSE_RING_ALERT_WEBHOOK` at
  anything and it works unchanged).
- **Model Performance** — the operating threshold is auto-selected by
  sweeping every threshold against real held-out predictions and picking
  the cost-minimizing one, not a slider a human drags.

`backend/data/processed/` holds the precomputed model outputs (metrics,
ranked cases, segment/spike analysis, the sqlite case store) the API
serves — already checked in, so the lightweight tier needs no GPU/training
step to run.

## Running it

**Full ML pipeline** (needs `torch`, `dgl`, `networkx`, `leidenalg`, etc. —
see `backend/requirements-ml.txt`):

```bash
cd backend
pip install -r requirements-ml.txt
python -m abuse_ring.train_eval --dataset all
python -m abuse_ring.export_cases --dataset all
python -m abuse_ring.export_analytics
```

**Just the API + UI** (lightweight — only `fastapi`, `uvicorn`, `gunicorn`,
`pydantic`, `httpx`, `python-multipart`; see `backend/requirements.txt`).
This is also what a platform deploy (e.g. Azure App Service) should
install — `app/main.py` never imports torch/dgl, which is what makes
deploying onto a small disk quota (like the Free F1 tier's 1GB) work:

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Wiring up a real Razorpay webhook (optional)

The Live Transaction Feed works with zero configuration via its
"Replay simulated traffic" button. To also accept real Razorpay test-mode
events:

1. Set `RAZORPAY_WEBHOOK_SECRET` to any string, on whatever host is
   running the app.
2. Razorpay Dashboard (Test Mode) → Account & Settings → Webhooks →
   + Add New Webhook → URL = `<your host>/webhooks/razorpay`, same secret,
   events `payment.captured` + `payment.failed`.
3. Trigger a real test-mode payment (a Payment Link paid with UPI VPA
   `success@razorpay`, or test card `4111 1111 1111 1111`) — it lands in
   the feed tagged `[REAL]`.

## Honest results, briefly

- Real, checkable results on published benchmarks: beats CARE-GNN on both
  Yelp-Fraud and Amazon-Fraud, with the ring mechanism proven essential on
  Amazon.
- Real financial data (Elliptic): loses to Random Forest on F1, reported
  honestly rather than hidden.
- Synthetic UPI mule-ring demo: camouflaged rings, ring mechanism lifts F1
  from 0.727 to 0.899 after adding IP/phone relations.
- The CSV-upload and Razorpay-feed demos use a disclosed structural
  heuristic (cluster size + number of distinct identifier types shared),
  not the trained CA-HGAT model — running that at request time on
  arbitrary uploads/webhooks would need the full torch/dgl stack loaded
  into the serving process, which the lightweight deploy deliberately
  doesn't carry.
