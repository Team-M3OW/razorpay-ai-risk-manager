# RiskOps

Graph-based detection of coordinated fraud rings, built for a Razorpay
hackathon submission.

## Overview

Most fraud detectors score one account at a time. RiskOps instead looks
for groups of accounts, reviews, or transactions that share an identifier
(device, IP, bank account, phone number, or a review-level relation such
as same product and rating within a short time window) and scores the
group. The core model is CA-HGAT, a camouflage-aware hypergraph attention
network trained and evaluated on four datasets:

- Yelp-Fraud and Amazon-Fraud (the CARE-GNN benchmark from Dou et al.,
  CIKM 2020). Results beat the published baseline on both. On Amazon, a
  version of the model without the group mechanism drops to zero recall,
  which is the evidence that group-level signal is doing real work.
- Elliptic (real Bitcoin transaction graph, real illicit-wallet labels
  from Weber et al., 2019). This model loses to a plain Random Forest on
  F1; that result is reported as-is.
- A synthetic UPI dataset with deliberately partial identifier sharing
  and injected bystander accounts, built to avoid trivially easy
  separability.

## What is in the repository

`backend/abuse_ring/` contains the model, training and evaluation code,
ring extraction (Leiden community detection), cost-curve threshold
selection, and relation-importance analysis.

`backend/app/` contains a FastAPI backend and a plain HTML/CSS/JS frontend
(no build step) that serves:

- A case queue with filtering, bulk actions, and CSV export.
- A watchlist: confirming a ring adds its shared identifiers, and any
  future case touching one of them is flagged regardless of model score.
- A policy engine that auto-confirms high-confidence large groups,
  auto-clears near-zero scores, and routes everything else to review.
- Webhook alerting on confirmation, configurable via
  `ABUSE_RING_ALERT_WEBHOOK`.
- A threshold selection panel that sweeps every threshold against held-out
  predictions and reports the cost-minimizing point, rather than a
  manually chosen value.
- An identifier lookup tool that queries the exported ring index directly.
- A CSV upload tool that builds a shared-identifier graph over arbitrary
  log files using union-find, independent of the trained model.
- A Razorpay webhook receiver at `/webhooks/razorpay` that verifies the
  HMAC-SHA256 signature Razorpay uses for its payment webhooks and applies
  the same identifier-matching logic to incoming payment events. A
  companion endpoint generates signed synthetic traffic through the same
  code path for testing without a live Razorpay account.

`backend/data/processed/` contains precomputed model outputs and the
sqlite case store, checked into the repository so the API runs without a
training step.

## Running locally

Full pipeline, including training (requires torch, dgl, and the packages
in `backend/requirements-ml.txt`):

```bash
cd backend
pip install -r requirements-ml.txt
python -m abuse_ring.train_eval --dataset all
python -m abuse_ring.export_cases --dataset all
python -m abuse_ring.export_analytics
```

API and frontend only (requires only `backend/requirements.txt`; the
model does not need to be retrained since its outputs are already in
`data/processed/`):

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Configuring a Razorpay webhook

1. Set the `RAZORPAY_WEBHOOK_SECRET` environment variable on the host
   running the app.
2. In the Razorpay dashboard (test mode), go to Account & Settings,
   Webhooks, Add New Webhook. Set the URL to `<host>/webhooks/razorpay`,
   use the same secret, and select the `payment.captured` and
   `payment.failed` events.
3. A test-mode payment (for example, a Payment Link paid with the test
   UPI ID `success@razorpay`) will trigger a signed webhook that appears
   in the app's live transaction feed.

## Results summary

| Dataset | Metric | Result |
|---|---|---|
| Yelp-Fraud | AUC | 0.907 (paper: 0.757, DGL reproduction: 0.687) |
| Amazon-Fraud | Recall, with vs. without the group mechanism | 0.924 vs. 0.0 |
| Elliptic | F1, this model vs. Random Forest | 0.424 vs. 0.788 |
| UPI (synthetic) | F1, with vs. without the group mechanism | 0.899 vs. 0.530 |

The CSV upload tool and the Razorpay webhook feed use a disclosed
structural heuristic (cluster size and number of distinct identifier
types shared), not the trained model. Loading the full model stack for
arbitrary request-time input would require torch and dgl in the serving
process, which the deployed API does not include.
