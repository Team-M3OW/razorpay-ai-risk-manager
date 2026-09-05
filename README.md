# Abuse Ring Sentinel — AI Risk Manager

A camouflage-resistant graph model for detecting coordinated fraud rings (not
just individual bad actors), built for the AI Risk Manager track. Defense-only.

## What's here

- `backend/abuse_ring/` — the ML pipeline: data loaders (Yelp/Amazon/Elliptic/
  UPI-synthetic), ring extraction (Leiden community detection + connected
  components), the CA-HGAT model (camouflage-aware hypergraph attention
  network), training/evaluation, cost-curve threshold selection, relation
  importance, camouflage-frontier sweeps, and segment/spike analytics.
- `backend/app/` — the FastAPI + vanilla-JS case-queue product: ring case
  browser, live cost-threshold slider, watchlist, auto-action policy engine,
  webhook alerting, bulk actions, CSV export.
- `backend/data/processed/` — precomputed model outputs (metrics, ranked
  cases, segment/spike analysis) the API serves.

## Running it

Full ML pipeline (needs `torch`, `dgl`, `networkx`, `leidenalg`, etc. — see
`backend/requirements-ml.txt`):

```bash
cd backend
pip install -r requirements-ml.txt
python -m abuse_ring.train_eval --dataset all
python -m abuse_ring.export_cases --dataset all
python -m abuse_ring.export_analytics
```

Just the API + UI (lightweight — only `fastapi`, `uvicorn`, `pydantic`,
`httpx`, see `backend/requirements.txt`; the precomputed JSON/sqlite in
`data/processed/` is already checked in). This is also what a platform
deploy — e.g. Azure App Service — should install; keeping it separate from
`requirements-ml.txt` is what makes deploying onto a small disk quota (like
the Free F1 tier's 1GB) actually work, since `app/main.py` never imports
torch/dgl at all:

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Honest results, briefly

- Real, checkable results on published benchmarks: beats CARE-GNN (Dou et
  al., CIKM 2020) on both Yelp-Fraud and Amazon-Fraud, with the ring
  mechanism proven essential on Amazon (a plain baseline collapses to 0
  recall without it).
- Real financial data (Elliptic Bitcoin AML dataset, Weber et al. 2019):
  loses to Random Forest on F1, reported honestly rather than hidden.
- Synthetic UPI mule-ring demo: camouflaged rings (partial identifier
  sharing, injected bystanders), ring mechanism lifts F1 from 0.727 to 0.899
  after adding IP/phone relations.
- A tried-and-discarded hypothesis (gating relations by Louvain modularity)
  and the causally-correct replacement (post-hoc relation importance) are
  both documented in `ring_extraction.py` and `train_eval.py`.
# razorpay-ai-risk-manager
# razorpay-ai-risk-manager
# razorpay-ai-risk-manager
