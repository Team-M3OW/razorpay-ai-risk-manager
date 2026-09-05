# Ringfence — demo video script (~3 min)

Tone: confident, fast, no filler. Every claim on screen should be something
the app is actually doing live, not a slide.

---

### 0:00–0:15 — Hook

**Say:** "Fraud rings don't look suspicious one account at a time. Ten
accounts, each spending a normal amount, from a normal city, on a normal
day — the only thing wrong is they all share one bank account behind the
scenes. That's what Ringfence finds: not bad actors, bad *rings*."

**Show:** Ringfence Live Risk Check tab, idle, clean UI, ambient network
background animating.

---

### 0:15–0:45 — The detection core is real, and honestly reported

**Say:** "This runs on a camouflage-aware graph model, CA-HGAT, and I'm
not going to just show you a number — I'm showing you the receipts.
On Yelp and Amazon's published fraud-ring benchmark, it beats the paper's
own baseline. On Amazon specifically, if you remove the ring mechanism
entirely, the model collapses to zero recall — the ring signal isn't
decoration, it's load-bearing. On real Bitcoin transactions, it actually
*loses* to a plain Random Forest — I'm reporting that because hiding a
loss is worse than having one."

**Show:** Model Performance tab. Switch between Yelp / Amazon / Elliptic /
UPI in the sidebar. Point at the auto-selected cost-threshold curve —
mention: "this threshold isn't hand-picked, it's swept against real
held-out predictions and chosen to minimize dollar cost."

---

### 0:45–1:15 — Live Risk Check

**Say:** "Here's the product an analyst actually uses. I type in a UPI
handle —"

**Show:** Type one of the example chips (a real flagged VPA) into Live
Risk Check, hit Screen. Result card appears: ring match, score, evidence,
live force graph.

**Say:** "— real match, real evidence: this account shares a bank account
with five others, model score 0.99. Type something that isn't in any
ring —"

**Show:** Type a random string, hit Screen → clean result.

**Say:** "— clean. Not a canned demo response, an actual query against the
exported ring index."

---

### 1:15–1:45 — Network Explorer

**Say:** "Zoom out and this is what the model itself sees — the real
bipartite structure between accounts and the shared identifiers that
connect them."

**Show:** Network Explorer tab, drag a node, scroll to zoom, hover for a
tooltip, click a hub to open its case.

---

### 1:45–2:20 — Upload & Trace

**Say:** "If you've got your own logs, drop them in."

**Show:** Upload & Trace tab, click "Use sample log file instead," watch
the console feed print detections, force graph forms clusters live.

**Say:** "Real union-find over shared device, IP, phone, bank-account
columns — splitting a flat CSV into connected rings in real time. This one
uses a transparent scoring rule instead of the full graph model, and I say
so on screen — no GPU needed for this path, so it stays deployable
anywhere."

---

### 2:20–2:55 — The integration: Razorpay Live Transaction Feed

**Say:** "And here's the part that makes this more than a research demo.
This is a real webhook endpoint, speaking Razorpay's actual
`payment.captured` payload shape, with real HMAC-SHA256 signature
verification — not a mock."

**Show:** Scroll to Live Transaction Feed panel. Click "Replay simulated
traffic."

**Say:** "Watch — twenty-odd payments streaming in… and there — six of
them just lit up as a ring, all sharing one phone number across different
UPI handles. Same code path a real Razorpay test-mode webhook would hit —
[if real webhook wired up: "in fact, here's a real one arriving right
now" — trigger the payment link] — this isn't simulated versus real, it's
one pipeline that doesn't care which."

**Show:** Ring forms in the console + graph, "RING DETECTED" line
highlighted red.

---

### 2:55–3:05 — Close

**Say:** "Four honestly-benchmarked datasets, a real analyst product, and
a real integration point into an actual payment platform. That's
Ringfence."

**Show:** Cut back to the sidebar, sitting on Live Risk Check.

---

## Notes for recording

- Do the Razorpay section **last** in case the real-webhook bonus doesn't
  fire cleanly live — simulated traffic alone still lands the whole beat.
- Reset the feed (`Reset feed` button) before recording so the first ring
  detected on camera is unambiguous, not buried among earlier test runs.
- If narrating live rather than voicing over afterward, practice the
  Live Risk Check example values once beforehand — pick one that you know
  is a real hit (use one of the suggestion chips) rather than guessing on
  camera.
