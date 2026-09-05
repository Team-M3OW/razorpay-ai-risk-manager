"""Precomputes segment-threshold and ring-formation-spike analysis into JSON,
the same pattern as export_cases.py - so the API serves real, already-computed
results instantly instead of retraining a model on every request.
"""

from __future__ import annotations

import json
from pathlib import Path

from abuse_ring.segment_analysis import per_segment_threshold, ring_formation_spikes

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def export_all(device: str = "cpu", max_epoch: int = 150):
    segments = per_segment_threshold(device=device, max_epoch=max_epoch)
    (OUT_DIR / "segments_upi.json").write_text(json.dumps(segments, indent=2))
    print(f"wrote {len(segments)} segments to segments_upi.json")

    for ds in ("elliptic", "upi"):
        spikes = ring_formation_spikes(ds, device=device, max_epoch=max_epoch)
        (OUT_DIR / f"spikes_{ds}.json").write_text(json.dumps(spikes, indent=2))
        n_spikes = sum(1 for s in spikes if s["spike"])
        print(f"wrote {len(spikes)} buckets ({n_spikes} spikes) to spikes_{ds}.json")


if __name__ == "__main__":
    import argparse

    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epoch", type=int, default=150)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    export_all(device=args.device, max_epoch=args.max_epoch)
