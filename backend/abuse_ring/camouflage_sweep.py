"""Camouflage-frontier sweep: how low can a ring's shared-identity fraction
go before CA-HGAT stops detecting it?

The original UPI generator drew each ring's device/bank sharing fraction
from a random range (0.35-0.75). That's one point on a curve, not the curve
- it doesn't say anything about where detection actually breaks down. This
fixes that fraction to a single exact value per run and sweeps it, which is
the adversarial-testing fix from the open limitations list: instead of
"here's one hard-but-detectable camouflage level," this produces "here's
the recall/precision curve as a fraudster shares less and less," which is
the actual answer to "how much camouflage would it take to evade this."
"""

from __future__ import annotations

import json
from pathlib import Path

from abuse_ring.train_eval import train
from abuse_ring.upi_synthetic import generate_upi_graph

SHARE_FRACTIONS = [0.75, 0.55, 0.35, 0.20, 0.10, 0.05]


def run_sweep(max_epoch: int = 100, device: str = "cpu", out_path: str | None = None):
    results = []
    for frac in SHARE_FRACTIONS:
        print(f"\n########## camouflage sweep: share_fraction={frac} ##########")
        loader = lambda f=frac: generate_upi_graph(seed=0, share_fraction_override=f)[0]
        _, _, _, _, _, result = train("upi", device=device, max_epoch=max_epoch, loader_override=loader)
        entry = {
            "share_fraction": frac,
            "node_test_metrics": result["node_test_metrics"],
            "node_test_auc_ci95": result.get("node_test_auc_ci95"),
        }
        results.append(entry)
        m = result["node_test_metrics"]
        print(f"  -> share_fraction={frac}: AUC={m['auc']:.3f} Recall={m['recall']:.3f} Precision={m['precision']:.3f} F1={m['f1']:.3f}")

    print("\n=== camouflage frontier summary ===")
    print(f"{'share_fraction':>14s} {'AUC':>8s} {'Recall':>8s} {'Precision':>10s} {'F1':>8s}")
    for r in results:
        m = r["node_test_metrics"]
        print(f"{r['share_fraction']:>14.2f} {m['auc']:>8.3f} {m['recall']:>8.3f} {m['precision']:>10.3f} {m['f1']:>8.3f}")

    if out_path:
        Path(out_path).write_text(json.dumps(results, indent=2))
        print(f"wrote {out_path}")
    return results


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "processed" / "camouflage_sweep.json"))
    args = parser.parse_args()
    run_sweep(max_epoch=args.max_epoch, device=args.device, out_path=args.out)
