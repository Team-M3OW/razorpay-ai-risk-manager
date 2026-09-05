"""Policy engine: maps a case's score, size, and watchlist status to a
recommended action.

Evaluated in order:
  1. Watchlist match on any member's evidence entity.
  2. Score and size thresholds, split into three bands (auto-confirm,
     needs-review, auto-clear).
"""

from __future__ import annotations

from dataclasses import dataclass

AUTO_CONFIRM_SCORE = 0.90
AUTO_CONFIRM_MIN_SIZE = 4
AUTO_CLEAR_SCORE = 0.05


@dataclass
class PolicyDecision:
    action: str          # "auto_confirm" | "auto_clear" | "needs_review"
    reason: str
    triggered_by: str    # "watchlist" | "score_threshold" | "default"


def evaluate_policy(case: dict, watchlist_hit: bool) -> PolicyDecision:
    if watchlist_hit:
        return PolicyDecision(
            action="auto_confirm",
            reason="shares a watchlisted identifier with a previously confirmed ring",
            triggered_by="watchlist",
        )

    score = case["score"]
    size = case["size"]

    if score >= AUTO_CONFIRM_SCORE and size >= AUTO_CONFIRM_MIN_SIZE:
        return PolicyDecision(
            action="auto_confirm",
            reason=f"score {score:.3f} >= {AUTO_CONFIRM_SCORE} and size {size} >= {AUTO_CONFIRM_MIN_SIZE}",
            triggered_by="score_threshold",
        )
    if score <= AUTO_CLEAR_SCORE:
        return PolicyDecision(
            action="auto_clear",
            reason=f"score {score:.3f} <= {AUTO_CLEAR_SCORE}",
            triggered_by="score_threshold",
        )
    return PolicyDecision(
        action="needs_review",
        reason=f"score {score:.3f} in the ambiguous band, or size {size} too small to auto-confirm",
        triggered_by="default",
    )
