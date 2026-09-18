"""
week8_decision_rule.py

Week 8, step 1: an explicit, configurable accept/NO_RELIABLE_RESULT rule for
the Top-3 whole-string majority vote, which is now the provisional default
per the supervisor (Week 7 established it as the best accuracy/cost
trade-off; no more fusion-variant exploration this week unless a specific
failure demands it).

Rule (defaults, all configurable via AcceptanceConfig):
  - select the top_k (default 3) highest-scoring usable observations
    (usable = detection + OCR both succeeded), using the same composite
    score from week7_selection.py
  - require at least min_usable_observations (default 2) selected
    observations to even consider a result
  - among the selected observations' NORMALIZED plate text (see
    plate_normalize.py), require the winning candidate to have at least
    min_agreement_count (default 2) votes
  - otherwise: NO_RELIABLE_RESULT

A track with zero usable observations at all is tagged
"complete_detection_failure" — distinct from an ordinary
NO_RELIABLE_RESULT, consistent with the Week 7 correction (a detector
total-miss should never be silently folded into an "abstention" count).
"""

from dataclasses import dataclass
from typing import Optional

from week7_selection import compute_composite_scores, select_top_k, usable_observations
from plate_normalize import normalize_plate_text

NO_RELIABLE_RESULT = "no_reliable_result"
COMPLETE_DETECTION_FAILURE = "complete_detection_failure"


@dataclass
class AcceptanceConfig:
    top_k: int = 3
    min_usable_observations: int = 2
    min_agreement_count: int = 2


@dataclass
class TrackDecision:
    status: str  # "ok" | "no_reliable_result" | "complete_detection_failure"
    text: Optional[str]
    agreement_count: int
    num_selected: int
    num_usable_total: int
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def decide(observations: list, config: Optional[AcceptanceConfig] = None) -> TrackDecision:
    config = config or AcceptanceConfig()
    usable = usable_observations(observations)

    if not usable:
        return TrackDecision(status=COMPLETE_DETECTION_FAILURE, text=None, agreement_count=0,
                              num_selected=0, num_usable_total=0, reason="zero usable observations")

    scores = compute_composite_scores(observations)
    selected = select_top_k(observations, scores, config.top_k)

    if len(selected) < config.min_usable_observations:
        return TrackDecision(
            status=NO_RELIABLE_RESULT, text=None, agreement_count=0,
            num_selected=len(selected), num_usable_total=len(usable),
            reason=f"only {len(selected)} usable observation(s), need >= {config.min_usable_observations}",
        )

    counts = {}
    for o in selected:
        norm = normalize_plate_text(o.ocr_text)
        counts[norm] = counts.get(norm, 0) + 1
    winner, count = max(counts.items(), key=lambda kv: kv[1])

    if count < config.min_agreement_count:
        return TrackDecision(
            status=NO_RELIABLE_RESULT, text=None, agreement_count=count,
            num_selected=len(selected), num_usable_total=len(usable),
            reason=f"top candidate agreement {count} < {config.min_agreement_count}",
        )

    return TrackDecision(status="ok", text=winner, agreement_count=count,
                          num_selected=len(selected), num_usable_total=len(usable), reason=None)


def summarize_decisions(decisions_with_gt: list) -> dict:
    """decisions_with_gt: list of (TrackDecision, gt_text) pairs, one per
    encounter. Returns accepted-result precision, coverage, and
    incorrect-accepted count/rate — the three numbers Week 8 step 1 asks
    for."""
    n_total = len(decisions_with_gt)
    n_accepted = sum(1 for d, _ in decisions_with_gt if d.status == "ok")
    n_correct = sum(1 for d, gt in decisions_with_gt
                     if d.status == "ok" and normalize_plate_text(d.text) == normalize_plate_text(gt))
    n_incorrect_accepted = n_accepted - n_correct
    n_complete_failures = sum(1 for d, _ in decisions_with_gt if d.status == COMPLETE_DETECTION_FAILURE)
    n_no_reliable_result = sum(1 for d, _ in decisions_with_gt if d.status == NO_RELIABLE_RESULT)

    return {
        "n_total": n_total,
        "n_accepted": n_accepted,
        "n_correct": n_correct,
        "n_incorrect_accepted": n_incorrect_accepted,
        "n_no_reliable_result": n_no_reliable_result,
        "n_complete_detection_failures": n_complete_failures,
        "accepted_result_precision": (n_correct / n_accepted) if n_accepted else None,
        "coverage": (n_accepted / n_total) if n_total else None,
        "incorrect_accepted_rate_of_total": (n_incorrect_accepted / n_total) if n_total else None,
    }
