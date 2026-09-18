"""
week9_decision_rule.py

Week 9, step 1: adds an explicit failure_kind to the accept/NO_RELIABLE_RESULT
decision, so the rescue policy (week9_rescue_policy.py) can distinguish:

  - ZERO_OBSERVATIONS / INSUFFICIENT_OBSERVATIONS: detection-related
    failures (nothing, or too little, to work with) — rescue IS allowed
    to attempt these.
  - DISAGREEMENT: there were enough usable observations, but they didn't
    agree — a real OCR conflict among otherwise-good detections. Tiling
    more of the same already-detected frames cannot resolve a conflict
    in what the recognizer read; rescue must NOT be attempted here.

week8_decision_rule.py is left completely unchanged (kept as the frozen
Video Pipeline V1 rollback/reference, per instruction). This module
duplicates its short decide() logic with one addition rather than
modifying the original.
"""

from dataclasses import dataclass
from typing import Optional

from week7_selection import compute_composite_scores, select_top_k, usable_observations
from plate_normalize import normalize_plate_text
from week8_decision_rule import AcceptanceConfig, NO_RELIABLE_RESULT, COMPLETE_DETECTION_FAILURE

ZERO_OBSERVATIONS = "zero_observations"
INSUFFICIENT_OBSERVATIONS = "insufficient_observations"
DISAGREEMENT = "disagreement"

# failure kinds rescue is allowed to attempt — everything else, it must not
RESCUABLE_FAILURE_KINDS = (ZERO_OBSERVATIONS, INSUFFICIENT_OBSERVATIONS)


@dataclass
class TrackDecisionV2:
    status: str  # "ok" | "no_reliable_result" | "complete_detection_failure"
    text: Optional[str]
    agreement_count: int
    num_selected: int
    num_usable_total: int
    reason: Optional[str] = None
    failure_kind: Optional[str] = None  # None if status == "ok"

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def is_detection_related_failure(self) -> bool:
        """True only for failures the hardened rescue policy is allowed to
        attempt. False for DISAGREEMENT (and for status == 'ok')."""
        return self.failure_kind in RESCUABLE_FAILURE_KINDS


def decide_v2(observations: list, config: Optional[AcceptanceConfig] = None) -> TrackDecisionV2:
    config = config or AcceptanceConfig()
    usable = usable_observations(observations)

    if not usable:
        return TrackDecisionV2(
            status=COMPLETE_DETECTION_FAILURE, text=None, agreement_count=0,
            num_selected=0, num_usable_total=0,
            reason="zero usable observations", failure_kind=ZERO_OBSERVATIONS,
        )

    scores = compute_composite_scores(observations)
    selected = select_top_k(observations, scores, config.top_k)

    if len(selected) < config.min_usable_observations:
        return TrackDecisionV2(
            status=NO_RELIABLE_RESULT, text=None, agreement_count=0,
            num_selected=len(selected), num_usable_total=len(usable),
            reason=f"only {len(selected)} usable observation(s), need >= {config.min_usable_observations}",
            failure_kind=INSUFFICIENT_OBSERVATIONS,
        )

    counts = {}
    for o in selected:
        norm = normalize_plate_text(o.ocr_text)
        counts[norm] = counts.get(norm, 0) + 1
    winner, count = max(counts.items(), key=lambda kv: kv[1])

    if count < config.min_agreement_count:
        return TrackDecisionV2(
            status=NO_RELIABLE_RESULT, text=None, agreement_count=count,
            num_selected=len(selected), num_usable_total=len(usable),
            reason=f"top candidate agreement {count} < {config.min_agreement_count}",
            failure_kind=DISAGREEMENT,
        )

    return TrackDecisionV2(
        status="ok", text=winner, agreement_count=count,
        num_selected=len(selected), num_usable_total=len(usable),
        reason=None, failure_kind=None,
    )
