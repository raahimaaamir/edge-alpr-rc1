"""
week9_rescue_policy.py

Week 9, step 1: the hardened rescue policy, fixing the specific bug found
on the Week 8 test set (track 0015: a single rescue observation blended
into an already-disagreeing pool tipped the vote to a confident WRONG
answer).

Two changes from Week 8's rescue behavior:

  1. TRIGGER: rescue is attempted only when the normal-path decision is a
     detection-related failure (decision.is_detection_related_failure()) —
     zero or too-few usable observations. It is NEVER triggered for
     DISAGREEMENT (enough valid observations that simply don't agree).
     Tiling the same already-detected frames cannot resolve a genuine OCR
     conflict, and attempting to "help" there is exactly what caused the
     Week 8 bug.

  2. ACCEPTANCE: rescue-provided observations are decided ENTIRELY ON
     THEIR OWN — never blended with the original (inconclusive) pool
     before voting. Rescue succeeds only if its own observations pass
     the SAME final decision rule as the normal path (as of the RC1
     reliability layer: decision_rule_rc1.decide_rc1 — temporal
     agreement + confidence + applicable plate-profile validation, not
     just agreement). If rescue does not reach that bar, the ORIGINAL
     decision is kept unchanged — rescue can only ever turn a failure
     into an (independently-verified, fully-qualified) success; it can
     no longer silently corrupt an already-uncertain result, AND it can
     no longer authorize a plate through a weaker path than the normal
     one.

  UPDATED per supervisor instruction (RC1 finalization): rescue's
  TRIGGER logic remains separate and unchanged (only detection-related
  failures are rescuable — see is_detection_related_failure() below).
  But rescue no longer has its OWN weaker final-acceptance rule — once
  it produces candidate observations, they are decided by the exact
  same decide_rc1() as the normal path, including the confidence and
  plate-profile checks. Rescue generates additional evidence; it does
  not get a separate way to authorize a plate. (An earlier version of
  this module used decide_v2 — agreement only — for this step; that is
  what changed here.)
"""

from dataclasses import dataclass, field
from typing import Optional

from week9_decision_rule import TrackDecisionV2, AcceptanceConfig
from decision_rule_rc1 import decide_rc1, MIN_WINNING_CONFIDENCE
from plate_profile import PlateProfile
from tiled_detection_rescue import tiled_rescue_detect_and_recognize, TiledRescueConfig
from track_types import FrameObservation


def _default_rescue_acceptance() -> AcceptanceConfig:
    # top_k large enough to include every rescue observation found (rescue
    # only ever looks at a handful of candidate frames, so this never
    # meaningfully caps anything) — but still require >=2 independent
    # rescue observations to agree, per the plan.
    return AcceptanceConfig(top_k=50, min_usable_observations=2, min_agreement_count=2)


@dataclass
class RescuePolicyConfig:
    max_candidate_frames: int = 3
    tiled_config: TiledRescueConfig = field(default_factory=TiledRescueConfig)
    rescue_acceptance: AcceptanceConfig = field(default_factory=_default_rescue_acceptance)


@dataclass
class RescueOutcome:
    triggered: bool
    decision: TrackDecisionV2  # the FINAL decision to use — either the rescued one, or the original unchanged
    rescue_latency_ms: float
    num_new_observations: int

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["decision"] = self.decision.to_dict()
        return d


def maybe_rescue(
    pipeline,
    all_frames: list,
    normal_decision: TrackDecisionV2,
    pick_candidate_frames_fn,
    config: Optional[RescuePolicyConfig] = None,
    profile: Optional[PlateProfile] = None,
    min_winning_confidence: float = MIN_WINNING_CONFIDENCE,
) -> RescueOutcome:
    """
    pipeline: an AlprPipeline instance (reused unchanged).
    all_frames: the FULL, original (non-strided) frame list for this
        track/video, so rescue can look at full-resolution frames the
        normal (strided) pass may have skipped.
    normal_decision: the TrackDecisionV2 from the normal path (decide_rc1
        on the tracker's pooled observations).
    pick_candidate_frames_fn: e.g. run_week8_rescue_experiment.pick_candidate_frames
        — reused unchanged; picks evenly-spaced full frames to examine.
    profile, min_winning_confidence: passed straight through to the SAME
        decide_rc1() call used for rescue-generated observations — these
        should be the SAME values the caller uses for its own normal-path
        decision, so rescue is held to an identical standard, not a
        separately-configured one. Defaults match decide_rc1()'s own
        defaults (no profile, standard confidence threshold) for callers
        that don't have a specific jurisdiction/profile in mind.
    """
    config = config or RescuePolicyConfig()

    if not normal_decision.is_detection_related_failure():
        # the hardened trigger condition — disagreement is never rescuable
        return RescueOutcome(triggered=False, decision=normal_decision,
                              rescue_latency_ms=0.0, num_new_observations=0)

    candidates = pick_candidate_frames_fn(all_frames, config.max_candidate_frames)
    rescue_latency_ms = 0.0
    new_observations = []
    for frame_idx, _ts, image in candidates:
        results, tiling_latency_ms = tiled_rescue_detect_and_recognize(pipeline, image, config.tiled_config)
        rescue_latency_ms += tiling_latency_ms
        for r in results:
            if r.status == "ok":
                new_observations.append(FrameObservation.from_plate_result(frame_idx, None, r))

    # decided on THEIR OWN — never blended with the original ambiguous pool —
    # but through the EXACT SAME final decision rule as the normal path
    # (agreement + confidence + applicable plate-profile validation).
    # Rescue generates additional evidence; it does not get a separate,
    # weaker way to authorize a plate. (Confirmed per the supervisor's
    # explicit RC1-finalization instruction — see module docstring.)
    rescue_only_decision = decide_rc1(new_observations, config.rescue_acceptance,
                                       profile=profile, min_winning_confidence=min_winning_confidence)

    if rescue_only_decision.status == "ok":
        final_decision = rescue_only_decision
    else:
        # rescue attempted but didn't clear its own bar -- keep the
        # ORIGINAL decision unchanged, never downgrade further
        final_decision = normal_decision

    return RescueOutcome(triggered=True, decision=final_decision,
                          rescue_latency_ms=rescue_latency_ms, num_new_observations=len(new_observations))
