"""
decision_rule_rc1.py

FROZEN — Release Candidate 1's decision rule, replacing decide_v2 as the
active policy. Approved by the supervisor after the track 0080 diagnostic
package and the 4-rule comparison (run_acceptance_rule_comparison.py).

Decision path (matches the supervisor's own stated architecture):
    adequate observations -> temporal agreement -> confidence check
    -> applicable plate-profile validation -> ACCEPT / NO_RELIABLE_RESULT

IMPORTANT — why this rule was chosen, stated plainly for the record:
B (agreement + confidence), E (agreement + profile), and F (agreement +
confidence + profile) produced IDENTICAL results on the 22-track UFPR
validation set: precision 1.0000, coverage 0.9545, zero incorrect
accepts, one abstention (track 0080 in all three cases). F — this rule —
was chosen NOT because this validation set demonstrates an additive
accuracy benefit from combining confidence and profile checks, but
because it performs both INDEPENDENT reliability checks the supervisor
asked for (a probabilistic one and a structural one) at negligible
extra computational cost. A future validation set that exercises the two
checks' disagreement region (a confident-but-structurally-invalid result,
or a structurally-valid-but-low-confidence one) could reveal a real
difference between B, E, and F that this dataset does not.

The confidence threshold (MIN_WINNING_CONFIDENCE, default 0.90) is a
VALIDATION-SELECTED CONSERVATIVE HEURISTIC, not a calibrated probability
— it was chosen as a single round default before running against real
data, then confirmed (not tuned) against the validation set. It is a
parameter, not a hard-coded constant, so it can be adjusted without
editing this module.

The plate-profile check is OPTIONAL and applies ONLY when the caller
supplies a profile for a known jurisdiction/plate class. When profile is
None (the default), plate_profile.validate() itself returns
(None, "no_profile") — this rule treats that as "check unavailable, skip
it," never as a rejection. A result is never abstained on merely because
its jurisdiction is unknown.

Image-quality measurements (plate size, sharpness, exposure, and the
continuous edge-margin metric — see track_types.FrameObservation) are
recorded on every observation as telemetry, but are NOT used as a hard
gate here. Per the supervisor's explicit instruction: the validation set
does not span enough genuinely degraded (small/blurred/clipped/low-light)
examples to derive a defensible operating threshold, and inventing one
anyway would be exactly the kind of brittle heuristic this project is
trying to avoid. A data-derived visual-quality gate remains future work,
pending a broader validation set — see model_card_recognizer_v1_1.md's
limitations section.
"""

from week7_selection import compute_composite_scores, select_top_k, usable_observations
from plate_normalize import normalize_plate_text
from week8_decision_rule import NO_RELIABLE_RESULT, COMPLETE_DETECTION_FAILURE
from week9_decision_rule import AcceptanceConfig, TrackDecisionV2, ZERO_OBSERVATIONS, INSUFFICIENT_OBSERVATIONS, DISAGREEMENT
from plate_profile import validate as validate_plate_profile, PlateProfile

MIN_WINNING_CONFIDENCE = 0.90  # validation-selected conservative heuristic — see module docstring


def decide_rc1(observations: list, config: AcceptanceConfig = None,
                profile: PlateProfile = None,
                min_winning_confidence: float = MIN_WINNING_CONFIDENCE) -> TrackDecisionV2:
    """The frozen RC1 decision rule: agreement -> confidence -> applicable
    plate-profile validation.

    profile: PlateProfile for the known jurisdiction/plate class, or None
    if unknown. When None, the profile check is skipped entirely — never
    treated as a failure.
    """
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

    # --- temporal agreement ---
    if count < config.min_agreement_count:
        return TrackDecisionV2(
            status=NO_RELIABLE_RESULT, text=None, agreement_count=count,
            num_selected=len(selected), num_usable_total=len(usable),
            reason=f"top candidate agreement {count} < {config.min_agreement_count}",
            failure_kind=DISAGREEMENT,
        )

    # --- confidence check ---
    winner_obs = [o for o in selected if normalize_plate_text(o.ocr_text) == winner]
    mean_conf = sum(o.overall_conf or 0.0 for o in winner_obs) / len(winner_obs)
    if mean_conf < min_winning_confidence:
        return TrackDecisionV2(
            status=NO_RELIABLE_RESULT, text=None, agreement_count=count,
            num_selected=len(selected), num_usable_total=len(usable),
            reason=f"winning mean confidence {mean_conf:.3f} < {min_winning_confidence}",
            failure_kind=DISAGREEMENT,
        )

    # --- applicable plate-profile validation (skipped entirely if profile is None) ---
    is_valid, profile_reason = validate_plate_profile(winner, profile)
    if is_valid is False:
        return TrackDecisionV2(
            status=NO_RELIABLE_RESULT, text=None, agreement_count=count,
            num_selected=len(selected), num_usable_total=len(usable),
            reason=f"failed plate-profile validation: {profile_reason}",
            failure_kind=DISAGREEMENT,
        )
    # is_valid is True, or None (profile unavailable) -- either way, proceed to accept.

    return TrackDecisionV2(
        status="ok", text=winner, agreement_count=count,
        num_selected=len(selected), num_usable_total=len(usable),
        reason=None, failure_kind=None,
    )
