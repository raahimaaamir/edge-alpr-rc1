"""
run_acceptance_rule_comparison.py

Compares 4 acceptance-rule variants on the 22 UFPR VALIDATION tracks only
(never the 13 reserved local plates, never the test set), per the
supervisor's instruction after track 0080's diagnostic package showed the
recognizer's own probabilities never support "J" as a credible
alternative at the disputed position -- this is a genuine recognizer
limitation, not something a decision-layer patch should paper over. The
fix under consideration is therefore general and minimal: structural
plate-format validity (plate_profile.py) as an independent check, not a
track-0080-specific correction.

All 4 rules share the IDENTICAL Top-K selection (compute_composite_scores
+ select_top_k, unchanged) -- only the final accept/abstain gate differs.
This means:
  - ZERO_OBSERVATIONS / INSUFFICIENT_OBSERVATIONS classification is
    IDENTICAL across all 4 rules (it's determined purely by how many
    usable/selected observations exist, never by the confidence/profile
    gates).
  - The hardened rescue policy (Week 9) is applied IDENTICALLY across all
    4 rules, completely unchanged, and only ever triggers on those two
    detection-related failure kinds -- never on disagreement, low
    confidence, or a profile-validation failure, exactly as the
    supervisor wants preserved.
  - Only tracks that reach the "count >= min_agreement_count" stage can
    differ between rules -- that's precisely where track 0080 lives.

Rules compared:
  A. CURRENT -- >= 2 of Top-3 agree (decide_v2, unchanged, the baseline).
  B. AGREEMENT + CONFIDENCE -- same as A, plus the winning votes' mean
     overall_conf must be >= MIN_WINNING_CONFIDENCE (0.90, a single
     round, pre-chosen threshold -- not tuned to flip any specific track).
  E. AGREEMENT + STRUCTURAL PROFILE -- same as A, plus the winning text
     must pass plate_profile.validate() against UFPR_PROFILE (accept/
     reject only, no reranking in this comparison -- a pure structural-
     validity gate). This is the rule that correctly rejects AP1382:
     6 characters is not a valid length for the 7-character UFPR format,
     independent of any confidence or character-level reasoning.
  F. AGREEMENT + CONFIDENCE + STRUCTURAL PROFILE -- B and E combined; all
     three independent checks (temporal agreement, confidence, format
     validity) must pass.

Reports, per rule, computed ONLY on validation data:
  - accepted-result precision (of tracks the rule accepted, what
    fraction were exactly correct)
  - coverage (fraction of all tracks accepted)
  - incorrect accepted tracks (the actual list, not just a count)
  - abstentions (tracks the rule declined, excluding detection-related
    complete failures, which are identical across all rules and reported
    separately as a sanity check)

Run from inside the alpr-train container, from the pipeline/ directory:
    python3 run_acceptance_rule_comparison.py --v1-1-onnx-path <path>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.alpr_pipeline import AlprPipeline
from pipeline.recognizer import PlateRecognizer

from tracker import IoUTracker
from stage3_config import load_config
from ufpr_video_loader import get_track_ids, load_track, UFPR_ROOT_DEFAULT
from video_pipeline import VideoAlprPipeline
from week7_selection import compute_composite_scores, select_top_k, usable_observations
from plate_normalize import normalize_plate_text
from week8_decision_rule import NO_RELIABLE_RESULT, COMPLETE_DETECTION_FAILURE
from week9_decision_rule import (
    decide_v2, AcceptanceConfig, TrackDecisionV2,
    ZERO_OBSERVATIONS, INSUFFICIENT_OBSERVATIONS, DISAGREEMENT,
)
from week9_rescue_policy import maybe_rescue, RescuePolicyConfig
from plate_profile import validate as validate_plate_profile, UFPR_PROFILE
from run_week8_rescue_experiment import pick_candidate_frames

MANIFEST_VAL_CSV = Path("/workspace/home/alpr-week5/data/interim/manifest_val_full.csv")
DEFAULT_CONFIG_PATH = "/workspace/home/alpr-week5/weights/cct_s_v2_global_plate_config.yaml"
STRIDE = 2
RESCUE_CONFIG = RescuePolicyConfig()

MIN_WINNING_CONFIDENCE = 0.90  # Rule B — single round default, chosen before looking at any specific track's outcome
MIN_CHAR_CONFIDENCE = 0.60     # Rule D — single round default, chosen before looking at any specific track's outcome
WEAK_EVIDENCE = "weak_evidence"  # new failure_kind for Rule D — NOT in RESCUABLE_FAILURE_KINDS, so rescue correctly never fires on it


def _select_and_count(observations: list, config: AcceptanceConfig):
    """The shared first stage every rule uses identically. Returns either
    an early TrackDecisionV2 (zero/insufficient observations — identical
    outcome regardless of which rule is active) or (selected, counts,
    winner, count, n_usable) to let the calling rule apply its own final gate."""
    usable = usable_observations(observations)
    if not usable:
        return TrackDecisionV2(status=COMPLETE_DETECTION_FAILURE, text=None, agreement_count=0,
                                num_selected=0, num_usable_total=0,
                                reason="zero usable observations", failure_kind=ZERO_OBSERVATIONS)

    scores = compute_composite_scores(observations)
    selected = select_top_k(observations, scores, config.top_k)

    if len(selected) < config.min_usable_observations:
        return TrackDecisionV2(status=NO_RELIABLE_RESULT, text=None, agreement_count=0,
                                num_selected=len(selected), num_usable_total=len(usable),
                                reason=f"only {len(selected)} usable observation(s), need >= {config.min_usable_observations}",
                                failure_kind=INSUFFICIENT_OBSERVATIONS)

    counts = {}
    for o in selected:
        norm = normalize_plate_text(o.ocr_text)
        counts[norm] = counts.get(norm, 0) + 1
    winner, count = max(counts.items(), key=lambda kv: kv[1])
    return selected, counts, winner, count, len(usable)


def decide_rule_A_current(observations: list, config: AcceptanceConfig) -> TrackDecisionV2:
    """The current, unchanged rule — exact reuse of decide_v2."""
    return decide_v2(observations, config)


def decide_rule_B_agreement_plus_confidence(observations: list, config: AcceptanceConfig) -> TrackDecisionV2:
    stage1 = _select_and_count(observations, config)
    if isinstance(stage1, TrackDecisionV2):
        return stage1
    selected, counts, winner, count, n_usable = stage1

    if count < config.min_agreement_count:
        return TrackDecisionV2(status=NO_RELIABLE_RESULT, text=None, agreement_count=count,
                                num_selected=len(selected), num_usable_total=n_usable,
                                reason=f"top candidate agreement {count} < {config.min_agreement_count}",
                                failure_kind=DISAGREEMENT)

    winner_obs = [o for o in selected if normalize_plate_text(o.ocr_text) == winner]
    mean_conf = sum(o.overall_conf or 0.0 for o in winner_obs) / len(winner_obs)
    if mean_conf < MIN_WINNING_CONFIDENCE:
        return TrackDecisionV2(status=NO_RELIABLE_RESULT, text=None, agreement_count=count,
                                num_selected=len(selected), num_usable_total=n_usable,
                                reason=f"winning mean confidence {mean_conf:.3f} < {MIN_WINNING_CONFIDENCE}",
                                failure_kind=DISAGREEMENT)

    return TrackDecisionV2(status="ok", text=winner, agreement_count=count,
                            num_selected=len(selected), num_usable_total=n_usable, failure_kind=None)


def decide_rule_C_unanimity(observations: list, config: AcceptanceConfig) -> TrackDecisionV2:
    stage1 = _select_and_count(observations, config)
    if isinstance(stage1, TrackDecisionV2):
        return stage1
    selected, counts, winner, count, n_usable = stage1

    if count < len(selected):  # anything short of full unanimity among whatever was selected
        return TrackDecisionV2(status=NO_RELIABLE_RESULT, text=None, agreement_count=count,
                                num_selected=len(selected), num_usable_total=n_usable,
                                reason=f"not unanimous: {count} of {len(selected)} selected observations agree",
                                failure_kind=DISAGREEMENT)

    return TrackDecisionV2(status="ok", text=winner, agreement_count=count,
                            num_selected=len(selected), num_usable_total=n_usable, failure_kind=None)


def decide_rule_D_weak_evidence_abstain(observations: list, config: AcceptanceConfig) -> TrackDecisionV2:
    stage1 = _select_and_count(observations, config)
    if isinstance(stage1, TrackDecisionV2):
        return stage1
    selected, counts, winner, count, n_usable = stage1

    if count < config.min_agreement_count:
        return TrackDecisionV2(status=NO_RELIABLE_RESULT, text=None, agreement_count=count,
                                num_selected=len(selected), num_usable_total=n_usable,
                                reason=f"top candidate agreement {count} < {config.min_agreement_count}",
                                failure_kind=DISAGREEMENT)

    winner_obs = [o for o in selected if normalize_plate_text(o.ocr_text) == winner]
    winning_char_confs = [o.char_confs for o in winner_obs if o.char_confs]
    if winning_char_confs and len(set(len(c) for c in winning_char_confs)) == 1:
        min_per_position = [min(vals) for vals in zip(*winning_char_confs)]
        if min(min_per_position) < MIN_CHAR_CONFIDENCE:
            return TrackDecisionV2(status=NO_RELIABLE_RESULT, text=None, agreement_count=count,
                                    num_selected=len(selected), num_usable_total=n_usable,
                                    reason=f"weak per-character evidence: min position confidence "
                                           f"{min(min_per_position):.3f} < {MIN_CHAR_CONFIDENCE}",
                                    failure_kind=WEAK_EVIDENCE)

    return TrackDecisionV2(status="ok", text=winner, agreement_count=count,
                            num_selected=len(selected), num_usable_total=n_usable, failure_kind=None)


def decide_rule_E_agreement_plus_profile(observations: list, config: AcceptanceConfig) -> TrackDecisionV2:
    """Same as A (current), plus the winning text must pass structural
    validity against UFPR_PROFILE (accept/reject only — no reranking here,
    matching the supervisor's request to compare a pure structural-validity
    gate before adding anything more elaborate). A winner that clears the
    agreement bar but fails the profile check (wrong length or a
    character-class violation) is tagged DISAGREEMENT — never rescuable,
    same as a plain agreement failure."""
    stage1 = _select_and_count(observations, config)
    if isinstance(stage1, TrackDecisionV2):
        return stage1
    selected, counts, winner, count, n_usable = stage1

    if count < config.min_agreement_count:
        return TrackDecisionV2(status=NO_RELIABLE_RESULT, text=None, agreement_count=count,
                                num_selected=len(selected), num_usable_total=n_usable,
                                reason=f"top candidate agreement {count} < {config.min_agreement_count}",
                                failure_kind=DISAGREEMENT)

    is_valid, profile_reason = validate_plate_profile(winner, UFPR_PROFILE)
    if is_valid is False:
        return TrackDecisionV2(status=NO_RELIABLE_RESULT, text=None, agreement_count=count,
                                num_selected=len(selected), num_usable_total=n_usable,
                                reason=f"failed plate-profile validation: {profile_reason}",
                                failure_kind=DISAGREEMENT)

    return TrackDecisionV2(status="ok", text=winner, agreement_count=count,
                            num_selected=len(selected), num_usable_total=n_usable, failure_kind=None)


def decide_rule_F_agreement_confidence_profile(observations: list, config: AcceptanceConfig) -> TrackDecisionV2:
    """Agreement + confidence (Rule B) + structural profile (Rule E),
    combined — all three independent checks must pass."""
    stage1 = _select_and_count(observations, config)
    if isinstance(stage1, TrackDecisionV2):
        return stage1
    selected, counts, winner, count, n_usable = stage1

    if count < config.min_agreement_count:
        return TrackDecisionV2(status=NO_RELIABLE_RESULT, text=None, agreement_count=count,
                                num_selected=len(selected), num_usable_total=n_usable,
                                reason=f"top candidate agreement {count} < {config.min_agreement_count}",
                                failure_kind=DISAGREEMENT)

    winner_obs = [o for o in selected if normalize_plate_text(o.ocr_text) == winner]
    mean_conf = sum(o.overall_conf or 0.0 for o in winner_obs) / len(winner_obs)
    if mean_conf < MIN_WINNING_CONFIDENCE:
        return TrackDecisionV2(status=NO_RELIABLE_RESULT, text=None, agreement_count=count,
                                num_selected=len(selected), num_usable_total=n_usable,
                                reason=f"winning mean confidence {mean_conf:.3f} < {MIN_WINNING_CONFIDENCE}",
                                failure_kind=DISAGREEMENT)

    is_valid, profile_reason = validate_plate_profile(winner, UFPR_PROFILE)
    if is_valid is False:
        return TrackDecisionV2(status=NO_RELIABLE_RESULT, text=None, agreement_count=count,
                                num_selected=len(selected), num_usable_total=n_usable,
                                reason=f"failed plate-profile validation: {profile_reason}",
                                failure_kind=DISAGREEMENT)

    return TrackDecisionV2(status="ok", text=winner, agreement_count=count,
                            num_selected=len(selected), num_usable_total=n_usable, failure_kind=None)


RULES = {
    "A_current": decide_rule_A_current,
    "B_agreement_plus_confidence": decide_rule_B_agreement_plus_confidence,
    "E_agreement_plus_profile": decide_rule_E_agreement_plus_profile,
    "F_agreement_confidence_profile": decide_rule_F_agreement_confidence_profile,
}


def evaluate_all_rules_on_validation(pipeline: AlprPipeline) -> dict:
    config = load_config()
    acceptance_config = AcceptanceConfig()
    val_track_ids = sorted(get_track_ids(MANIFEST_VAL_CSV))

    per_rule_outcomes = {name: [] for name in RULES}

    for track_id in val_track_ids:
        try:
            loaded = load_track(track_id, ufpr_root=UFPR_ROOT_DEFAULT)
        except (FileNotFoundError, ValueError, IOError) as e:
            print(f"  SKIPPING {track_id}: {e}")
            continue

        all_frames = loaded["frames"]
        gt_text = normalize_plate_text(loaded["gt_text"])
        strided_frames = all_frames[::STRIDE]

        video_pipeline = VideoAlprPipeline(pipeline, config, tracker_cls=IoUTracker)
        _tr, predicted_tracks, _stats = video_pipeline.process_video(strided_frames)
        pooled_observations = []
        for t in predicted_tracks:
            pooled_observations.extend(t.observations)

        for rule_name, rule_fn in RULES.items():
            decision = rule_fn(pooled_observations, acceptance_config)
            if decision.is_detection_related_failure():
                outcome = maybe_rescue(pipeline, all_frames, decision, pick_candidate_frames, RESCUE_CONFIG)
                decision = outcome.decision

            per_rule_outcomes[rule_name].append({
                "track_id": track_id, "gt_text": gt_text,
                "predicted_text": decision.text, "status": decision.status,
                "exact_match": decision.status == "ok" and decision.text == gt_text,
                "incorrect_accepted": decision.status == "ok" and decision.text != gt_text,
            })

    return per_rule_outcomes


def summarize_rule(outcomes: list) -> dict:
    n = len(outcomes)
    accepted = [o for o in outcomes if o["status"] == "ok"]
    n_accepted = len(accepted)
    n_correct_accepted = sum(1 for o in accepted if o["exact_match"])
    incorrect_accepted = [o for o in accepted if o["incorrect_accepted"]]
    n_complete_failure = sum(1 for o in outcomes if o["status"] == COMPLETE_DETECTION_FAILURE)
    n_abstained = sum(1 for o in outcomes if o["status"] == NO_RELIABLE_RESULT)

    return {
        "n_total": n,
        "accepted_result_precision": n_correct_accepted / n_accepted if n_accepted else None,
        "coverage": n_accepted / n if n else None,
        "n_incorrect_accepted": len(incorrect_accepted),
        "incorrect_accepted_tracks": [(o["track_id"], o["gt_text"], o["predicted_text"]) for o in incorrect_accepted],
        "n_abstentions": n_abstained,
        "n_complete_detection_failure": n_complete_failure,
    }


def print_comparison(summaries: dict):
    print("=" * 100)
    print("ACCEPTANCE RULE COMPARISON — validation data only")
    print("=" * 100)
    header = f"{'Rule':<32} {'Precision':>10} {'Coverage':>10} {'Incorrect-accepted':>19} {'Abstentions':>12}"
    print(header)
    print("-" * len(header))
    for name, s in summaries.items():
        prec = f"{s['accepted_result_precision']:.4f}" if s['accepted_result_precision'] is not None else "n/a"
        cov = f"{s['coverage']:.4f}" if s['coverage'] is not None else "n/a"
        print(f"{name:<32} {prec:>10} {cov:>10} {s['n_incorrect_accepted']:>19} {s['n_abstentions']:>12}")
    print()
    print("Complete-detection-failure count (identical across all rules, sanity check):",
          {name: s["n_complete_detection_failure"] for name, s in summaries.items()})
    print()
    for name, s in summaries.items():
        if s["incorrect_accepted_tracks"]:
            print(f"{name} — incorrect accepted tracks: {s['incorrect_accepted_tracks']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-1-onnx-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-json", type=str, default="acceptance_rule_comparison.json")
    args = parser.parse_args()

    pipeline = AlprPipeline(recognizer=PlateRecognizer(model_path=args.v1_1_onnx_path, config_path=args.config_path))

    print("Running all 4 rules on the 22 UFPR validation tracks...")
    per_rule_outcomes = evaluate_all_rules_on_validation(pipeline)
    summaries = {name: summarize_rule(outcomes) for name, outcomes in per_rule_outcomes.items()}

    print_comparison(summaries)

    import json
    with open(args.output_json, "w") as f:
        json.dump({"summaries": summaries, "per_rule_outcomes": per_rule_outcomes}, f, indent=2, default=str)
    print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
