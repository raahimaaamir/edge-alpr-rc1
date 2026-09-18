"""
run_rc1_final_test_regression.py

The final, one-time verification of the COMPLETE frozen RC1 system —
Recognizer V1.1 + decide_rc1 (agreement + confidence + applicable
plate-profile) + the corrected rescue policy (no longer with its own
weaker acceptance rule) — on the 23 UFPR TEST tracks.

Per this project's own established test-set discipline (touched only
when freezing a pipeline version — first touched in Week 8 for V1's
freeze, now touched a second time for RC1's freeze): this run is
STRICTLY VERIFICATION. Nothing here should be used to select or adjust
any parameter — if anything looks wrong, the fix happens on validation
data, and this script is rerun once, not iterated against.

Run from inside the alpr-train container, from the pipeline/ directory:
    python3 run_rc1_final_test_regression.py --v1-1-onnx-path <path>
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.alpr_pipeline import AlprPipeline
from pipeline.recognizer import PlateRecognizer

from tracker import IoUTracker
from stage3_config import load_config
from ufpr_video_loader import get_track_ids, load_track, UFPR_ROOT_DEFAULT
from video_pipeline import VideoAlprPipeline
from plate_normalize import normalize_plate_text
from decision_rule_rc1 import decide_rc1, AcceptanceConfig
from week9_rescue_policy import maybe_rescue, RescuePolicyConfig
from run_week8_rescue_experiment import pick_candidate_frames

MANIFEST_TEST_CSV = Path("/workspace/home/alpr-week5/data/interim/manifest_test_full.csv")
DEFAULT_CONFIG_PATH = "/workspace/home/alpr-week5/weights/cct_s_v2_global_plate_config.yaml"
STRIDE = 2
ACCEPTANCE_CONFIG = AcceptanceConfig()
RESCUE_CONFIG = RescuePolicyConfig()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-1-onnx-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-json", type=str, default="rc1_final_test_regression.json")
    args = parser.parse_args()

    pipeline = AlprPipeline(recognizer=PlateRecognizer(model_path=args.v1_1_onnx_path, config_path=args.config_path))
    config = load_config()
    test_track_ids = sorted(get_track_ids(MANIFEST_TEST_CSV))
    print(f"Found {len(test_track_ids)} UFPR test tracks (expect 23). Running RC1 (V1.1 + decide_rc1 + fixed rescue).")

    per_encounter = []
    t_start = time.perf_counter()

    for track_id in test_track_ids:
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

        decision = decide_rc1(pooled_observations, ACCEPTANCE_CONFIG)  # no profile: jurisdiction not wired at this level
        rescue_triggered = False
        if decision.is_detection_related_failure():
            outcome = maybe_rescue(pipeline, all_frames, decision, pick_candidate_frames, RESCUE_CONFIG)
            rescue_triggered = outcome.triggered
            decision = outcome.decision

        per_encounter.append({
            "track_id": track_id, "gt_text": gt_text,
            "predicted_text": decision.text, "status": decision.status,
            "exact_match": decision.status == "ok" and decision.text == gt_text,
            "incorrect_accepted": decision.status == "ok" and decision.text != gt_text,
            "rescue_triggered": rescue_triggered,
        })
        print(f"  {track_id}: status={decision.status}, text={decision.text!r}, gt={gt_text!r}, "
              f"rescue={rescue_triggered}, exact_match={per_encounter[-1]['exact_match']}")

    total_runtime_s = time.perf_counter() - t_start
    n = len(per_encounter)
    n_exact = sum(1 for e in per_encounter if e["exact_match"])
    n_coverage = sum(1 for e in per_encounter if e["status"] == "ok")
    n_incorrect = sum(1 for e in per_encounter if e["incorrect_accepted"])
    n_rescued = sum(1 for e in per_encounter if e["rescue_triggered"])
    incorrect_tracks = [(e["track_id"], e["gt_text"], e["predicted_text"]) for e in per_encounter if e["incorrect_accepted"]]

    print()
    print("=" * 80)
    print("RC1 FINAL TEST-SET REGRESSION (23 UFPR test tracks) — VERIFICATION ONLY")
    print("=" * 80)
    print(f"n = {n}")
    print(f"Exact accuracy:            {n_exact/n:.4f}")
    print(f"Coverage:                  {n_coverage/n:.4f}")
    print(f"Incorrect-accepted rate:   {n_incorrect/n:.4f}  ({n_incorrect}/{n})")
    print(f"Rescue triggered on:       {n_rescued}/{n} tracks")
    print(f"Total runtime:             {total_runtime_s:.2f}s")
    if incorrect_tracks:
        print(f"Incorrect accepted tracks: {incorrect_tracks}")

    import json
    with open(args.output_json, "w") as f:
        json.dump({
            "n": n, "exact_accuracy": n_exact/n, "coverage": n_coverage/n,
            "incorrect_accepted_rate": n_incorrect/n, "n_rescued": n_rescued,
            "total_runtime_s": total_runtime_s, "per_encounter": per_encounter,
        }, f, indent=2, default=str)
    print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
