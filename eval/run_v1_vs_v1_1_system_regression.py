"""
run_v1_vs_v1_1_system_regression.py

Per the supervisor's post-freeze instruction: runs V1 and V1.1 side-by-side
through the SAME frozen Week 8 video pipeline (detector, tracking, Top-3
selection, majority voting, rescue, decision rule ALL unchanged) on the
22 UFPR validation tracks. This is a regression check, not a tuning
exercise — nothing here is adjusted based on the result.

Compares, per model:
  - encounter-level exact accuracy
  - coverage (fraction with status == 'ok')
  - incorrect-accepted rate (status == 'ok' but text != ground truth)
  - NO_RELIABLE_RESULT rate (status != 'ok' and not complete_detection_failure)
  - total wall-clock runtime

Only the recognizer differs between the two runs — everything else
(stride=2, Top-3 majority, AcceptanceConfig, tiled rescue) is the exact
frozen Week 8/9 configuration, loaded via stage3_config.load_config().

Run from inside the alpr-train container, from the pipeline/ directory:
    python3 run_v1_vs_v1_1_system_regression.py --v1-1-onnx-path <path>
"""

import argparse
import sys
import time
from pathlib import Path

# alpr_pipeline.py imports its siblings as `pipeline.xxx` — put the
# directory ABOVE pipeline/ on sys.path so that resolves correctly.
# Everything else here (video_pipeline, tracker, decision rule, rescue)
# uses flat sibling imports and has no dependency on alpr_pipeline.py at
# all (confirmed: it receives an already-built single-frame pipeline via
# dependency injection), so those stay as ordinary flat imports below.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.alpr_pipeline import AlprPipeline
from pipeline.recognizer import PlateRecognizer

from tracker import IoUTracker
from stage3_config import load_config
from ufpr_video_loader import get_track_ids, load_track, UFPR_ROOT_DEFAULT
from video_pipeline import VideoAlprPipeline
from week9_decision_rule import decide_v2, AcceptanceConfig
from week9_rescue_policy import maybe_rescue, RescuePolicyConfig
from run_week8_rescue_experiment import pick_candidate_frames

MANIFEST_VAL_CSV = Path("/workspace/home/alpr-week5/data/interim/manifest_val_full.csv")
DEFAULT_V1_ONNX = "/workspace/home/alpr-week5/output/full_run1/onnx/best.onnx"
DEFAULT_CONFIG_PATH = "/workspace/home/alpr-week5/weights/cct_s_v2_global_plate_config.yaml"

STRIDE = 2
ACCEPTANCE_CONFIG = AcceptanceConfig()
RESCUE_CONFIG = RescuePolicyConfig()


def run_full_video_pipeline_for_model(single_frame_pipeline: AlprPipeline) -> dict:
    """Runs the frozen Week 8/9 video pipeline (unchanged: stride, tracker,
    Top-3 majority fusion, decision rule, rescue) over all 22 UFPR
    validation tracks, using the given single-frame pipeline (differing
    only by which recognizer it wraps). Returns per-encounter outcomes
    and summary rates."""
    config = load_config()
    val_track_ids = sorted(get_track_ids(MANIFEST_VAL_CSV))

    per_encounter = []
    t_start = time.perf_counter()

    for track_id in val_track_ids:
        try:
            loaded = load_track(track_id, ufpr_root=UFPR_ROOT_DEFAULT)
        except (FileNotFoundError, ValueError, IOError) as e:
            print(f"  SKIPPING {track_id}: {e}")
            continue

        all_frames = loaded["frames"]
        gt_text = loaded["gt_text"]
        strided_frames = all_frames[::STRIDE]

        video_pipeline = VideoAlprPipeline(single_frame_pipeline, config, tracker_cls=IoUTracker)
        _tr, predicted_tracks, _stats = video_pipeline.process_video(strided_frames)

        pooled_observations = []
        for t in predicted_tracks:
            pooled_observations.extend(t.observations)

        decision = decide_v2(pooled_observations, ACCEPTANCE_CONFIG)
        if decision.is_detection_related_failure():
            outcome = maybe_rescue(single_frame_pipeline, all_frames, decision, pick_candidate_frames, RESCUE_CONFIG)
            decision = outcome.decision

        per_encounter.append({
            "track_id": track_id,
            "gt_text": gt_text,
            "predicted_text": decision.text,
            "status": decision.status,
            "exact_match": decision.status == "ok" and decision.text == gt_text,
            "incorrect_accepted": decision.status == "ok" and decision.text != gt_text,
        })

    total_runtime_s = time.perf_counter() - t_start
    n = len(per_encounter)

    n_exact = sum(1 for e in per_encounter if e["exact_match"])
    n_coverage = sum(1 for e in per_encounter if e["status"] == "ok")
    n_incorrect_accepted = sum(1 for e in per_encounter if e["incorrect_accepted"])
    n_no_reliable = sum(1 for e in per_encounter if e["status"] not in ("ok", "complete_detection_failure"))

    return {
        "n_encounters": n,
        "exact_accuracy": n_exact / n if n else None,
        "coverage": n_coverage / n if n else None,
        "incorrect_accepted_rate": n_incorrect_accepted / n if n else None,
        "no_reliable_result_rate": n_no_reliable / n if n else None,
        "total_runtime_s": total_runtime_s,
        "per_encounter": per_encounter,
    }


def print_comparison(v1_result: dict, v1_1_result: dict):
    print("=" * 80)
    print("V1 vs V1.1 — SYSTEM-LEVEL REGRESSION CHECK (frozen video pipeline, unchanged)")
    print("=" * 80)
    fields = [
        ("n_encounters", "n"),
        ("exact_accuracy", "Exact accuracy"),
        ("coverage", "Coverage"),
        ("incorrect_accepted_rate", "Incorrect-accepted rate"),
        ("no_reliable_result_rate", "NO_RELIABLE_RESULT rate"),
        ("total_runtime_s", "Total runtime (s)"),
    ]
    header = f"{'Metric':<28} {'V1':>12} {'V1.1':>12}"
    print(header)
    print("-" * len(header))
    for key, label in fields:
        v1_val = v1_result[key]
        v11_val = v1_1_result[key]
        fmt = (lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))
        print(f"{label:<28} {fmt(v1_val):>12} {fmt(v11_val):>12}")

    print()
    print("Per-encounter differences (where V1 and V1.1 disagree on status or text):")
    v1_by_track = {e["track_id"]: e for e in v1_result["per_encounter"]}
    v11_by_track = {e["track_id"]: e for e in v1_1_result["per_encounter"]}
    any_diff = False
    for track_id in v1_by_track:
        a, b = v1_by_track[track_id], v11_by_track.get(track_id)
        if b and (a["status"] != b["status"] or a["predicted_text"] != b["predicted_text"]):
            any_diff = True
            print(f"  {track_id}: gt={a['gt_text']!r}  V1=({a['status']}, {a['predicted_text']!r})  "
                  f"V1.1=({b['status']}, {b['predicted_text']!r})")
    if not any_diff:
        print("  (none)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-onnx-path", type=str, default=DEFAULT_V1_ONNX)
    parser.add_argument("--v1-1-onnx-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-json", type=str, default="v1_vs_v1_1_system_regression.json")
    args = parser.parse_args()

    print(f"Building V1 pipeline from {args.v1_onnx_path}")
    pipeline_v1 = AlprPipeline(recognizer=PlateRecognizer(model_path=args.v1_onnx_path, config_path=args.config_path))
    print(f"Building V1.1 pipeline from {args.v1_1_onnx_path}")
    pipeline_v1_1 = AlprPipeline(recognizer=PlateRecognizer(model_path=args.v1_1_onnx_path, config_path=args.config_path))

    print("\nRunning V1 through the frozen video pipeline...")
    v1_result = run_full_video_pipeline_for_model(pipeline_v1)
    print("\nRunning V1.1 through the frozen video pipeline...")
    v1_1_result = run_full_video_pipeline_for_model(pipeline_v1_1)

    print()
    print_comparison(v1_result, v1_1_result)

    import json
    with open(args.output_json, "w") as f:
        json.dump({"v1": v1_result, "v1_1": v1_1_result}, f, indent=2, default=str)
    print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
