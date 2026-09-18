"""
run_week8_final_test_evaluation.py

Week 8, step 5 — THE ONLY SCRIPT THIS PROJECT TOUCHES THE TEST SET WITH
SINCE THE WEEK 5 FREEZE. Every threshold below was chosen entirely from
validation-set evidence (Weeks 6-8): stride=2 (step 2), Top-3 majority
vote with the explicit decision rule (step 1), default IoU tracker
parameters (Week 6), and the tiled rescue path kept (step 3 showed clear
benefit: 1 real recovery, 0 false detections, on the validation set).
Nothing here is tuned against test-set results — this script runs the
frozen candidate exactly once and reports what it gets.

FROZEN WEEK 8 CANDIDATE ("Video Pipeline V1"):
    video -> stride=2 frame sampling -> detect -> crop -> quality
    -> Recognizer V1 -> IoU/displacement tracking (default config)
    -> Top-3 selection -> whole-string majority vote
    -> decision rule (min_usable=2, min_agreement=2)
    -> tiled rescue path IF the above produced no accepted result
    -> final plate or NO_RELIABLE_RESULT

Applies to the 23 UFPR test tracks (real video). RodoSol's 3,000 test
images have no video/track structure, so they are NOT re-run here — their
Week 5 frozen single-frame numbers are unchanged and are simply carried
forward into the final comparison table, exactly as "frozen" implies.

Run from inside the alpr-train container:
    cd /workspace/home/alpr-week5/pipeline
    python3 run_week8_final_test_evaluation.py

Writes results to week8_final_test_evaluation_results.json.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpr_pipeline import AlprPipeline
from tracker import IoUTracker
from stage3_config import load_config
from ufpr_video_loader import get_track_ids, load_track, UFPR_ROOT_DEFAULT
from video_pipeline import VideoAlprPipeline
from track_types import FrameObservation
from week8_decision_rule import decide, AcceptanceConfig, COMPLETE_DETECTION_FAILURE, NO_RELIABLE_RESULT
from tiled_detection_rescue import tiled_rescue_detect_and_recognize, TiledRescueConfig
from run_week8_rescue_experiment import pick_candidate_frames
from plate_normalize import normalize_plate_text

MANIFEST_TEST_CSV = Path("/workspace/home/alpr-week5/data/interim/manifest_test_full.csv")
OUTPUT_JSON = Path(__file__).parent / "week8_final_test_evaluation_results.json"

# --- FROZEN Week 8 candidate parameters, chosen entirely from validation
# data in steps 1-3 above. Do not tune against what this script prints. ---
STRIDE = 2
ACCEPTANCE_CONFIG = AcceptanceConfig()          # top_k=3, min_usable_observations=2, min_agreement_count=2
RESCUE_CONFIG = TiledRescueConfig()             # grid=(2,2), overlap_frac=0.15, nms_iou_thresh=0.3, max_results=3
MAX_RESCUE_CANDIDATE_FRAMES = 3

# --- Frozen Week 5 baseline numbers, UNCHANGED, carried forward for
# comparison (not recomputed — that is what "frozen" means) ---
WEEK5_UFPR_BASELINE = {
    "unit": "per-frame (690 individual frames, no tracking/fusion)",
    "exact_accuracy": 0.7536231884057971,
    "coverage": 0.8579710144927536,
    "incorrect_accepted_rate": 0.10434782608695652,
    "n": 690,
}
WEEK5_RODOSOL_BASELINE = {
    "unit": "per-image (3000 individual images, no video exists for RodoSol)",
    "exact_accuracy": 0.7783333333333333,
    "coverage": 0.812,
    "incorrect_accepted_rate": 0.033666666666666664,
    "n": 3000,
}
WEEK5_COMBINED_BASELINE = {
    "unit": "per-sample (3690 total: 690 UFPR frames + 3000 RodoSol images)",
    "exact_accuracy": 0.7737127371273713,
    "coverage": 0.8205962059620596,
    "incorrect_accepted_rate": 0.046883468834688344,
    "n": 3690,
}


def run_video_pipeline_v1_on_track(pipeline, config, loaded):
    """Runs the complete frozen Week 8 candidate on one UFPR track:
    stride -> tracker -> Top-3 decision -> rescue-if-needed.
    Returns (final_decision, latency_stats_dict)."""
    all_frames = loaded["frames"]
    strided_frames = all_frames[::STRIDE]

    t0 = time.time()
    video_pipeline = VideoAlprPipeline(pipeline, config, tracker_cls=IoUTracker)
    _track_results, predicted_tracks, stats = video_pipeline.process_video(strided_frames)
    main_pass_s = time.time() - t0

    pooled_observations = []
    for t in predicted_tracks:
        pooled_observations.extend(t.observations)

    decision = decide(pooled_observations, ACCEPTANCE_CONFIG)
    rescue_latency_ms = 0.0
    rescue_triggered = False
    num_frames_detected_on = len(strided_frames)
    num_frames_ocrd = decision.num_selected if decision.status != COMPLETE_DETECTION_FAILURE else 0

    if decision.status != "ok":
        rescue_triggered = True
        candidates = pick_candidate_frames(all_frames, MAX_RESCUE_CANDIDATE_FRAMES)
        new_observations = []
        for frame_idx, _ts, image in candidates:
            results, tiling_latency_ms = tiled_rescue_detect_and_recognize(pipeline, image, RESCUE_CONFIG)
            rescue_latency_ms += tiling_latency_ms
            for r in results:
                if r.status == "ok":
                    new_observations.append(FrameObservation.from_plate_result(frame_idx, None, r))
        merged = pooled_observations + new_observations
        decision = decide(merged, ACCEPTANCE_CONFIG)
        num_frames_detected_on += len(candidates)
        num_frames_ocrd = decision.num_selected if decision.status not in (COMPLETE_DETECTION_FAILURE,) else num_frames_ocrd

    total_latency_s = main_pass_s + (rescue_latency_ms / 1000.0)
    return decision, {
        "main_pass_s": main_pass_s,
        "rescue_latency_ms": rescue_latency_ms,
        "rescue_triggered": rescue_triggered,
        "total_latency_s": total_latency_s,
        "num_frames_detected_on": num_frames_detected_on,
        "num_frames_ocrd": num_frames_ocrd,
    }


def main():
    config = load_config()
    pipeline = AlprPipeline()

    test_track_ids = sorted(get_track_ids(MANIFEST_TEST_CSV))
    print(f"Found {len(test_track_ids)} UFPR TEST tracks (expect 23). "
          f"This is the FIRST and ONLY test-set run for the Week 8 candidate.")

    per_track_results = []
    skipped = []

    for track_id in test_track_ids:
        try:
            loaded = load_track(track_id, ufpr_root=UFPR_ROOT_DEFAULT)
        except (FileNotFoundError, ValueError, IOError) as e:
            print(f"  SKIPPING {track_id}: {e}")
            skipped.append({"track_id": track_id, "reason": str(e)})
            continue

        gt_text = loaded["gt_text"]
        decision, latency = run_video_pipeline_v1_on_track(pipeline, config, loaded)
        correct = decision.status == "ok" and normalize_plate_text(decision.text) == normalize_plate_text(gt_text)

        print(f"  track {track_id}: gt={gt_text}, status={decision.status}, text={decision.text}, "
              f"correct={correct}, rescue_triggered={latency['rescue_triggered']}, "
              f"latency={latency['total_latency_s']:.2f}s")

        per_track_results.append({
            "track_id": track_id, "gt_text": gt_text, "status": decision.status,
            "text": decision.text, "correct": correct, **latency,
        })

    n = len(per_track_results)
    n_ok = sum(1 for r in per_track_results if r["status"] == "ok")
    n_correct = sum(1 for r in per_track_results if r["correct"])
    n_incorrect_accepted = n_ok - n_correct
    n_complete_failures = sum(1 for r in per_track_results if r["status"] == COMPLETE_DETECTION_FAILURE)
    n_no_reliable_result = sum(1 for r in per_track_results if r["status"] == NO_RELIABLE_RESULT)
    avg_frames_detected = sum(r["num_frames_detected_on"] for r in per_track_results) / n if n else None
    avg_frames_ocrd = sum(r["num_frames_ocrd"] for r in per_track_results) / n if n else None
    avg_latency_s = sum(r["total_latency_s"] for r in per_track_results) / n if n else None
    n_rescue_triggered = sum(1 for r in per_track_results if r["rescue_triggered"])

    video_pipeline_v1_ufpr = {
        "unit": f"per-encounter ({n} physical UFPR test tracks, video pipeline)",
        "exact_accuracy": n_correct / n if n else None,
        "coverage": n_ok / n if n else None,
        "incorrect_accepted_rate": n_incorrect_accepted / n if n else None,
        "n_complete_detection_failures": n_complete_failures,
        "n_no_reliable_result": n_no_reliable_result,
        "avg_frames_detected_on": avg_frames_detected,
        "avg_frames_ocrd": avg_frames_ocrd,
        "avg_latency_s": avg_latency_s,
        "n_rescue_triggered": n_rescue_triggered,
        "n": n,
    }

    print()
    print("=" * 90)
    print("WEEK 8 FINAL TEST-SET EVALUATION — Video Pipeline V1 vs. frozen Week 5 single-frame baseline")
    print("=" * 90)
    print("\n-- UFPR (video pipeline vs. single-frame; different units, see 'unit' field) --")
    print(f"Week 5 single-frame (per-frame, n=690): accuracy={WEEK5_UFPR_BASELINE['exact_accuracy']:.4f}, "
          f"coverage={WEEK5_UFPR_BASELINE['coverage']:.4f}, incorrect_accepted_rate={WEEK5_UFPR_BASELINE['incorrect_accepted_rate']:.4f}")
    print(f"Week 8 Video Pipeline V1 (per-encounter, n={n}): accuracy={video_pipeline_v1_ufpr['exact_accuracy']:.4f}, "
          f"coverage={video_pipeline_v1_ufpr['coverage']:.4f}, incorrect_accepted_rate={video_pipeline_v1_ufpr['incorrect_accepted_rate']:.4f}")
    print(f"  complete detection failures: {n_complete_failures}/{n}  |  rescue triggered on {n_rescue_triggered}/{n} encounters")
    print(f"  avg frames detected-on: {avg_frames_detected:.2f}  |  avg frames OCR'd: {avg_frames_ocrd:.2f}  |  avg latency: {avg_latency_s:.2f}s/encounter")

    print("\n-- RodoSol (no video exists — Week 5 single-frame numbers unchanged, carried forward) --")
    print(f"accuracy={WEEK5_RODOSOL_BASELINE['exact_accuracy']:.4f}, coverage={WEEK5_RODOSOL_BASELINE['coverage']:.4f}, "
          f"incorrect_accepted_rate={WEEK5_RODOSOL_BASELINE['incorrect_accepted_rate']:.4f}, n={WEEK5_RODOSOL_BASELINE['n']}")

    if skipped:
        print(f"\n{len(skipped)} track(s) skipped:")
        for s in skipped:
            print(f"  {s['track_id']}: {s['reason']}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "scope": "UFPR TEST split (manifest_test_full.csv) — first and only test-set run of the Week 8 candidate",
            "frozen_candidate_config": {
                "stride": STRIDE,
                "acceptance_config": ACCEPTANCE_CONFIG.__dict__,
                "rescue_config": RESCUE_CONFIG.__dict__,
            },
            "video_pipeline_v1_ufpr": video_pipeline_v1_ufpr,
            "week5_ufpr_baseline": WEEK5_UFPR_BASELINE,
            "week5_rodosol_baseline_unchanged": WEEK5_RODOSOL_BASELINE,
            "week5_combined_baseline": WEEK5_COMBINED_BASELINE,
            "per_track_results": per_track_results,
            "skipped_tracks": skipped,
        }, f, indent=2)
    print(f"\nWrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
