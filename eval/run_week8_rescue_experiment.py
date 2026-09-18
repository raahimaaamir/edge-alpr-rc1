"""
run_week8_rescue_experiment.py

Week 8, step 3 driver. Runs the normal pipeline (stride=2 — the Week 8
step-2 winner — + Top-3 majority vote + the explicit decision rule from
step 1) on all 22 UFPR validation encounters, identifies which ones FAIL
(complete_detection_failure or no_reliable_result), and — ONLY for those —
applies the tiled-detection rescue path to a few full-resolution candidate
frames. Measures whether the rescue recovers the correct plate, whether it
introduces any new wrong-but-confident answers, and how much extra latency
it costs — charged only against the failing encounters it actually ran on,
never against the normal per-frame cost.

Run from inside the alpr-train container:
    cd /workspace/home/alpr-week5/pipeline
    python3 run_week8_rescue_experiment.py

Writes results to week8_rescue_experiment_results.json.
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
from plate_normalize import normalize_plate_text

MANIFEST_VAL_CSV = Path("/workspace/home/alpr-week5/data/interim/manifest_val_full.csv")
OUTPUT_JSON = Path(__file__).parent / "week8_rescue_experiment_results.json"

STRIDE = 2  # Week 8 step-2 winner
ACCEPTANCE_CONFIG = AcceptanceConfig()
RESCUE_CONFIG = TiledRescueConfig()
MAX_CANDIDATE_FRAMES = 3  # cap how many full frames the rescue path examines per failing encounter


def pick_candidate_frames(all_frames: list, n: int) -> list:
    """Evenly spaced full-resolution frames across the whole track (not
    just the strided subset) — the rescue path gets to look at frames the
    normal stride=2 pass may have skipped entirely."""
    if len(all_frames) <= n:
        return all_frames
    step = len(all_frames) / n
    return [all_frames[int(i * step)] for i in range(n)]


def main():
    config = load_config()
    pipeline = AlprPipeline()

    val_track_ids = sorted(get_track_ids(MANIFEST_VAL_CSV))
    print(f"Found {len(val_track_ids)} UFPR validation tracks (expect 22).")

    failing_encounters = []  # list of dicts, one per failing encounter, filled in as we go

    for track_id in val_track_ids:
        try:
            loaded = load_track(track_id, ufpr_root=UFPR_ROOT_DEFAULT)
        except (FileNotFoundError, ValueError, IOError) as e:
            print(f"  SKIPPING {track_id}: {e}")
            continue

        all_frames = loaded["frames"]
        gt_text = loaded["gt_text"]
        strided_frames = all_frames[::STRIDE]

        video_pipeline = VideoAlprPipeline(pipeline, config, tracker_cls=IoUTracker)
        _track_results, predicted_tracks, _stats = video_pipeline.process_video(strided_frames)
        pooled_observations = []
        for t in predicted_tracks:
            pooled_observations.extend(t.observations)

        before_decision = decide(pooled_observations, ACCEPTANCE_CONFIG)

        if before_decision.status == "ok":
            continue  # not a failure case — rescue path never runs here

        print(f"  track {track_id}: FAILING under normal pipeline "
              f"(status={before_decision.status}, reason={before_decision.reason}) — attempting rescue")

        candidates = pick_candidate_frames(all_frames, MAX_CANDIDATE_FRAMES)
        rescue_latency_total_ms = 0.0
        new_observations = []
        for frame_idx, _ts, image in candidates:
            results, tiling_latency_ms = tiled_rescue_detect_and_recognize(pipeline, image, RESCUE_CONFIG)
            rescue_latency_total_ms += tiling_latency_ms
            for r in results:
                if r.status == "ok":
                    new_observations.append(FrameObservation.from_plate_result(frame_idx, None, r))

        merged_observations = pooled_observations + new_observations
        after_decision = decide(merged_observations, ACCEPTANCE_CONFIG)

        outcome = "unchanged"
        if before_decision.status != "ok" and after_decision.status == "ok":
            if normalize_plate_text(after_decision.text) == normalize_plate_text(gt_text):
                outcome = "recovered_correct"
            else:
                outcome = "recovered_but_wrong"  # a new false-detection-driven wrong answer

        print(f"    rescue found {len(new_observations)} new usable observation(s) across "
              f"{len(candidates)} candidate frame(s); outcome={outcome}, "
              f"extra latency={rescue_latency_total_ms:.1f}ms")

        failing_encounters.append({
            "track_id": track_id,
            "gt_text": gt_text,
            "before_status": before_decision.status,
            "before_reason": before_decision.reason,
            "num_candidate_frames_tried": len(candidates),
            "num_new_observations_found": len(new_observations),
            "after_status": after_decision.status,
            "after_text": after_decision.text,
            "outcome": outcome,
            "rescue_latency_ms": rescue_latency_total_ms,
        })

    n_failing = len(failing_encounters)
    n_recovered_correct = sum(1 for e in failing_encounters if e["outcome"] == "recovered_correct")
    n_recovered_wrong = sum(1 for e in failing_encounters if e["outcome"] == "recovered_but_wrong")
    n_unchanged = sum(1 for e in failing_encounters if e["outcome"] == "unchanged")
    total_rescue_latency_ms = sum(e["rescue_latency_ms"] for e in failing_encounters)
    mean_rescue_latency_ms = total_rescue_latency_ms / n_failing if n_failing else None

    print()
    print("=" * 70)
    print("WEEK 8 STEP 3 — TILED RESCUE RESULTS")
    print("=" * 70)
    print(f"Encounters failing under normal pipeline: {n_failing}")
    print(f"  Recovered (correct plate): {n_recovered_correct}")
    print(f"  Recovered but WRONG (false detection introduced): {n_recovered_wrong}")
    print(f"  Unchanged (still failing): {n_unchanged}")
    print(f"Mean rescue latency per failing encounter: {mean_rescue_latency_ms}")
    print(f"Total rescue latency across all failing encounters: {total_rescue_latency_ms:.1f}ms")

    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "scope": f"UFPR validation split, stride={STRIDE} + Top-3 majority + Week8 decision rule as the baseline pipeline; rescue applied only to failing encounters",
            "acceptance_config": ACCEPTANCE_CONFIG.__dict__,
            "rescue_config": RESCUE_CONFIG.__dict__,
            "n_failing_encounters": n_failing,
            "n_recovered_correct": n_recovered_correct,
            "n_recovered_but_wrong": n_recovered_wrong,
            "n_unchanged": n_unchanged,
            "mean_rescue_latency_ms": mean_rescue_latency_ms,
            "total_rescue_latency_ms": total_rescue_latency_ms,
            "failing_encounters": failing_encounters,
        }, f, indent=2)
    print(f"\nWrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
