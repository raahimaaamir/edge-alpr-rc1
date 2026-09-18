"""
run_week9_rescue_hardening_comparison.py

Week 9, step 1 deliverable: runs BOTH the old (Week 8 V1, blended-pool,
unrestricted trigger) and the new (Week 9 hardened, restricted trigger,
independent acceptance) rescue policies on the SAME 22 UFPR validation
encounters, in one pass, for a clean side-by-side.

Everything upstream of the decision/rescue layer (stride=2 frame sampling,
tracking, Top-3 selection) is identical for both — only the
decide()/rescue behavior differs, isolating exactly what this week's
change affects.

Run from inside the alpr-train container:
    cd /workspace/home/alpr-week5/pipeline
    python3 run_week9_rescue_hardening_comparison.py

Writes results to week9_rescue_hardening_comparison_results.json.
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
from plate_normalize import normalize_plate_text

# OLD (V1, unchanged) — the frozen Week 8 rollback reference
from week8_decision_rule import decide as decide_v1, AcceptanceConfig as AcceptanceConfigV1
from tiled_detection_rescue import tiled_rescue_detect_and_recognize, TiledRescueConfig
from run_week8_rescue_experiment import pick_candidate_frames

# NEW (V2, hardened) — this week's fix
from week9_decision_rule import decide_v2, RESCUABLE_FAILURE_KINDS
from week9_rescue_policy import maybe_rescue, RescuePolicyConfig

MANIFEST_VAL_CSV = Path("/workspace/home/alpr-week5/data/interim/manifest_val_full.csv")
OUTPUT_JSON = Path(__file__).parent / "week9_rescue_hardening_comparison_results.json"

STRIDE = 2
V1_ACCEPTANCE = AcceptanceConfigV1()
RESCUE_CONFIG_TILED = TiledRescueConfig()
MAX_CANDIDATE_FRAMES = 3


def run_v1_rescue(pipeline, all_frames, pooled_observations, normal_decision_v1, gt_text):
    """Reproduces Week 8's exact rescue behavior: triggers on ANY non-'ok'
    decision, blends new observations into the ORIGINAL pool, re-decides
    with decide_v1 on the blended pool."""
    if normal_decision_v1.status == "ok":
        return normal_decision_v1, 0.0, False

    candidates = pick_candidate_frames(all_frames, MAX_CANDIDATE_FRAMES)
    rescue_latency_ms = 0.0
    new_observations = []
    for frame_idx, _ts, image in candidates:
        results, tiling_latency_ms = tiled_rescue_detect_and_recognize(pipeline, image, RESCUE_CONFIG_TILED)
        rescue_latency_ms += tiling_latency_ms
        for r in results:
            if r.status == "ok":
                new_observations.append(FrameObservation.from_plate_result(frame_idx, None, r))

    merged = pooled_observations + new_observations  # THE OLD BEHAVIOR: blend into original pool
    final_decision = decide_v1(merged, V1_ACCEPTANCE)
    return final_decision, rescue_latency_ms, True


def main():
    config = load_config()
    pipeline = AlprPipeline()

    val_track_ids = sorted(get_track_ids(MANIFEST_VAL_CSV))
    print(f"Found {len(val_track_ids)} UFPR validation tracks (expect 22).")

    v1_results, v2_results = [], []

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
        _tr, predicted_tracks, _stats = video_pipeline.process_video(strided_frames)
        pooled_observations = []
        for t in predicted_tracks:
            pooled_observations.extend(t.observations)

        # --- V1 (old, unrestricted, blended) ---
        normal_decision_v1 = decide_v1(pooled_observations, V1_ACCEPTANCE)
        final_v1, latency_v1_ms, triggered_v1 = run_v1_rescue(
            pipeline, all_frames, pooled_observations, normal_decision_v1, gt_text
        )
        correct_v1 = final_v1.status == "ok" and normalize_plate_text(final_v1.text) == normalize_plate_text(gt_text)

        # --- V2 (new, hardened) ---
        normal_decision_v2 = decide_v2(pooled_observations)
        outcome_v2 = maybe_rescue(pipeline, all_frames, normal_decision_v2, pick_candidate_frames)
        final_v2 = outcome_v2.decision
        correct_v2 = final_v2.status == "ok" and normalize_plate_text(final_v2.text) == normalize_plate_text(gt_text)

        print(f"  {track_id}: gt={gt_text} | V1: status={final_v1.status}, text={final_v1.text}, "
              f"correct={correct_v1}, triggered={triggered_v1} | V2: status={final_v2.status}, "
              f"text={final_v2.text}, correct={correct_v2}, triggered={outcome_v2.triggered} "
              f"(kind={normal_decision_v2.failure_kind})")

        v1_results.append({"track_id": track_id, "gt_text": gt_text, "status": final_v1.status,
                            "text": final_v1.text, "correct": correct_v1, "rescue_triggered": triggered_v1,
                            "rescue_latency_ms": latency_v1_ms})
        v2_results.append({"track_id": track_id, "gt_text": gt_text, "status": final_v2.status,
                            "text": final_v2.text, "correct": correct_v2, "rescue_triggered": outcome_v2.triggered,
                            "rescue_latency_ms": outcome_v2.rescue_latency_ms,
                            "normal_failure_kind": normal_decision_v2.failure_kind})

    def summarize(results):
        n = len(results)
        n_ok = sum(1 for r in results if r["status"] == "ok")
        n_correct = sum(1 for r in results if r["correct"])
        n_incorrect_accepted = n_ok - n_correct
        n_rescue_triggered = sum(1 for r in results if r["rescue_triggered"])
        n_rescue_recovered_correct = sum(1 for r in results if r["rescue_triggered"] and r["correct"])
        total_rescue_latency_ms = sum(r["rescue_latency_ms"] for r in results)
        return {
            "n": n, "coverage": n_ok / n if n else None, "accuracy": n_correct / n if n else None,
            "n_incorrect_accepted": n_incorrect_accepted, "n_rescue_triggered": n_rescue_triggered,
            "n_rescue_recovered_correct": n_rescue_recovered_correct,
            "total_rescue_latency_ms": total_rescue_latency_ms,
            "mean_rescue_latency_ms_when_triggered": (
                total_rescue_latency_ms / n_rescue_triggered if n_rescue_triggered else None
            ),
        }

    summary_v1 = summarize(v1_results)
    summary_v2 = summarize(v2_results)

    print()
    print("=" * 90)
    print("WEEK 9 STEP 1 — RESCUE HARDENING: BEFORE (V1) vs AFTER (V2 hardened)")
    print("=" * 90)
    print(f"{'Metric':<35}{'V1 (before)':<20}{'V2 (hardened, after)'}")
    print(f"{'Accuracy':<35}{summary_v1['accuracy']:<20}{summary_v2['accuracy']}")
    print(f"{'Coverage':<35}{summary_v1['coverage']:<20}{summary_v2['coverage']}")
    print(f"{'Incorrect-accepted':<35}{summary_v1['n_incorrect_accepted']:<20}{summary_v2['n_incorrect_accepted']}")
    print(f"{'Rescue triggered on':<35}{summary_v1['n_rescue_triggered']:<20}{summary_v2['n_rescue_triggered']}")
    print(f"{'Rescue recovered correctly':<35}{summary_v1['n_rescue_recovered_correct']:<20}{summary_v2['n_rescue_recovered_correct']}")
    print(f"{'Total rescue latency (ms)':<35}{round(summary_v1['total_rescue_latency_ms'],1):<20}{round(summary_v2['total_rescue_latency_ms'],1)}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "scope": "UFPR validation split, stride=2 + Top-3 majority, comparing V1 (Week 8) vs V2 (Week 9 hardened) rescue policy",
            "v1_summary": summary_v1, "v2_summary": summary_v2,
            "v1_per_track": v1_results, "v2_per_track": v2_results,
        }, f, indent=2)
    print(f"\nWrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
