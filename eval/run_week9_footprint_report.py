"""
run_week9_footprint_report.py

Week 9, step 4: documents the ACTUAL final computational cost of the
hardened candidate (stride=2, Top-3 majority vote, the explicit decision
rule, the hardened rescue path) on the 22 UFPR validation encounters.
This is a measurement pass, not another optimization study — the
stride=2 + Top-3 trade-off was already established in Weeks 7-8; this
just documents its real footprint precisely, per the plan.

Reports, per the plan's exact list:
  - detector latency (mean detect_ms per processed frame)
  - OCR latency (mean recognize_ms per processed frame)
  - tracking/fusion overhead (tracker.update() + decide_rc1() wall time,
    measured directly, separate from the detect/recognize stage costs)
  - total processing time per encounter
  - number of frames sampled (after stride) vs. total frames in the video
  - number of frames actually OCR'd
  - rescue frequency and rescue cost
  - peak memory (via resource.getrusage — Linux/POSIX, available in this
    container)
  - ONNX model sizes (recognizer + detector, wherever found on disk)

HONEST NOTE on "frames actually OCR'd": today, OCR runs on every sampled
frame that passes detection + quality gating — Top-K only decides which
of those already-computed observations are TRUSTED for the final vote,
it does not currently skip the OCR call itself for non-selected frames.
So "frames OCR'd" here equals "frames that passed detect+quality", not a
smaller number reflecting Top-K. A structural optimization that actually
skips OCR on non-selected frames is possible (see Week 7's latency
estimate, which modeled this) but was not implemented in the pipeline
itself — this script reports what the code actually does today, not a
hypothetical.

Run from inside the alpr-train container:
    cd /workspace/home/alpr-week5/pipeline
    python3 run_week9_footprint_report.py
"""

import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpr_pipeline import AlprPipeline
from tracker import IoUTracker
from stage3_config import load_config
from ufpr_video_loader import get_track_ids, load_track, UFPR_ROOT_DEFAULT
from video_pipeline import VideoAlprPipeline
from decision_rule_rc1 import decide_rc1, AcceptanceConfig
from week9_rescue_policy import maybe_rescue, RescuePolicyConfig
from run_week8_rescue_experiment import pick_candidate_frames

MANIFEST_VAL_CSV = Path("/workspace/home/alpr-week5/data/interim/manifest_val_full.csv")
OUTPUT_JSON = Path(__file__).parent / "week9_footprint_report_results.json"

STRIDE = 2
ACCEPTANCE_CONFIG = AcceptanceConfig()
RESCUE_CONFIG = RescuePolicyConfig()


def find_onnx_model_sizes() -> dict:
    """Searches a handful of plausible locations for ONNX model files and
    reports their sizes — does not assume a fixed path, since the
    detector's ONNX file location depends on where open_image_models
    cached it, which varies by environment."""
    search_roots = [
        Path("/workspace/home/alpr-week5/output"),
        Path.home() / ".cache",
        Path("/root/.cache"),
    ]
    found = {}
    for root in search_roots:
        if not root.exists():
            continue
        for onnx_path in root.rglob("*.onnx"):
            try:
                size_mb = onnx_path.stat().st_size / (1024 * 1024)
                found[str(onnx_path)] = round(size_mb, 2)
            except OSError:
                continue
    return found


def main():
    config = load_config()
    pipeline = AlprPipeline()

    val_track_ids = sorted(get_track_ids(MANIFEST_VAL_CSV))
    print(f"Found {len(val_track_ids)} UFPR validation tracks (expect 22).")

    per_encounter = []
    detect_ms_all, recognize_ms_all = [], []
    tracking_fusion_overhead_ms_all = []
    rescue_triggered_count = 0
    rescue_latency_ms_all = []

    for track_id in val_track_ids:
        try:
            loaded = load_track(track_id, ufpr_root=UFPR_ROOT_DEFAULT)
        except (FileNotFoundError, ValueError, IOError) as e:
            print(f"  SKIPPING {track_id}: {e}")
            continue

        all_frames = loaded["frames"]
        strided_frames = all_frames[::STRIDE]
        n_total_frames = len(all_frames)
        n_sampled_frames = len(strided_frames)

        t_encounter_start = time.perf_counter()

        video_pipeline = VideoAlprPipeline(pipeline, config, tracker_cls=IoUTracker)
        _tr, predicted_tracks, stats = video_pipeline.process_video(strided_frames)

        pooled_observations = []
        for t in predicted_tracks:
            pooled_observations.extend(t.observations)

        n_ocrd = sum(
            1 for o in pooled_observations if o.stage_timings_ms and o.stage_timings_ms.get("recognize_ms", 0.0) > 0
        )
        for o in pooled_observations:
            if o.stage_timings_ms:
                detect_ms_all.append(o.stage_timings_ms.get("detect_ms", 0.0))
                if o.stage_timings_ms.get("recognize_ms", 0.0) > 0:
                    recognize_ms_all.append(o.stage_timings_ms["recognize_ms"])

        # tracking/fusion overhead, measured directly and separately from
        # detect/recognize (which are already counted above, inside
        # process_video's per-frame pipeline.process() calls)
        t_decision_start = time.perf_counter()
        decision = decide_rc1(pooled_observations, ACCEPTANCE_CONFIG)
        decision_ms = (time.perf_counter() - t_decision_start) * 1000
        tracking_fusion_overhead_ms_all.append(decision_ms)

        rescue_triggered = False
        rescue_latency_ms = 0.0
        if decision.is_detection_related_failure():
            outcome = maybe_rescue(pipeline, all_frames, decision, pick_candidate_frames, RESCUE_CONFIG)
            rescue_triggered = outcome.triggered
            rescue_latency_ms = outcome.rescue_latency_ms
            decision = outcome.decision
            if rescue_triggered:
                rescue_triggered_count += 1
                rescue_latency_ms_all.append(rescue_latency_ms)

        total_encounter_ms = (time.perf_counter() - t_encounter_start) * 1000 + rescue_latency_ms

        per_encounter.append({
            "track_id": track_id,
            "n_total_frames": n_total_frames,
            "n_sampled_frames": n_sampled_frames,
            "n_frames_ocrd": n_ocrd,
            "total_processing_ms": total_encounter_ms,
            "rescue_triggered": rescue_triggered,
            "rescue_latency_ms": rescue_latency_ms,
            "final_status": decision.status,
        })
        print(f"  {track_id}: sampled={n_sampled_frames}/{n_total_frames}, ocrd={n_ocrd}, "
              f"total_ms={total_encounter_ms:.1f}, rescue={rescue_triggered}")

    peak_memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: KB
    onnx_sizes = find_onnx_model_sizes()

    n = len(per_encounter)
    summary = {
        "n_encounters": n,
        "mean_detect_ms_per_frame": sum(detect_ms_all) / len(detect_ms_all) if detect_ms_all else None,
        "mean_recognize_ms_per_frame": sum(recognize_ms_all) / len(recognize_ms_all) if recognize_ms_all else None,
        "mean_tracking_fusion_overhead_ms": (
            sum(tracking_fusion_overhead_ms_all) / len(tracking_fusion_overhead_ms_all)
            if tracking_fusion_overhead_ms_all else None
        ),
        "mean_total_processing_ms_per_encounter": (
            sum(e["total_processing_ms"] for e in per_encounter) / n if n else None
        ),
        "mean_frames_sampled": sum(e["n_sampled_frames"] for e in per_encounter) / n if n else None,
        "mean_frames_total": sum(e["n_total_frames"] for e in per_encounter) / n if n else None,
        "mean_frames_ocrd": sum(e["n_frames_ocrd"] for e in per_encounter) / n if n else None,
        "rescue_frequency": rescue_triggered_count / n if n else None,
        "mean_rescue_latency_ms_when_triggered": (
            sum(rescue_latency_ms_all) / len(rescue_latency_ms_all) if rescue_latency_ms_all else None
        ),
        "peak_memory_kb": peak_memory_kb,
        "peak_memory_mb": round(peak_memory_kb / 1024, 1),
        "onnx_model_sizes_mb": onnx_sizes,
    }

    print()
    print("=" * 70)
    print("WEEK 9 STEP 4 — COMPUTATIONAL FOOTPRINT (hardened candidate)")
    print("=" * 70)
    for k, v in summary.items():
        print(f"{k}: {v}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump({"summary": summary, "per_encounter": per_encounter}, f, indent=2)
    print(f"\nWrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
