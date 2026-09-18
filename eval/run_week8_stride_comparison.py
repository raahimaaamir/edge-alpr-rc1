"""
run_week8_stride_comparison.py

Week 8, step 2: does detection need to run on every frame? Subsample each
validation video's frames by a stride (1 = every frame, 2 = every 2nd, 3 =
every 3rd) BEFORE running them through the tracker + single-frame pipeline
— this actually reduces detection work (unlike Week 7's Top-K, which only
reduced OCR work; detection still ran on every frame there). Track
association, Top-3 selection, and the Week 8 decision rule (step 1) all run
unchanged on whatever frames survive the stride.

Per the plan: use the cheaper stride if it doesn't meaningfully hurt
reliability. Evaluation stays on the 22 UFPR validation encounters, at the
encounter level, consistent with every prior week's process correction.

Run from inside the alpr-train container:
    cd /workspace/home/alpr-week5/pipeline
    python3 run_week8_stride_comparison.py

Writes results to week8_stride_comparison_results.json.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpr_pipeline import AlprPipeline
from tracker import IoUTracker, TrackerConfig
from tracker_eval import evaluate_tracker, TrackerEvalConfig, TrackerEvalReport
from stage3_config import load_config
from ufpr_video_loader import get_track_ids, load_track, UFPR_ROOT_DEFAULT
from video_pipeline import VideoAlprPipeline
from week8_decision_rule import decide, summarize_decisions, AcceptanceConfig

MANIFEST_VAL_CSV = Path("/workspace/home/alpr-week5/data/interim/manifest_val_full.csv")
OUTPUT_JSON = Path(__file__).parent / "week8_stride_comparison_results.json"

STRIDES = [1, 2, 3]
ACCEPTANCE_CONFIG = AcceptanceConfig()  # untouched Week 8 step-1 defaults (top_k=3, min_usable=2, min_agreement=2)


def main():
    config = load_config()
    pipeline = AlprPipeline()

    val_track_ids = sorted(get_track_ids(MANIFEST_VAL_CSV))
    print(f"Found {len(val_track_ids)} UFPR validation tracks (expect 22).")

    results_by_stride = {}

    for stride in STRIDES:
        print(f"\n{'=' * 70}\nSTRIDE = {stride} (every {stride}{'st' if stride==1 else ('nd' if stride==2 else 'rd')} frame)\n{'=' * 70}")

        decisions_with_gt = []
        per_video_eval_reports = []
        skipped = []
        total_wall_time_s = 0.0

        for track_id in val_track_ids:
            try:
                loaded = load_track(track_id, ufpr_root=UFPR_ROOT_DEFAULT)
            except (FileNotFoundError, ValueError, IOError) as e:
                skipped.append({"track_id": track_id, "reason": str(e)})
                continue

            all_frames = loaded["frames"]
            gt_text = loaded["gt_text"]
            gt_boxes_per_frame = loaded["gt_boxes_per_frame"]

            # subsample BEFORE running the pipeline — this is what actually
            # reduces detection work, unlike Week 7's Top-K (OCR-only saving)
            strided_frames = all_frames[::stride]

            video_pipeline = VideoAlprPipeline(pipeline, config, tracker_cls=IoUTracker)
            t0 = time.time()
            _track_results, predicted_tracks, stats = video_pipeline.process_video(strided_frames)
            total_wall_time_s += (time.time() - t0)

            pooled_observations = []
            for t in predicted_tracks:
                pooled_observations.extend(t.observations)

            decision = decide(pooled_observations, ACCEPTANCE_CONFIG)
            decisions_with_gt.append((decision, gt_text))

            # association quality on the strided frames (fragmentation etc.
            # can behave differently at lower stride — worth tracking)
            video_eval = evaluate_tracker(predicted_tracks, gt_boxes_per_frame, TrackerEvalConfig())
            per_video_eval_reports.append(video_eval)

        summary = summarize_decisions(decisions_with_gt)

        merged_eval = TrackerEvalReport()
        purity_values = []
        for r in per_video_eval_reports:
            merged_eval.num_fragmentations += r.num_fragmentations
            merged_eval.num_id_switches += r.num_id_switches
            merged_eval.num_missed_detections += r.num_missed_detections
            merged_eval.num_gt_observations += r.num_gt_observations
            purity_values.extend(r.per_track_purity.values())
        merged_eval.mean_track_purity = sum(purity_values) / len(purity_values) if purity_values else 0.0

        results_by_stride[stride] = {
            "decision_summary": summary,
            "num_fragmentations": merged_eval.num_fragmentations,
            "num_id_switches": merged_eval.num_id_switches,
            "num_missed_detections": merged_eval.num_missed_detections,
            "num_gt_observations": merged_eval.num_gt_observations,
            "mean_track_purity": merged_eval.mean_track_purity,
            "total_wall_time_s": total_wall_time_s,
            "skipped_tracks": skipped,
        }

        print(f"accepted_result_precision={summary['accepted_result_precision']}, "
              f"coverage={summary['coverage']}, "
              f"incorrect_accepted={summary['n_incorrect_accepted']}, "
              f"fragmentations={merged_eval.num_fragmentations}, "
              f"total_wall_time_s={total_wall_time_s:.2f}")

    print(f"\n{'=' * 70}\nSTRIDE COMPARISON SUMMARY\n{'=' * 70}")
    print(f"{'stride':<8}{'precision':<12}{'coverage':<12}{'incorrect':<12}{'fragment':<10}{'missed_det':<12}{'wall_time_s'}")
    for stride in STRIDES:
        r = results_by_stride[stride]
        s = r["decision_summary"]
        print(f"{stride:<8}{str(round(s['accepted_result_precision'],4) if s['accepted_result_precision'] is not None else None):<12}"
              f"{str(round(s['coverage'],4) if s['coverage'] is not None else None):<12}{s['n_incorrect_accepted']:<12}"
              f"{r['num_fragmentations']:<10}{r['num_missed_detections']:<12}{round(r['total_wall_time_s'],2)}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "scope": "UFPR validation split, encounter-level, comparing frame stride 1/2/3 under Top-3 majority vote + Week 8 explicit decision rule",
            "acceptance_config": ACCEPTANCE_CONFIG.__dict__,
            "results_by_stride": {str(k): v for k, v in results_by_stride.items()},
        }, f, indent=2)
    print(f"\nWrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
