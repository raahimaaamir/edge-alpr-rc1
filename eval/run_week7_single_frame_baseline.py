"""
run_week7_single_frame_baseline.py

Closes a gap in the Week 7 comparison: the plan calls for comparing the
track-level baselines (best-frame, whole-string majority) against "the
current single-frame result" — but every single-frame number produced so
far (Week 5's report) was measured on the TEST set, not validation. Since
Week 7's track-level numbers are measured on the 22 UFPR VALIDATION
encounters, comparing them against a test-set single-frame number isn't
apples-to-apples.

This is exactly run_week5_eval.py's logic (evaluate_single_frame.py,
unchanged), pointed at manifest_val_full.csv instead of
manifest_test_full.csv — no fusion, no tracking, just the frozen
single-frame pipeline's raw per-frame accuracy on the same 660 frames
(22 tracks x 30 frames) that Week 7's track-level results are built from.

Run from inside the alpr-train container:
    cd /workspace/home/alpr-week5/pipeline
    python3 run_week7_single_frame_baseline.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpr_pipeline import AlprPipeline
from evaluate_single_frame import evaluate_single_frame, print_report
from ufpr_video_loader import get_track_ids, load_track, UFPR_ROOT_DEFAULT

MANIFEST_VAL_CSV = Path("/workspace/home/alpr-week5/data/interim/manifest_val_full.csv")
OUTPUT_JSON = Path(__file__).parent / "week7_single_frame_baseline_results.json"


def main():
    pipeline = AlprPipeline()

    val_track_ids = sorted(get_track_ids(MANIFEST_VAL_CSV))
    print(f"Found {len(val_track_ids)} UFPR validation tracks (expect 22).")

    all_results = []
    all_gt = []
    skipped = []

    for track_id in val_track_ids:
        try:
            loaded = load_track(track_id, ufpr_root=UFPR_ROOT_DEFAULT)
        except (FileNotFoundError, ValueError, IOError) as e:
            print(f"  SKIPPING {track_id}: {e}")
            skipped.append({"track_id": track_id, "reason": str(e)})
            continue
        gt_text = loaded["gt_text"]
        for frame_idx, _ts, image in loaded["frames"]:
            result = pipeline.process(image, image_id=f"{track_id}_frame{frame_idx}")
            all_results.append(result)
            all_gt.append(gt_text)

    print(f"\nEvaluating {len(all_results)} frames (no tracking, no fusion — raw per-frame)...\n")
    report = evaluate_single_frame(all_results, all_gt)
    print_report(report)

    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "scope": "UFPR validation split, single-frame (no tracking/fusion) — the apples-to-apples baseline for Week 7's track-level comparison",
            "report": report.to_dict(),
            "n_frames_evaluated": len(all_results),
            "skipped_tracks": skipped,
        }, f, indent=2)
    print(f"\nWrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
