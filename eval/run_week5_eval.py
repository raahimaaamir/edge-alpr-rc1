"""
run_week5_eval.py

Driver for step 2 of the Week 5 checklist: run the frozen single-frame
pipeline (detector -> crop -> quality -> Recognizer V1) over real held-out
test data, frame by frame, and score it with the one standardized
evaluate_single_frame() function.

Uses the same 23 held-out UFPR test tracks (690 frames total) already
validated in the video-tracking stage — full raw frames with per-frame
ground truth, loaded via ufpr_video_loader.py. Each frame is evaluated
independently here (no tracking/fusion — that's stage 3, not this).

NOTE: this covers the UFPR portion of the test set only. manifest_test_full.csv
also has ~3,000 RodoSol test rows, but those point to pre-cropped plate
images (used for recognizer training), not full frames — running the full
detect+recognize pipeline on an already-cropped plate isn't a meaningful
test. A RodoSol full-frame loader (analogous to ufpr_video_loader.py) would
be needed to extend this; not built yet. State this scope explicitly in
the Week 5 report rather than implying full-test-set coverage.

Run from inside the alpr-train container:
    cd /workspace/home/alpr-week5/pipeline
    python3 run_week5_eval.py

Writes results to week5_eval_results.json in the same folder.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # see run_stage3_on_ufpr_test.py for why

from alpr_pipeline import AlprPipeline
from evaluate_single_frame import evaluate_single_frame, print_report
from ufpr_video_loader import get_test_track_ids, load_track, UFPR_ROOT_DEFAULT

MANIFEST_TEST_CSV = Path("/workspace/home/alpr-week5/data/interim/manifest_test_full.csv")
OUTPUT_JSON = Path(__file__).parent / "week5_eval_results.json"


def main():
    pipeline = AlprPipeline()

    test_track_ids = sorted(get_test_track_ids(MANIFEST_TEST_CSV))
    print(f"Found {len(test_track_ids)} held-out UFPR test tracks (expect 23).")

    all_results = []
    all_gt = []
    skipped_tracks = []

    for ufpr_track_id in test_track_ids:
        try:
            loaded = load_track(ufpr_track_id, ufpr_root=UFPR_ROOT_DEFAULT)
        except (FileNotFoundError, ValueError, IOError) as e:
            print(f"  SKIPPING track {ufpr_track_id}: {e}")
            skipped_tracks.append({"track_id": ufpr_track_id, "reason": str(e)})
            continue

        frames = loaded["frames"]
        gt_text = loaded["gt_text"]
        print(f"  track {ufpr_track_id}: {len(frames)} frames, gt='{gt_text}'")

        for frame_idx, _timestamp, image in frames:
            result = pipeline.process(image, image_id=f"{ufpr_track_id}_frame{frame_idx}")
            all_results.append(result)
            all_gt.append(gt_text)  # ground truth is the same plate for every frame in a track

    print(f"\nEvaluating {len(all_results)} frames...\n")
    report = evaluate_single_frame(all_results, all_gt)
    print_report(report)

    if skipped_tracks:
        print(f"\n{len(skipped_tracks)} track(s) skipped:")
        for s in skipped_tracks:
            print(f"  {s['track_id']}: {s['reason']}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "report": report.to_dict(),
            "n_tracks_found": len(test_track_ids),
            "n_tracks_evaluated": len(test_track_ids) - len(skipped_tracks),
            "n_frames_evaluated": len(all_results),
            "skipped_tracks": skipped_tracks,
            "scope_note": "UFPR held-out test tracks only (full raw frames); RodoSol test rows not included — see module docstring",
        }, f, indent=2)
    print(f"\nWrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
