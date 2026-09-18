"""
run_stage3_on_ufpr_test.py

The real-data driver: runs the frozen single-frame AlprPipeline + the new
tracker/fusion stage over every UFPR track in this project's held-out test
split (manifest_test_full.csv), then produces the evaluate_stage3.py report
(single-frame vs. fused accuracy per method, track coverage, tracker
association failures, frames-used, timing).

Run from inside the alpr-train container:
    cd /workspace/home/alpr-week5/pipeline
    python3 run_stage3_on_ufpr_test.py

Writes results to stage3_ufpr_test_results.json in the same folder.

IMPORTANT: this only touches the 23 held-out UFPR test tracks — the same
test set Recognizer V1 was frozen against. Per the supervisor's instruction,
this is measurement only; nothing here re-tunes the recognizer.
"""

import json
import sys
import time
from pathlib import Path

# alpr_pipeline.py (existing, frozen) imports as `from pipeline.detector import ...`,
# i.e. it expects the PARENT of this pipeline/ folder to be on sys.path so that
# `pipeline` resolves as a package. Add it here so this script can be run directly
# (`python3 run_stage3_on_ufpr_test.py` from inside pipeline/) without needing
# `python3 -m pipeline.run_stage3_on_ufpr_test` from one directory up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpr_pipeline import AlprPipeline
from tracker import IoUTracker
from tracker_eval import evaluate_tracker, TrackerEvalReport, TrackerEvalConfig
from fusion import fuse, FUSION_METHODS
from evaluate_stage3 import evaluate_stage3, print_report
from stage3_config import load_config
from ufpr_video_loader import get_test_track_ids, load_track, UFPR_ROOT_DEFAULT
from video_pipeline import VideoAlprPipeline

MANIFEST_TEST_CSV = Path("/workspace/home/alpr-week5/data/interim/manifest_test_full.csv")
OUTPUT_JSON = Path(__file__).parent / "stage3_ufpr_test_results.json"


def main():
    config = load_config()
    single_frame_pipeline = AlprPipeline()  # uses the frozen Recognizer V1 + detector, per existing alpr_pipeline.py defaults

    test_track_ids = sorted(get_test_track_ids(MANIFEST_TEST_CSV))
    print(f"Found {len(test_track_ids)} held-out UFPR test tracks in manifest (expect 23).")

    all_tracks = []          # list[Track], track_id renumbered globally to stay unique across videos
    all_track_results = []   # list[TrackResult], primary fusion method
    gt_text_by_track_id = {}
    per_video_eval_reports = []  # list[TrackerEvalReport], one per UFPR track/video
    extra_results_by_method = {m: [] for m in FUSION_METHODS}

    total_pipeline_time_ms = 0.0
    total_frames_processed = 0
    next_global_track_id = 0
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

        # run this single video through the tracker fresh (each UFPR track
        # is one physical plate across its own frame sequence — a separate
        # "video" from every other track)
        video_pipeline = VideoAlprPipeline(single_frame_pipeline, config, tracker_cls=IoUTracker)
        t0 = time.time()
        track_results, tracks, stats = video_pipeline.process_video(frames)
        elapsed_ms = (time.time() - t0) * 1000.0

        total_pipeline_time_ms += stats.total_pipeline_time_ms
        total_frames_processed += stats.num_frames

        if len(tracks) != 1:
            print(f"    NOTE: tracker produced {len(tracks)} tracks for this single-plate video "
                  f"(expected 1) — likely a fragmentation or a false split; kept for tracker_eval to quantify.")

        # tracker_eval must see ALL of this video's predicted tracks together
        # (not one at a time) — fragmentation is defined as ">1 predicted
        # track matched to the same GT track", which is structurally
        # invisible if you evaluate one predicted track in isolation. Frame
        # indices are still video-local at this point (renumbering below
        # only touches track_id, not frame_idx), so this must run BEFORE
        # renumbering, against this video's own gt_boxes_per_frame.
        video_eval_report = evaluate_tracker(tracks, loaded["gt_boxes_per_frame"], TrackerEvalConfig())
        per_video_eval_reports.append(video_eval_report)

        # renumber every predicted track's id to a global id so tracks from
        # different videos never collide when concatenated together below
        for track, result in zip(tracks, track_results):
            global_id = next_global_track_id
            next_global_track_id += 1
            track.track_id = global_id
            result.track_id = global_id
            all_tracks.append(track)
            all_track_results.append(result)
            gt_text_by_track_id[global_id] = gt_text

            for method_name in FUSION_METHODS:
                extra_results_by_method[method_name].append(fuse(track, method=method_name, config=config.fusion))

    # --- merge the per-video tracker_eval reports (each already correctly
    # scored against its own video's full set of predicted tracks) ---
    merged_eval = TrackerEvalReport()
    purity_values = []
    for r in per_video_eval_reports:
        merged_eval.num_gt_tracks += r.num_gt_tracks
        merged_eval.num_gt_observations += r.num_gt_observations
        merged_eval.num_predicted_tracks += r.num_predicted_tracks
        merged_eval.num_missed_detections += r.num_missed_detections
        merged_eval.num_false_positive_tracks += r.num_false_positive_tracks
        merged_eval.num_id_switches += r.num_id_switches
        merged_eval.num_fragmentations += r.num_fragmentations
        purity_values.extend(r.per_track_purity.values())
    merged_eval.mean_track_purity = sum(purity_values) / len(purity_values) if purity_values else 0.0

    report = evaluate_stage3(
        tracks=all_tracks,
        track_results=all_track_results,
        gt_text_by_track_id=gt_text_by_track_id,
        total_pipeline_time_ms=total_pipeline_time_ms,
        num_frames_processed=total_frames_processed,
        tracker_eval_report=merged_eval,
        extra_fused_results_by_method=extra_results_by_method,
    )

    print()
    print_report(report)
    if skipped_tracks:
        print(f"\n{len(skipped_tracks)} track(s) skipped (see JSON output for details):")
        for s in skipped_tracks:
            print(f"  {s['track_id']}: {s['reason']}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "report": report.to_dict(),
            "skipped_tracks": skipped_tracks,
            "num_test_tracks_found": len(test_track_ids),
            "num_test_tracks_evaluated": len(test_track_ids) - len(skipped_tracks),
        }, f, indent=2)
    print(f"\nWrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
