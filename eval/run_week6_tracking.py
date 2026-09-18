"""
run_week6_tracking.py

Week 6 deliverable: video -> detections -> associated plate tracks ->
multiple structured observations per track, plus tracking-quality
evaluation and two simple track-level reference baselines.

Per the Week 6 correction on test-set discipline: this runs on
manifest_val_full.csv (the 22 UFPR validation tracks), NOT
manifest_test_full.csv. The earlier stage-3 exploration
(run_stage3_on_ufpr_test.py) ran on the test split before that correction
was given — treat that run as exploratory, not the frozen Week 6 result.
The test set is touched again only when freezing a new pipeline version.

What this does NOT do (explicitly out of scope for Week 6, per the
supervisor's instructions):
  - No Kalman filtering or ByteTrack — the IoU/centre-displacement tracker
    (tracker.py) is used as-is. Only reach for something heavier if THIS
    week's numbers show a real problem the simple approach can't handle.
  - No fusion-method comparison or tuning — best_single_frame and
    majority_vote are computed once, unweighted and untuned, purely as
    reference points. The full fusion comparison is Week 7's job.
  - No threshold tuning sweep — uses the existing default TrackerConfig
    (stage3_config.py) as the "simplest practical" starting point. Whether
    these defaults need tuning is itself one of this week's questions,
    answered by the association-quality numbers below; tuning (against
    validation data, never test) is follow-up work.

Association-quality metrics reported, computed per video (each UFPR track
folder = one video containing exactly one physical plate) then aggregated:
  - correct track association rate: fraction of videos where the tracker
    produced exactly one predicted track, matched fully to the true plate
    (purity 1.0, no fragmentation, no false-positive tracks)
  - track fragmentation: count/rate of videos where the true plate was
    split across >1 predicted track
  - incorrect merges / identity switches: from tracker_eval's id-switch
    count. NOTE: every UFPR video here contains exactly one physical
    plate, so a genuine cross-identity merge cannot be demonstrated by
    this data — this number will read ~0 by construction, not because
    merging is solved. A real identity-switch test needs multi-plate
    scenes, which UFPR's track-per-video structure doesn't provide.
  - missed observations: ground-truth frames with no matching predicted
    observation (detector misses)
  - average observations per track

Run from inside the alpr-train container:
    cd /workspace/home/alpr-week5/pipeline
    python3 run_week6_tracking.py

Writes results to week6_tracking_results.json.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpr_pipeline import AlprPipeline
from tracker import IoUTracker, TrackerConfig
from tracker_eval import evaluate_tracker, TrackerEvalConfig, TrackerEvalReport
from fusion import fuse
from stage3_config import load_config
from ufpr_video_loader import get_track_ids, load_track, UFPR_ROOT_DEFAULT
from video_pipeline import VideoAlprPipeline

MANIFEST_VAL_CSV = Path("/workspace/home/alpr-week5/data/interim/manifest_val_full.csv")
OUTPUT_JSON = Path(__file__).parent / "week6_tracking_results.json"

REFERENCE_FUSION_METHODS = ("best_single_frame", "majority_vote")


def main():
    config = load_config()  # unmodified defaults — see module docstring
    pipeline = AlprPipeline()

    val_track_ids = sorted(get_track_ids(MANIFEST_VAL_CSV))
    print(f"Found {len(val_track_ids)} UFPR validation tracks (expect 22).")

    per_video_eval_reports = []
    per_video_summaries = []  # one dict per video: association-quality verdict
    reference_results = {m: [] for m in REFERENCE_FUSION_METHODS}
    gt_text_by_global_track_id = {}
    all_tracks_by_method = {m: [] for m in REFERENCE_FUSION_METHODS}

    next_global_track_id = 0
    skipped = []
    total_pipeline_ms = 0.0
    total_frames = 0

    for ufpr_track_id in val_track_ids:
        try:
            loaded = load_track(ufpr_track_id, ufpr_root=UFPR_ROOT_DEFAULT)
        except (FileNotFoundError, ValueError, IOError) as e:
            print(f"  SKIPPING track {ufpr_track_id}: {e}")
            skipped.append({"track_id": ufpr_track_id, "reason": str(e)})
            continue

        frames = loaded["frames"]
        gt_text = loaded["gt_text"]
        gt_boxes_per_frame = loaded["gt_boxes_per_frame"]
        print(f"  track {ufpr_track_id}: {len(frames)} frames, gt='{gt_text}'")

        video_pipeline = VideoAlprPipeline(pipeline, config, tracker_cls=IoUTracker)
        t0 = time.time()
        _track_results, tracks, stats = video_pipeline.process_video(frames)
        elapsed_ms = (time.time() - t0) * 1000.0
        total_pipeline_ms += stats.total_pipeline_time_ms
        total_frames += stats.num_frames

        # --- association-quality evaluation, this video only ---
        video_eval = evaluate_tracker(tracks, gt_boxes_per_frame, TrackerEvalConfig())
        per_video_eval_reports.append(video_eval)

        is_correctly_associated = (
            len(tracks) == 1
            and video_eval.num_fragmentations == 0
            and video_eval.num_false_positive_tracks == 0
            and video_eval.mean_track_purity == 1.0
        )
        per_video_summaries.append({
            "track_id": ufpr_track_id,
            "num_predicted_tracks": len(tracks),
            "num_gt_observations": video_eval.num_gt_observations,
            "num_missed_detections": video_eval.num_missed_detections,
            "num_fragmentations": video_eval.num_fragmentations,
            "num_false_positive_tracks": video_eval.num_false_positive_tracks,
            "num_id_switches": video_eval.num_id_switches,
            "mean_track_purity": video_eval.mean_track_purity,
            "correctly_associated": is_correctly_associated,
            "avg_observations_per_predicted_track": (
                sum(t.num_observations for t in tracks) / len(tracks) if tracks else 0.0
            ),
        })

        # --- reference baselines: best_single_frame and majority_vote only,
        # untuned defaults, no comparison/optimization (that's Week 7) ---
        for track in tracks:
            global_id = next_global_track_id
            next_global_track_id += 1
            gt_text_by_global_track_id[global_id] = gt_text
            for method in REFERENCE_FUSION_METHODS:
                result = fuse(track, method=method, config=config.fusion)
                result.track_id = global_id
                reference_results[method].append(result)

    # --- merge per-video tracker_eval reports into one association report ---
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

    num_videos = len(per_video_summaries)
    num_correctly_associated = sum(1 for s in per_video_summaries if s["correctly_associated"])
    avg_obs_per_track = (
        sum(s["avg_observations_per_predicted_track"] for s in per_video_summaries) / num_videos
        if num_videos else 0.0
    )

    print()
    print("=" * 60)
    print("WEEK 6 — ASSOCIATION QUALITY (UFPR validation, n={} videos)".format(num_videos))
    print("=" * 60)
    print(f"Correct track association: {num_correctly_associated}/{num_videos} videos "
          f"({num_correctly_associated / num_videos:.1%})" if num_videos else "n/a")
    print(f"Track fragmentation: {merged_eval.num_fragmentations} "
          f"(videos affected: {sum(1 for s in per_video_summaries if s['num_fragmentations'] > 0)}/{num_videos})")
    print(f"Incorrect merges / identity switches: {merged_eval.num_id_switches}  "
          f"[caveat: every video here has exactly 1 physical plate — this metric reads "
          f"~0 by construction, not evidence merging is solved; see module docstring]")
    print(f"Missed observations: {merged_eval.num_missed_detections}/{merged_eval.num_gt_observations} "
          f"({merged_eval.num_missed_detections / merged_eval.num_gt_observations:.1%})"
          if merged_eval.num_gt_observations else "n/a")
    print(f"Average observations per predicted track: {avg_obs_per_track:.2f}")
    print(f"False-positive predicted tracks: {merged_eval.num_false_positive_tracks}")
    print(f"Mean track purity: {merged_eval.mean_track_purity:.4f}")

    # --- reference baselines ---
    print()
    print("=" * 60)
    print("REFERENCE BASELINES (untuned, not optimized — Week 7 does the real comparison)")
    print("=" * 60)
    reference_summary = {}
    for method in REFERENCE_FUSION_METHODS:
        results = reference_results[method]
        n_ok = sum(1 for r in results if r.status == "ok")
        n_correct = sum(1 for r in results if r.status == "ok" and r.text == gt_text_by_global_track_id.get(r.track_id))
        n_total = len(results)
        acc = n_correct / n_total if n_total else None
        coverage = n_ok / n_total if n_total else None
        reference_summary[method] = {"n_total": n_total, "n_ok": n_ok, "n_correct": n_correct,
                                      "accuracy": acc, "coverage": coverage}
        print(f"{method}: accuracy={acc}, coverage={coverage} (n={n_total})")

    if skipped:
        print(f"\n{len(skipped)} track(s) skipped:")
        for s in skipped:
            print(f"  {s['track_id']}: {s['reason']}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "scope": "UFPR validation split (manifest_val_full.csv) — NOT test set, per Week 6 correction",
            "num_videos": num_videos,
            "association_quality": {
                "num_correctly_associated": num_correctly_associated,
                "correct_association_rate": num_correctly_associated / num_videos if num_videos else None,
                "num_fragmentations": merged_eval.num_fragmentations,
                "num_id_switches": merged_eval.num_id_switches,
                "num_missed_detections": merged_eval.num_missed_detections,
                "num_gt_observations": merged_eval.num_gt_observations,
                "num_false_positive_tracks": merged_eval.num_false_positive_tracks,
                "mean_track_purity": merged_eval.mean_track_purity,
                "avg_observations_per_predicted_track": avg_obs_per_track,
            },
            "per_video_summaries": per_video_summaries,
            "reference_baselines": reference_summary,
            "total_pipeline_time_ms": total_pipeline_ms,
            "total_frames_processed": total_frames,
            "skipped_tracks": skipped,
            "id_switch_caveat": "All UFPR videos here contain exactly one physical plate; id_switches/incorrect-merge metrics cannot be meaningfully exercised by this data and will read ~0 regardless of tracker quality on that dimension.",
        }, f, indent=2)
    print(f"\nWrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
