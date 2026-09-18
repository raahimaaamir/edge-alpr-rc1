"""
run_week8_multivehicle_review.py

Week 8, step 4: run the frozen pipeline + tracker on the user's own
recorded videos (multi-vehicle scenes UFPR structurally cannot provide,
since every UFPR video contains exactly one physical plate) and produce
something a human can actually review — this is deliberately NOT an
automatic accuracy script, since there's no ground truth for these videos.
Per the plan: "you do not need a large annotation project... we mainly
need enough examples to verify that the current tracker does not
incorrectly merge two different plates."

For each video, for each resulting track, this saves the crop from the
track's single highest-quality observation (via week7_selection's
composite score) to disk, plus a text summary of every distinct OCR
reading within that track. Open the saved crops and check:
  - do all crops within ONE track show the same physical plate?
    (if not: an incorrect merge — two different plates joined into one
    track)
  - does the SAME plate appear as two or more SEPARATE tracks in the same
    video? (if so: an incorrect split / identity switch — the tracker lost
    and re-acquired a plate it should have kept as one track, or two
    plates share overlapping trajectories that confused association)
  - are there vehicles with visible plates that produced NO track at all?
    (a complete miss)

Run from inside the alpr-train container:
    cd /workspace/home/alpr-week5/pipeline
    python3 run_week8_multivehicle_review.py

Crops and a summary JSON are written to:
    /workspace/home/alpr-week5/pipeline/week8_multivehicle_review/<video_name>/
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpr_pipeline import AlprPipeline
from tracker import IoUTracker, TrackerConfig
from track_types import FrameObservation
from stage3_config import load_config
from week7_selection import compute_composite_scores
from own_video_loader import extract_frames, list_videos

VIDEO_FOLDER = Path("/workspace/home/alpr-week5/data/raw/week1_videos")
OUTPUT_ROOT = Path(__file__).parent / "week8_multivehicle_review"
STRIDE = 2  # consistent with the Week 8 step-2 default; change here if you want a denser look


def main():
    config = load_config()
    pipeline = AlprPipeline()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    videos = list_videos(VIDEO_FOLDER)
    print(f"Found {len(videos)} video(s) in {VIDEO_FOLDER} (expect 7).")

    overall_summary = {}

    for video_path in videos:
        video_name = video_path.stem
        print(f"\n{'=' * 70}\n{video_name}\n{'=' * 70}")
        crop_dir = OUTPUT_ROOT / video_name
        crop_dir.mkdir(parents=True, exist_ok=True)

        frames = extract_frames(video_path, stride=STRIDE)
        print(f"  extracted {len(frames)} frames (stride={STRIDE})")

        tracker = IoUTracker(config.tracker)
        for frame_idx, timestamp, image in frames:
            image_id = f"frame{frame_idx:05d}"
            result = pipeline.process(image, save_crop_dir=str(crop_dir), image_id=image_id)
            observations = []
            if result.status != "no_detection":
                det_conf = getattr(result, "detector_confidence", None)
                if det_conf is None or det_conf >= config.detector_conf_thresh:
                    obs = FrameObservation.from_plate_result(frame_idx, timestamp, result)
                    obs.crop_path = result.crop_path  # not a FrameObservation field by default; attach for this review only
                    observations.append(obs)
            tracker.update(frame_idx, observations)

        tracks = tracker.finalize()
        print(f"  {len(tracks)} track(s) produced")

        video_summary = {"num_frames": len(frames), "num_tracks": len(tracks), "tracks": []}

        for track in tracks:
            usable = [o for o in track.observations if o.status == "ok" and o.ocr_text]
            if not usable:
                video_summary["tracks"].append({
                    "track_id": track.track_id, "num_observations": track.num_observations,
                    "note": "no usable OCR observations in this track", "distinct_readings": {},
                    "representative_crop": None,
                })
                continue

            scores = compute_composite_scores(track.observations)
            best_obs = max(usable, key=lambda o: scores.get(o.frame_idx, 0.0))

            readings = {}
            for o in usable:
                readings[o.ocr_text] = readings.get(o.ocr_text, 0) + 1

            frame_range = (min(o.frame_idx for o in track.observations), max(o.frame_idx for o in track.observations))
            print(f"    track {track.track_id}: frames {frame_range}, {len(usable)} usable obs, "
                  f"readings={readings}, representative crop={getattr(best_obs, 'crop_path', None)}")

            video_summary["tracks"].append({
                "track_id": track.track_id,
                "num_observations": track.num_observations,
                "num_usable_observations": len(usable),
                "frame_range": frame_range,
                "distinct_readings": readings,
                "representative_crop": getattr(best_obs, "crop_path", None),
            })

        overall_summary[video_name] = video_summary

    with open(OUTPUT_ROOT / "review_summary.json", "w") as f:
        json.dump(overall_summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Wrote crops + summary to {OUTPUT_ROOT}")
    print("Open review_summary.json and the saved crops. For each video, check:")
    print("  1. Does every track's crops show the SAME physical plate throughout? (else: incorrect merge)")
    print("  2. Does any single plate appear split across 2+ separate track_ids? (else: fragmentation/identity switch)")
    print("  3. Any vehicle with a visible plate that produced NO track at all? (complete miss)")


if __name__ == "__main__":
    main()
