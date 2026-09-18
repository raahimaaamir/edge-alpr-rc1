"""
diagnose_plate_size.py

One-off diagnostic, not part of the frozen pipeline: checks whether small
ground-truth plate size (in pixels) correlates with the detector-miss
pattern seen on tracks 0064, 0091, 0101 in the Week 6 run. Uses ONLY the
ground-truth bboxes from the annotation files (via ufpr_video_loader) —
not detector output — so plate size here is what was actually in the
frame, independent of whether the detector found it.

Run from inside the alpr-train container:
    cd /workspace/home/alpr-week5/pipeline
    python3 diagnose_plate_size.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ufpr_video_loader import get_track_ids, load_track, UFPR_ROOT_DEFAULT

MANIFEST_VAL_CSV = Path("/workspace/home/alpr-week5/data/interim/manifest_val_full.csv")
WEEK6_RESULTS_JSON = Path(__file__).parent / "week6_tracking_results.json"


def mean_gt_plate_size(gt_boxes_per_frame: dict) -> tuple:
    """Returns (mean_width_px, mean_height_px, mean_diag_px) over every
    ground-truth box across all frames in this track."""
    widths, heights = [], []
    for boxes in gt_boxes_per_frame.values():
        for bbox, _gt_id in boxes:
            x1, y1, x2, y2 = bbox
            widths.append(x2 - x1)
            heights.append(y2 - y1)
    if not widths:
        return (None, None, None)
    mean_w = sum(widths) / len(widths)
    mean_h = sum(heights) / len(heights)
    mean_diag = (mean_w ** 2 + mean_h ** 2) ** 0.5
    return (mean_w, mean_h, mean_diag)


def main():
    # pull miss counts from the Week 6 run already on disk, so this
    # doesn't require re-running the real pipeline
    week6 = json.load(open(WEEK6_RESULTS_JSON))
    miss_by_track = {s["track_id"]: s["num_missed_detections"] for s in week6["per_video_summaries"]}
    correctly_associated_by_track = {s["track_id"]: s["correctly_associated"] for s in week6["per_video_summaries"]}

    val_track_ids = sorted(get_track_ids(MANIFEST_VAL_CSV))
    rows = []
    for track_id in val_track_ids:
        try:
            loaded = load_track(track_id, ufpr_root=UFPR_ROOT_DEFAULT)
        except (FileNotFoundError, ValueError, IOError) as e:
            print(f"  SKIPPING {track_id}: {e}")
            continue
        mean_w, mean_h, mean_diag = mean_gt_plate_size(loaded["gt_boxes_per_frame"])
        rows.append({
            "track_id": track_id,
            "mean_plate_width_px": round(mean_w, 1) if mean_w else None,
            "mean_plate_height_px": round(mean_h, 1) if mean_h else None,
            "mean_plate_diag_px": round(mean_diag, 1) if mean_diag else None,
            "num_missed_detections": miss_by_track.get(track_id),
            "correctly_associated": correctly_associated_by_track.get(track_id),
        })

    rows.sort(key=lambda r: r["mean_plate_diag_px"] or 0)

    print(f"{'track':<8}{'width_px':<11}{'height_px':<12}{'diag_px':<10}{'missed':<9}{'correctly_assoc'}")
    for r in rows:
        print(f"{r['track_id']:<8}{r['mean_plate_width_px']:<11}{r['mean_plate_height_px']:<12}"
              f"{r['mean_plate_diag_px']:<10}{r['num_missed_detections']:<9}{r['correctly_associated']}")

    # quick correlation summary: split into "small plate" (below median diag)
    # vs "large plate" (above median), compare mean miss count
    diags = [r["mean_plate_diag_px"] for r in rows if r["mean_plate_diag_px"] is not None]
    diags.sort()
    median_diag = diags[len(diags) // 2]
    small = [r for r in rows if r["mean_plate_diag_px"] is not None and r["mean_plate_diag_px"] < median_diag]
    large = [r for r in rows if r["mean_plate_diag_px"] is not None and r["mean_plate_diag_px"] >= median_diag]
    mean_miss_small = sum(r["num_missed_detections"] or 0 for r in small) / len(small) if small else None
    mean_miss_large = sum(r["num_missed_detections"] or 0 for r in large) / len(large) if large else None

    print()
    print(f"Median plate diagonal across validation set: {median_diag:.1f}px")
    print(f"Below-median-size tracks (n={len(small)}): mean missed detections = {mean_miss_small:.2f}/30")
    print(f"Above-median-size tracks (n={len(large)}): mean missed detections = {mean_miss_large:.2f}/30")
    print()
    print("Tracks flagged in the earlier discussion (0064, 0091, 0101):")
    for tid in ("0064", "0091", "0101"):
        r = next((r for r in rows if r["track_id"] == tid), None)
        if r:
            rank = rows.index(r) + 1
            print(f"  {tid}: diag={r['mean_plate_diag_px']}px (rank {rank}/{len(rows)}, "
                  f"1=smallest), missed={r['num_missed_detections']}/30")


if __name__ == "__main__":
    main()
