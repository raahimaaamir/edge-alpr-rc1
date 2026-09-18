"""
run_quality_gate_derivation.py

Quantifies OCR correctness (per-observation, not per-track) as a function
of plate crop size, sharpness, and crop completeness, across all 22 UFPR
validation tracks — then derives a conservative operating threshold from
that data, per the supervisor's explicit instruction not to pick a pixel
threshold arbitrarily.

This is deliberately kept separate from plate-format validity
(plate_profile.py) and from the acceptance-rule comparison
(run_acceptance_rule_comparison.py) — the supervisor's point that track
0080 demonstrates a plate can PASS a quality gate and still fail OCR
means quality and reliability are separate concepts, and this script
only measures the quality side.

Quality dimensions measured per usable observation:
  - plate size: min(plate_width_px, plate_height_px) — the smaller of the
    two crop dimensions, since a plate can be long but very short
    (or vice versa) and either failure mode limits legible detail
  - sharpness: blur_score (Laplacian variance — higher is sharper,
    consistent with its use everywhere else in this project)
  - crop completeness: whether the detected bounding box touches the
    frame's edge (a proxy for the plate possibly being cut off — this is
    the one dimension not already stored per observation, so it's
    computed here from the original frame dimensions)

For each dimension, this reports the FULL accuracy-vs-threshold curve
(not just one derived number) — "if we required this metric to clear a
given value, what accuracy and coverage would result" — so the derived
threshold is transparent and inspectable, not asserted.

Derivation rule (stated plainly, not hidden in code): the recommended
threshold is the SMALLEST value at or above which per-observation
accuracy reaches TARGET_ACCURACY. This is one reasonable, conservative,
data-derived rule — the full curve is printed so a different target can
be chosen by inspection if TARGET_ACCURACY's choice doesn't seem right.

Run from inside the alpr-train container, from the pipeline/ directory:
    python3 run_quality_gate_derivation.py --v1-1-onnx-path <path>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.alpr_pipeline import AlprPipeline
from pipeline.recognizer import PlateRecognizer

from tracker import IoUTracker
from stage3_config import load_config
from ufpr_video_loader import get_track_ids, load_track, UFPR_ROOT_DEFAULT
from video_pipeline import VideoAlprPipeline
from week7_selection import usable_observations
from plate_normalize import normalize_plate_text

MANIFEST_VAL_CSV = Path("/workspace/home/alpr-week5/data/interim/manifest_val_full.csv")
DEFAULT_CONFIG_PATH = "/workspace/home/alpr-week5/weights/cct_s_v2_global_plate_config.yaml"
STRIDE = 2
TARGET_ACCURACY = 0.97  # stated derivation target, not hidden — see module docstring


def collect_observation_records(pipeline: AlprPipeline) -> list:
    """One record per usable observation across all 22 validation tracks
    — per-FRAME correctness, not per-track decision outcome, since the
    quality-vs-accuracy relationship needs to be measured before any
    temporal fusion or decision-layer logic is applied."""
    config = load_config()
    val_track_ids = sorted(get_track_ids(MANIFEST_VAL_CSV))
    records = []

    for track_id in val_track_ids:
        try:
            loaded = load_track(track_id, ufpr_root=UFPR_ROOT_DEFAULT)
        except (FileNotFoundError, ValueError, IOError) as e:
            print(f"  SKIPPING {track_id}: {e}")
            continue

        all_frames = loaded["frames"]
        gt_text = normalize_plate_text(loaded["gt_text"])
        strided_frames = all_frames[::STRIDE]
        frame_lookup = {fidx: img for fidx, _ts, img in strided_frames}

        video_pipeline = VideoAlprPipeline(pipeline, config, tracker_cls=IoUTracker)
        _tr, predicted_tracks, _stats = video_pipeline.process_video(strided_frames)
        pooled_observations = []
        for t in predicted_tracks:
            pooled_observations.extend(t.observations)

        for o in usable_observations(pooled_observations):
            frame_image = frame_lookup.get(o.frame_idx)
            edge_touch = None
            if frame_image is not None and o.bbox is not None:
                x1, y1, x2, y2 = o.bbox
                h, w = frame_image.shape[:2]
                edge_touch = (x1 <= 1 or y1 <= 1 or x2 >= w - 1 or y2 >= h - 1)

            min_dim = None
            if o.plate_width_px and o.plate_height_px:
                min_dim = min(o.plate_width_px, o.plate_height_px)

            records.append({
                "track_id": track_id,
                "frame_idx": o.frame_idx,
                "is_correct": normalize_plate_text(o.ocr_text) == gt_text,
                "min_dim_px": min_dim,
                "blur_score": o.blur_score,
                "edge_touch": edge_touch,
            })

    return records


def accuracy_at_or_above_threshold(records: list, key: str, threshold: float) -> tuple:
    """Returns (accuracy, coverage_count) for observations where
    record[key] >= threshold. coverage_count is the number of
    observations meeting the threshold (out of len(records))."""
    subset = [r for r in records if r[key] is not None and r[key] >= threshold]
    if not subset:
        return None, 0
    n_correct = sum(1 for r in subset if r["is_correct"])
    return n_correct / len(subset), len(subset)


def print_curve_and_derive(records: list, key: str, label: str, candidate_thresholds: list):
    n_total = len(records)
    print(f"\n--- {label} ({key}) ---")
    print(f"{'threshold':>12} {'accuracy (>= threshold)':>26} {'coverage':>10} {'n':>6}")
    derived = None
    for t in candidate_thresholds:
        acc, n = accuracy_at_or_above_threshold(records, key, t)
        if acc is None:
            continue
        coverage = n / n_total
        print(f"{t:>12.1f} {acc:>26.4f} {coverage:>10.4f} {n:>6}")
        if derived is None and acc >= TARGET_ACCURACY:
            derived = t
    if derived is not None:
        acc, n = accuracy_at_or_above_threshold(records, key, derived)
        print(f"DERIVED THRESHOLD for {label}: {derived} "
              f"(accuracy {acc:.4f} >= target {TARGET_ACCURACY}, coverage {n/n_total:.4f})")
    else:
        print(f"No threshold in the candidates tried reaches target accuracy {TARGET_ACCURACY} for {label} — "
              f"widen the candidate range or reconsider the target.")
    return derived


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-1-onnx-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-json", type=str, default="quality_gate_derivation.json")
    args = parser.parse_args()

    pipeline = AlprPipeline(recognizer=PlateRecognizer(model_path=args.v1_1_onnx_path, config_path=args.config_path))

    print("Collecting per-observation quality/correctness records across the 22 validation tracks...")
    records = collect_observation_records(pipeline)
    n_total = len(records)
    overall_acc = sum(1 for r in records if r["is_correct"]) / n_total if n_total else None
    print(f"Total usable observations: {n_total}, overall per-observation accuracy: {overall_acc:.4f}")

    # --- crop completeness (edge-touching), reported as a simple split, not a threshold curve ---
    touching = [r for r in records if r["edge_touch"] is True]
    not_touching = [r for r in records if r["edge_touch"] is False]
    acc_touching = sum(1 for r in touching if r["is_correct"]) / len(touching) if touching else None
    acc_not_touching = sum(1 for r in not_touching if r["is_correct"]) / len(not_touching) if not_touching else None
    print(f"\n--- Crop completeness (bbox touching frame edge) ---")
    print(f"Edge-touching crops:     n={len(touching)}, accuracy={acc_touching}")
    print(f"Non-edge-touching crops: n={len(not_touching)}, accuracy={acc_not_touching}")

    min_dims = sorted(set(r["min_dim_px"] for r in records if r["min_dim_px"] is not None))
    size_candidates = list(range(int(min(min_dims)), int(max(min_dims)) + 1, 5)) if min_dims else []
    derived_size = print_curve_and_derive(records, "min_dim_px", "Plate size (min dimension, px)", size_candidates)

    blur_scores = [r["blur_score"] for r in records if r["blur_score"] is not None]
    if blur_scores:
        lo, hi = min(blur_scores), max(blur_scores)
        step = (hi - lo) / 20 if hi > lo else 1
        blur_candidates = [lo + i * step for i in range(21)]
    else:
        blur_candidates = []
    derived_blur = print_curve_and_derive(records, "blur_score", "Sharpness (blur_score)", blur_candidates)

    summary = {
        "n_total_observations": n_total,
        "overall_accuracy": overall_acc,
        "target_accuracy": TARGET_ACCURACY,
        "edge_touching_accuracy": acc_touching,
        "non_edge_touching_accuracy": acc_not_touching,
        "derived_min_dim_threshold_px": derived_size,
        "derived_blur_score_threshold": derived_blur,
        "records": records,
    }
    import json
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
