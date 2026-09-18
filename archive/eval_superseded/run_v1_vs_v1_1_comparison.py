"""
run_v1_vs_v1_1_comparison.py

Week 9, V1.1: evaluates Recognizer V1 (frozen) vs Recognizer V1.1
(candidate) separately on the four groups the plan specifies:
  1. 6-character development plates (from the V1.1 dataset's val split)
  2. 7-character development plates (from the V1.1 dataset's val split)
  3. the existing (original, UFPR/RodoSol) validation set
  4. the 13 local plates — the untouched, held-out final sanity check

For each group and each recognizer, reports: exact match, plate-length
accuracy (predicted length == ground-truth length), insertion/deletion
error counts, and character accuracy — using plate_length_analysis.py's
already-tested edit-distance machinery, unchanged.

This script swaps ONLY the recognizer's weights file between runs (via
AlprPipeline(recognizer=PlateRecognizer(weights_path=...))) — detector,
tracker, and all other pipeline code stay completely untouched, so any
difference in these numbers is attributable to the recognizer change
alone.

Run from inside the alpr-train container, once V1.1 has finished training
and been located (e.g. output/v1_1_finetune/<timestamp>/best.keras):
    cd /workspace/home/alpr-week5/pipeline
    python3 run_v1_vs_v1_1_comparison.py --v1-1-weights output/v1_1_finetune/<timestamp>/best.keras
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpr_pipeline import AlprPipeline
from pipeline.recognizer import PlateRecognizer
from plate_length_analysis import analyze_set, print_report

# The 13 local plates (7 videos + 6 photos) — untouched hold-out set
LOCAL_13_GROUND_TRUTH = {
    # 7 own videos (majority reading already established — but for a fair
    # V1-vs-V1.1 comparison, this script re-runs the recognizer on the
    # SAME representative crop per video, not the pre-computed majority;
    # see --local-crops-dir)
    "video-01": "CMK7507", "video-02": "38P697", "video-03": "CTY8283",
    "video-04": "CSV4780", "video-05": "BPM9818", "video-06": "CYZ7703", "video-07": "BCP7506",
    # 6 new photos
    "plate01": "PVV055", "plate02": "DJPT60", "plate03": "BPD845",
    "plate04": "NEL248", "plate05": "726NNS", "plate06": "242MMQ",
}


def load_dev_split_by_length(ledger_csv: Path, split: str = "val"):
    """Reads source_ledger.csv (from build_v1_1_training_manifest.py) and
    returns {6: [(image_path, plate_text), ...], 7: [...]} for the
    requested split, EXCLUDING synthetic rows so this measures real-image
    generalization, not just synthetic recall. (Change include_synthetic
    below if you want synthetic included too.)"""
    by_length = {6: [], 7: []}
    include_synthetic = False
    with open(ledger_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            if row["source"] == "synthetic" and not include_synthetic:
                continue
            length = int(row["length"])
            if length in by_length:
                by_length[length].append((row["image_path"], row["plate_text"]))
    return by_length


def evaluate_recognizer_on_pairs(pipeline: AlprPipeline, pairs: list) -> dict:
    """pairs: list of (image_path, ground_truth). Runs the FULL pipeline
    (detect+crop+quality+recognize) per image, then the length/domain
    analysis on the results."""
    predictions_with_gt = []
    for image_path, gt_text in pairs:
        image = cv2.imread(str(image_path))
        if image is None:
            predictions_with_gt.append((None, gt_text))
            continue
        result = pipeline.process(image)
        predicted = result.plate_text if result.status == "ok" else None
        predictions_with_gt.append((predicted, gt_text))
    return analyze_set(predictions_with_gt)


def evaluate_local_13(pipeline: AlprPipeline, local_crops_dir: Path) -> dict:
    predictions_with_gt = []
    for key, gt_text in LOCAL_13_GROUND_TRUTH.items():
        # expects one representative image per key, e.g. video-01.jpg / plate01.jpg
        candidates = list(local_crops_dir.glob(f"{key}.*"))
        if not candidates:
            predictions_with_gt.append((None, gt_text))
            continue
        image = cv2.imread(str(candidates[0]))
        if image is None:
            predictions_with_gt.append((None, gt_text))
            continue
        result = pipeline.process(image)
        predicted = result.plate_text if result.status == "ok" else None
        predictions_with_gt.append((predicted, gt_text))
    return analyze_set(predictions_with_gt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-weights", type=str, default="output/full_run1/2026-08-26_17-15-13/best.keras")
    parser.add_argument("--v1-1-weights", type=str, required=True)
    parser.add_argument("--ledger-csv", type=str, default="data/interim/v1_1_combined/source_ledger.csv")
    parser.add_argument("--existing-val-manifest", type=str, default="data/interim/manifest_val_full.csv")
    parser.add_argument("--local-crops-dir", type=str,
                         default="/workspace/home/alpr-week5/data/raw/week9_local_13_crops",
                         help="Directory with one representative image per local plate, named video-01.jpg ... plate06.jpg")
    parser.add_argument("--output-json", type=str, default="v1_vs_v1_1_comparison_results.json")
    args = parser.parse_args()

    print("Loading V1 (frozen)...")
    pipeline_v1 = AlprPipeline(recognizer=PlateRecognizer(weights_path=args.v1_weights))
    print("Loading V1.1 (candidate)...")
    pipeline_v1_1 = AlprPipeline(recognizer=PlateRecognizer(weights_path=args.v1_1_weights))

    dev_by_length = load_dev_split_by_length(Path(args.ledger_csv))

    results = {}
    for label, pipeline in [("V1", pipeline_v1), ("V1.1", pipeline_v1_1)]:
        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")

        print("\n-- 6-character development plates --")
        r6 = evaluate_recognizer_on_pairs(pipeline, dev_by_length[6])
        print_report(r6)

        print("\n-- 7-character development plates --")
        r7 = evaluate_recognizer_on_pairs(pipeline, dev_by_length[7])
        print_report(r7)

        print("\n-- 13 local plates (final sanity check) --")
        r_local = evaluate_local_13(pipeline, Path(args.local_crops_dir))
        print_report(r_local)

        results[label] = {"six_char_dev": r6, "seven_char_dev": r7, "local_13": r_local}

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.output_json}")
    print("\nNOTE: the 'existing validation set' comparison (UFPR/RodoSol, "
          "manifest_val_full.csv) should be run via the existing frozen "
          "evaluate_single_frame.py / run_week5_eval_full.py machinery with "
          "the recognizer swapped, to stay consistent with every other "
          "week's numbers on that set — not duplicated here.")


if __name__ == "__main__":
    main()
