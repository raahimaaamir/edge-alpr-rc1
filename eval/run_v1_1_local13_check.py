"""
run_v1_1_local13_check.py

The FINAL, one-time independent comparison of Recognizer V1 (frozen) vs
V1.1 (candidate) on the 13 reserved local plates — per the supervisor's
explicit instruction, this is run ONCE, after V1.1 has already been
selected from the internal validation results (run_v1_1_validation_eval.py),
never used to pick epochs or tune anything.

This is intentionally narrow: the grouped 6-char/7-char/real/retained-legacy
comparison already happened correctly in run_v1_1_validation_eval.py,
which calls PlateRecognizer.recognize() directly on pre-cropped images.
The 13 local plates are full SCENE photos/frames, so they need the full
AlprPipeline (detect -> crop -> quality -> recognize), not the recognizer
alone — that is the only reason this separate script exists.

Import note: alpr_pipeline.py's own internal imports
("from pipeline.detector import ...", etc.) mean it must be imported as
part of a `pipeline` package from ONE LEVEL ABOVE pipeline/ — i.e. this
script must be run with /workspace/home/alpr-week5 on sys.path, using
`from pipeline.alpr_pipeline import AlprPipeline`, not a flat import.
(An earlier draft of this script got this wrong — flat-imported
alpr_pipeline while only adding the parent dir to sys.path, which does
not match how alpr_pipeline.py imports its own sibling modules.)

PlateRecognizer loads a model via onnxruntime.InferenceSession, so both
--v1-onnx-path and --v1-1-onnx-path must point at EXPORTED ONNX files,
never a .keras checkpoint.

Run from inside the alpr-train container, from /workspace/home/alpr-week5:
    python3 pipeline/run_v1_1_local13_check.py \
        --v1-1-onnx-path output/v1_1_finetune_balanced/onnx/best.onnx
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

# alpr_pipeline.py imports its siblings as `pipeline.xxx` — put the
# directory ABOVE pipeline/ on sys.path so that resolves correctly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.alpr_pipeline import AlprPipeline
from pipeline.recognizer import PlateRecognizer
from plate_length_analysis import analyze_set, print_report

DEFAULT_V1_ONNX = "/workspace/home/alpr-week5/output/full_run1/onnx/best.onnx"
DEFAULT_CONFIG_PATH = "/workspace/home/alpr-week5/weights/cct_s_v2_global_plate_config.yaml"


def load_ground_truth(gt_json_path: Path) -> dict:
    """{key: plate_text} for the 13 local plates — key matches the image
    filename stem (e.g. 'video-01', 'plate06')."""
    with open(gt_json_path) as f:
        return json.load(f)


def evaluate_local_13(pipeline: AlprPipeline, crops_dir: Path, ground_truth: dict) -> dict:
    predictions_with_gt = []
    for key, gt_text in ground_truth.items():
        candidates = list(crops_dir.glob(f"{key}.*"))
        if not candidates:
            print(f"  WARNING: no image found for '{key}' in {crops_dir}")
            predictions_with_gt.append((None, gt_text))
            continue
        image = cv2.imread(str(candidates[0]))
        if image is None:
            print(f"  WARNING: failed to load {candidates[0]}")
            predictions_with_gt.append((None, gt_text))
            continue
        result = pipeline.process(image, image_id=key)
        predicted = result.plate_text if result.status == "ok" else None
        predictions_with_gt.append((predicted, gt_text))
    return analyze_set(predictions_with_gt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-onnx-path", type=str, default=DEFAULT_V1_ONNX)
    parser.add_argument("--v1-1-onnx-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--local-crops-dir", type=str,
                         default="/workspace/home/alpr-week5/data/raw/week9_local_13_crops")
    parser.add_argument("--ground-truth-json", type=str,
                         default="/workspace/home/alpr-week5/data/raw/week9_local_13_crops/ground_truth.json")
    parser.add_argument("--output-json", type=str, default="v1_1_local13_results.json")
    args = parser.parse_args()

    ground_truth = load_ground_truth(Path(args.ground_truth_json))
    print(f"Loaded ground truth for {len(ground_truth)} local plates")

    print(f"Loading V1 (frozen) from {args.v1_onnx_path}")
    pipeline_v1 = AlprPipeline(recognizer=PlateRecognizer(model_path=args.v1_onnx_path, config_path=args.config_path))
    print(f"Loading V1.1 (candidate) from {args.v1_1_onnx_path}")
    pipeline_v1_1 = AlprPipeline(recognizer=PlateRecognizer(model_path=args.v1_1_onnx_path, config_path=args.config_path))

    results = {}
    for label, pipeline in [("V1", pipeline_v1), ("V1.1", pipeline_v1_1)]:
        print(f"\n{'=' * 60}\n{label} — 13 local plates (final, one-time check)\n{'=' * 60}")
        report = evaluate_local_13(pipeline, Path(args.local_crops_dir), ground_truth)
        print_report(report)
        results[label] = report

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
