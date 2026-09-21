"""
run_examples.py

Runs RC1 (the frozen live pipeline) on the 13 representative example
images in examples/inputs/ and compares each prediction against the
known ground truth in examples/ground_truth.json.

These are the project's own collected images (6 photos + 7 video
frames) — NOT UFPR-ALPR data, which cannot be redistributed under its
license agreement. This is exactly why these 13 exist: they're the
only real-world images in this whole project safe to ship alongside
the code as a "does this actually work" sanity check for whoever
receives this package.

Run from the examples/ directory, or from the package root:
    python3 examples/run_examples.py
"""

import json
import sys
from pathlib import Path

# Reach pipeline/ regardless of where this script is invoked from —
# same pattern as tests/test_stage3_synthetic.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from pipeline.alpr_pipeline import AlprPipeline

EXAMPLES_DIR = Path(__file__).resolve().parent
INPUTS_DIR = EXAMPLES_DIR / "inputs"
GROUND_TRUTH_PATH = EXAMPLES_DIR / "ground_truth.json"


def main():
    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = json.load(f)

    pipeline = AlprPipeline()  # frozen V1.1 recognizer, loaded automatically

    results = []
    n_correct = 0
    print(f"Running RC1 on {len(ground_truth)} representative example images...\n")

    for key, gt_text in sorted(ground_truth.items()):
        candidates = list(INPUTS_DIR.glob(f"{key}.*"))
        if not candidates:
            print(f"  {key}: MISSING input file, skipped")
            continue
        image = cv2.imread(str(candidates[0]))
        if image is None:
            print(f"  {key}: FAILED to load image")
            continue

        result = pipeline.process(image, image_id=key)
        predicted = result.plate_text if result.status == "ok" else None
        correct = predicted == gt_text
        n_correct += int(correct)

        status_str = "MATCH" if correct else ("MISMATCH" if predicted else "NO_RESULT")
        print(f"  {key}: predicted={predicted!r:>10}  ground_truth={gt_text!r:>10}  [{status_str}]")

        results.append({
            "key": key, "ground_truth": gt_text, "predicted": predicted,
            "status": result.status, "correct": correct,
        })

    print(f"\n{n_correct}/{len(results)} correct.")
    print("\nNote: this uses the single-image pipeline (no multi-frame tracking/")
    print("fusion/reliability layer, since these are individual example images,")
    print("not video tracks) — so results may differ from the full video-pipeline")
    print("numbers reported in the model card, which pool multiple observations")
    print("per vehicle before deciding.")

    output_path = EXAMPLES_DIR / "example_run_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
