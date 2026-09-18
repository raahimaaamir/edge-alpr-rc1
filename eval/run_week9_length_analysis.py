"""
run_week9_length_analysis.py

Week 9, step 2 driver. Runs the frozen single-frame pipeline over a small,
development-only folder of plate photos (and/or the existing 7 own videos'
majority readings) against manually-provided ground truth, and produces
the length/domain-bias report from plate_length_analysis.py.

SETUP: put your new photos in
    /workspace/home/alpr-week5/data/raw/week9_length_test/
and create a ground truth file next to them,
    /workspace/home/alpr-week5/data/raw/week9_length_test/ground_truth.json
as a simple {"filename.jpg": "TRUEPLATETEXT", ...} mapping — e.g.:
    {
      "plate01.jpg": "6ABC123",
      "plate02.jpg": "XYZ12"
    }

Run from inside the alpr-train container:
    cd /workspace/home/alpr-week5/pipeline
    python3 run_week9_length_analysis.py

This is deliberately small and manual (per the plan: "you do not need a
large annotation project") — no automatic loader/dataset needed, just a
folder of images and a hand-written ground-truth file.
"""

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpr_pipeline import AlprPipeline
from plate_length_analysis import analyze_set, print_report

PHOTO_DIR = Path("/workspace/home/alpr-week5/data/raw/week9_length_test")
GROUND_TRUTH_JSON = PHOTO_DIR / "ground_truth.json"
OUTPUT_JSON = Path(__file__).parent / "week9_length_analysis_results.json"

# The 7 own videos' already-established majority readings, included by
# default so this report combines everything available rather than
# starting over. Set to [] if you only want the new photos.
EXISTING_VIDEO_RESULTS = [
    ("CMK7507", "CMK7507"),   # video-01
    ("3BP6997", "38P697"),    # video-02 -- disputed length, see report
    ("CTY8283", "CTY8283"),   # video-03
    ("CSV4780", "CSV4780"),   # video-04
    ("BPM9818", "BPM9818"),   # video-05
    ("CYZ7703", "CYZ7703"),   # video-06
    (None, "BCP7506"),        # video-07 -- no stable reading
]


def main():
    pipeline = AlprPipeline()
    pairs = list(EXISTING_VIDEO_RESULTS)

    if GROUND_TRUTH_JSON.exists():
        ground_truth = json.loads(GROUND_TRUTH_JSON.read_text())
        print(f"Found {len(ground_truth)} new photo(s) with ground truth in {GROUND_TRUTH_JSON}")
        for filename, gt_text in ground_truth.items():
            image_path = PHOTO_DIR / filename
            if not image_path.exists():
                print(f"  SKIPPING {filename}: file not found at {image_path}")
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"  SKIPPING {filename}: cv2 failed to read it")
                continue
            result = pipeline.process(image, image_id=filename)
            predicted = result.plate_text if result.status == "ok" else None
            print(f"  {filename}: gt={gt_text}, predicted={predicted}, status={result.status}")
            pairs.append((predicted, gt_text))
    else:
        print(f"No ground_truth.json found at {GROUND_TRUTH_JSON} yet — reporting on the "
              f"existing 7 videos only. Add photos + ground_truth.json there and re-run "
              f"for a stronger, more length-diverse result.")

    report = analyze_set(pairs)
    print()
    print_report(report)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
