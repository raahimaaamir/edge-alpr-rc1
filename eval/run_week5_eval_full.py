"""
run_week5_eval_full.py

Extends run_week5_eval.py to cover the FULL held-out test set — UFPR frames
(via ufpr_video_loader.py) AND RodoSol images (via rodosol_loader.py) —
closing the scope gap noted in run_week5_eval.py's docstring. Same
standardized evaluate_single_frame() scoring, same "measurement only,
nothing in the frozen pipeline changes" spirit.

Run from inside the alpr-train container:
    cd /workspace/home/alpr-week5/pipeline
    python3 run_week5_eval_full.py

Writes results to week5_eval_full_results.json, and ALSO prints separate
per-dataset breakdowns (UFPR-only / RodoSol-only / combined) since the two
datasets have very different image characteristics (UFPR: sequential dashcam
frames; RodoSol: single toll-booth photos) and mixing them into one number
without also showing the breakdown would hide that.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpr_pipeline import AlprPipeline
from evaluate_single_frame import evaluate_single_frame, print_report
from ufpr_video_loader import get_test_track_ids, load_track, UFPR_ROOT_DEFAULT
from rodosol_loader import get_test_image_ids, load_image, RODOSOL_ROOT_DEFAULT

MANIFEST_TEST_CSV = Path("/workspace/home/alpr-week5/data/interim/manifest_test_full.csv")
OUTPUT_JSON = Path(__file__).parent / "week5_eval_full_results.json"


def run_ufpr(pipeline) -> tuple:
    results, gts = [], []
    skipped = []
    test_track_ids = sorted(get_test_track_ids(MANIFEST_TEST_CSV))
    print(f"UFPR: {len(test_track_ids)} held-out tracks (expect 23).")
    for track_id in test_track_ids:
        try:
            loaded = load_track(track_id, ufpr_root=UFPR_ROOT_DEFAULT)
        except (FileNotFoundError, ValueError, IOError) as e:
            print(f"  SKIPPING UFPR track {track_id}: {e}")
            skipped.append({"track_id": track_id, "reason": str(e)})
            continue
        gt_text = loaded["gt_text"]
        for frame_idx, _ts, image in loaded["frames"]:
            result = pipeline.process(image, image_id=f"ufpr_{track_id}_frame{frame_idx}")
            results.append(result)
            gts.append(gt_text)
    print(f"UFPR: {len(results)} frames evaluated, {len(skipped)} tracks skipped.")
    return results, gts, skipped


def run_rodosol(pipeline) -> tuple:
    results, gts = [], []
    skipped = []
    test_image_ids = sorted(get_test_image_ids(MANIFEST_TEST_CSV))
    print(f"RodoSol: {len(test_image_ids)} held-out images.")
    for image_id in test_image_ids:
        try:
            loaded = load_image(image_id, rodosol_root=RODOSOL_ROOT_DEFAULT)
        except (FileNotFoundError, ValueError, IOError) as e:
            skipped.append({"image_id": image_id, "reason": str(e)})
            continue
        result = pipeline.process(loaded["image"], image_id=f"rodosol_{image_id}")
        results.append(result)
        gts.append(loaded["gt_text"])
        if len(results) % 250 == 0:
            print(f"  ...{len(results)} RodoSol images done")
    print(f"RodoSol: {len(results)} images evaluated, {len(skipped)} skipped.")
    if skipped[:5]:
        print(f"  first few skip reasons: {skipped[:5]}")
    return results, gts, skipped


def main():
    pipeline = AlprPipeline()

    ufpr_results, ufpr_gts, ufpr_skipped = run_ufpr(pipeline)
    print()
    rodosol_results, rodosol_gts, rodosol_skipped = run_rodosol(pipeline)
    print()

    print("=" * 60)
    print("UFPR-only report:")
    ufpr_report = evaluate_single_frame(ufpr_results, ufpr_gts)
    print_report(ufpr_report)

    print()
    print("=" * 60)
    print("RodoSol-only report:")
    rodosol_report = evaluate_single_frame(rodosol_results, rodosol_gts)
    print_report(rodosol_report)

    print()
    print("=" * 60)
    print("COMBINED report (full held-out test set):")
    combined_results = ufpr_results + rodosol_results
    combined_gts = ufpr_gts + rodosol_gts
    combined_report = evaluate_single_frame(combined_results, combined_gts)
    print_report(combined_report)

    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "ufpr_report": ufpr_report.to_dict(),
            "rodosol_report": rodosol_report.to_dict(),
            "combined_report": combined_report.to_dict(),
            "ufpr_n_frames": len(ufpr_results),
            "rodosol_n_images": len(rodosol_results),
            "ufpr_skipped_tracks": ufpr_skipped,
            "rodosol_skipped_images": rodosol_skipped,
        }, f, indent=2)
    print(f"\nWrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
