"""
run_track0080_diagnostic.py

Detailed inspection of a single track's decision-layer evidence, using
the SAME frozen video pipeline (V1.1 recognizer, stride=2, tracker,
decide_rc1) as production. Originally built for track 0080 (per the
supervisor's request during the track-0080 investigation); the
--track-id argument makes it reusable for any track — e.g. for
inspecting why a test-set track abstained under the frozen RC1 system.
Prints every piece of evidence the decision layer had available:

  - all usable observations, with per-observation detector/OCR confidence,
    per-character confidences, and frame-quality (blur) score
  - the Top-3 composite-score selection (which 3 of the usable
    observations were chosen, and why — the composite score components)
  - whole-string vote counts among the Top-3
  - decide_rc1's accept/abstain decision and its exact reason (whether
    that's an agreement failure, a confidence failure, a plate-profile
    failure, or acceptance) — and whether the failure was
    detection-related (rescue-eligible) or not

This is diagnostic only — it doesn't change any code or any parameter,
and reading this on the test set is pure inspection, not tuning.

Run from inside the alpr-train container, from the pipeline/ directory:
    python3 run_track0080_diagnostic.py --v1-1-onnx-path <path>
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
from week7_selection import compute_composite_scores, select_top_k, usable_observations
from plate_normalize import normalize_plate_text
from decision_rule_rc1 import decide_rc1, AcceptanceConfig

MANIFEST_VAL_CSV = Path("/workspace/home/alpr-week5/data/interim/manifest_val_full.csv")
DEFAULT_CONFIG_PATH = "/workspace/home/alpr-week5/weights/cct_s_v2_global_plate_config.yaml"
STRIDE = 2
TRACK_ID = "0080"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-1-onnx-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--track-id", type=str, default=TRACK_ID)
    args = parser.parse_args()

    pipeline = AlprPipeline(recognizer=PlateRecognizer(model_path=args.v1_1_onnx_path, config_path=args.config_path))
    config = load_config()
    acceptance_config = AcceptanceConfig()

    loaded = load_track(args.track_id, ufpr_root=UFPR_ROOT_DEFAULT)
    all_frames = loaded["frames"]
    gt_text = loaded["gt_text"]
    strided_frames = all_frames[::STRIDE]

    video_pipeline = VideoAlprPipeline(pipeline, config, tracker_cls=IoUTracker)
    _tr, predicted_tracks, _stats = video_pipeline.process_video(strided_frames)

    pooled_observations = []
    for t in predicted_tracks:
        pooled_observations.extend(t.observations)

    print("=" * 90)
    print(f"TRACK {args.track_id} DIAGNOSTIC — ground truth: {gt_text!r}")
    print("=" * 90)
    print(f"Total pooled observations: {len(pooled_observations)}")

    usable = usable_observations(pooled_observations)
    print(f"Usable observations (status=='ok' and has text): {len(usable)}")
    print()
    print("All usable observations:")
    print(f"{'frame':>6} {'ocr_text':>10} {'normalized':>10} {'overall_conf':>13} {'detector_conf':>14} "
          f"{'blur_score':>11} {'plate_wxh':>10} {'char_confs'}")
    for o in usable:
        wxh = f"{o.plate_width_px:.0f}x{o.plate_height_px:.0f}" if o.plate_width_px else "?"
        char_confs_str = "[" + ", ".join(f"{c:.3f}" for c in (o.char_confs or [])) + "]"
        print(f"{o.frame_idx:>6} {o.ocr_text!r:>10} {normalize_plate_text(o.ocr_text)!r:>10} "
              f"{o.overall_conf:>13.4f} {o.detector_conf:>14.4f} {o.blur_score:>11.2f} {wxh:>10} {char_confs_str}")

    scores = compute_composite_scores(pooled_observations)
    selected = select_top_k(pooled_observations, scores, acceptance_config.top_k)

    print()
    print(f"Top-{acceptance_config.top_k} selected (by composite score, descending):")
    for o in selected:
        print(f"  frame={o.frame_idx}, composite_score={scores[o.frame_idx]:.4f}, "
              f"ocr_text={o.ocr_text!r}, overall_conf={o.overall_conf:.4f}, "
              f"detector_conf={o.detector_conf:.4f}, blur_score={o.blur_score:.2f}, "
              f"char_confs={[round(c, 3) for c in (o.char_confs or [])]}")

    counts = {}
    for o in selected:
        norm = normalize_plate_text(o.ocr_text)
        counts[norm] = counts.get(norm, 0) + 1
    print()
    print(f"Whole-string vote counts among Top-{acceptance_config.top_k}: {counts}")

    decision = decide_rc1(pooled_observations, acceptance_config)  # no profile: jurisdiction not wired at this level, matches production default
    print(f"Detection-related failure (rescue-eligible)? {decision.is_detection_related_failure()}")
    print()
    print("Current rule's decision:")
    print(f"  status={decision.status}, text={decision.text!r}, agreement_count={decision.agreement_count}, "
          f"reason={decision.reason!r}")
    if decision.status == "ok":
        winner_obs = [o for o in selected if normalize_plate_text(o.ocr_text) == decision.text]
        print()
        print(f"Why the current rule accepted {decision.text!r}: {decision.agreement_count} of "
              f"{len(selected)} selected observations voted for it "
              f"(>= min_agreement_count={acceptance_config.min_agreement_count}), from frames "
              f"{[o.frame_idx for o in winner_obs]}.")
        if gt_text and decision.text != normalize_plate_text(gt_text):
            print(f"  This is INCORRECT — ground truth is {gt_text!r}, not {decision.text!r}.")
            winning_char_confs = [o.char_confs for o in winner_obs if o.char_confs]
            if winning_char_confs:
                min_per_position = [min(vals) for vals in zip(*winning_char_confs)]
                print(f"  Per-position MINIMUM confidence across the winning votes: "
                      f"{[round(v, 3) for v in min_per_position]}")
                print(f"  (any low value here indicates the winning votes had weak evidence at that "
                      f"character position, despite unanimous agreement on the decoded character)")


if __name__ == "__main__":
    main()
