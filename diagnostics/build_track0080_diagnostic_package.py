"""
build_track0080_diagnostic_package.py

Builds the full diagnostic package for track 0080 that the supervisor
requested, before any decision-rule change. This is diagnostic only —
no correction of any kind is applied for this specific plate.

For each of the Top-3 observations actually used by the decision layer:
  - the original-resolution crop, exactly as the pipeline cropped it
    (re-extracted from the original frame using the SAME bbox-clamping
    logic as alpr_pipeline.py, via the bbox already stored on the
    observation — no re-detection needed)
  - the crop exactly as it enters Recognizer V1.1 after preprocessing
    (resize + RGB conversion, via recognizer.preprocess(), saved as a
    viewable image)
  - bounding-box coordinates and crop width/height in pixels
  - raw OCR string (from ocr_text_raw — NOT ocr_text, which is already
    normalized), overall confidence, per-character confidence, and the
    top-3 character probabilities at EVERY decoder slot position
    (obtained by calling the ONNX session directly and reading the full
    softmax array, bypassing recognize()'s decode-only return — the
    model's own reported alternatives, not anything derived or guessed)
  - detector confidence, blur/sharpness score, brightness, exposure flags

For all 15 usable observations: the full OCR output / vote-count table.

Everything is written to one output folder, plus a JSON summary and a
CSV vote table, per the supervisor's request.

Run from inside the alpr-train container, from the pipeline/ directory:
    python3 build_track0080_diagnostic_package.py --v1-1-onnx-path <path>
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.alpr_pipeline import AlprPipeline
from pipeline.recognizer import PlateRecognizer

from tracker import IoUTracker
from stage3_config import load_config
from ufpr_video_loader import load_track, UFPR_ROOT_DEFAULT
from video_pipeline import VideoAlprPipeline
from week7_selection import compute_composite_scores, select_top_k, usable_observations
from plate_normalize import normalize_plate_text
from week9_decision_rule import decide_v2, AcceptanceConfig

DEFAULT_CONFIG_PATH = "/workspace/home/alpr-week5/weights/cct_s_v2_global_plate_config.yaml"
STRIDE = 2
TRACK_ID = "0080"
OUTPUT_DIR = Path("/workspace/home/alpr-week5/data/interim/track0080_diagnostic_package")


def crop_from_bbox(image: np.ndarray, bbox: tuple) -> np.ndarray:
    """Exactly replicates alpr_pipeline.py's own crop logic (bbox clamped
    to image bounds), so this reproduces precisely what the real pipeline
    used — not an approximation."""
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(image.shape[1], int(x2)), min(image.shape[0], int(y2))
    return image[y1:y2, x1:x2]


def raw_topk_probs(recognizer: PlateRecognizer, crop_bgr: np.ndarray, k: int = 3) -> list:
    """Calls the ONNX session directly (bypassing recognize()'s
    decode-only return) to get the full per-slot softmax array, and
    returns the top-k (char, probability) pairs at EVERY slot position —
    the model's own reported alternatives, including the pad character
    where relevant, so nothing is hidden."""
    x = np.expand_dims(recognizer.preprocess(crop_bgr), axis=0)
    outputs = recognizer.session.run(None, {recognizer.input_name: x})
    plate_pred = outputs[recognizer.plate_output_idx][0]  # [10, 37]

    per_slot = []
    for slot_idx, slot in enumerate(plate_pred):
        top_indices = np.argsort(slot)[::-1][:k]
        top_k = [(recognizer.alphabet[i], float(slot[i])) for i in top_indices]
        per_slot.append({"slot": slot_idx, "top_k": top_k})
    return per_slot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-1-onnx-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--track-id", type=str, default=TRACK_ID)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recognizer = PlateRecognizer(model_path=args.v1_1_onnx_path, config_path=args.config_path)
    pipeline = AlprPipeline(recognizer=recognizer)
    config = load_config()
    acceptance_config = AcceptanceConfig()

    loaded = load_track(args.track_id, ufpr_root=UFPR_ROOT_DEFAULT)
    all_frames = loaded["frames"]
    gt_text = loaded["gt_text"]
    strided_frames = all_frames[::STRIDE]
    frame_lookup = {fidx: img for fidx, _ts, img in strided_frames}

    video_pipeline = VideoAlprPipeline(pipeline, config, tracker_cls=IoUTracker)
    _tr, predicted_tracks, _stats = video_pipeline.process_video(strided_frames)
    pooled_observations = []
    for t in predicted_tracks:
        pooled_observations.extend(t.observations)

    usable = usable_observations(pooled_observations)
    scores = compute_composite_scores(pooled_observations)
    selected = select_top_k(pooled_observations, scores, acceptance_config.top_k)
    decision = decide_v2(pooled_observations, acceptance_config)

    # --- all 15 usable observations: full OCR/vote table ---
    all_observations_table = []
    for o in usable:
        all_observations_table.append({
            "frame_idx": o.frame_idx,
            "raw_ocr_text": o.ocr_text_raw,
            "normalized_ocr_text": o.ocr_text,
            "overall_conf": o.overall_conf,
            "detector_conf": o.detector_conf,
            "blur_score": o.blur_score,
            "mean_brightness": o.mean_brightness,
            "is_overexposed": o.is_overexposed,
            "is_underexposed": o.is_underexposed,
            "plate_width_px": o.plate_width_px,
            "plate_height_px": o.plate_height_px,
            "char_confs": o.char_confs,
            "in_top3": o.frame_idx in [s.frame_idx for s in selected],
        })

    vote_counts = {}
    for o in selected:
        vote_counts[o.ocr_text] = vote_counts.get(o.ocr_text, 0) + 1

    # --- Top-3 crops + preprocessing + raw model output ---
    top3_details = []
    for o in selected:
        frame_image = frame_lookup.get(o.frame_idx)
        if frame_image is None:
            print(f"  WARNING: no original frame found for frame_idx={o.frame_idx}")
            continue

        original_crop = crop_from_bbox(frame_image, o.bbox)
        original_path = output_dir / f"frame{o.frame_idx}_original_{original_crop.shape[1]}x{original_crop.shape[0]}.jpg"
        cv2.imwrite(str(original_path), original_crop)

        preprocessed_rgb_float = recognizer.preprocess(original_crop)  # RGB, float32, resized
        preprocessed_bgr_uint8 = cv2.cvtColor(preprocessed_rgb_float.astype(np.uint8), cv2.COLOR_RGB2BGR)
        preprocessed_path = output_dir / f"frame{o.frame_idx}_preprocessed_{recognizer.img_w}x{recognizer.img_h}.jpg"
        cv2.imwrite(str(preprocessed_path), preprocessed_bgr_uint8)

        per_slot_topk = raw_topk_probs(recognizer, original_crop, k=3)

        top3_details.append({
            "frame_idx": o.frame_idx,
            "bbox": list(o.bbox) if o.bbox else None,
            "original_crop_path": str(original_path),
            "original_crop_width_px": original_crop.shape[1],
            "original_crop_height_px": original_crop.shape[0],
            "preprocessed_crop_path": str(preprocessed_path),
            "preprocessed_size": f"{recognizer.img_w}x{recognizer.img_h}",
            "raw_ocr_text": o.ocr_text_raw,
            "overall_conf": o.overall_conf,
            "per_char_conf": o.char_confs,
            "detector_conf": o.detector_conf,
            "blur_score": o.blur_score,
            "mean_brightness": o.mean_brightness,
            "is_overexposed": o.is_overexposed,
            "is_underexposed": o.is_underexposed,
            "composite_score": scores.get(o.frame_idx),
            "top_k_probabilities_per_slot": per_slot_topk,
        })

    # --- jurisdiction / plate-type note ---
    # UFPR-ALPR is a Brazilian license plate dataset (Universidade Federal
    # do Paraná, Curitiba). Per the dataset's originating paper and a
    # follow-up analysis of it (Laroca et al., IJCNN 2018; cited
    # secondary source discussing the same dataset), plates in Paraná
    # (where this dataset was collected) follow the older Brazilian
    # standard format — 3 letters + 4 digits (LLL-DDDD) — and range from
    # AAA-0001 to BEZ-9999, meaning the leading letter is documented to be
    # A or B. Ground truth 'APJ3829' matches this format exactly (3
    # letters, 4 digits) and its leading letter 'A' falls inside the
    # documented AAA-BEZ range. This is the dataset's documented format,
    # independent of this specific track — not inferred from this plate.
    jurisdiction_note = (
        "UFPR-ALPR dataset (Brazil, Universidade Federal do Paraná, Curitiba). "
        "Plates in this dataset (collected in Paraná) follow the older "
        "Brazilian standard format: 3 letters + 4 digits (LLL-DDDD), "
        "documented to range from AAA-0001 to BEZ-9999. Ground truth "
        "'APJ3829' matches this format exactly and its leading letter "
        "falls inside the documented range."
    )

    summary = {
        "track_id": args.track_id,
        "ground_truth": gt_text,
        "jurisdiction_note": jurisdiction_note,
        "n_usable_observations": len(usable),
        "top3_vote_counts": vote_counts,
        "decision": decision.to_dict(),
        "top3_details": top3_details,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(output_dir / "all_observations.csv", "w", newline="") as f:
        fieldnames = list(all_observations_table[0].keys()) if all_observations_table else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_observations_table)

    print(f"Wrote diagnostic package to {output_dir}")
    print(f"  - summary.json")
    print(f"  - all_observations.csv")
    print(f"  - {len(top3_details) * 2} crop images (original + preprocessed, x{len(top3_details)} Top-3 frames)")


if __name__ == "__main__":
    main()
