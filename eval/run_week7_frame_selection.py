"""
run_week7_frame_selection.py

Week 7: for each of the 22 UFPR validation tracks (physical plates), pool
ALL observations that video produced — regardless of how many predicted
tracks the Week 6 tracker split it into — and compare frame-selection sizes
(Top-1, Top-3, Top-5, all) x fusion methods (best_frame, whole_string_
majority, per_char_majority, confidence_weighted_char_quality).

Per this week's correction: evaluation is at the ENCOUNTER level — each of
the 22 physical plate tracks contributes exactly one result per (K, method)
cell, never more. A track that fragmented into 2 predicted tracks does NOT
become 2 evaluation samples: its observations are pooled back into one
encounter before selection/fusion. A track with zero detections (0064) is
recorded as a distinct "complete_detection_failure" outcome for every cell
— not silently dropped, and not folded into ordinary abstention counts.
Predicted-track-level structure (fragmentation, missed detections) is
still available from Week 6's output and is not recomputed here — this
driver's job is the frame-selection/fusion question, on the fixed
encounter-level evaluation set.

Nothing about the detector, recognizer, or tracker changes this week —
this driver only re-uses their already-recorded outputs (observations)
differently.

Run from inside the alpr-train container:
    cd /workspace/home/alpr-week5/pipeline
    python3 run_week7_frame_selection.py

Writes results to week7_frame_selection_results.json.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpr_pipeline import AlprPipeline
from tracker import IoUTracker
from fusion import fuse, fuse_confidence_weighted_char_quality, FusionConfig
from stage3_config import load_config
from ufpr_video_loader import get_track_ids, load_track, UFPR_ROOT_DEFAULT
from video_pipeline import VideoAlprPipeline
from track_types import Track
from week7_selection import compute_composite_scores, select_top_k, usable_observations
from plate_normalize import normalize_plate_text

MANIFEST_VAL_CSV = Path("/workspace/home/alpr-week5/data/interim/manifest_val_full.csv")
OUTPUT_JSON = Path(__file__).parent / "week7_frame_selection_results.json"

TOP_K_VALUES = [1, 3, 5, None]  # None = "all"
SIMPLE_METHODS = ("best_single_frame", "majority_vote", "per_char_majority")  # existing fuse() dispatch
QUALITY_WEIGHTED_METHOD = "confidence_weighted_char_quality"  # needs quality_weights, called separately

COMPLETE_DETECTION_FAILURE = "complete_detection_failure"


def k_label(k):
    return "all" if k is None else f"top{k}"


def evaluate_cell(pooled_observations, scores, k, method, gt_text, fusion_config):
    """Returns a dict: {status, correct, text, num_frames_used}."""
    selected = select_top_k(pooled_observations, scores, k)
    if not selected:
        return {"status": COMPLETE_DETECTION_FAILURE, "correct": False, "text": None, "num_frames_used": 0}

    ephemeral_track = Track(track_id=-1, observations=selected)
    if method == QUALITY_WEIGHTED_METHOD:
        result = fuse_confidence_weighted_char_quality(ephemeral_track, quality_weights=scores, config=fusion_config)
    else:
        result = fuse(ephemeral_track, method=method, config=fusion_config)

    if result.status == "ok":
        correct = normalize_plate_text(result.text) == normalize_plate_text(gt_text)
        return {"status": "ok", "correct": correct, "text": result.text, "num_frames_used": result.num_frames_used}
    else:
        return {"status": "no_reliable_result", "correct": False, "text": None, "num_frames_used": 0}


def main():
    config = load_config()
    fusion_config = FusionConfig()  # untouched defaults — no tuning this week
    pipeline = AlprPipeline()
    all_methods = list(SIMPLE_METHODS) + [QUALITY_WEIGHTED_METHOD]

    val_track_ids = sorted(get_track_ids(MANIFEST_VAL_CSV))
    print(f"Found {len(val_track_ids)} UFPR validation tracks (expect 22).")

    # cell_results[(k_label, method)] = list of per-encounter dicts
    cell_results = {(k_label(k), m): [] for k in TOP_K_VALUES for m in all_methods}
    latency_estimates = {(k_label(k), m): [] for k in TOP_K_VALUES for m in all_methods}

    num_complete_failures = 0
    num_encounters = 0
    skipped = []

    for ufpr_track_id in val_track_ids:
        try:
            loaded = load_track(ufpr_track_id, ufpr_root=UFPR_ROOT_DEFAULT)
        except (FileNotFoundError, ValueError, IOError) as e:
            print(f"  SKIPPING track {ufpr_track_id}: {e}")
            skipped.append({"track_id": ufpr_track_id, "reason": str(e)})
            continue

        frames = loaded["frames"]
        gt_text = loaded["gt_text"]
        print(f"  track {ufpr_track_id}: {len(frames)} frames, gt='{gt_text}'")
        num_encounters += 1

        video_pipeline = VideoAlprPipeline(pipeline, config, tracker_cls=IoUTracker)
        _track_results, predicted_tracks, stats = video_pipeline.process_video(frames)

        # pool ALL observations across however many predicted tracks this
        # video produced (fragmentation doesn't lose observations — see
        # module docstring) into one encounter
        pooled_observations = []
        for t in predicted_tracks:
            pooled_observations.extend(t.observations)
        pooled_usable = usable_observations(pooled_observations)

        if not pooled_usable:
            num_complete_failures += 1

        scores = compute_composite_scores(pooled_observations)

        # mean per-stage latency for THIS video's usable observations, for
        # the latency estimate below
        detect_ms_vals = [o.stage_timings_ms.get("detect_ms", 0.0) for o in pooled_usable if o.stage_timings_ms]
        crop_ms_vals = [o.stage_timings_ms.get("crop_ms", 0.0) for o in pooled_usable if o.stage_timings_ms]
        quality_ms_vals = [o.stage_timings_ms.get("quality_ms", 0.0) for o in pooled_usable if o.stage_timings_ms]
        recognize_ms_vals = [o.stage_timings_ms.get("recognize_ms", 0.0) for o in pooled_usable if o.stage_timings_ms]
        mean_detect_ms = sum(detect_ms_vals) / len(detect_ms_vals) if detect_ms_vals else 0.0
        mean_crop_ms = sum(crop_ms_vals) / len(crop_ms_vals) if crop_ms_vals else 0.0
        mean_quality_ms = sum(quality_ms_vals) / len(quality_ms_vals) if quality_ms_vals else 0.0
        mean_recognize_ms = sum(recognize_ms_vals) / len(recognize_ms_vals) if recognize_ms_vals else 0.0
        # unavoidable per-frame cost regardless of K: detection+crop+quality
        # must run on every candidate frame to know which ones are good
        # enough to rank/select in the first place
        fixed_cost_all_frames_ms = len(frames) * (mean_detect_ms + mean_crop_ms + mean_quality_ms)

        for k in TOP_K_VALUES:
            for method in all_methods:
                key = (k_label(k), method)
                if not pooled_usable:
                    cell_results[key].append({"status": COMPLETE_DETECTION_FAILURE, "correct": False,
                                               "text": None, "num_frames_used": 0,
                                               "track_id": ufpr_track_id, "gt_text": gt_text})
                    latency_estimates[key].append(fixed_cost_all_frames_ms)  # detection still ran on all frames
                    continue
                outcome = evaluate_cell(pooled_observations, scores, k, method, gt_text, fusion_config)
                outcome["track_id"] = ufpr_track_id
                outcome["gt_text"] = gt_text
                cell_results[key].append(outcome)
                # OCR (recognize) cost only for the frames actually used;
                # detect/crop/quality cost is fixed regardless of K (see above)
                estimated_ms = fixed_cost_all_frames_ms + outcome["num_frames_used"] * mean_recognize_ms
                latency_estimates[key].append(estimated_ms)

    # --- aggregate per cell ---
    print()
    print("=" * 100)
    print(f"WEEK 7 — FRAME SELECTION x FUSION METHOD (UFPR validation, n={num_encounters} encounters, "
          f"{num_complete_failures} complete detection failures)")
    print("=" * 100)
    header = f"{'K':<8}{'method':<32}{'accuracy':<12}{'coverage':<12}{'incorrect':<12}{'avg_frames':<12}{'est_latency_ms'}"
    print(header)
    summary_table = []
    for k in TOP_K_VALUES:
        for method in all_methods:
            key = (k_label(k), method)
            outcomes = cell_results[key]
            n = len(outcomes)
            n_correct = sum(1 for o in outcomes if o["status"] == "ok" and o["correct"])
            n_ok = sum(1 for o in outcomes if o["status"] == "ok")
            n_incorrect_accepted = sum(1 for o in outcomes if o["status"] == "ok" and not o["correct"])
            n_failures = sum(1 for o in outcomes if o["status"] == COMPLETE_DETECTION_FAILURE)
            accuracy = n_correct / n if n else None
            coverage = n_ok / n if n else None
            avg_frames = sum(o["num_frames_used"] for o in outcomes) / n if n else None
            avg_latency = sum(latency_estimates[key]) / len(latency_estimates[key]) if latency_estimates[key] else None
            row = {
                "k": k_label(k), "method": method, "n": n, "n_correct": n_correct, "n_ok": n_ok,
                "n_incorrect_accepted": n_incorrect_accepted, "n_complete_detection_failures": n_failures,
                "accuracy": accuracy, "coverage": coverage, "avg_frames_used": avg_frames,
                "estimated_latency_ms": avg_latency,
            }
            summary_table.append(row)
            print(f"{k_label(k):<8}{method:<32}{str(round(accuracy,4) if accuracy is not None else None):<12}"
                  f"{str(round(coverage,4) if coverage is not None else None):<12}{n_incorrect_accepted:<12}"
                  f"{str(round(avg_frames,2) if avg_frames is not None else None):<12}"
                  f"{round(avg_latency,1) if avg_latency is not None else None}")

    if skipped:
        print(f"\n{len(skipped)} track(s) skipped:")
        for s in skipped:
            print(f"  {s['track_id']}: {s['reason']}")

    # --- per-encounter breakdown: which specific track(s) are wrong/abstained,
    # and how does that verdict change across methods and K? Focused on
    # top3 (this week's headline K) plus "all" for comparison. ---
    print()
    print("=" * 100)
    print("PER-ENCOUNTER BREAKDOWN — top3, all 4 methods")
    print("=" * 100)
    print(f"{'track':<8}{'gt_text':<10}" + "".join(f"{m:<34}" for m in all_methods))
    per_encounter_top3 = {}
    for track_id in val_track_ids:
        row_outcomes = []
        found_any = False
        for method in all_methods:
            outcome = next((o for o in cell_results[("top3", method)] if o["track_id"] == track_id), None)
            if outcome is None:
                continue
            found_any = True
            row_outcomes.append((method, outcome))
        if not found_any:
            continue
        per_encounter_top3[track_id] = {m: o for m, o in row_outcomes}
        gt_text = row_outcomes[0][1]["gt_text"]
        cells = []
        for method, o in row_outcomes:
            if o["status"] == COMPLETE_DETECTION_FAILURE:
                cells.append("DETECTION_FAILURE")
            elif o["status"] == "no_reliable_result":
                cells.append("ABSTAINED")
            elif o["correct"]:
                cells.append(f"OK:{o['text']}")
            else:
                cells.append(f"WRONG:{o['text']}")
        print(f"{track_id:<8}{gt_text:<10}" + "".join(f"{c:<34}" for c in cells))

    print()
    print("PER-ENCOUNTER BREAKDOWN — all (every usable frame), all 4 methods")
    print("=" * 100)
    print(f"{'track':<8}{'gt_text':<10}" + "".join(f"{m:<34}" for m in all_methods))
    per_encounter_all = {}
    for track_id in val_track_ids:
        row_outcomes = []
        found_any = False
        for method in all_methods:
            outcome = next((o for o in cell_results[("all", method)] if o["track_id"] == track_id), None)
            if outcome is None:
                continue
            found_any = True
            row_outcomes.append((method, outcome))
        if not found_any:
            continue
        per_encounter_all[track_id] = {m: o for m, o in row_outcomes}
        gt_text = row_outcomes[0][1]["gt_text"]
        cells = []
        for method, o in row_outcomes:
            if o["status"] == COMPLETE_DETECTION_FAILURE:
                cells.append("DETECTION_FAILURE")
            elif o["status"] == "no_reliable_result":
                cells.append("ABSTAINED")
            elif o["correct"]:
                cells.append(f"OK:{o['text']}")
            else:
                cells.append(f"WRONG:{o['text']}")
        print(f"{track_id:<8}{gt_text:<10}" + "".join(f"{c:<34}" for c in cells))

    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "scope": "UFPR validation split (manifest_val_full.csv), encounter-level (22 physical tracks, one result per cell)",
            "num_encounters": num_encounters,
            "num_complete_detection_failures": num_complete_failures,
            "scoring_formula": "mean(norm(width*height), norm(blur_score), detector_conf, ocr_conf), min-max normalized per-track — see week7_selection.py docstring",
            "results_table": summary_table,
            "per_encounter_top3": per_encounter_top3,
            "per_encounter_all": per_encounter_all,
            "skipped_tracks": skipped,
        }, f, indent=2)
    print(f"\nWrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
