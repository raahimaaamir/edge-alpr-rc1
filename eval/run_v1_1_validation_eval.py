"""
run_v1_1_validation_eval.py

Evaluates Recognizer V1 and V1.1 on the internal validation set, broken
out by (source, length) group exactly as the supervisor requested:
  - synthetic 6-char
  - synthetic 7-char
  - real (kaggle_verified) 6-char
  - real (kaggle_verified) 7-char
  - retained UFPR/RodoSol legacy (7-char) — the catastrophic-forgetting check

Per the supervisor's instructions, this does NOT touch the 13 reserved
local plates — those are evaluated once, separately, only after V1.1 is
frozen from these validation results.

Recognizer inputs here are ALREADY-CROPPED plate images (matching the
training data format), so this calls PlateRecognizer.recognize() DIRECTLY
— not the full AlprPipeline.process() (which expects a full scene to
detect within). Using the full pipeline on pre-cropped images would be
testing the wrong thing.

Decision metrics per group, per model:
  - exact-match accuracy
  - predicted-length accuracy (fraction where predicted length == true length)
  - insertion / deletion rate (per plate, from edit_distance_with_ops)
  - character accuracy (ops.matches / ground-truth length, averaged)

Run from inside the alpr-train container, from the pipeline/ directory:
    python3 run_v1_1_validation_eval.py --v1-1-model-path <path to v1.1 onnx>
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2

from plate_length_analysis import analyze_set
from recognizer import PlateRecognizer, DEFAULT_MODEL_PATH as V1_MODEL_PATH, DEFAULT_CONFIG_PATH

LEDGER_PATH = Path("/workspace/home/alpr-week5/data/interim/v1_1_combined/source_ledger.csv")


def load_val_groups(ledger_path: Path) -> dict:
    """Returns {(source, length): [(image_path, plate_text), ...]} for val rows only."""
    groups = defaultdict(list)
    with open(ledger_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != "val":
                continue
            key = (row["source"], int(row["length"]))
            groups[key].append((row["image_path"], row["plate_text"]))
    return groups


def run_recognizer_on_group(recognizer: PlateRecognizer, rows: list) -> list:
    """Returns [(predicted_text_or_None, ground_truth), ...] for one group."""
    results = []
    for image_path, gt_text in rows:
        image = cv2.imread(image_path)
        if image is None:
            results.append((None, gt_text))
            continue
        text, _confs, _overall = recognizer.recognize(image)
        results.append((text if text else None, gt_text))
    return results


def char_accuracy(report: dict) -> float:
    """Mean, over plates with a usable comparison, of ops.matches / gt_length —
    the fraction of ground-truth characters correctly recovered."""
    vals = []
    for c in report["comparisons"]:
        if c["ops"] is None or c["gt_length"] == 0:
            continue
        vals.append(c["ops"]["matches"] / c["gt_length"])
    return sum(vals) / len(vals) if vals else None


def ins_del_rate(report: dict) -> float:
    """(total insertions + total deletions) / n_total — per-plate rate,
    not per-character, matching how the supervisor's requested metric
    ('insertion/deletion rate') is most directly interpretable at a glance."""
    n = report["n_total"]
    return (report["total_insertions"] + report["total_deletions"]) / n if n else None


def summarize_group(rows: list, recognizer: PlateRecognizer) -> dict:
    predictions = run_recognizer_on_group(recognizer, rows)
    report = analyze_set(predictions)
    return {
        "n": report["n_total"],
        "exact_match_acc": report["exact_match_rate"],
        "length_acc": report["n_same_length"] / report["n_total"] if report["n_total"] else None,
        "char_acc": char_accuracy(report),
        "ins_del_rate": ins_del_rate(report),
        "total_insertions": report["total_insertions"],
        "total_deletions": report["total_deletions"],
        "total_substitutions": report["total_substitutions"],
    }


def print_comparison_table(all_results: dict):
    group_order = [
        ("synthetic", 6), ("synthetic", 7),
        ("kaggle_verified", 6), ("kaggle_verified", 7),
        ("existing_retained", 7),
    ]
    print("=" * 100)
    print("V1 vs V1.1 — VALIDATION RESULTS BY (SOURCE, LENGTH) GROUP")
    print("=" * 100)
    header = f"{'Group':<28} {'n':>5} | {'V1 exact':>9} {'V1.1 exact':>10} | {'V1 len':>8} {'V1.1 len':>9} | {'V1 char':>8} {'V1.1 char':>10} | {'V1 ins/del':>11} {'V1.1 ins/del':>13}"
    print(header)
    print("-" * len(header))
    for key in group_order:
        if key not in all_results:
            continue
        label = f"{key[0]}, {key[1]}-char"
        v1 = all_results[key]["v1"]
        v11 = all_results[key]["v1_1"]
        n = v1["n"]

        def fmt(x):
            return f"{x:.4f}" if x is not None else "n/a"

        print(f"{label:<28} {n:>5} | {fmt(v1['exact_match_acc']):>9} {fmt(v11['exact_match_acc']):>10} | "
              f"{fmt(v1['length_acc']):>8} {fmt(v11['length_acc']):>9} | "
              f"{fmt(v1['char_acc']):>8} {fmt(v11['char_acc']):>10} | "
              f"{fmt(v1['ins_del_rate']):>11} {fmt(v11['ins_del_rate']):>13}")
    print()
    print("Note: 'existing_retained' (7-char, UFPR/RodoSol legacy) is the")
    print("catastrophic-forgetting check — V1.1 should not be materially worse")
    print("than V1 here, even as 6-char groups hopefully improve.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-1-model-path", type=str, required=True,
                         help="Path to the exported V1.1 ONNX model")
    parser.add_argument("--config-path", type=str, default=DEFAULT_CONFIG_PATH,
                         help="Plate config YAML — same for V1 and V1.1 (architecture unchanged)")
    parser.add_argument("--ledger-path", type=str, default=str(LEDGER_PATH))
    parser.add_argument("--output-json", type=str, default="v1_1_validation_eval_results.json")
    args = parser.parse_args()

    print(f"Loading V1 recognizer from {V1_MODEL_PATH}")
    v1_recognizer = PlateRecognizer(model_path=V1_MODEL_PATH, config_path=args.config_path)
    print(f"Loading V1.1 recognizer from {args.v1_1_model_path}")
    v1_1_recognizer = PlateRecognizer(model_path=args.v1_1_model_path, config_path=args.config_path)

    groups = load_val_groups(Path(args.ledger_path))
    print(f"Loaded {sum(len(v) for v in groups.values())} val rows across {len(groups)} groups")

    all_results = {}
    for key, rows in groups.items():
        print(f"  Evaluating {key[0]}, {key[1]}-char ({len(rows)} rows)...")
        all_results[key] = {
            "v1": summarize_group(rows, v1_recognizer),
            "v1_1": summarize_group(rows, v1_1_recognizer),
        }

    print_comparison_table(all_results)

    # exploratory (other-length) rows — reported separately, never used to
    # optimize or select V1.1, per the supervisor's instruction
    # Exploratory (other-length) rows — read from the LEDGER (absolute
    # paths, same as the val groups above), NOT from
    # manifest_exploratory_other_length.csv directly. That file's paths
    # are written relative to the manifest's own folder (needed for the
    # fast-plate-ocr trainer, which resolves relative to the CSV's own
    # directory) — reading it here with a plain cv2.imread() from this
    # script's working directory would silently fail for every row. The
    # ledger's exploratory rows are the SAME underlying images with
    # absolute paths already resolved, so use those instead.
    exploratory_rows = []
    with open(args.ledger_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "exploratory_other_length":
                exploratory_rows.append((row["image_path"], row["plate_text"]))
    if exploratory_rows:
        print()
        print(f"Exploratory (other-length, n={len(exploratory_rows)}) — NOT used for model selection:")
        v1_exp = summarize_group(exploratory_rows, v1_recognizer)
        v11_exp = summarize_group(exploratory_rows, v1_1_recognizer)
        print(f"  V1:   exact_match_acc={v1_exp['exact_match_acc']}, char_acc={v1_exp['char_acc']}")
        print(f"  V1.1: exact_match_acc={v11_exp['exact_match_acc']}, char_acc={v11_exp['char_acc']}")
        all_results[("exploratory", "other")] = {"v1": v1_exp, "v1_1": v11_exp}

    serializable = {f"{k[0]}_{k[1]}": v for k, v in all_results.items()}
    with open(args.output_json, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
