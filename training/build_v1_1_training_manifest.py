"""
build_v1_1_training_manifest.py

Week 9, V1.1 dataset strategy: merges every data source into the final
training manifests, and produces the dataset summary report the plan
requires BEFORE any training happens (total samples; real vs synthetic;
6-char vs 7-char; source; train/val split; character-frequency
distribution).

Sources (each optional except synthetic, so this can run with whatever is
ready so far — CENPARMI can be added later without rebuilding anything):
  - synthetic:        manifest_synthetic.csv from build_synthetic_dataset.py
  - kaggle_verified:   kaggle_candidates.csv from run_week9_kaggle_candidate_labeling.py,
                       filtered to rows with a non-empty verified_text
  - cenparmi:          same 3-column schema, once available (not yet)
  - existing_retained: a SAMPLE of the original manifest_train_full.csv,
                       so fine-tuning on the 6-char fix doesn't erase
                       7-character performance (per the plan)

Produces TWO kinds of output:
  1. The exact 3-column CSVs fast_plate_ocr's trainer expects
     (image_path, plate_text, plate_region) — manifest_train_v1_1.csv and
     manifest_val_v1_1.csv. image_path is written ABSOLUTE so these are
     valid regardless of where the CSV itself is read from.
  2. A companion source ledger (source_ledger.csv) with one row per image
     — path, plate_text, length, source, split — for full audit/tracking,
     kept OUT of the trainer-facing CSVs since fast_plate_ocr warns on any
     column beyond the fixed three.

The 13 local hold-out plates and the CENPARMI/Kaggle raw pools are never
touched by this script except through their already-verified subsets —
nothing here can accidentally leak the hold-out set into training.
"""

import argparse
import csv
import os
import random
from collections import Counter
from pathlib import Path


def _read_manifest(path: Path):
    rows = []
    if not path.exists():
        return rows
    base = path.parent
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img_path = Path(row["image_path"])
            if not img_path.is_absolute():
                img_path = (base / img_path).resolve()
            rows.append({"image_path": str(img_path), "plate_text": row["plate_text"].strip()})
    return rows


def _read_kaggle_verified(path: Path):
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            verified = (row.get("verified_text") or "").strip()
            if not verified:
                continue  # blank verified_text = rejected/skipped, per the review workflow
            rows.append({"image_path": row["image_path"], "plate_text": verified})
    return rows


def _sample_existing(path: Path, n: int, seed: int) -> list:
    rows = _read_manifest(path)
    if not rows:
        return []
    random.seed(seed)
    if len(rows) > n:
        rows = random.sample(rows, n)
    return rows


def build(
    synthetic_manifest: Path,
    kaggle_verified_csv: Path,
    cenparmi_manifest: Path,
    existing_train_manifest: Path,
    n_existing_retained: int,
    output_dir: Path,
    val_fraction: float = 0.1,
    seed: int = 2026,
):
    """
    Per the supervisor's Week 9 follow-up instructions:
      - the 17 real "other"-length (not 6 or 7) examples are too few to
        train on meaningfully — they are split out entirely into a
        separate exploratory file, never entering train or val.
      - train/val split is STRATIFIED by (source, length) so every group
        (synthetic-6, synthetic-7, kaggle-6, kaggle-7, existing_retained)
        is proportionally represented in both splits — required for the
        grouped validation reporting the supervisor asked for.
      - within the TRAIN split only, the minority length (6-char) is
        oversampled (duplicated, with replacement) so 6- and 7-character
        examples have approximately equal opportunity during training.
        This does NOT touch the retained 7-char legacy rows or the val
        split — the legacy data stays fully intact (protecting against
        catastrophic forgetting is the whole point of retaining it), and
        val composition is left representative (not rebalanced) since
        it's used for per-group reporting, not training influence.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = []

    def add(rows, source):
        for r in rows:
            length = len(r["plate_text"])
            ledger.append({"image_path": r["image_path"], "plate_text": r["plate_text"],
                            "length": length, "source": source})

    add(_read_manifest(synthetic_manifest), "synthetic")
    add(_read_kaggle_verified(kaggle_verified_csv), "kaggle_verified")
    add(_read_manifest(cenparmi_manifest), "cenparmi")
    add(_sample_existing(existing_train_manifest, n_existing_retained, seed), "existing_retained")

    if not ledger:
        raise ValueError("No data found from any source — check the paths passed in.")

    random.seed(seed)

    # --- split out "other"-length rows into their own exploratory bucket,
    # never entering train or val ---
    trainable_rows = [r for r in ledger if r["length"] in (6, 7)]
    exploratory_rows = [r for r in ledger if r["length"] not in (6, 7)]
    for r in exploratory_rows:
        r["split"] = "exploratory_other_length"

    # --- stratified train/val split, by (source, length) group ---
    groups = {}
    for r in trainable_rows:
        groups.setdefault((r["source"], r["length"]), []).append(r)

    train_rows, val_rows = [], []
    for key, rows in groups.items():
        random.shuffle(rows)
        n_val = max(1, int(len(rows) * val_fraction)) if len(rows) > 1 else 0
        val_rows.extend(rows[:n_val])
        train_rows.extend(rows[n_val:])
    for r in val_rows:
        r["split"] = "val"
    for r in train_rows:
        r["split"] = "train"

    # --- oversample the minority length WITHIN TRAIN ONLY, so 6- and
    # 7-character examples get approximately equal training opportunity.
    # Legacy 7-char rows are never removed; only 6-char rows are duplicated. ---
    train_by_length = {6: [r for r in train_rows if r["length"] == 6],
                        7: [r for r in train_rows if r["length"] == 7]}
    n6, n7 = len(train_by_length[6]), len(train_by_length[7])
    minority_len, majority_len = (6, 7) if n6 < n7 else (7, 6)
    minority_rows, majority_count = train_by_length[minority_len], len(train_by_length[majority_len])

    oversampled_extra = []
    if minority_rows and len(minority_rows) < majority_count:
        deficit = majority_count - len(minority_rows)
        oversampled_extra = [dict(random.choice(minority_rows)) for _ in range(deficit)]
        for r in oversampled_extra:
            r["split"] = "train"
            r["oversampled_duplicate"] = True

    train_rows_balanced = train_rows + oversampled_extra

    # --- write trainer-facing CSVs: EXACTLY the 3 allowed columns ---
    def write_trainer_csv(rows, path):
        # Paths are written RELATIVE to output_dir (the manifest's own
        # folder) — matching the original manifest_train_full.csv
        # convention that V1 was successfully trained with. An earlier
        # version of this function wrote ABSOLUTE paths on the assumption
        # that fast_plate_ocr's loader would safely discard the CSV's own
        # folder when joining an absolute image_path (the way Python's
        # pathlib / and os.path.join both do) — but the actual training
        # data loader (dataset.py / core/process.py) does plain string
        # concatenation instead, with no such safety check, producing a
        # doubled/broken path. Confirmed by reproducing the real error
        # against the real training run. Relative paths are the format
        # that's actually been proven to work.
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "plate_text", "plate_region"])
            writer.writeheader()
            for r in rows:
                rel_path = os.path.relpath(r["image_path"], start=output_dir)
                writer.writerow({"image_path": rel_path, "plate_text": r["plate_text"], "plate_region": "Unknown"})

    train_path = output_dir / "manifest_train_v1_1.csv"
    val_path = output_dir / "manifest_val_v1_1.csv"
    write_trainer_csv(train_rows_balanced, train_path)
    write_trainer_csv(val_rows, val_path)

    # --- exploratory (other-length) file, same 3-column trainer format,
    # kept SEPARATE — never fed to training, only for later exploratory checks ---
    exploratory_path = output_dir / "manifest_exploratory_other_length.csv"
    if exploratory_rows:
        write_trainer_csv(exploratory_rows, exploratory_path)

    # --- write the full source ledger (audit trail, NOT fed to the trainer).
    # `ledger` already reflects every distinct real row exactly once, with
    # "split" correctly set (train/val/exploratory_other_length), because
    # train_rows/val_rows/exploratory_rows hold the SAME dict objects — the
    # in-place r["split"]=... assignments above are visible here too.
    # Oversampled duplicates are intentionally NOT included in this ledger
    # (they're training-only artifacts, not distinct real images). ---
    ledger_path = output_dir / "source_ledger.csv"
    with open(ledger_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "plate_text", "length", "source", "split"])
        writer.writeheader()
        writer.writerows(ledger)

    balancing_info = {
        "train_n6_before_oversample": n6, "train_n7_before_oversample": n7,
        "minority_length": minority_len, "n_oversampled_duplicates_added": len(oversampled_extra),
        "train_n_total_after_oversample": len(train_rows_balanced),
        "n_exploratory_other_length": len(exploratory_rows),
    }
    return ledger, train_path, val_path, ledger_path, balancing_info


def print_summary(ledger: list, balancing_info: dict = None) -> dict:
    n_total = len(ledger)
    by_source = Counter(r["source"] for r in ledger)
    by_length = Counter(r["length"] for r in ledger)
    by_split = Counter(r["split"] for r in ledger)
    n_real = sum(v for k, v in by_source.items() if k != "synthetic")
    n_synthetic = by_source.get("synthetic", 0)

    char_counts = Counter()
    for r in ledger:
        char_counts.update(r["plate_text"])

    # per-(source, length) breakdown, in val specifically, since that's what
    # the supervisor's grouped-reporting requirement is scored against
    val_group_counts = Counter((r["source"], r["length"]) for r in ledger if r["split"] == "val")

    print("=" * 60)
    print("V1.1 DATASET SUMMARY (report this before training)")
    print("=" * 60)
    print(f"Total samples (train + val + exploratory): {n_total}")
    print(f"Real vs synthetic: real={n_real} ({n_real/n_total:.1%}), synthetic={n_synthetic} ({n_synthetic/n_total:.1%})")
    print(f"By source: {dict(by_source)}")
    print(f"6-character vs 7-character (train+val only, pre-oversample counts): "
          f"6-char={by_length.get(6,0)}, 7-char={by_length.get(7,0)}")
    print(f"Excluded from train/val as exploratory-only (other length, e.g. 5/8-char): "
          f"{by_split.get('exploratory_other_length', 0)}")
    print(f"Train/val split (pre-oversample, distinct real images): "
          f"train={by_split.get('train',0)}, val={by_split.get('val',0)}")
    if balancing_info:
        print(f"Train-set length balancing: {balancing_info['train_n6_before_oversample']} six-char vs "
              f"{balancing_info['train_n7_before_oversample']} seven-char before oversampling; "
              f"{balancing_info['n_oversampled_duplicates_added']} duplicate six-char rows added "
              f"(minority length={balancing_info['minority_length']}) to reach approximate parity. "
              f"Final training file size: {balancing_info['train_n_total_after_oversample']} rows.")
    print(f"Validation set, per (source, length) group — the grouped breakdown for decision metrics:")
    for (source, length), count in sorted(val_group_counts.items()):
        print(f"  {source}, {length}-char: {count}")
    print(f"Character frequency (top 15): {char_counts.most_common(15)}")
    print(f"Character frequency (bottom 10, check for under-representation): {char_counts.most_common()[-10:]}")

    return {
        "n_total": n_total, "n_real": n_real, "n_synthetic": n_synthetic,
        "by_source": dict(by_source), "by_length": dict(by_length), "by_split": dict(by_split),
        "val_group_counts": {f"{k[0]}_{k[1]}char": v for k, v in val_group_counts.items()},
        "balancing_info": balancing_info,
        "char_frequency": dict(char_counts),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-manifest", type=str,
                         default="/workspace/home/alpr-week5/data/interim/synthetic_v1_1/manifest_synthetic.csv")
    parser.add_argument("--kaggle-verified-csv", type=str,
                         default="/workspace/home/alpr-week5/data/interim/kaggle_review/kaggle_candidates.csv")
    parser.add_argument("--cenparmi-manifest", type=str,
                         default="/workspace/home/alpr-week5/data/interim/cenparmi/manifest_cenparmi.csv")
    parser.add_argument("--existing-train-manifest", type=str,
                         default="/workspace/home/alpr-week5/data/interim/manifest_train_full.csv")
    parser.add_argument("--n-existing-retained", type=int, default=5000)
    parser.add_argument("--output-dir", type=str,
                         default="/workspace/home/alpr-week5/data/interim/v1_1_combined")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    ledger, train_path, val_path, ledger_path, balancing_info = build(
        synthetic_manifest=Path(args.synthetic_manifest),
        kaggle_verified_csv=Path(args.kaggle_verified_csv),
        cenparmi_manifest=Path(args.cenparmi_manifest),
        existing_train_manifest=Path(args.existing_train_manifest),
        n_existing_retained=args.n_existing_retained,
        output_dir=Path(args.output_dir),
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print_summary(ledger, balancing_info)
    print(f"\nWrote {train_path}")
    print(f"Wrote {val_path}")
    print(f"Wrote {ledger_path}")
