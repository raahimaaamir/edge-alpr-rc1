"""
run_v1_1_finetune.py

Week 9, V1.1: fine-tunes Recognizer V1.1 FROM the frozen V1 checkpoint,
using fast_plate_ocr's own train CLI (confirmed against its real source —
fast_plate_ocr/cli/train.py).

Key differences from how V1 was originally trained:
  - --weights-path points at V1's best.keras (NOT the original pretrained
    cct_s_v2_global.keras) — this is fine-tuning from V1, not training
    from scratch.
  - --lr is reduced 10x from V1's 0.001 to 0.0001 — "low learning rate"
    per the plan, standard practice for fine-tuning without catastrophic
    forgetting of the strong 7-character performance already achieved.
  - --epochs reduced from 30 to 15 (fine-tuning from a good starting
    point typically needs fewer epochs; early stopping — same
    patience=3 as V1 — will stop sooner if it plateaus earlier).
  - --annotations / --val-annotations point at the NEW combined manifest
    (build_v1_1_training_manifest.py's output), not the original V1 data.
  - --seed is set (2026) for reproducibility — V1's run did not set one.

Everything else (architecture config, batch size, loss/label-smoothing
setup, weight decay, clip norm) is kept IDENTICAL to V1's training run, so
the only things that changed are: the starting weights, the learning
rate, the epoch budget, and the training data.

NOTE on "balanced sampling by plate length": fast_plate_ocr's CLI has no
built-in stratified-sampling flag. Balance is instead achieved entirely
at the DATA level — build_v1_1_training_manifest.py enforces a 50/50
synthetic 6-/7-character split and the Kaggle review queue is
length-interleaved — so the training CSV itself is already balanced by
the time this script runs. This is documented here explicitly so it
isn't mistaken for a missed requirement.

Run from inside the alpr-train container (after build_v1_1_training_manifest.py
has produced manifest_train_v1_1.csv / manifest_val_v1_1.csv):
    cd /workspace/home/alpr-week5
    python3 pipeline/run_v1_1_finetune.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

V1_MODEL_CONFIG = "weights/cct_s_v2_global_model_config.yaml"
V1_PLATE_CONFIG = "weights/cct_s_v2_global_plate_config.yaml"
V1_WEIGHTS = "output/full_run1/2026-08-26_17-15-13/best.keras"


def build_command(
    annotations: str,
    val_annotations: str,
    weights_path: str = V1_WEIGHTS,
    model_config_file: str = V1_MODEL_CONFIG,
    plate_config_file: str = V1_PLATE_CONFIG,
    lr: float = 0.0001,
    epochs: int = 15,
    batch_size: int = 16,
    early_stopping_patience: int = 3,
    output_dir: str = "output/v1_1_finetune",
    seed: int = 2026,
) -> list:
    return [
        sys.executable, "-m", "fast_plate_ocr.cli.cli", "train",
        "--model-config-file", model_config_file,
        "--plate-config-file", plate_config_file,
        "--annotations", annotations,
        "--val-annotations", val_annotations,
        "--lr", str(lr),
        "--final-lr-factor", "0.01",       # unchanged from V1
        "--warmup-fraction", "0.05",       # unchanged from V1
        "--weight-decay", "0.01",          # unchanged from V1
        "--clipnorm", "1.0",               # unchanged from V1
        "--plate-loss", "cce",             # unchanged from V1
        "--label-smoothing", "0.01",       # unchanged from V1
        "--plate-loss-weight", "0.9",      # unchanged from V1
        "--region-loss-weight", "0.1",     # unchanged from V1
        "--batch-size", str(batch_size),   # unchanged from V1 (16)
        "--output-dir", output_dir,
        "--epochs", str(epochs),
        "--early-stopping-patience", str(early_stopping_patience),  # unchanged from V1 (3)
        "--early-stopping-metric", "val_plate_acc",  # unchanged from V1
        "--weights-path", weights_path,    # <-- THE key change: fine-tune FROM V1, not from scratch
        "--seed", str(seed),
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=str, default="data/interim/v1_1_combined/manifest_train_v1_1.csv")
    parser.add_argument("--val-annotations", type=str, default="data/interim/v1_1_combined/manifest_val_v1_1.csv")
    parser.add_argument("--weights-path", type=str, default=V1_WEIGHTS)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--output-dir", type=str, default="output/v1_1_finetune")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it")
    args = parser.parse_args()

    cmd = build_command(
        annotations=args.annotations, val_annotations=args.val_annotations,
        weights_path=args.weights_path, lr=args.lr, epochs=args.epochs, output_dir=args.output_dir,
    )

    print("Command to run:")
    print(" ".join(cmd))
    print()

    if args.dry_run:
        print("(--dry-run: not executing)")
    else:
        for p in (args.annotations, args.val_annotations, args.weights_path):
            if not Path(p).exists():
                print(f"ERROR: required file not found: {p}")
                sys.exit(1)
        subprocess.run(cmd, check=True)
