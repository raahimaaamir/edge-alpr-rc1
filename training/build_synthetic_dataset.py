"""
build_synthetic_dataset.py

Week 9, V1.1 dataset strategy, item 1: generates the full synthetic plate
dataset (default 15,000 images, balanced 50/50 between 6- and 7-character
plates, multiple plausible layouts per length, ~20% left clean per the
plan's "substantial clean subset" instruction) and writes a manifest CSV
in the exact schema fast_plate_ocr's train.py expects
(image_path, plate_text, plate_region — confirmed against
train/data/annotations.py source).

Run from inside the alpr-train container:
    cd /workspace/home/alpr-week5/pipeline
    python3 build_synthetic_dataset.py --n 15000 --output-dir /workspace/home/alpr-week5/data/interim/synthetic_v1_1

Images are written under <output-dir>/images/, and the manifest CSV at
<output-dir>/manifest_synthetic.csv, with image_path written RELATIVE to
the manifest's own folder (consistent with this project's existing
manifest convention — see /areas/alpr-research.md project notes: paths
resolve relative to the CSV's own folder).
"""

import argparse
import csv
import random
from pathlib import Path

from synth_plate_generator import generate_plate_image, random_plate_text


def build_dataset(n: int, output_dir: Path, clean_fraction: float = 0.2, seed: int = 1234):
    random.seed(seed)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest_synthetic.csv"
    rows = []

    n_six = n // 2
    n_seven = n - n_six

    plan = [("six", 6)] * n_six + [("seven", 7)] * n_seven
    random.shuffle(plan)

    for i, (label, length) in enumerate(plan):
        text = random_plate_text(length)
        is_clean = random.random() < clean_fraction
        img = generate_plate_image(text, clean_probability=1.0 if is_clean else 0.0)

        filename = f"synth_{i:06d}_{label}_{text}.jpg"
        img_path = images_dir / filename
        img.save(img_path, format="JPEG", quality=random.randint(80, 95))

        rows.append({
            "image_path": f"images/{filename}",  # relative to manifest's own folder
            "plate_text": text,
            "plate_region": "Unknown",
        })

        if (i + 1) % 1000 == 0:
            print(f"  generated {i + 1}/{n}")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "plate_text", "plate_region"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} images to {images_dir}")
    print(f"Wrote manifest to {manifest_path}")
    print(f"  6-character: {n_six}, 7-character: {n_seven}, clean fraction target: {clean_fraction}")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=15000, help="Total number of synthetic images to generate")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--clean-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    build_dataset(n=args.n, output_dir=Path(args.output_dir), clean_fraction=args.clean_fraction, seed=args.seed)
