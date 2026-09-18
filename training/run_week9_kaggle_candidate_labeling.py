"""
run_week9_kaggle_candidate_labeling.py

Week 9, V1.1 dataset strategy, item 2: generates INITIAL candidate
transcriptions for the Kaggle US plate images using the frozen
Recognizer V1 (per the plan: "You can use an OCR model to generate
initial transcription candidates, but the final plate text used for
training must be manually verified").

This script does NOT produce verified training labels by itself — it
produces a review queue: a CSV with a candidate reading + confidence per
image, plus contact-sheet image grids (25 images each) so verification can
be done by scanning a grid and correcting a short text list, rather than
opening thousands of files individually.

Candidates are INTERLEAVED between predicted-6-character and
predicted-7-character buckets (each internally sorted by confidence,
highest first) — per the plan: "prioritize a balanced mixture of 6- and
7-character plates rather than simply labeling images in dataset order."
Working through the CSV top-to-bottom yields a length-balanced verified
set even if verification stops partway through.

Run from inside the alpr-train container, AFTER inspecting the actual
downloaded folder structure (the --image-root default below is a guess
pending that inspection — override with --image-root if the real layout
differs):
    cd /workspace/home/alpr-week5/pipeline
    python3 run_week9_kaggle_candidate_labeling.py \\
        --image-root /workspace/home/alpr-week5/data/raw/kaggle_us_plates \\
        --max-images 6000 \\
        --output-dir /workspace/home/alpr-week5/data/interim/kaggle_review
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpr_pipeline import AlprPipeline

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def find_images(root: Path) -> list:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def find_images_by_state(root: Path) -> dict:
    """Groups images by their immediate parent-folder name (the state
    label in this dataset's directory structure, e.g. 'new plates/test/TEXAS').
    Used for STRATIFIED sampling — see the module docstring update below
    on why sampling by predicted length doesn't work for this dataset."""
    by_state = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            state = p.parent.name
            by_state.setdefault(state, []).append(p)
    return by_state


def stratified_sample_by_state(by_state: dict, max_images: int, seed: int = 42) -> list:
    """Samples roughly evenly across every state folder, rather than
    uniformly at random across the pooled dataset. This matters
    specifically because V1's own predicted plate length turned out
    NOT to be a usable signal for finding non-7-character plates in this
    dataset (see run notes: 99.93% of 5,869 real candidates were
    predicted as 7 characters, regardless of the true plate format) —
    so length diversity has to come from sampling across states with
    different real-world formats, not from filtering on V1's guess."""
    random.seed(seed)
    states = sorted(by_state.keys())
    n_states = len(states)
    per_state_target = max(1, max_images // n_states)

    sampled = []
    for state in states:
        images = by_state[state]
        take = min(per_state_target, len(images))
        sampled.extend((state, p) for p in random.sample(images, take))

    # if under target (some states had fewer images than per_state_target),
    # top up from states with leftover images, still round-robin across states
    if len(sampled) < max_images:
        leftovers = {s: [p for p in by_state[s] if (s, p) not in sampled] for s in states}
        remaining_needed = max_images - len(sampled)
        state_cycle = [s for s in states if leftovers[s]]
        i = 0
        while remaining_needed > 0 and state_cycle:
            s = state_cycle[i % len(state_cycle)]
            if leftovers[s]:
                sampled.append((s, leftovers[s].pop()))
                remaining_needed -= 1
            else:
                state_cycle.remove(s)
                continue
            i += 1

    random.shuffle(sampled)
    print(f"Stratified sample: {len(sampled)} images across {n_states} states "
          f"(~{per_state_target}/state target)")
    return sampled


def generate_candidates(image_root: Path, max_images: int, seed: int = 42) -> list:
    pipeline = AlprPipeline()
    by_state = find_images_by_state(image_root)
    total_found = sum(len(v) for v in by_state.values())
    print(f"Found {total_found} image(s) under {image_root}, across {len(by_state)} state folder(s)")

    sampled = stratified_sample_by_state(by_state, max_images, seed=seed)

    candidates = []
    for i, (state, image_path) in enumerate(sampled):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        result = pipeline.process(image, image_id=image_path.stem)
        if result.status == "ok" and result.plate_text:
            candidates.append({
                "image_path": str(image_path),
                "predicted_text": result.plate_text,
                "predicted_length": len(result.plate_text),
                "confidence": result.recognition_confidence,
                "state": state,
            })
        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(sampled)}")

    return candidates


# Heuristic expected standard-plate character count by state, aggregated
# from general reference sources (not DMV primary sources) — see project
# notes. NOT authoritative: formats change over time (e.g. Rhode Island's
# own source notes a 2023 change), older vehicles may carry older-format
# plates, and this dataset's images could span any era. This is used ONLY
# to decide REVIEW ORDER (front-load states more likely to hold the
# 6-character examples we most need) — it never changes which images were
# sampled, and every reading is still visually verified by a human either
# way; a wrong prior here costs review time, not correctness.
STATE_EXPECTED_LENGTH = {
    "ALASKA": 6, "ARKANSAS": 6, "COLORADO": 6, "DELAWARE": 6,
    "DISTRICT OF COLUMBIA": 6, "FLORIDA": 6, "HAWAII": 6, "INDIANA": 6,
    "IOWA": 6, "KENTUCKY": 6, "LOUISIANA": 6, "MAINE": 6, "MASSACHUSETTS": 6,
    "MINNESOTA": 6, "MISSISSIPPI": 6, "MISSOURI": 6, "NEBRASKA": 6,
    "NEVADA": 6, "NEW JERSEY": 6, "NEW MEXICO": 6, "NORTH DAKOTA": 6,
    "OKLAHOMA": 6, "OREGON": 6, "SOUTH CAROLINA": 6, "SOUTH DAKOTA": 6,
    "UTAH": 6, "VERMONT": 6,
    "RHODE ISLAND": 5,  # per-source: changed to 2 letters + 3 numbers in 2023
    "ARIZONA": 7, "CALIFORNIA": 7, "CONNECTICUT": 7, "GEORGIA": 7,
    "IDAHO": 7, "ILLINOIS": 7, "KANSAS": 7, "MARYLAND": 7, "MICHIGAN": 7,
    "MONTANA": 7, "NEW HAMPSHIRE": 7, "NEW YORK": 7, "NORTH CAROLINA": 7,
    "OHIO": 7, "PENNSYLVANIA": 7, "TENNESSEE": 7, "TEXAS": 7, "VIRGINIA": 7,
    "WASHINGTON": 7, "WEST VIRGINIA": 7, "WISCONSIN": 7,
}


def _state_review_priority(state_folder_name: str) -> int:
    """Lower = reviewed earlier. Prioritizes states heuristically expected
    to be non-7-character (we already have abundant confirmed 7-character
    examples; the scarce ones are what this ordering is for)."""
    expected = STATE_EXPECTED_LENGTH.get(state_folder_name.upper().replace("_", " "))
    if expected is None:
        return 1  # unknown/varies — review after the likely-6/5, before the likely-7
    if expected == 7:
        return 2  # deprioritized — we already have plenty of confirmed 7-char examples
    return 0  # expected 5 or 6 — highest priority


def interleave_by_state(candidates: list) -> list:
    """Round-robins across STATE folders (not predicted length — see the
    finding above on why that signal doesn't work here), ordering states
    by _state_review_priority so likely-non-7-character states surface
    first. Within each state, sorts by confidence descending so the
    easiest-to-confirm readings for that state come first."""
    buckets = {}
    for c in candidates:
        buckets.setdefault(c["state"], []).append(c)
    for state in buckets:
        buckets[state].sort(key=lambda c: c["confidence"] or 0.0, reverse=True)

    states = sorted(buckets.keys(), key=lambda s: (_state_review_priority(s), s))
    queues = [iter(buckets[s]) for s in states]

    result = []
    active = list(range(len(queues)))
    while active:
        for idx in list(active):
            item = next(queues[idx], None)
            if item is None:
                active.remove(idx)
            else:
                result.append(item)
    return result


def write_review_csv(ordered_candidates: list, output_path: Path) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rank", "image_path", "state", "predicted_text", "predicted_length", "confidence", "verified_text", "notes",
        ])
        writer.writeheader()
        for rank, c in enumerate(ordered_candidates, start=1):
            writer.writerow({
                "rank": rank, "image_path": c["image_path"], "state": c["state"],
                "predicted_text": c["predicted_text"],
                "predicted_length": c["predicted_length"], "confidence": round(c["confidence"], 4) if c["confidence"] else None,
                "verified_text": "",  # fill this in during manual review; leave blank to reject/skip
                "notes": "",
            })


def build_contact_sheets(ordered_candidates: list, output_dir: Path, per_sheet: int = 25) -> int:
    """Tiles images with their predicted text as a caption, in batches, so
    manual review can happen by scanning a grid instead of opening each
    file. Returns the number of sheets written."""
    from PIL import Image, ImageDraw, ImageFont

    sheets_dir = output_dir / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    thumb_w, thumb_h = 220, 90
    caption_h = 24
    cols = 5
    rows = (per_sheet + cols - 1) // cols

    n_sheets = (len(ordered_candidates) + per_sheet - 1) // per_sheet
    for sheet_idx in range(n_sheets):
        batch = ordered_candidates[sheet_idx * per_sheet: (sheet_idx + 1) * per_sheet]
        sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + caption_h)), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default()

        for i, c in enumerate(batch):
            r, col = divmod(i, cols)
            x0, y0 = col * thumb_w, r * (thumb_h + caption_h)
            try:
                img = Image.open(c["image_path"]).convert("RGB")
                img.thumbnail((thumb_w - 4, thumb_h - 4))
                paste_x = x0 + (thumb_w - img.width) // 2
                paste_y = y0 + (thumb_h - img.height) // 2
                sheet.paste(img, (paste_x, paste_y))
            except Exception:
                pass
            conf_str = f"{c['confidence']:.2f}" if c["confidence"] else "?"
            caption = f"#{sheet_idx*per_sheet + i + 1} [{c.get('state','?')}]: {c['predicted_text']} ({conf_str})"
            draw.text((x0 + 2, y0 + thumb_h + 2), caption, fill=(0, 0, 0), font=font)
            draw.rectangle([x0, y0, x0 + thumb_w - 1, y0 + thumb_h + caption_h - 1], outline=(200, 200, 200))

        sheet.save(sheets_dir / f"sheet_{sheet_idx:03d}.png")

    return n_sheets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=str, default="/workspace/home/alpr-week5/data/raw/kaggle_us_plates")
    parser.add_argument("--max-images", type=int, default=6000,
                         help="Cap on how many images to run candidate generation on (not the final verified count)")
    parser.add_argument("--output-dir", type=str, default="/workspace/home/alpr-week5/data/interim/kaggle_review")
    parser.add_argument("--per-sheet", type=int, default=25)
    parser.add_argument("--skip-contact-sheets", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = generate_candidates(Path(args.image_root), args.max_images)
    print(f"\n{len(candidates)} candidate(s) with a non-empty OCR reading")

    ordered = interleave_by_state(candidates)
    csv_path = output_dir / "kaggle_candidates.csv"
    write_review_csv(ordered, csv_path)
    print(f"Wrote review queue to {csv_path}")

    length_counts = {}
    for c in candidates:
        length_counts[c["predicted_length"]] = length_counts.get(c["predicted_length"], 0) + 1
    print(f"Predicted-length distribution: {length_counts}")

    if not args.skip_contact_sheets:
        n_sheets = build_contact_sheets(ordered, output_dir, per_sheet=args.per_sheet)
        print(f"Wrote {n_sheets} contact sheet(s) to {output_dir / 'contact_sheets'}")

    print("\nNext step: open kaggle_candidates.csv (and/or the contact sheets) and fill in "
          "'verified_text' for each row you confirm or correct. Leave 'verified_text' blank "
          "for anything you want to reject/skip. Work top-to-bottom for a length-balanced subset.")


if __name__ == "__main__":
    main()
