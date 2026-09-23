"""
rodosol_loader.py

Loads real RodoSol-ALPR full images (not the pre-cropped recognizer-training
crops) + their ground-truth annotations, for running the full single-frame
pipeline (detector -> crop -> quality -> recognizer) end to end — same
purpose as ufpr_video_loader.py serves for UFPR, adapted to RodoSol's
different (simpler, non-track) layout.

Confirmed on-disk layout:
    data/raw/rodosol-alpr/tbFcZE-RodoSol-ALPR/images/{cars-br,cars-me,motorcycles-br,motorcycles-me}/
        img_NNNNNN.jpg   — full image
        img_NNNNNN.txt   — annotation, same key format as UFPR's:
            plate: PLATETEXT
            corners: x1,y1 x2,y2 x3,y3 x4,y4

manifest_test_full.csv references these as e.g.
    crops/rodosol_img_000010_img_000010.jpg,OYE3384,Unknown
i.e. the image id "img_000010" is embedded as `rodosol_{id}_{id}.jpg` — get_test_image_ids()
extracts that id so only the actual held-out test images are evaluated.
"""

import re
from pathlib import Path
from typing import Optional

try:
    import cv2
except ImportError:
    cv2 = None

RODOSOL_ROOT_DEFAULT = Path("/workspace/home/alpr-week5/data/raw/rodosol-alpr/tbFcZE-RodoSol-ALPR/images")
RODOSOL_CATEGORY_DIRS = ("cars-br", "cars-me", "motorcycles-br", "motorcycles-me")

# e.g. "crops/rodosol_img_000010_img_000010.jpg" -> "img_000010"
IMAGE_ID_IN_MANIFEST_PATH_RE = re.compile(r"rodosol_(img_\d+)_")


def get_test_image_ids(manifest_test_csv: Path) -> set:
    """Extract the set of RodoSol image ids (e.g. 'img_000010') that appear
    in this project's manifest_test_full.csv. Skips UFPR rows (no such
    pattern there)."""
    ids = set()
    with open(manifest_test_csv, "r", encoding="utf-8") as f:
        for line in f:
            m = IMAGE_ID_IN_MANIFEST_PATH_RE.search(line)
            if m:
                ids.add(m.group(1))
    return ids


def find_image_files(image_id: str, rodosol_root: Path = RODOSOL_ROOT_DEFAULT) -> Optional[tuple]:
    """image_id: e.g. 'img_000010'. Searches all four category subfolders.
    Returns (jpg_path, txt_path) or None if not found."""
    for category in RODOSOL_CATEGORY_DIRS:
        jpg_path = rodosol_root / category / f"{image_id}.jpg"
        txt_path = rodosol_root / category / f"{image_id}.txt"
        if jpg_path.exists() and txt_path.exists():
            return (jpg_path, txt_path)
    return None


def parse_annotation(txt_path: Path) -> dict:
    """Returns {'plate_text': str, 'bbox': (x1,y1,x2,y2)}. Same key format
    as UFPR's annotations (plate:, corners:), so this mirrors
    ufpr_video_loader.parse_annotation exactly — kept as its own copy here
    so this loader has no cross-file dependency."""
    plate_text = None
    bbox = None
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("plate:"):
                plate_text = line.split(":", 1)[1].strip()
            elif line.startswith("corners:"):
                coords_str = line.split(":", 1)[1].strip()
                points = []
                for pair in coords_str.split():
                    x_str, y_str = pair.split(",")
                    points.append((float(x_str), float(y_str)))
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                bbox = (min(xs), min(ys), max(xs), max(ys))
    if plate_text is None or bbox is None:
        raise ValueError(f"could not find both 'plate:' and 'corners:' in {txt_path}")
    return {"plate_text": plate_text, "bbox": bbox}


def load_image(image_id: str, rodosol_root: Path = RODOSOL_ROOT_DEFAULT) -> dict:
    """Loads one RodoSol image + its ground truth.

    Returns:
        {
          "image_id": str,
          "image": ndarray (or path string if cv2 unavailable — see
                    ufpr_video_loader for the same fallback convention),
          "gt_text": str,
          "gt_bbox": (x1,y1,x2,y2),
        }
    """
    found = find_image_files(image_id, rodosol_root)
    if found is None:
        raise FileNotFoundError(f"no image+annotation pair found for RodoSol id {image_id} under {rodosol_root}")
    jpg_path, txt_path = found

    ann = parse_annotation(txt_path)

    if cv2 is not None:
        image = cv2.imread(str(jpg_path))
        if image is None:
            raise IOError(f"cv2 failed to read {jpg_path}")
    else:
        image = str(jpg_path)

    return {
        "image_id": image_id,
        "image": image,
        "gt_text": ann["plate_text"],
        "gt_bbox": ann["bbox"],
    }
