"""
ufpr_video_loader.py

Bridges the real UFPR-ALPR raw dataset on disk to the stage-3 code
(video_pipeline.py / tracker_eval.py / evaluate_stage3.py), which only know
about generic (frame_idx, timestamp, image) tuples and
{frame_idx: [(bbox, gt_track_id)]} ground truth.

Matches the on-disk layout confirmed against the real dataset:
    data/raw/ufpr-alpr/UFPR-ALPR dataset/{training,testing,validation}/trackNNNN/
        trackNNNN[FF].png   — frame image
        trackNNNN[FF].txt   — annotation for that frame, containing at least:
            plate: PLATETEXT
            corners: x1,y1 x2,y2 x3,y3 x4,y4     (plate quadrilateral)

Only `plate:` and `corners:` are parsed — everything else in the annotation
(camera, vehicle make/model, per-character boxes) is ignored; it isn't
needed for tracking/fusion evaluation.

NOTE on splits: the UFPR-ALPR dataset's own training/testing/validation
folders are NOT the same as this project's manifest_{train,val,test}_full.csv
split (that split was built fresh, by track_id, 105/22/23). A given
track number can live in any of the three on-disk folders — search_track_dir
checks all three. Determine which tracks to evaluate on from the relevant
manifest_{train,val,test}_full.csv via get_track_ids() (alias of
get_test_track_ids, despite the name).

NOTE on test-set discipline (per Week 6 correction): manifest_val_full.csv
is for selecting/tuning thresholds and parameters (tracking, fusion,
confidence, etc.) during Weeks 6-9. manifest_test_full.csv is touched only
when freezing a meaningful new pipeline version — not for iterative
development.
"""

import re
from pathlib import Path
from typing import Optional

try:
    import cv2
except ImportError:
    cv2 = None  # loader can still be unit-tested without cv2/real images


UFPR_ROOT_DEFAULT = Path("/workspace/home/alpr-week5/data/raw/ufpr-alpr/UFPR-ALPR dataset")
UFPR_SPLIT_DIRS = ("training", "testing", "validation")

# e.g. "crops/ufpr_track0093_track0093[01].png" -> "0093"
TRACK_ID_IN_MANIFEST_PATH_RE = re.compile(r"ufpr_track(\d{4})_")

# e.g. "track0021[07].png" -> track "0021", frame "07"
FRAME_FILENAME_RE = re.compile(r"track(\d{4})\[(\d+)\]\.(png|jpg|jpeg)$", re.IGNORECASE)


def get_test_track_ids(manifest_test_csv: Path) -> set:
    """Extract the set of 4-digit UFPR track-id strings (e.g. '0093') that
    appear in a manifest CSV built by this project's split pipeline —
    works for manifest_train_full.csv, manifest_val_full.csv, or
    manifest_test_full.csv alike, despite the name (kept for backward
    compatibility with earlier callers). Skips RodoSol rows (they have no
    such pattern). See get_track_ids() for the split-agnostic name."""
    ids = set()
    with open(manifest_test_csv, "r", encoding="utf-8") as f:
        for line in f:
            m = TRACK_ID_IN_MANIFEST_PATH_RE.search(line)
            if m:
                ids.add(m.group(1))
    return ids


# Split-agnostic alias — prefer this name in new code. Per the Week 6
# correction: use manifest_val_full.csv (validation/development data) for
# selecting tracking thresholds, frame-selection rules, fusion parameters,
# etc. manifest_test_full.csv is reserved for freezing a pipeline version,
# not for iterative tuning.
get_track_ids = get_test_track_ids


def find_track_dir(track_id: str, ufpr_root: Path = UFPR_ROOT_DEFAULT) -> Optional[Path]:
    """track_id: 4-digit string, e.g. '0093'. Searches all three on-disk
    splits since track numbering isn't unique to one of them."""
    for split in UFPR_SPLIT_DIRS:
        candidate = ufpr_root / split / f"track{track_id}"
        if candidate.is_dir():
            return candidate
    return None


def parse_annotation(txt_path: Path) -> dict:
    """Returns {'plate_text': str, 'bbox': (x1,y1,x2,y2)} from one frame's
    .txt annotation. bbox is the axis-aligned min/max of the 4 plate
    corners (the annotation gives a quadrilateral, not an axis-aligned box —
    detectors/this project's bbox convention are axis-aligned, so we take
    the enclosing rectangle)."""
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


def load_track(track_id: str, ufpr_root: Path = UFPR_ROOT_DEFAULT) -> dict:
    """Loads one full track: every (frame_idx, image, gt_bbox, gt_text)
    in frame order.

    Returns:
        {
          "track_dir": Path,
          "frames": [(frame_idx:int, timestamp:None, image:ndarray), ...],
          "gt_boxes_per_frame": {frame_idx: [(bbox, track_id_str)]},
          "gt_text": str,   # ground-truth plate text for the whole track
                             # (assumed constant across frames; asserted below)
        }
    """
    track_dir = find_track_dir(track_id, ufpr_root)
    if track_dir is None:
        raise FileNotFoundError(f"no on-disk folder found for UFPR track {track_id} under {ufpr_root}")

    # NOTE: cannot use track_dir.glob("track{id}[*].png") here — the real
    # filenames contain literal square brackets (e.g. "track0021[01].png"),
    # and glob patterns treat [ and ] as "match one of these characters"
    # syntax, not literal characters. That pattern silently matches nothing
    # (a valid empty match, not an error) rather than raising, so listing +
    # regex-filtering is used instead of globbing.
    all_files = sorted(track_dir.iterdir())
    png_files = [
        p for p in all_files
        if p.suffix.lower() in (".png", ".jpg", ".jpeg") and FRAME_FILENAME_RE.search(p.name)
    ]

    frames = []
    gt_boxes_per_frame = {}
    gt_texts_seen = set()

    for png_path in png_files:
        m = FRAME_FILENAME_RE.search(png_path.name)
        if not m:
            continue
        frame_idx = int(m.group(2))
        txt_path = png_path.with_suffix(".txt")
        if not txt_path.exists():
            continue  # frame image with no annotation — skip, can't get GT for it

        ann = parse_annotation(txt_path)
        gt_texts_seen.add(ann["plate_text"])
        gt_boxes_per_frame[frame_idx] = [(ann["bbox"], track_id)]

        if cv2 is not None:
            image = cv2.imread(str(png_path))
            if image is None:
                raise IOError(f"cv2 failed to read {png_path}")
        else:
            image = str(png_path)  # fallback for environments without cv2/real images (unit tests)
        frames.append((frame_idx, None, image))

    frames.sort(key=lambda t: t[0])

    if len(gt_texts_seen) > 1:
        # UFPR ground truth is per-track in this project's usage, but the
        # dataset technically stores plate text per frame. Surface a
        # disagreement loudly rather than silently picking one — this
        # would indicate a data issue worth checking before trusting results.
        raise ValueError(
            f"track {track_id}: ground-truth plate text is not constant across frames: {gt_texts_seen}"
        )

    gt_text = gt_texts_seen.pop() if gt_texts_seen else None

    return {
        "track_dir": track_dir,
        "frames": frames,
        "gt_boxes_per_frame": gt_boxes_per_frame,
        "gt_text": gt_text,
    }
