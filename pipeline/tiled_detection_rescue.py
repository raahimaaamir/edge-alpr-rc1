"""
tiled_detection_rescue.py

Week 8, step 3: a lightweight rescue path for encounters where the normal
pipeline produced no track or too few usable detections. Splits one or a
few candidate FULL-RESOLUTION frames into overlapping tiles, runs the
EXISTING, UNMODIFIED detector (pipeline.detector.PlateDetector) on each
tile, maps detections back to full-frame coordinates, deduplicates
overlapping detections (simple greedy IoU suppression), and runs the
EXISTING, UNMODIFIED recognizer + quality check on any surviving
detections — reusing the frozen components exactly as alpr_pipeline.py
does, just orchestrated differently for the tiled case (which
AlprPipeline.process() does not support, since it always takes the single
top detection on one full image).

Nothing about the detector or recognizer changes. This module is meant to
be invoked ONLY for failure-case frames identified by the normal pipeline —
never on every frame — per the plan.
"""

import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from pipeline.result_types import BoundingBox, PlateResult, StageTimings, QualityIndicators
from pipeline.quality import assess_quality


def generate_tiles(image: np.ndarray, grid: Tuple[int, int] = (2, 2), overlap_frac: float = 0.15) -> list:
    """Splits image into an n_rows x n_cols grid of overlapping tiles.
    Returns list of (tile_image, x_offset, y_offset) — offsets are needed
    to map tile-local detections back to full-frame coordinates."""
    h, w = image.shape[:2]
    n_rows, n_cols = grid
    tile_h, tile_w = h / n_rows, w / n_cols
    overlap_h, overlap_w = tile_h * overlap_frac, tile_w * overlap_frac

    tiles = []
    for r in range(n_rows):
        for c in range(n_cols):
            y1 = max(0, int(r * tile_h - overlap_h))
            y2 = min(h, int((r + 1) * tile_h + overlap_h))
            x1 = max(0, int(c * tile_w - overlap_w))
            x2 = min(w, int((c + 1) * tile_w + overlap_w))
            tiles.append((image[y1:y2, x1:x2], x1, y1))
    return tiles


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    ax1, ay1, ax2, ay2 = a.x1, a.y1, a.x2, a.y2
    bx1, by1, bx2, by2 = b.x1, b.y1, b.x2, b.y2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _greedy_nms(detections: list, iou_thresh: float = 0.3) -> list:
    """detections: list of (BoundingBox, confidence). Standard greedy NMS —
    keep the highest-confidence detection, discard anything overlapping it
    above iou_thresh, repeat. Needed because overlapping tiles will often
    detect the same plate twice."""
    ranked = sorted(detections, key=lambda d: d[1], reverse=True)
    kept = []
    for bbox, conf in ranked:
        if all(_iou(bbox, kept_bbox) < iou_thresh for kept_bbox, _ in kept):
            kept.append((bbox, conf))
    return kept


@dataclass
class TiledRescueConfig:
    grid: Tuple[int, int] = (2, 2)
    overlap_frac: float = 0.15
    nms_iou_thresh: float = 0.3
    max_results: int = 3  # cap how many candidate plates we bother recognizing per frame


def tiled_rescue_detect_and_recognize(pipeline, image: np.ndarray, config: TiledRescueConfig = None) -> Tuple[List[PlateResult], float]:
    """pipeline: an AlprPipeline instance — reuses its .detector and
    .recognizer directly, unmodified. Returns (list[PlateResult],
    tiling_latency_ms) — the latency covers ONLY this tiled-rescue call,
    reported separately since it should only ever run on failure-case
    frames, never as part of the normal per-frame cost.
    """
    config = config or TiledRescueConfig()
    t0 = time.perf_counter()

    tiles = generate_tiles(image, config.grid, config.overlap_frac)
    all_detections = []
    for tile_img, x_off, y_off in tiles:
        if tile_img.size == 0:
            continue
        tile_detections = pipeline.detector.detect(tile_img)
        for bbox, conf in tile_detections:
            mapped = BoundingBox(x1=bbox.x1 + x_off, y1=bbox.y1 + y_off,
                                  x2=bbox.x2 + x_off, y2=bbox.y2 + y_off)
            all_detections.append((mapped, conf))

    deduped = _greedy_nms(all_detections, config.nms_iou_thresh)
    deduped.sort(key=lambda d: d[1], reverse=True)
    deduped = deduped[: config.max_results]

    results = []
    for bbox, det_conf in deduped:
        timings = StageTimings()  # only crop/quality/recognize measured per-result; tiling cost reported separately
        t_stage = time.perf_counter()
        x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
        x2, y2 = min(image.shape[1], int(bbox.x2)), min(image.shape[0], int(bbox.y2))
        crop = image[y1:y2, x1:x2]
        timings.crop_ms = (time.perf_counter() - t_stage) * 1000
        if crop.size == 0:
            results.append(PlateResult(status="error", reason="empty crop after clamping", bbox=bbox,
                                        detector_confidence=det_conf, timings=timings))
            continue

        t_stage = time.perf_counter()
        quality = assess_quality(crop)
        timings.quality_ms = (time.perf_counter() - t_stage) * 1000

        if not quality.min_dimension_ok:
            results.append(PlateResult(status="low_quality", reason=f"crop too small ({quality.crop_width}x{quality.crop_height})",
                                        bbox=bbox, detector_confidence=det_conf, quality=quality, timings=timings))
            continue

        t_stage = time.perf_counter()
        try:
            text, per_char_conf, overall_conf = pipeline.recognizer.recognize(crop)
        except Exception as e:
            results.append(PlateResult(status="error", reason=f"recognizer_exception: {e}",
                                        bbox=bbox, detector_confidence=det_conf, quality=quality, timings=timings))
            continue
        timings.recognize_ms = (time.perf_counter() - t_stage) * 1000

        if not text:
            results.append(PlateResult(status="low_quality", reason="recognizer produced empty output",
                                        bbox=bbox, detector_confidence=det_conf, quality=quality, timings=timings))
            continue

        results.append(PlateResult(
            status="ok", bbox=bbox, detector_confidence=det_conf, plate_text=text,
            recognition_confidence=overall_conf, per_char_confidence=per_char_conf,
            quality=quality, timings=timings,
        ))

    tiling_latency_ms = (time.perf_counter() - t0) * 1000
    return results, tiling_latency_ms
