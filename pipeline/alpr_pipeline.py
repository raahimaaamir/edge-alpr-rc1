"""
Deterministic single-frame ALPR pipeline:
image -> detect -> crop -> quality check -> recognize -> structured result.

Detector and recognizer are independent, replaceable components (see
detector.py / recognizer.py). Nothing in this file needs to change if either
component is swapped for a different implementation with the same interface.
"""
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pipeline.detector import PlateDetector
from pipeline.recognizer import PlateRecognizer
from pipeline.quality import assess_quality
from pipeline.result_types import PlateResult, StageTimings


class AlprPipeline:
    def __init__(self, detector: Optional[PlateDetector] = None, recognizer: Optional[PlateRecognizer] = None):
        self.detector = detector or PlateDetector()
        self.recognizer = recognizer or PlateRecognizer()

    def process(self, image: np.ndarray, save_crop_dir: Optional[str] = None, image_id: str = "frame") -> PlateResult:
        t_start = time.perf_counter()
        timings = StageTimings()

        # --- Detect ---
        t0 = time.perf_counter()
        try:
            detections = self.detector.detect(image)
        except Exception as e:
            timings.total_ms = (time.perf_counter() - t_start) * 1000
            return PlateResult(status="error", reason=f"detector_exception: {e}", timings=timings)
        timings.detect_ms = (time.perf_counter() - t0) * 1000

        if not detections:
            timings.total_ms = (time.perf_counter() - t_start) * 1000
            return PlateResult(status="no_detection", reason="no plate detected above confidence threshold", timings=timings)

        bbox, det_conf = detections[0]  # highest-confidence detection

        # --- Crop ---
        t0 = time.perf_counter()
        x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
        x2, y2 = min(image.shape[1], int(bbox.x2)), min(image.shape[0], int(bbox.y2))
        crop = image[y1:y2, x1:x2]
        timings.crop_ms = (time.perf_counter() - t0) * 1000

        if crop.size == 0:
            timings.total_ms = (time.perf_counter() - t_start) * 1000
            return PlateResult(
                status="error", reason="empty crop after clamping bbox to image bounds",
                bbox=bbox, detector_confidence=det_conf, timings=timings,
            )

        # --- Quality ---
        t0 = time.perf_counter()
        quality = assess_quality(crop)
        timings.quality_ms = (time.perf_counter() - t0) * 1000

        crop_path = None
        if save_crop_dir:
            Path(save_crop_dir).mkdir(parents=True, exist_ok=True)
            crop_path = str(Path(save_crop_dir) / f"{image_id}_crop.jpg")
            cv2.imwrite(crop_path, crop)

        if not quality.min_dimension_ok:
            timings.total_ms = (time.perf_counter() - t_start) * 1000
            return PlateResult(
                status="low_quality", reason=f"crop too small ({quality.crop_width}x{quality.crop_height})",
                bbox=bbox, detector_confidence=det_conf, quality=quality, timings=timings, crop_path=crop_path,
            )

        # --- Recognize ---
        t0 = time.perf_counter()
        try:
            text, per_char_conf, overall_conf = self.recognizer.recognize(crop)
        except Exception as e:
            timings.total_ms = (time.perf_counter() - t_start) * 1000
            return PlateResult(
                status="error", reason=f"recognizer_exception: {e}",
                bbox=bbox, detector_confidence=det_conf, quality=quality, timings=timings, crop_path=crop_path,
            )
        timings.recognize_ms = (time.perf_counter() - t0) * 1000
        timings.total_ms = (time.perf_counter() - t_start) * 1000

        if not text:
            return PlateResult(
                status="low_quality", reason="recognizer produced empty output",
                bbox=bbox, detector_confidence=det_conf, quality=quality, timings=timings, crop_path=crop_path,
            )

        return PlateResult(
            status="ok",
            bbox=bbox,
            detector_confidence=det_conf,
            plate_text=text,
            recognition_confidence=overall_conf,
            per_char_confidence=per_char_conf,
            quality=quality,
            timings=timings,
            crop_path=crop_path,
        )
