"""
Deterministic single-frame ALPR pipeline:
image -> detect -> crop -> quality check -> recognize -> structured result.

Detector and recognizer are independent, replaceable components (see
detector.py / recognizer.py). Nothing in this file needs to change if either
component is swapped for a different implementation with the same interface.

TWO ENTRY POINTS, deliberately different reliability semantics — see
README.md for the full explanation:

  process()     -> single PlateResult, from the single HIGHEST-CONFIDENCE
                    detection in the frame. This is a raw single-frame OCR
                    candidate — it does NOT apply the RC1 temporal
                    agreement / confidence / plate-profile reliability
                    layer (that only exists at the video/ALPRSystem level,
                    where multiple frames of the same vehicle are pooled).
                    Unchanged since RC1 was frozen — every existing caller
                    (examples/, tests/, the image entry point in README.md)
                    keeps working exactly as before.

  process_all()  -> list[PlateResult], one per detection in the frame
                     (multiple plates supported). Added for multi-plate
                     support (per supervisor instruction, item 1 of the
                     post-handoff engineering pass) — video_pipeline.py's
                     process_video(..., multi_plate=True) uses this to let
                     more than one plate per frame reach the tracker.
                     Same per-detection logic as process() (crop, quality
                     check, recognize), just applied to every detection
                     instead of only the top one. Also has no temporal
                     reliability layer of its own — that still only exists
                     at the video/ALPRSystem level, now correctly applied
                     per-plate once the tracker has grouped observations
                     across frames.
"""
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from pipeline.detector import PlateDetector
from pipeline.recognizer import PlateRecognizer
from pipeline.quality import assess_quality, MIN_WIDTH, MIN_HEIGHT
from pipeline.result_types import PlateResult, StageTimings


class AlprPipeline:
    def __init__(self, detector: Optional[PlateDetector] = None, recognizer: Optional[PlateRecognizer] = None,
                 min_crop_width: int = MIN_WIDTH, min_crop_height: int = MIN_HEIGHT):
        """min_crop_width/min_crop_height: the basic sanity-guard crop-size
        thresholds (see quality.py's module docstring — this is NOT a
        validated OCR-quality cutoff). Configurable here since they are a
        guard, not a calibrated one-true-value; default to quality.py's
        own defaults."""
        self.detector = detector or PlateDetector()
        self.recognizer = recognizer or PlateRecognizer()
        self.min_crop_width = min_crop_width
        self.min_crop_height = min_crop_height

    def _process_one_detection(self, image: np.ndarray, bbox, det_conf: float,
                                save_crop_dir: Optional[str], image_id: str,
                                timings: StageTimings, t_start: float) -> PlateResult:
        """Crop -> quality check -> recognize, for ONE already-detected
        bounding box. Shared by process() and process_all() so both stay
        behaviorally identical for a given detection — this is the only
        code path either method uses for this part of the work, so there
        is exactly one place to get it right."""
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

        t0 = time.perf_counter()
        quality = assess_quality(crop, min_width=self.min_crop_width, min_height=self.min_crop_height)
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

    def process(self, image: np.ndarray, save_crop_dir: Optional[str] = None, image_id: str = "frame") -> PlateResult:
        """UNCHANGED since RC1 was frozen. Own control flow, own detect()
        call, own exception/empty-detection handling — deliberately not
        implemented in terms of process_all(), so nothing about this
        method's behavior can be affected by process_all() existing."""
        t_start = time.perf_counter()
        timings = StageTimings()

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
        return self._process_one_detection(image, bbox, det_conf, save_crop_dir, image_id, timings, t_start)

    def process_all(self, image: np.ndarray, save_crop_dir: Optional[str] = None, image_id: str = "frame") -> List[PlateResult]:
        """Same per-detection logic as process() (via the shared helper),
        applied to EVERY detection in the frame, not just the top one.

        Returns:
          - [] if the detector finds zero detections (nothing to report
            per-item; an empty list is the natural representation)
          - [PlateResult(status="error", ...)] if the detector itself
            raises (a single item describing what broke, matching
            process()'s own error-reporting shape for this case)
          - one PlateResult per detection otherwise, in the same order
            self.detector.detect() returned them (typically confidence-
            descending, matching process()'s own detections[0] convention)
        """
        t_start = time.perf_counter()
        detect_timings = StageTimings()

        t0 = time.perf_counter()
        try:
            detections = self.detector.detect(image)
        except Exception as e:
            detect_timings.total_ms = (time.perf_counter() - t_start) * 1000
            return [PlateResult(status="error", reason=f"detector_exception: {e}", timings=detect_timings)]
        detect_ms = (time.perf_counter() - t0) * 1000

        if not detections:
            return []

        results = []
        for i, (bbox, det_conf) in enumerate(detections):
            timings = StageTimings(detect_ms=detect_ms)
            per_detection_t_start = time.perf_counter()
            detection_image_id = image_id if len(detections) == 1 else f"{image_id}_det{i}"
            results.append(self._process_one_detection(
                image, bbox, det_conf, save_crop_dir, detection_image_id, timings, per_detection_t_start,
            ))
        return results
