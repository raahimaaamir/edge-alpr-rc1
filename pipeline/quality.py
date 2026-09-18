"""Basic crop-quality indicators, used to decide whether a crop is worth recognizing."""
import cv2
import numpy as np
from pipeline.result_types import QualityIndicators

MIN_WIDTH = 40
MIN_HEIGHT = 15
BLUR_THRESHOLD = 60.0  # variance of Laplacian below this = likely too blurry
OVEREXPOSED_MEAN_THRESHOLD = 200.0  # mean grayscale brightness (0-255) above this = likely overexposed
UNDEREXPOSED_MEAN_THRESHOLD = 50.0  # mean grayscale brightness below this = likely underexposed


def assess_quality(crop_bgr: np.ndarray) -> QualityIndicators:
    h, w = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_brightness = float(gray.mean())
    return QualityIndicators(
        crop_width=w,
        crop_height=h,
        blur_score=blur_score,
        is_blurry=blur_score < BLUR_THRESHOLD,
        min_dimension_ok=(w >= MIN_WIDTH and h >= MIN_HEIGHT),
        mean_brightness=mean_brightness,
        # measurement only, per Week 5 scope — not wired into any accept/
        # reject decision yet (is_blurry/min_dimension_ok already were;
        # this stays purely informational until a later stage says otherwise)
        is_overexposed=mean_brightness > OVEREXPOSED_MEAN_THRESHOLD,
        is_underexposed=mean_brightness < UNDEREXPOSED_MEAN_THRESHOLD,
    )
