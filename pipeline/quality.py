"""Basic crop-quality indicators, used to decide whether a crop is worth
recognizing.

IMPORTANT (per supervisor's post-handoff engineering pass): the minimum-
dimension check below (min_dimension_ok, gating on MIN_WIDTH/MIN_HEIGHT)
is a BASIC SANITY GUARD — it exists to reject crops too small to
plausibly contain readable characters at all (a handful of pixels), not
a validated, data-derived OCR-quality threshold. The project's
validation data was explicitly found not to span enough genuinely
degraded (small/blurred/clipped/low-light) examples to derive a real
operating threshold defensibly — see model_card_recognizer_v1_1.md's
limitations and decision_rule_rc1.py's module docstring for the full
reasoning; the same reasoning applies here. MIN_WIDTH/MIN_HEIGHT below
are the sane, conservative defaults, not tuned/validated numbers — and
are configurable per-call (see assess_quality's parameters) precisely
because they are a guard, not a calibrated cutoff, so a caller with
different crop-size expectations (a different camera setup, a different
detector) isn't stuck with these specific values.

blur_score/is_blurry and the exposure indicators remain measurement-only
telemetry, exactly as before — NOT used as an accept/reject gate
anywhere in the live pipeline, unlike min_dimension_ok."""
import cv2
import numpy as np
from pipeline.result_types import QualityIndicators

MIN_WIDTH = 40
MIN_HEIGHT = 15
BLUR_THRESHOLD = 60.0  # variance of Laplacian below this = likely too blurry
OVEREXPOSED_MEAN_THRESHOLD = 200.0  # mean grayscale brightness (0-255) above this = likely overexposed
UNDEREXPOSED_MEAN_THRESHOLD = 50.0  # mean grayscale brightness below this = likely underexposed


def assess_quality(crop_bgr: np.ndarray, min_width: int = MIN_WIDTH, min_height: int = MIN_HEIGHT) -> QualityIndicators:
    """min_width/min_height: the basic sanity-guard thresholds (see module
    docstring) — default to this module's own MIN_WIDTH/MIN_HEIGHT, but
    configurable per-call since they are not a validated, one-true-value
    cutoff."""
    h, w = crop_bgr.shape[:2]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_brightness = float(gray.mean())
    return QualityIndicators(
        crop_width=w,
        crop_height=h,
        blur_score=blur_score,
        is_blurry=blur_score < BLUR_THRESHOLD,
        min_dimension_ok=(w >= min_width and h >= min_height),
        mean_brightness=mean_brightness,
        # measurement only, per Week 5 scope — not wired into any accept/
        # reject decision yet (is_blurry/min_dimension_ok already were;
        # this stays purely informational until a later stage says otherwise)
        is_overexposed=mean_brightness > OVEREXPOSED_MEAN_THRESHOLD,
        is_underexposed=mean_brightness < UNDEREXPOSED_MEAN_THRESHOLD,
    )
