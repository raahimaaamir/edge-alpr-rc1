"""
Comprehensive tests around the actual RC1 decision path (decide_rc1,
plate-profile validation, and the rescue policy) — per supervisor
instruction, item 4 of the post-handoff engineering pass. Complements
test_stage3_synthetic.py (which tests the underlying tracker/fusion
utilities in isolation) by testing the FINAL decision layer itself,
which is what RC1 is actually frozen against.

Covers every scenario the supervisor listed explicitly:
  1. correct agreement -> accepted
  2. low confidence -> NO_RELIABLE_RESULT
  3. known profile + invalid structure -> rejected
  4. unknown jurisdiction/profile -> profile check skipped
  5. rescue candidate accepted when it passes RC1
  6. rescue candidate rejected by confidence/profile
  7. zero-detection rescue
  8. multiple detections in one frame -- NOT duplicated here; already
     covered end-to-end by test_multi_plate.py (item 1)
  9. output schema consistency across every status
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import types
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

# week9_rescue_policy imports tiled_detection_rescue, which imports
# `pipeline.result_types` and `pipeline.quality` at module load time —
# stub these BEFORE importing week9_rescue_policy, same pattern used in
# test_multi_plate.py.
pipeline_pkg = types.ModuleType("pipeline")
result_types_mod = types.ModuleType("pipeline.result_types")
quality_mod = types.ModuleType("pipeline.quality")


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class QualityIndicators:
    crop_width: int
    crop_height: int
    blur_score: float
    is_blurry: bool
    min_dimension_ok: bool
    mean_brightness: Optional[float] = None
    is_overexposed: Optional[bool] = None
    is_underexposed: Optional[bool] = None


@dataclass
class StageTimings:
    detect_ms: float = 0.0
    crop_ms: float = 0.0
    quality_ms: float = 0.0
    recognize_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class PlateResult:
    status: str
    reason: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    detector_confidence: Optional[float] = None
    plate_text: Optional[str] = None
    recognition_confidence: Optional[float] = None
    per_char_confidence: Optional[List[float]] = None
    quality: Optional[QualityIndicators] = None
    timings: StageTimings = field(default_factory=StageTimings)
    crop_path: Optional[str] = None


result_types_mod.BoundingBox = BoundingBox
result_types_mod.PlateResult = PlateResult
result_types_mod.StageTimings = StageTimings
result_types_mod.QualityIndicators = QualityIndicators
quality_mod.assess_quality = lambda crop: QualityIndicators(crop.shape[1], crop.shape[0], 100.0, False, True)
sys.modules["pipeline"] = pipeline_pkg
sys.modules["pipeline.result_types"] = result_types_mod
sys.modules["pipeline.quality"] = quality_mod

from decision_rule_rc1 import decide_rc1, MIN_WINNING_CONFIDENCE
from week9_decision_rule import AcceptanceConfig
from week9_rescue_policy import maybe_rescue, RescuePolicyConfig
from plate_profile import PlateProfile, validate as validate_plate_profile
from track_types import FrameObservation
from tiled_detection_rescue import TiledRescueConfig


def make_observation(frame_idx, ocr_text, overall_conf, detector_conf=0.9, status="ok"):
    """A minimally-complete FrameObservation for decision-path testing —
    only the fields decide_rc1's own logic actually inspects need
    meaningful values; the rest get sane, unremarkable defaults."""
    return FrameObservation(
        frame_idx=frame_idx, timestamp=frame_idx / 30.0, bbox=(10, 10, 60, 40),
        detector_conf=detector_conf, status=status, ocr_text=ocr_text,
        ocr_text_raw=ocr_text, char_confs=[overall_conf] * len(ocr_text or ""),
        overall_conf=overall_conf, plate_width_px=50, plate_height_px=25,
        blur_score=100.0, is_blurry=False, min_dimension_ok=True,
    )


# A profile that ONLY accepts 7-character LLLDDDD (matching the real
# UFPR_PROFILE convention) -- used for the profile-check tests.
STRICT_PROFILE = PlateProfile(name="test-strict", formats={7: "LLLDDDD"})


def test_correct_agreement_accepted():
    """3 observations agree on a valid 7-char plate at high confidence,
    no profile supplied -> ACCEPT."""
    obs = [make_observation(i, "ABC1234", 0.95) for i in range(3)]
    decision = decide_rc1(obs, AcceptanceConfig())
    assert decision.status == "ok", decision.reason
    assert decision.text == "ABC1234"
    print(f"PASS: correct agreement -> accepted (status={decision.status}, text={decision.text})")


def test_low_confidence_no_reliable_result():
    """3 observations agree perfectly, but confidence is below the
    0.90 threshold -> NO_RELIABLE_RESULT, not an accept."""
    obs = [make_observation(i, "ABC1234", 0.50) for i in range(3)]
    decision = decide_rc1(obs, AcceptanceConfig())
    assert decision.status == "no_reliable_result", decision.status
    assert "confidence" in decision.reason.lower()
    print(f"PASS: low confidence -> NO_RELIABLE_RESULT (reason={decision.reason})")


def test_known_profile_invalid_structure_rejected():
    """Observations agree at high confidence on a 6-character string,
    but the supplied profile only accepts 7-character LLLDDDD ->
    rejected by the profile check specifically (not agreement or
    confidence, which both pass cleanly here)."""
    obs = [make_observation(i, "ABC123", 0.95) for i in range(3)]  # 6 chars -- doesn't match LLLDDDD (7)
    decision = decide_rc1(obs, AcceptanceConfig(), profile=STRICT_PROFILE)
    assert decision.status == "no_reliable_result", decision.status
    assert "profile" in decision.reason.lower()
    print(f"PASS: known profile + invalid structure -> rejected (reason={decision.reason})")


def test_unknown_jurisdiction_profile_check_skipped():
    """The SAME 6-character string that test 3 rejected, but with NO
    profile supplied -> the profile check is skipped entirely, and the
    result is accepted (since agreement and confidence both pass)."""
    obs = [make_observation(i, "ABC123", 0.95) for i in range(3)]
    decision = decide_rc1(obs, AcceptanceConfig(), profile=None)
    assert decision.status == "ok", decision.reason
    assert decision.text == "ABC123"
    print(f"PASS: unknown jurisdiction -> profile check skipped, accepted (text={decision.text})")


class FakeDetector:
    """Stands in for pipeline.detector — used directly by
    tiled_rescue_detect_and_recognize, not through process()/process_all()."""
    def __init__(self, detections):
        self._detections = detections  # list of (BoundingBox, conf)
    def detect(self, image):
        return self._detections


class FakeRecognizer:
    """Stands in for pipeline.recognizer — also called directly."""
    def __init__(self, text, conf):
        self._text, self._conf = text, conf
    def recognize(self, crop):
        return (self._text, [self._conf] * len(self._text), self._conf)


class FakePipeline:
    """tiled_rescue_detect_and_recognize reaches into pipeline.detector
    and pipeline.recognizer directly (bypassing process()/process_all()
    entirely) — so this needs to expose those two attributes, matching
    AlprPipeline's real structure, not a process()-level mock."""
    def __init__(self, detector, recognizer):
        self.detector = detector
        self.recognizer = recognizer


def _fake_pick_candidates(all_frames, max_frames):
    return all_frames[:max_frames]


def _make_fake_frame():
    """A real numpy image (tiled rescue calls .shape and slices it) —
    grid=(1,1) in the tests below means exactly one tile == the whole
    frame, so tiling math stays trivial and doesn't affect the result."""
    return np.full((100, 100, 3), 128, dtype="uint8")


def test_rescue_accepted_when_it_passes_rc1():
    """Normal path is a detection-related failure (zero observations);
    rescue examines candidate frames and finds an agreeing, confident
    observation -> rescue's own decide_rc1 call passes, and its
    decision REPLACES the original failure."""
    normal_decision = decide_rc1([], AcceptanceConfig())  # zero observations -> detection failure
    assert normal_decision.is_detection_related_failure()

    detector = FakeDetector([(BoundingBox(10, 10, 60, 40), 0.9)])
    recognizer = FakeRecognizer("ABC1234", 0.95)
    pipeline = FakePipeline(detector, recognizer)
    all_frames = [(i, i / 30.0, _make_fake_frame()) for i in range(6)]

    config = RescuePolicyConfig(max_candidate_frames=3, tiled_config=TiledRescueConfig(grid=(1, 1)))
    outcome = maybe_rescue(pipeline, all_frames, normal_decision, _fake_pick_candidates, config)
    assert outcome.triggered
    assert outcome.decision.status == "ok", outcome.decision.reason
    assert outcome.decision.text == "ABC1234"
    print(f"PASS: rescue accepted when it independently passes RC1 (text={outcome.decision.text})")


def test_rescue_rejected_by_confidence_keeps_original():
    """Same detection-failure starting point, but rescue's own
    observation is low-confidence -> rescue's decide_rc1 call fails its
    OWN bar, so the ORIGINAL (failure) decision is kept unchanged, not
    silently downgraded or replaced with a low-confidence accept."""
    normal_decision = decide_rc1([], AcceptanceConfig())

    detector = FakeDetector([(BoundingBox(10, 10, 60, 40), 0.9)])
    recognizer = FakeRecognizer("ABC1234", 0.40)  # below MIN_WINNING_CONFIDENCE
    pipeline = FakePipeline(detector, recognizer)
    all_frames = [(i, i / 30.0, _make_fake_frame()) for i in range(6)]

    config = RescuePolicyConfig(max_candidate_frames=3, tiled_config=TiledRescueConfig(grid=(1, 1)))
    outcome = maybe_rescue(pipeline, all_frames, normal_decision, _fake_pick_candidates, config)
    assert outcome.triggered  # rescue WAS attempted...
    assert outcome.decision.status == normal_decision.status  # ...but the original decision is kept
    assert outcome.decision.text is None
    print(f"PASS: rescue rejected by confidence -> original decision kept unchanged "
          f"(status={outcome.decision.status})")


def test_zero_detection_rescue():
    """The empty-track case: the plate was never detected in ANY sampled
    frame (no Track object even exists), but rescue is still offered
    the chance to recover it from full-resolution candidate frames."""
    empty_decision = decide_rc1([], AcceptanceConfig())
    assert empty_decision.status == "complete_detection_failure"
    assert empty_decision.is_detection_related_failure()

    detector = FakeDetector([(BoundingBox(10, 10, 60, 40), 0.9)])
    recognizer = FakeRecognizer("XYZ9876", 0.95)
    pipeline = FakePipeline(detector, recognizer)
    all_frames = [(i, i / 30.0, _make_fake_frame()) for i in range(6)]

    config = RescuePolicyConfig(max_candidate_frames=3, tiled_config=TiledRescueConfig(grid=(1, 1)))
    outcome = maybe_rescue(pipeline, all_frames, empty_decision, _fake_pick_candidates, config)
    assert outcome.triggered
    assert outcome.decision.status == "ok"
    assert outcome.decision.text == "XYZ9876"
    print(f"PASS: zero-detection case correctly offered to rescue, and recovered (text={outcome.decision.text})")
def test_output_schema_consistency():
    """Every decide_rc1() result -- accept, reject, or complete failure
    -- must expose the SAME dict keys, so a caller never has to
    special-case which fields exist based on status."""
    accept = decide_rc1([make_observation(i, "ABC1234", 0.95) for i in range(3)], AcceptanceConfig())
    reject = decide_rc1([make_observation(i, "ABC1234", 0.50) for i in range(3)], AcceptanceConfig())
    empty = decide_rc1([], AcceptanceConfig())

    keys_accept = set(accept.to_dict().keys())
    keys_reject = set(reject.to_dict().keys())
    keys_empty = set(empty.to_dict().keys())

    assert keys_accept == keys_reject == keys_empty, (
        f"schema mismatch: accept={keys_accept}, reject={keys_reject}, empty={keys_empty}"
    )
    print(f"PASS: output schema is identical across all statuses ({sorted(keys_accept)})")


if __name__ == "__main__":
    test_correct_agreement_accepted()
    test_low_confidence_no_reliable_result()
    test_known_profile_invalid_structure_rejected()
    test_unknown_jurisdiction_profile_check_skipped()
    test_rescue_accepted_when_it_passes_rc1()
    test_rescue_rejected_by_confidence_keeps_original()
    test_zero_detection_rescue()
    test_output_schema_consistency()
    print("\nALL RC1 DECISION PATH TESTS PASSED")
    print("(item 8 from the supervisor's list -- multiple detections in one frame -- ")
    print(" is covered by test_multi_plate.py, not duplicated here)")
