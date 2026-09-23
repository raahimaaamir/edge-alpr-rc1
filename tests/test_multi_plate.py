"""
Regression test for multi-plate support (per supervisor instruction,
post-handoff engineering pass, item 1). Confirms:

  1. AlprPipeline.process() is UNCHANGED — still only ever returns the
     single highest-confidence detection, exactly as before this change.
  2. AlprPipeline.process_all() returns every detection in a frame,
     each processed independently and correctly.
  3. VideoAlprPipeline.process_video(..., multi_plate=True) correctly
     produces TWO SEPARATE TRACKS for two plates present in every frame
     of a synthetic video — the actual end-to-end scenario the
     supervisor asked to be covered by a test.
  4. multi_plate=False (the default) is completely unaffected — old
     behavior, old output shape, process_all() never even called.

Uses fake detector/recognizer components (no real ONNX model needed),
same pattern as test_stage3_synthetic.py / test_video_pipeline_integration.py.
"""

import sys
from pathlib import Path

# alpr_pipeline.py and tiled_detection_rescue.py (a transitive import via
# week9_rescue_policy, not used directly here, but video_pipeline.py's
# sibling modules don't need this) import their siblings as `pipeline.xxx`
# — put the directory ABOVE pipeline/ on sys.path for that. Flat imports
# (tracker, fusion, video_pipeline itself) need pipeline/ itself on
# sys.path — same dual-path pattern used throughout this project's Week 9
# scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass, field
from typing import Optional, List
import types
import numpy as np

# --- minimal real-shaped stubs for pipeline.result_types / pipeline.quality,
# so alpr_pipeline.py's real AlprPipeline can be imported and run without
# the actual ONNX detector/recognizer or the real quality.py. ---
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
quality_mod.assess_quality = lambda crop, min_width=40, min_height=15: QualityIndicators(crop.shape[1], crop.shape[0], 100.0, False, True)
quality_mod.MIN_WIDTH = 40
quality_mod.MIN_HEIGHT = 15

# alpr_pipeline.py imports these at module load time regardless of
# whether the defaults are ever instantiated (we always pass explicit
# detector=/recognizer= below) — still need real modules present.
detector_mod = types.ModuleType("pipeline.detector")
recognizer_mod = types.ModuleType("pipeline.recognizer")
detector_mod.PlateDetector = object  # never actually instantiated in this test
recognizer_mod.PlateRecognizer = object

sys.modules["pipeline"] = pipeline_pkg
sys.modules["pipeline.result_types"] = result_types_mod
sys.modules["pipeline.quality"] = quality_mod
sys.modules["pipeline.detector"] = detector_mod
sys.modules["pipeline.recognizer"] = recognizer_mod

from alpr_pipeline import AlprPipeline
from video_pipeline import VideoAlprPipeline, VideoPipelineConfig
from tracker import IoUTracker, TrackerConfig
from fusion import FusionConfig


class FakeDetector:
    """Always finds the same two plates, at the same two locations,
    confidence-descending (matching the real detector's convention)."""
    def detect(self, image):
        return [
            (BoundingBox(10, 10, 60, 40), 0.95),
            (BoundingBox(200, 10, 250, 40), 0.90),
        ]


class FakeRecognizer:
    """Returns a different plate depending on which bbox was cropped —
    matching real crop coordinates back to a specific plate string, so
    the test can independently verify BOTH plates end up correct, not
    just that two results exist."""
    def recognize(self, crop):
        # left plate crop is 50px wide (x1=10,x2=60); right plate crop
        # is also 50px wide but we can't distinguish by width alone here,
        # so use mean pixel value as a stand-in "which crop is this"
        # signal, seeded differently per bbox in the test frames below.
        if crop.mean() < 128:
            return ("ABC1234", [0.9] * 7, 0.9)
        return ("XYZ9876", [0.85] * 7, 0.85)


def make_frame():
    """One frame containing two distinguishable regions: darker on the
    left (where bbox1 crops from) and lighter on the right (bbox2)."""
    frame = np.full((100, 300, 3), 200, dtype="uint8")  # light background
    frame[10:40, 10:60] = 50  # dark region under bbox1 -> 'ABC1234'
    return frame


def test_process_unchanged():
    """process() must still return ONLY the single highest-confidence
    detection — exactly the pre-existing behavior."""
    pipeline = AlprPipeline(detector=FakeDetector(), recognizer=FakeRecognizer())
    frame = make_frame()
    result = pipeline.process(frame)
    assert result.status == "ok"
    assert result.detector_confidence == 0.95  # the higher-confidence one
    assert result.plate_text == "ABC1234"  # the dark (left) region
    print("PASS: process() unchanged — still returns only the top detection")


def test_process_all_returns_both():
    """process_all() must return BOTH detections, each correctly matched
    to its own crop and recognized independently."""
    pipeline = AlprPipeline(detector=FakeDetector(), recognizer=FakeRecognizer())
    frame = make_frame()
    results = pipeline.process_all(frame)
    assert len(results) == 2
    texts = sorted(r.plate_text for r in results)
    assert texts == ["ABC1234", "XYZ9876"]
    print(f"PASS: process_all() returns both plates independently: {[r.plate_text for r in results]}")


def test_two_plates_produce_two_tracks():
    """The end-to-end scenario: two plates present in every frame of a
    short synthetic video must produce TWO SEPARATE TRACKS when
    multi_plate=True, each correctly read across all frames — using the
    real tracker and fusion code, not mocks of them."""
    single_frame_pipeline = AlprPipeline(detector=FakeDetector(), recognizer=FakeRecognizer())
    frames = [(i, i / 30.0, make_frame()) for i in range(6)]
    config = VideoPipelineConfig(tracker=TrackerConfig(), fusion=FusionConfig(), fusion_method="majority_vote")
    video_pipeline = VideoAlprPipeline(single_frame_pipeline, config, tracker_cls=IoUTracker)

    track_results, tracks, stats = video_pipeline.process_video(frames, multi_plate=True)

    assert len(tracks) == 2, f"expected 2 separate tracks (one per plate), got {len(tracks)}"
    texts = sorted(tr.text for tr in track_results if tr.status == "ok")
    assert texts == ["ABC1234", "XYZ9876"], f"expected both plates tracked and fused correctly, got {texts}"
    for tr in track_results:
        assert tr.num_frames_used == 6, f"expected all 6 frames used for {tr.text}, got {tr.num_frames_used}"
    print(f"PASS: multi_plate=True produces 2 separate tracks, both correctly read across all 6 frames: {texts}")


def test_multi_plate_false_default_unaffected():
    """multi_plate defaults to False and must behave EXACTLY as before
    this change — single highest-confidence detection only, one track."""
    single_frame_pipeline = AlprPipeline(detector=FakeDetector(), recognizer=FakeRecognizer())
    frames = [(i, i / 30.0, make_frame()) for i in range(6)]
    config = VideoPipelineConfig(tracker=TrackerConfig(), fusion=FusionConfig(), fusion_method="majority_vote")
    video_pipeline = VideoAlprPipeline(single_frame_pipeline, config, tracker_cls=IoUTracker)

    track_results, tracks, stats = video_pipeline.process_video(frames)  # multi_plate not passed -> False

    assert len(tracks) == 1, f"default (multi_plate=False) should produce exactly 1 track, got {len(tracks)}"
    assert track_results[0].text == "ABC1234"  # only ever the top detection
    print("PASS: multi_plate=False (default) unaffected — exactly 1 track, only the top detection")


if __name__ == "__main__":
    test_process_unchanged()
    test_process_all_returns_both()
    test_two_plates_produce_two_tracks()
    test_multi_plate_false_default_unaffected()
    print("\nALL MULTI-PLATE TESTS PASSED")
