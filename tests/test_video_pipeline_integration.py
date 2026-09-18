"""Proves video_pipeline.py's wiring (mock single-frame pipeline -> tracker
-> fusion -> TrackResult) actually runs end to end. Uses a fake
single_frame_pipeline standing in for the real AlprPipeline so this runs
without the actual detector/recognizer models."""

import sys
from pathlib import Path
# Points at pipeline/ relative to this file's own location — see
# test_stage3_synthetic.py's comment for why this matters.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from dataclasses import dataclass
from typing import Optional

from video_pipeline import VideoAlprPipeline, VideoPipelineConfig
from tracker import TrackerConfig
from fusion import FusionConfig


@dataclass
class FakeQuality:
    blur_score: float = 200.0
    is_blurry: bool = False
    min_dimension_ok: bool = True


@dataclass
class FakeTimings:
    total_ms: float = 50.0
    def to_dict(self):
        return {"total_ms": self.total_ms}


@dataclass
class FakePlateResult:
    status: str
    bbox: Optional[tuple] = None
    detector_confidence: Optional[float] = None
    plate_text: Optional[str] = None
    per_char_confidence: Optional[list] = None
    recognition_confidence: Optional[float] = None
    quality: Optional[FakeQuality] = None
    timings: Optional[FakeTimings] = None


class FakeSingleFramePipeline:
    """Simulates one moving plate, correctly read every frame except frame 4
    (one OCR slip), plus one frame (frame 6) with no detection at all."""

    def process(self, image, image_id=None):
        frame_idx = image["frame_idx"]
        if frame_idx == 6:
            return FakePlateResult(status="no_detection")
        x = 100 + frame_idx * 10
        text = "ABC1234" if frame_idx != 4 else "ABC1284"
        return FakePlateResult(
            status="ok",
            bbox=(x, 100, x + 80, 140),
            detector_confidence=0.9,
            plate_text=text,
            per_char_confidence=[0.9] * 7,
            recognition_confidence=0.9 if frame_idx != 4 else 0.4,
            quality=FakeQuality(),
            timings=FakeTimings(),
        )


def main():
    fake_pipeline = FakeSingleFramePipeline()
    config = VideoPipelineConfig(
        tracker=TrackerConfig(iou_threshold=0.3, max_center_disp_frac=0.6, max_missing_frames=2),
        fusion=FusionConfig(min_agreement_ratio=0.5),
        fusion_method="majority_vote",
    )
    pipeline = VideoAlprPipeline(fake_pipeline, config)

    frames = [(f, f / 30.0, {"frame_idx": f}) for f in range(9)]
    track_results, tracks, stats = pipeline.process_video(frames)

    print("stats:", stats.to_dict())
    print("num tracks:", len(tracks))
    for tr in track_results:
        print(" ", tr.to_dict())

    assert len(tracks) == 1, f"expected 1 track (frame 6 no-detection should be absorbed by grace period), got {len(tracks)}"
    assert track_results[0].status == "ok"
    assert track_results[0].text == "ABC1234"
    assert stats.num_frames == 9
    assert stats.num_frames_with_detection == 8
    print("\nINTEGRATION TEST PASSED")


if __name__ == "__main__":
    main()
