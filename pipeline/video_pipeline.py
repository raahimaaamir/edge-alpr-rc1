"""
video_pipeline.py

Stage 3 top-level orchestration:
    video / frame sequence
      -> per-frame AlprPipeline.process() [existing, frozen single-frame code]
      -> IoUTracker (or any BaseTracker)
      -> fuse() per track
      -> list[TrackResult]  (one plate string, or NO_RELIABLE_RESULT, per track)

Deliberately thin: this file's only job is wiring, so the tracker and fusion
stages stay independently testable and swappable (per supervisor's
instruction). It does not import cv2/video-decoding directly — callers pass
in an iterable of (frame_idx, timestamp, image) so this same code works for
a live video reader, a directory of extracted frames, or a synthetic test.
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional

from track_types import FrameObservation, Track, TrackResult
from tracker import BaseTracker, IoUTracker, TrackerConfig
from fusion import fuse, FusionConfig


@dataclass
class VideoPipelineConfig:
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    fusion_method: str = "majority_vote"
    detector_conf_thresh: float = 0.25  # detections below this never reach the tracker


@dataclass
class VideoRunStats:
    num_frames: int = 0
    num_frames_with_detection: int = 0
    num_tracks_total: int = 0
    num_tracks_ok: int = 0
    num_tracks_no_reliable_result: int = 0
    avg_frames_per_track: float = 0.0
    total_pipeline_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class VideoAlprPipeline:
    """Wires the single-frame AlprPipeline, a tracker, and a fusion method
    into one video-level entry point.

    single_frame_pipeline must expose .process(image, image_id=...) ->
    PlateResult (used when multi_plate=False, the default) and
    .process_all(image, image_id=...) -> list[PlateResult] (used when
    multi_plate=True) — matching alpr_pipeline.py's AlprPipeline exactly.
    """

    def __init__(
        self,
        single_frame_pipeline,
        config: Optional[VideoPipelineConfig] = None,
        tracker_cls=IoUTracker,
    ):
        self.single_frame_pipeline = single_frame_pipeline
        self.config = config or VideoPipelineConfig()
        self.tracker: BaseTracker = tracker_cls(self.config.tracker)

    def process_video(
        self,
        frames: Iterable,
        multi_plate: bool = False,
    ) -> tuple:
        """frames: iterable of (frame_idx: int, timestamp: Optional[float], image).
        Returns (list[TrackResult], list[Track], VideoRunStats).

        If multi_plate is True, calls single_frame_pipeline.process_all()
        instead of .process() — one detector call, every detection above
        threshold becomes its own observation candidate for that frame,
        and the tracker associates across all of them, so more than one
        vehicle's plate can be tracked from the same frame. Default False
        calls .process() (single highest-confidence detection per frame),
        matching RC1's original single-plate behavior exactly.
        """
        stats = VideoRunStats()

        for frame_idx, timestamp, image in frames:
            stats.num_frames += 1
            if multi_plate:
                results_list = self.single_frame_pipeline.process_all(image, image_id=f"frame_{frame_idx}")
            else:
                results_list = [self.single_frame_pipeline.process(image, image_id=f"frame_{frame_idx}")]

            observations = []
            for r in results_list:
                if r is None:
                    continue
                det_conf = getattr(r, "detector_confidence", None)
                if r.status != "no_detection" and (det_conf is None or det_conf >= self.config.detector_conf_thresh):
                    edge_margin_px = None
                    bbox = getattr(r, "bbox", None)
                    frame_shape = getattr(image, "shape", None)
                    if bbox is not None and frame_shape is not None and len(frame_shape) >= 2:
                        frame_h, frame_w = frame_shape[0], frame_shape[1]
                        edge_margin_px = min(bbox.x1, bbox.y1, frame_w - bbox.x2, frame_h - bbox.y2)
                    observations.append(FrameObservation.from_plate_result(frame_idx, timestamp, r,
                                                                            edge_margin_px=edge_margin_px))
                timings = getattr(r, "timings", None)
                if timings is not None:
                    total_ms = getattr(timings, "total_ms", None)
                    if total_ms is not None:
                        stats.total_pipeline_time_ms += total_ms

            if observations:
                stats.num_frames_with_detection += 1

            self.tracker.update(frame_idx, observations)

        tracks = self.tracker.finalize()
        stats.num_tracks_total = len(tracks)

        track_results = []
        for track in tracks:
            result = fuse(track, method=self.config.fusion_method, config=self.config.fusion)
            track_results.append(result)
            if result.status == "ok":
                stats.num_tracks_ok += 1
            else:
                stats.num_tracks_no_reliable_result += 1

        if tracks:
            stats.avg_frames_per_track = sum(t.num_observations for t in tracks) / len(tracks)

        return track_results, tracks, stats
