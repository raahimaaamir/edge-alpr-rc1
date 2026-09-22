"""
alpr_system.py

Week 9, step 3: makes the core inference path usable as a component, not
just an experiment script. Exposes:

    process_image(image) -> dict            # single image, no tracking
    process_frame(stream_id, frame, ...)    -> dict   # one frame of a video/stream
    finalize_track(track, ...) -> dict      # one finished track -> plate or NO_RELIABLE_RESULT
    finalize_stream(stream_id, ...) -> list[dict]   # every track in a finished/ended stream

The streaming path does NOT require a completed video file — process_frame
is called once per frame as frames arrive, with per-stream tracker state
kept internally (IoUTracker.update() is already inherently incremental;
this class just exposes that directly instead of only through the
batch-oriented VideoAlprPipeline.process_video() wrapper used in earlier
weeks' scripts). An MP4 runner is provided as a thin WRAPPER
(week9_mp4_runner.py) that happens to read a file, but the core class
itself works identically fed from a live camera loop, a socket, or
anything else that hands it frames one at a time.

ALL tunable parameters live in SystemConfig — frame stride, tracker
thresholds, Top-K, minimum agreement, rescue enable/config, and (with the
one documented limitation below) quality thresholds. Nothing here is
hard-coded.

LIMITATION, stated plainly rather than overclaimed: AlprPipeline.process()
(the frozen Week 5 single-frame pipeline) makes its own low_quality
accept/reject decision internally, using quality.py's own hard-coded
thresholds (MIN_WIDTH=40, MIN_HEIGHT=15, BLUR_THRESHOLD=60.0), BEFORE this
class ever sees the result — by the time process() returns, a low_quality
rejection has already happened and recognition was never attempted.
SystemConfig's quality_* fields are recorded and reported for
documentation/reproducibility, but per the instruction to keep the frozen
pipeline unchanged, they do not currently override that internal gate —
doing so would require either modifying quality.py (not done, per
instructions) or re-implementing crop/quality/recognize outside
AlprPipeline.process() (a larger change than this week's scope). This is
flagged here explicitly rather than silently claimed as configurable.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from tracker import IoUTracker, TrackerConfig
from track_types import FrameObservation, Track
from week9_decision_rule import AcceptanceConfig
from decision_rule_rc1 import decide_rc1, MIN_WINNING_CONFIDENCE
from week9_rescue_policy import maybe_rescue, RescuePolicyConfig
from plate_profile import PlateProfile


@dataclass
class SystemConfig:
    # frame sampling
    stride: int = 2
    detector_conf_thresh: float = 0.25

    # multi-plate support: when True, process_frame() calls
    # pipeline.process_all() and lets every detection above
    # detector_conf_thresh become its own observation, so more than one
    # vehicle's plate can be tracked from the same frame. Default False
    # matches RC1's original single-plate-per-frame behavior exactly —
    # existing callers see no change unless they opt in.
    multi_plate: bool = False

    # tracking (IoU / centre-displacement / max-missed-frames — Week 6)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)

    # Top-K selection + accept/NO_RELIABLE_RESULT rule (Week 7/8)
    acceptance: AcceptanceConfig = field(default_factory=AcceptanceConfig)

    # RC1 decision rule (supersedes plain agreement-only decide_v2): the
    # confidence threshold is a validation-selected conservative heuristic,
    # not a calibrated probability (see decision_rule_rc1.py). plate_profile
    # is optional — None means jurisdiction/plate class is unknown, so the
    # structural check is skipped entirely, never treated as a rejection.
    min_winning_confidence: float = MIN_WINNING_CONFIDENCE
    plate_profile: Optional[PlateProfile] = None

    # rescue path (Week 8, hardened Week 9)
    rescue_enabled: bool = True
    rescue: RescuePolicyConfig = field(default_factory=RescuePolicyConfig)

    # quality thresholds — recorded for documentation/reproducibility;
    # see the LIMITATION note in the module docstring for why these are
    # not (yet) live overrides of the frozen pipeline's internal gate.
    quality_min_width: int = 40
    quality_min_height: int = 15
    quality_blur_threshold: float = 60.0

    def to_dict(self) -> dict:
        return {
            "stride": self.stride,
            "detector_conf_thresh": self.detector_conf_thresh,
            "multi_plate": self.multi_plate,
            "tracker": self.tracker.__dict__,
            "acceptance": self.acceptance.__dict__,
            "rescue_enabled": self.rescue_enabled,
            "rescue": {
                "max_candidate_frames": self.rescue.max_candidate_frames,
                "tiled_config": self.rescue.tiled_config.__dict__,
                "rescue_acceptance": self.rescue.rescue_acceptance.__dict__,
            },
            "quality_min_width": self.quality_min_width,
            "quality_min_height": self.quality_min_height,
            "quality_blur_threshold": self.quality_blur_threshold,
        }


def pick_candidate_frames_evenly_spaced(all_frames: list, n: int) -> list:
    """Same logic as run_week8_rescue_experiment.pick_candidate_frames,
    duplicated here so alpr_system.py has no dependency on a
    driver/experiment script — a component should not import from a
    one-off script."""
    if len(all_frames) <= n:
        return all_frames
    step = len(all_frames) / n
    return [all_frames[int(i * step)] for i in range(n)]


class ALPRSystem:
    """The core, reusable inference component. Wraps an AlprPipeline
    instance (unchanged, frozen) plus tracking/selection/decision/rescue
    logic (Weeks 6-9) behind a small, stable interface."""

    def __init__(self, pipeline, config: Optional[SystemConfig] = None):
        """pipeline: an AlprPipeline instance (from alpr_pipeline.py,
        unmodified). Passed in rather than constructed here, so callers
        can supply a specific recognizer weights file (e.g. to compare V1
        vs V1.1) without this class needing to know about that."""
        self.pipeline = pipeline
        self.config = config or SystemConfig()
        self._streams: dict = {}

    # ---- single image, no tracking ----
    def process_image(self, image, image_id: str = "frame") -> dict:
        result = self.pipeline.process(image, image_id=image_id)
        return result.to_dict()

    # ---- streaming / video path ----
    def _get_stream(self, stream_id: str) -> dict:
        if stream_id not in self._streams:
            self._streams[stream_id] = {"tracker": IoUTracker(self.config.tracker), "frame_count": 0}
        return self._streams[stream_id]

    def process_frame(self, stream_id: str, frame, timestamp: Optional[float] = None,
                       frame_idx: Optional[int] = None) -> dict:
        """Feed ONE frame of a stream. Call this repeatedly as frames
        arrive — no need to have the whole video available up front.

        If self.config.multi_plate is True, every detection in the frame
        (not just the highest-confidence one) becomes its own observation
        candidate — see SystemConfig.multi_plate. The returned dict's
        shape differs in that case (a "results" list instead of a single
        "plate_text"/"status" pair), since there may be more than one
        plate to report on."""
        stream = self._get_stream(stream_id)
        idx = frame_idx if frame_idx is not None else stream["frame_count"]
        stream["frame_count"] += 1

        if idx % self.config.stride != 0:
            return {"stream_id": stream_id, "frame_idx": idx, "sampled": False, "status": "skipped_by_stride"}

        if self.config.multi_plate:
            results = self.pipeline.process_all(frame, image_id=f"{stream_id}_frame{idx}")
        else:
            results = [self.pipeline.process(frame, image_id=f"{stream_id}_frame{idx}")]

        observations = []
        for result in results:
            if result.status != "no_detection":
                det_conf = getattr(result, "detector_confidence", None)
                if det_conf is None or det_conf >= self.config.detector_conf_thresh:
                    observations.append(FrameObservation.from_plate_result(idx, timestamp, result))
        stream["tracker"].update(idx, observations)

        if self.config.multi_plate:
            return {
                "stream_id": stream_id, "frame_idx": idx, "sampled": True,
                "num_detections": len(results),
                "results": [{"status": r.status, "plate_text": getattr(r, "plate_text", None)} for r in results],
                "num_active_tracks": len(stream["tracker"]._active),
            }

        result = results[0]
        return {
            "stream_id": stream_id, "frame_idx": idx, "sampled": True, "status": result.status,
            "plate_text": getattr(result, "plate_text", None),
            "num_active_tracks": len(stream["tracker"]._active),
        }

    def finalize_track(self, track: Track, all_frames_for_rescue: Optional[list] = None) -> dict:
        """Applies Top-K selection + the decision rule + (if enabled and
        candidate frames are supplied) the hardened rescue path to ONE
        finished track. Returns the final decision as a JSON-safe dict."""
        decision = decide_rc1(track.observations, self.config.acceptance,
                               profile=self.config.plate_profile,
                               min_winning_confidence=self.config.min_winning_confidence)

        if (self.config.rescue_enabled and all_frames_for_rescue is not None
                and decision.is_detection_related_failure()):
            outcome = maybe_rescue(
                self.pipeline, all_frames_for_rescue, decision,
                pick_candidate_frames_evenly_spaced, self.config.rescue,
                profile=self.config.plate_profile,
                min_winning_confidence=self.config.min_winning_confidence,
            )
            decision = outcome.decision

        return decision.to_dict()

    def finalize_stream(self, stream_id: str,
                         candidate_frame_provider: Optional[Callable[[Optional[Track]], list]] = None) -> list:
        """Call once the stream has ended. Finalizes every track the
        tracker produced.

        candidate_frame_provider: optional callable taking a Track (or
        None) and returning full-resolution (frame_idx, ts, image) tuples
        for the rescue path to examine — e.g. the MP4 wrapper can supply
        "all frames of this video" since it has them; a live-camera caller
        might supply a bounded recent-frame buffer, or None to disable
        rescue for that stream (the core never buffers raw frames itself,
        keeping it lightweight for true streaming use).

        IMPORTANT: if the tracker produced ZERO tracks for the whole
        stream (a total detection failure across every sampled frame —
        e.g. the plate was never once detected), there is no Track object
        to attach a result to, but that is exactly the case rescue exists
        to recover. This method handles it explicitly: the decision is
        built from an empty observation pool (correctly resolving to
        complete_detection_failure) and, if rescue is enabled, still
        offered to it — candidate_frame_provider is called with track=None
        in this one case. Callers whose provider ignores its argument
        (the common case: `lambda track: all_frames`) need no changes."""
        stream = self._streams.pop(stream_id, None)
        if stream is None:
            return []
        tracks = stream["tracker"].finalize()

        if not tracks:
            empty_decision = decide_rc1([], self.config.acceptance,
                                         profile=self.config.plate_profile,
                                         min_winning_confidence=self.config.min_winning_confidence)
            if self.config.rescue_enabled and candidate_frame_provider is not None:
                frames_for_rescue = candidate_frame_provider(None)
                outcome = maybe_rescue(
                    self.pipeline, frames_for_rescue, empty_decision,
                    pick_candidate_frames_evenly_spaced, self.config.rescue,
                    profile=self.config.plate_profile,
                    min_winning_confidence=self.config.min_winning_confidence,
                )
                return [outcome.decision.to_dict()]
            return [empty_decision.to_dict()]

        results = []
        for track in tracks:
            frames_for_rescue = candidate_frame_provider(track) if candidate_frame_provider else None
            results.append(self.finalize_track(track, all_frames_for_rescue=frames_for_rescue))
        return results
