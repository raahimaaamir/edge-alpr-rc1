"""
tracker.py

Stage 3, step 1: associate per-frame plate detections into tracks.

Deliberately the simplest thing that could work: greedy IoU + centre-
displacement matching with a "max missing frames" grace period, no motion
model. Kept behind BaseTracker so it can be swapped for something like
ByteTrack later without touching video_pipeline.py or fusion.py.

Config is passed in explicitly (TrackerConfig) rather than hard-coded, per
supervisor's instruction to keep thresholds in configuration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from track_types import FrameObservation, Track


@dataclass
class TrackerConfig:
    iou_threshold: float = 0.3          # min IoU to consider two boxes "the same plate"
    max_center_disp_frac: float = 0.5   # max centre displacement, as a fraction of the
                                         # mean of the two boxes' diagonals, to still match
    max_missing_frames: int = 5         # frames a track can go undetected before closing
    require_both_gates: bool = False    # if True, a match must pass IoU AND centre-disp;
                                         # if False (default), either gate passing is enough
                                         # (handles both "camera pans" and "plate shrinks/grows")


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center(box: tuple) -> tuple:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _diag(box: tuple) -> float:
    x1, y1, x2, y2 = box
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def _center_disp_frac(a: tuple, b: tuple) -> float:
    ax, ay = _center(a)
    bx, by = _center(b)
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    mean_diag = (_diag(a) + _diag(b)) / 2.0
    return dist / mean_diag if mean_diag > 0 else float("inf")


class BaseTracker(ABC):
    """Interface every tracker implementation must satisfy, so video_pipeline
    and tracker_eval never need to know which concrete tracker is in use."""

    @abstractmethod
    def update(self, frame_idx: int, observations: list) -> None:
        """Feed one frame's observations (list[FrameObservation], one per
        detection that frame; may be empty) into the tracker."""
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> list:
        """Call once after the last frame. Returns list[Track] — every track
        seen, open or closed."""
        raise NotImplementedError


class IoUTracker(BaseTracker):
    """Greedy IoU/centre-displacement tracker with a missing-frame allowance."""

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        self._active: list = []   # list[Track], still eligible for matching
        self._closed: list = []   # list[Track], finalized (exceeded max_missing_frames)
        self._next_id = 0

    def _new_track(self, obs: FrameObservation) -> Track:
        t = Track(track_id=self._next_id, observations=[obs])
        self._next_id += 1
        return t

    def _match(self, track: Track, obs: FrameObservation) -> bool:
        last_box = track.last_bbox
        if last_box is None or obs.bbox is None:
            return False
        iou_ok = _iou(last_box, obs.bbox) >= self.config.iou_threshold
        disp_ok = _center_disp_frac(last_box, obs.bbox) <= self.config.max_center_disp_frac
        if self.config.require_both_gates:
            return iou_ok and disp_ok
        return iou_ok or disp_ok

    def update(self, frame_idx: int, observations: list) -> None:
        # only detections with a real bbox can be associated; "no_detection"
        # frames simply advance every active track's missing-frame counter
        detected = [o for o in observations if o.bbox is not None]
        unmatched = list(detected)
        matched_track_ids = set()

        # greedy matching, largest IoU first, one detection per track per frame
        pairs = []
        for track in self._active:
            for obs in detected:
                last_box = track.last_bbox
                if last_box is None:
                    continue
                score = _iou(last_box, obs.bbox)
                if self._match(track, obs):
                    pairs.append((score, track, obs))
        pairs.sort(key=lambda p: p[0], reverse=True)

        used_obs_ids = set()
        for score, track, obs in pairs:
            if track.track_id in matched_track_ids or id(obs) in used_obs_ids:
                continue
            track.observations.append(obs)
            track.frames_since_update = 0
            matched_track_ids.add(track.track_id)
            used_obs_ids.add(id(obs))

        unmatched = [o for o in detected if id(o) not in used_obs_ids]

        # age out tracks that weren't matched this frame
        still_active = []
        for track in self._active:
            if track.track_id not in matched_track_ids:
                track.frames_since_update += 1
                if track.frames_since_update > self.config.max_missing_frames:
                    track.closed = True
                    self._closed.append(track)
                    continue
            still_active.append(track)
        self._active = still_active

        # start new tracks for anything left over
        for obs in unmatched:
            self._active.append(self._new_track(obs))

    def finalize(self) -> list:
        for track in self._active:
            track.closed = True
        all_tracks = self._closed + self._active
        self._closed, self._active = [], []
        return sorted(all_tracks, key=lambda t: t.track_id)
