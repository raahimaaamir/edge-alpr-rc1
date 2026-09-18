"""
track_types.py

Data types for the video-tracking / multi-frame-fusion stage. These sit on
top of the existing single-frame result_types.py (BoundingBox, PlateResult,
etc.) without modifying it — a FrameObservation is built FROM a PlateResult
plus the frame index/timestamp, so nothing about the frozen single-frame
pipeline changes.

Design note: every dataclass here has a .to_dict() that returns only JSON
-safe primitives (same convention as PlateResult), so track-level results can
be dumped straight to JSON for the report.
"""

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Optional

from plate_normalize import normalize_plate_text


def _timings_to_dict(timings) -> Optional[dict]:
    """StageTimings (result_types.py) is a plain dataclass with NO .to_dict()
    method — only PlateResult has one, via asdict(). Confirmed against
    source, not assumed, after this exact assumption caused a bug once
    already. Handles a dict already, a dataclass instance, or None."""
    if timings is None:
        return None
    if isinstance(timings, dict):
        return timings
    if hasattr(timings, "to_dict"):
        return timings.to_dict()
    if is_dataclass(timings):
        return asdict(timings)
    return timings


@dataclass
class FrameObservation:
    """One frame's worth of single-frame-pipeline output, attached to a track.

    Built from a PlateResult (see result_types.py) plus frame context. Uses
    duck typing via `from_plate_result` so it works whether or not the exact
    PlateResult class is importable in a given environment.
    """

    frame_idx: int
    timestamp: Optional[float]  # seconds, if known; else None and frame_idx is authoritative
    bbox: tuple  # (x1, y1, x2, y2) pixel coords
    detector_conf: float
    status: str  # "ok" | "no_detection" | "low_quality" | "error"
    # ocr_text is the NORMALIZED prediction (uppercase, spacing/hyphen
    # stripped — see plate_normalize.py) and is what tracking/fusion/eval
    # code compares against ground truth. ocr_text_raw is the untouched
    # recognizer output, kept for audit/debugging — never discarded.
    ocr_text: Optional[str] = None
    ocr_text_raw: Optional[str] = None
    char_confs: Optional[list] = None
    overall_conf: Optional[float] = None
    plate_width_px: Optional[float] = None
    plate_height_px: Optional[float] = None
    blur_score: Optional[float] = None
    is_blurry: Optional[bool] = None
    min_dimension_ok: Optional[bool] = None
    mean_brightness: Optional[float] = None
    is_overexposed: Optional[bool] = None
    is_underexposed: Optional[bool] = None
    stage_timings_ms: Optional[dict] = None
    # Continuous crop-completeness telemetry — minimum pixel distance from
    # any bbox edge to the corresponding frame edge (min(x1, y1,
    # frame_w-x2, frame_h-y2)). Smaller means closer to being clipped by
    # the frame boundary; <= 0 means the bbox actually touches/exceeds it.
    # Replaces an earlier binary touch/no-touch check, which came back
    # completely uninformative on the validation set (0 of 294 crops ever
    # literally touched an edge, even including cases visibly close to
    # it). Recorded as telemetry only — NOT used as an acceptance gate
    # yet; per the supervisor's instruction, deriving a real operating
    # threshold needs a broader validation set containing genuinely
    # clipped/degraded examples, which the current validation set does
    # not have enough of.
    edge_margin_px: Optional[float] = None

    @staticmethod
    def _bbox_to_tuple(bbox) -> Optional[tuple]:
        """Accepts either a BoundingBox-like object (x1/y1/x2/y2 attributes,
        e.g. result_types.BoundingBox) or a plain (x1,y1,x2,y2) tuple/list."""
        if bbox is None:
            return None
        if hasattr(bbox, "x1"):
            return (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
        return tuple(bbox)

    @classmethod
    def from_plate_result(cls, frame_idx: int, timestamp: Optional[float], plate_result,
                           edge_margin_px: Optional[float] = None) -> "FrameObservation":
        """Build from a PlateResult-like object (duck-typed: works with the
        real result_types.PlateResult or any object exposing the same
        attribute names).

        edge_margin_px: optional, computed by the CALLER (video_pipeline.py
        has both the original frame and the bbox in scope at the call
        site; this classmethod deliberately does not reach into the
        frozen single-frame pipeline to get them itself)."""
        bbox = getattr(plate_result, "bbox", None)
        bbox_tuple = cls._bbox_to_tuple(bbox)
        quality = getattr(plate_result, "quality", None)
        timings = getattr(plate_result, "timings", None)
        raw_text = getattr(plate_result, "plate_text", None)
        return cls(
            frame_idx=frame_idx,
            timestamp=timestamp,
            bbox=bbox_tuple,
            detector_conf=getattr(plate_result, "detector_confidence", None),
            status=getattr(plate_result, "status", "error"),
            # field names below match the real result_types.PlateResult
            # (confirmed against source, not assumed): plate_text,
            # per_char_confidence, recognition_confidence — NOT text/
            # char_confidences/overall_confidence.
            ocr_text=normalize_plate_text(raw_text),
            ocr_text_raw=raw_text,
            char_confs=getattr(plate_result, "per_char_confidence", None),
            overall_conf=getattr(plate_result, "recognition_confidence", None),
            plate_width_px=getattr(quality, "crop_width", None) if quality else None,
            plate_height_px=getattr(quality, "crop_height", None) if quality else None,
            blur_score=getattr(quality, "blur_score", None) if quality else None,
            is_blurry=getattr(quality, "is_blurry", None) if quality else None,
            min_dimension_ok=getattr(quality, "min_dimension_ok", None) if quality else None,
            mean_brightness=getattr(quality, "mean_brightness", None) if quality else None,
            is_overexposed=getattr(quality, "is_overexposed", None) if quality else None,
            is_underexposed=getattr(quality, "is_underexposed", None) if quality else None,
            stage_timings_ms=_timings_to_dict(timings),
            edge_margin_px=edge_margin_px,
        )

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "bbox": list(self.bbox) if self.bbox else None,
            "detector_conf": self.detector_conf,
            "status": self.status,
            "ocr_text": self.ocr_text,
            "ocr_text_raw": self.ocr_text_raw,
            "char_confs": self.char_confs,
            "overall_conf": self.overall_conf,
            "plate_width_px": self.plate_width_px,
            "plate_height_px": self.plate_height_px,
            "blur_score": self.blur_score,
            "is_blurry": self.is_blurry,
            "min_dimension_ok": self.min_dimension_ok,
            "mean_brightness": self.mean_brightness,
            "is_overexposed": self.is_overexposed,
            "is_underexposed": self.is_underexposed,
            "stage_timings_ms": self.stage_timings_ms,
            "edge_margin_px": self.edge_margin_px,
        }


@dataclass
class Track:
    """A sequence of FrameObservations believed to be the same physical
    plate, produced by a Tracker implementation."""

    track_id: int
    observations: list = field(default_factory=list)
    frames_since_update: int = 0
    closed: bool = False

    @property
    def last_bbox(self) -> Optional[tuple]:
        for obs in reversed(self.observations):
            if obs.bbox is not None:
                return obs.bbox
        return None

    @property
    def num_observations(self) -> int:
        return len(self.observations)

    @property
    def num_ok_observations(self) -> int:
        return sum(1 for o in self.observations if o.status == "ok" and o.ocr_text)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "num_observations": self.num_observations,
            "observations": [o.to_dict() for o in self.observations],
        }


@dataclass
class TrackResult:
    """Final, fused, track-level output — the thing stage 3 is actually
    supposed to produce: one plate string (or an explicit abstention) per
    track, plus enough provenance to debug it."""

    track_id: int
    status: str  # "ok" | "no_reliable_result"
    text: Optional[str]
    confidence: Optional[float]
    fusion_method: str
    num_frames_total: int
    num_frames_used: int
    frame_idxs_used: list = field(default_factory=list)
    reason: Optional[str] = None  # populated when status == "no_reliable_result"
    candidates: Optional[dict] = None  # text -> vote/weight, for debugging disagreement

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "status": self.status,
            "text": self.text,
            "confidence": self.confidence,
            "fusion_method": self.fusion_method,
            "num_frames_total": self.num_frames_total,
            "num_frames_used": self.num_frames_used,
            "frame_idxs_used": self.frame_idxs_used,
            "reason": self.reason,
            "candidates": self.candidates,
        }
