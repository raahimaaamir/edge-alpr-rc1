"""Structured result types for the single-frame ALPR pipeline."""
from dataclasses import dataclass, field, asdict
from typing import Optional, List
@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    @property
    def height(self) -> float:
        return self.y2 - self.y1
@dataclass
class QualityIndicators:
    crop_width: int
    crop_height: int
    blur_score: float       # variance of Laplacian; higher = sharper
    is_blurry: bool
    min_dimension_ok: bool  # crop meets a minimum usable size
    mean_brightness: Optional[float] = None  # grayscale mean, 0-255
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
    # "ok" | "no_detection" | "low_quality" | "error"
    status: str
    reason: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    detector_confidence: Optional[float] = None
    plate_text: Optional[str] = None
    recognition_confidence: Optional[float] = None  # min over output characters
    per_char_confidence: Optional[List[float]] = None
    quality: Optional[QualityIndicators] = None
    timings: StageTimings = field(default_factory=StageTimings)
    crop_path: Optional[str] = None
    def to_dict(self) -> dict:
        """JSON-safe representation (no raw image arrays)."""
        return asdict(self)
