"""
week7_selection.py

Week 7: rank a track's observations using cheap, already-available signals,
and select the top K for fusion. Deliberately simple — per the plan, no
elaborate quality formula, and the exact calculation is documented here so
it can be reported plainly.

Composite score per observation, computed within one track's (encounter's)
own set of usable observations:

    composite = mean(
        normalized_plate_size,     # width_px * height_px, min-max normalized within the track
        normalized_sharpness,      # blur_score (higher = sharper), min-max normalized within the track
        detector_conf,             # already in [0, 1], used as-is
        ocr_conf,                  # overall_conf, already in [0, 1] (min over characters), used as-is
    )

Plate size and sharpness are normalized because they're in raw, unbounded
units (pixels^2, Laplacian variance) that aren't comparable to the two
confidence values out of the box. Min-max is computed PER TRACK (not
globally across the dataset) — this ranks a track's own frames against each
other, which is what frame selection needs; it says nothing about whether
one track's frames are better than another's, only which of THIS track's
frames to trust most. All four components are weighted equally (simple
average) — no attempt yet to learn or hand-tune relative weights.

If every observation in a track has the same plate size (or same sharpness),
min-max normalization would divide by zero — in that case the normalized
value is set to 1.0 for all observations (no basis to prefer one over
another on that dimension, so it doesn't penalize any of them).
"""

from typing import Optional


def _min_max_normalize(values: list) -> list:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def usable_observations(observations: list) -> list:
    """Observations usable for ranking/fusion: detection + OCR both
    succeeded (status "ok") and produced text. Matches fusion.py's
    _usable_observations definition, kept independent here since this
    module operates on plain observation lists, not Track objects."""
    return [o for o in observations if o.status == "ok" and o.ocr_text]


def compute_composite_scores(observations: list) -> dict:
    """observations: list of FrameObservation (already filtered to usable,
    or will be filtered here). Returns {frame_idx: composite_score}."""
    usable = usable_observations(observations)
    if not usable:
        return {}

    plate_sizes = [(o.plate_width_px or 0) * (o.plate_height_px or 0) for o in usable]
    sharpness = [o.blur_score or 0.0 for o in usable]
    detector_confs = [max(0.0, min(1.0, o.detector_conf or 0.0)) for o in usable]
    ocr_confs = [max(0.0, min(1.0, o.overall_conf or 0.0)) for o in usable]

    norm_size = _min_max_normalize(plate_sizes)
    norm_sharp = _min_max_normalize(sharpness)

    scores = {}
    for i, o in enumerate(usable):
        composite = (norm_size[i] + norm_sharp[i] + detector_confs[i] + ocr_confs[i]) / 4.0
        scores[o.frame_idx] = composite
    return scores


def select_top_k(observations: list, scores: dict, k: Optional[int]) -> list:
    """observations: usable observations (same set compute_composite_scores
    was called on). k=None means 'all' (still sorted by score, for
    consistency / so best_frame is trivially the first element).
    Returns observations sorted by score descending, truncated to k."""
    usable = usable_observations(observations)
    ranked = sorted(usable, key=lambda o: scores.get(o.frame_idx, 0.0), reverse=True)
    if k is None:
        return ranked
    return ranked[:k]
