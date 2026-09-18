"""
fusion.py

Stage 3, step 2: turn a Track (many per-frame OCR observations) into one
TrackResult, or an explicit NO_RELIABLE_RESULT abstention.

Four methods, in increasing sophistication, all sharing the same signature
so they're trivial to swap/compare in the report:
  1. best_single_frame       — take the observation with highest overall_conf
  2. majority_vote           — most common exact-match string wins
  3. confidence_weighted_vote— votes weighted by overall_conf instead of count
  4. per_char_confidence     — fuse per-character, argmax-of-summed-confidence
                                 per slot (falls back to weighted vote if
                                 texts have mismatched lengths or char confs
                                 are missing)

Every method is followed by the same abstention gate: if the winning
candidate doesn't clear min_agreement_ratio of the usable evidence, return
NO_RELIABLE_RESULT instead of forcing an answer.
"""

from dataclasses import dataclass
from typing import Optional

from track_types import Track, TrackResult

NO_RELIABLE_RESULT = "no_reliable_result"


@dataclass
class FusionConfig:
    min_agreement_ratio: float = 0.5   # winning candidate's vote share must be >= this
    min_frames_used: int = 1           # need at least this many usable ("ok") observations
    min_overall_conf: float = 0.0      # per-frame observations below this conf are dropped
                                        # before fusion (0.0 = no filtering)


def _usable_observations(track: Track, config: FusionConfig) -> list:
    return [
        o for o in track.observations
        if o.status == "ok" and o.ocr_text and (o.overall_conf or 0.0) >= config.min_overall_conf
    ]


def _abstain(track: Track, method: str, reason: str, num_total: int) -> TrackResult:
    return TrackResult(
        track_id=track.track_id,
        status=NO_RELIABLE_RESULT,
        text=None,
        confidence=None,
        fusion_method=method,
        num_frames_total=num_total,
        num_frames_used=0,
        frame_idxs_used=[],
        reason=reason,
    )


def fuse_best_single_frame(track: Track, config: Optional[FusionConfig] = None) -> TrackResult:
    config = config or FusionConfig()
    method = "best_single_frame"
    usable = _usable_observations(track, config)
    if len(usable) < config.min_frames_used:
        return _abstain(track, method, "no usable observations", track.num_observations)
    best = max(usable, key=lambda o: o.overall_conf or 0.0)
    return TrackResult(
        track_id=track.track_id,
        status="ok",
        text=best.ocr_text,
        confidence=best.overall_conf,
        fusion_method=method,
        num_frames_total=track.num_observations,
        num_frames_used=1,
        frame_idxs_used=[best.frame_idx],
    )


def fuse_majority_vote(track: Track, config: Optional[FusionConfig] = None) -> TrackResult:
    config = config or FusionConfig()
    method = "majority_vote"
    usable = _usable_observations(track, config)
    if len(usable) < config.min_frames_used:
        return _abstain(track, method, "no usable observations", track.num_observations)

    counts: dict = {}
    for o in usable:
        counts[o.ocr_text] = counts.get(o.ocr_text, 0) + 1
    total = sum(counts.values())
    winner, votes = max(counts.items(), key=lambda kv: kv[1])
    share = votes / total

    if share < config.min_agreement_ratio:
        return _abstain(track, method, f"top candidate share {share:.2f} < {config.min_agreement_ratio}", track.num_observations)

    used = [o.frame_idx for o in usable if o.ocr_text == winner]
    avg_conf = sum(o.overall_conf or 0.0 for o in usable if o.ocr_text == winner) / len(used)
    return TrackResult(
        track_id=track.track_id,
        status="ok",
        text=winner,
        confidence=avg_conf,
        fusion_method=method,
        num_frames_total=track.num_observations,
        num_frames_used=len(usable),
        frame_idxs_used=used,
        candidates=counts,
    )


def fuse_confidence_weighted_vote(track: Track, config: Optional[FusionConfig] = None) -> TrackResult:
    config = config or FusionConfig()
    method = "confidence_weighted_vote"
    usable = _usable_observations(track, config)
    if len(usable) < config.min_frames_used:
        return _abstain(track, method, "no usable observations", track.num_observations)

    weights: dict = {}
    for o in usable:
        weights[o.ocr_text] = weights.get(o.ocr_text, 0.0) + (o.overall_conf or 0.0)
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return _abstain(track, method, "zero total confidence weight", track.num_observations)
    winner, weight = max(weights.items(), key=lambda kv: kv[1])
    share = weight / total_weight

    if share < config.min_agreement_ratio:
        return _abstain(track, method, f"top candidate weight share {share:.2f} < {config.min_agreement_ratio}", track.num_observations)

    used = [o.frame_idx for o in usable if o.ocr_text == winner]
    return TrackResult(
        track_id=track.track_id,
        status="ok",
        text=winner,
        confidence=weight / len(used),
        fusion_method=method,
        num_frames_total=track.num_observations,
        num_frames_used=len(usable),
        frame_idxs_used=used,
        candidates={k: round(v, 4) for k, v in weights.items()},
    )


def fuse_per_char_confidence(track: Track, config: Optional[FusionConfig] = None) -> TrackResult:
    """Fuse per character: for each slot position, sum per-char confidence
    across all observations sharing that string length, pick the argmax
    char per slot. Falls back to confidence_weighted_vote if char_confs are
    missing or texts disagree on length (can't align slots meaningfully)."""
    config = config or FusionConfig()
    method = "per_char_confidence"
    usable = _usable_observations(track, config)
    if len(usable) < config.min_frames_used:
        return _abstain(track, method, "no usable observations", track.num_observations)

    lengths = {len(o.ocr_text) for o in usable}
    have_char_confs = all(o.char_confs and len(o.char_confs) == len(o.ocr_text) for o in usable)

    if len(lengths) != 1 or not have_char_confs:
        fallback = fuse_confidence_weighted_vote(track, config)
        fallback.fusion_method = f"{method} (fell back to confidence_weighted_vote: " + (
            "length mismatch across observations)" if len(lengths) != 1 else "missing char confidences)"
        )
        return fallback

    plate_len = lengths.pop()
    slot_scores = [dict() for _ in range(plate_len)]  # slot -> {char: summed_conf}
    for o in usable:
        for i, ch in enumerate(o.ocr_text):
            slot_scores[i][ch] = slot_scores[i].get(ch, 0.0) + o.char_confs[i]

    fused_chars = []
    per_slot_conf = []
    for slot in slot_scores:
        ch, score = max(slot.items(), key=lambda kv: kv[1])
        fused_chars.append(ch)
        per_slot_conf.append(score / len(usable))
    fused_text = "".join(fused_chars)
    overall_conf = min(per_slot_conf) if per_slot_conf else 0.0

    # agreement check: what fraction of observations already matched the
    # fused string outright (a proxy for how much true disagreement there was)
    exact_matches = sum(1 for o in usable if o.ocr_text == fused_text)
    share = exact_matches / len(usable) if usable else 0.0
    if share < config.min_agreement_ratio and exact_matches == 0:
        # nothing observed even agrees with the fused answer at all — treat
        # as unreliable rather than presenting a Frankenstein string nothing
        # actually produced
        return _abstain(track, method, "fused string matches zero individual observations", track.num_observations)

    return TrackResult(
        track_id=track.track_id,
        status="ok",
        text=fused_text,
        confidence=overall_conf,
        fusion_method=method,
        num_frames_total=track.num_observations,
        num_frames_used=len(usable),
        frame_idxs_used=[o.frame_idx for o in usable],
    )


def fuse_per_char_majority(track: Track, config: Optional[FusionConfig] = None) -> TrackResult:
    """Week 7: plain per-character MAJORITY VOTE (counts, not confidence) —
    for each slot position, the most common character wins. Distinct from
    fuse_per_char_confidence, which sums confidence rather than counting
    votes. Falls back to fuse_majority_vote if texts disagree on length."""
    config = config or FusionConfig()
    method = "per_char_majority"
    usable = _usable_observations(track, config)
    if len(usable) < config.min_frames_used:
        return _abstain(track, method, "no usable observations", track.num_observations)

    lengths = {len(o.ocr_text) for o in usable}
    if len(lengths) != 1:
        fallback = fuse_majority_vote(track, config)
        fallback.fusion_method = f"{method} (fell back to majority_vote: length mismatch across observations)"
        return fallback

    plate_len = lengths.pop()
    slot_counts = [dict() for _ in range(plate_len)]
    for o in usable:
        for i, ch in enumerate(o.ocr_text):
            slot_counts[i][ch] = slot_counts[i].get(ch, 0) + 1

    fused_chars = []
    per_slot_share = []
    for slot in slot_counts:
        ch, count = max(slot.items(), key=lambda kv: kv[1])
        fused_chars.append(ch)
        per_slot_share.append(count / len(usable))
    fused_text = "".join(fused_chars)
    overall_conf = min(per_slot_share) if per_slot_share else 0.0

    exact_matches = sum(1 for o in usable if o.ocr_text == fused_text)
    if exact_matches == 0:
        return _abstain(track, method, "fused string matches zero individual observations", track.num_observations)

    return TrackResult(
        track_id=track.track_id,
        status="ok",
        text=fused_text,
        confidence=overall_conf,
        fusion_method=method,
        num_frames_total=track.num_observations,
        num_frames_used=len(usable),
        frame_idxs_used=[o.frame_idx for o in usable],
    )


def fuse_confidence_weighted_char_quality(
    track: Track,
    quality_weights: Optional[dict] = None,
    config: Optional[FusionConfig] = None,
) -> TrackResult:
    """Week 7: per-character fusion where each observation's contribution
    is (char confidence x frame-quality weight), accumulated per slot —
    distinct from fuse_per_char_confidence, which uses raw char confidence
    only. quality_weights: {frame_idx: weight}, e.g. the composite score
    from week7_selection.py. A frame_idx missing from quality_weights gets
    weight 1.0 (no quality-based reweighting for that frame)."""
    config = config or FusionConfig()
    quality_weights = quality_weights or {}
    method = "confidence_weighted_char_quality"
    usable = _usable_observations(track, config)
    if len(usable) < config.min_frames_used:
        return _abstain(track, method, "no usable observations", track.num_observations)

    lengths = {len(o.ocr_text) for o in usable}
    have_char_confs = all(o.char_confs and len(o.char_confs) == len(o.ocr_text) for o in usable)

    if len(lengths) != 1 or not have_char_confs:
        fallback = fuse_confidence_weighted_vote(track, config)
        fallback.fusion_method = f"{method} (fell back to confidence_weighted_vote: " + (
            "length mismatch across observations)" if len(lengths) != 1 else "missing char confidences)"
        )
        return fallback

    plate_len = lengths.pop()
    slot_scores = [dict() for _ in range(plate_len)]
    total_weight = 0.0
    for o in usable:
        w = quality_weights.get(o.frame_idx, 1.0)
        total_weight += w
        for i, ch in enumerate(o.ocr_text):
            slot_scores[i][ch] = slot_scores[i].get(ch, 0.0) + o.char_confs[i] * w

    fused_chars = []
    per_slot_conf = []
    for slot in slot_scores:
        ch, score = max(slot.items(), key=lambda kv: kv[1])
        fused_chars.append(ch)
        per_slot_conf.append(score / total_weight if total_weight > 0 else 0.0)
    fused_text = "".join(fused_chars)
    overall_conf = min(per_slot_conf) if per_slot_conf else 0.0

    exact_matches = sum(1 for o in usable if o.ocr_text == fused_text)
    if exact_matches == 0:
        return _abstain(track, method, "fused string matches zero individual observations", track.num_observations)

    return TrackResult(
        track_id=track.track_id,
        status="ok",
        text=fused_text,
        confidence=overall_conf,
        fusion_method=method,
        num_frames_total=track.num_observations,
        num_frames_used=len(usable),
        frame_idxs_used=[o.frame_idx for o in usable],
    )


FUSION_METHODS = {
    "best_single_frame": fuse_best_single_frame,
    "majority_vote": fuse_majority_vote,
    "confidence_weighted_vote": fuse_confidence_weighted_vote,
    "per_char_confidence": fuse_per_char_confidence,
    "per_char_majority": fuse_per_char_majority,
}


def fuse(track: Track, method: str = "majority_vote", config: Optional[FusionConfig] = None) -> TrackResult:
    if method not in FUSION_METHODS:
        raise ValueError(f"unknown fusion method '{method}', choose from {list(FUSION_METHODS)}")
    return FUSION_METHODS[method](track, config)
