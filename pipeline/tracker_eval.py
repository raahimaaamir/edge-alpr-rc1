"""
tracker_eval.py

Stage 3, step 3: evaluate the *tracker* on its own, wherever ground-truth
track identity is available (e.g. UFPR-ALPR's real multi-frame tracks),
separately from OCR/fusion. This is what lets the report say "N% of
failures were missed detections, M% were broken/switched tracks, and the
rest were OCR/fusion" instead of one opaque end-to-end number.

Ground truth is supplied per frame as a list of (bbox, gt_track_id) — i.e.
whatever the dataset's own track/vehicle identity says was in that frame.
This is matched to the tracker's *predicted* observations by IoU (detector
boxes won't pixel-match annotation boxes exactly), not by object identity,
since the detector and the annotations are independent.
"""

from dataclasses import dataclass, field
from typing import Optional

from track_types import Track
from tracker import _iou  # reuse the same IoU implementation as the tracker itself


@dataclass
class TrackerEvalConfig:
    gt_match_iou_threshold: float = 0.5  # min IoU to say "this detection is of this GT plate"


@dataclass
class TrackerEvalReport:
    num_gt_tracks: int = 0
    num_gt_observations: int = 0
    num_predicted_tracks: int = 0
    num_missed_detections: int = 0          # GT observations with no matching predicted observation
    num_false_positive_tracks: int = 0      # predicted tracks with no GT match at all
    num_id_switches: int = 0                # frame-to-frame changes of dominant GT id within one predicted track
    num_fragmentations: int = 0             # GT tracks split across >1 predicted track (extra predicted tracks beyond the first)
    mean_track_purity: float = 0.0          # avg, across predicted tracks with any GT match, of (majority-GT-id count / matched observations)
    per_track_purity: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def _match_frame_gt(obs_bbox: tuple, gt_boxes: list, iou_threshold: float) -> Optional[str]:
    """Return the gt_track_id of the best-IoU ground-truth box on this frame,
    if it clears iou_threshold, else None."""
    best_id, best_iou = None, 0.0
    for gt_bbox, gt_id in gt_boxes:
        score = _iou(obs_bbox, gt_bbox)
        if score > best_iou:
            best_iou, best_id = score, gt_id
    return best_id if best_iou >= iou_threshold else None


def evaluate_tracker(
    predicted_tracks: list,
    gt_boxes_per_frame: dict,
    config: Optional[TrackerEvalConfig] = None,
) -> TrackerEvalReport:
    """
    predicted_tracks: list[Track], from tracker.finalize()
    gt_boxes_per_frame: {frame_idx: [(bbox, gt_track_id), ...]}
    """
    config = config or TrackerEvalConfig()
    report = TrackerEvalReport()

    gt_ids = {gid for boxes in gt_boxes_per_frame.values() for _, gid in boxes}
    report.num_gt_tracks = len(gt_ids)
    report.num_gt_observations = sum(len(boxes) for boxes in gt_boxes_per_frame.values())
    report.num_predicted_tracks = len(predicted_tracks)

    # match every predicted observation to a GT id (or None)
    track_gt_sequences: dict = {}  # track_id -> list of gt_id (or None) in frame order
    matched_gt_this_frame: dict = {fi: set() for fi in gt_boxes_per_frame}

    for track in predicted_tracks:
        seq = []
        for obs in track.observations:
            gt_id = None
            if obs.bbox is not None and obs.frame_idx in gt_boxes_per_frame:
                gt_id = _match_frame_gt(obs.bbox, gt_boxes_per_frame[obs.frame_idx], config.gt_match_iou_threshold)
                if gt_id is not None:
                    matched_gt_this_frame[obs.frame_idx].add(gt_id)
            seq.append(gt_id)
        track_gt_sequences[track.track_id] = seq

    # missed detections: GT observations no predicted observation matched
    for frame_idx, boxes in gt_boxes_per_frame.items():
        for _, gt_id in boxes:
            if gt_id not in matched_gt_this_frame.get(frame_idx, set()):
                report.num_missed_detections += 1

    # purity + id switches, per predicted track
    gt_track_to_predicted_tracks: dict = {}  # gt_id -> set(predicted track_id) that ever matched it
    purities = []
    for track in predicted_tracks:
        seq = [g for g in track_gt_sequences[track.track_id] if g is not None]
        if not seq:
            report.num_false_positive_tracks += 1
            report.per_track_purity[track.track_id] = 0.0
            continue
        counts: dict = {}
        for g in seq:
            counts[g] = counts.get(g, 0) + 1
            gt_track_to_predicted_tracks.setdefault(g, set()).add(track.track_id)
        majority_id, majority_count = max(counts.items(), key=lambda kv: kv[1])
        purity = majority_count / len(seq)
        purities.append(purity)
        report.per_track_purity[track.track_id] = round(purity, 4)

        # id switches: count changes in the dominant identity as we walk the
        # (gt-labeled-only) sequence — a rough proxy, good enough to flag
        # tracks that wandered onto a different plate mid-way
        switches = 0
        prev = seq[0]
        for g in seq[1:]:
            if g != prev:
                switches += 1
                prev = g
        report.num_id_switches += switches

    report.mean_track_purity = sum(purities) / len(purities) if purities else 0.0

    # fragmentation: for each GT track matched by >1 predicted track, count
    # the extra predicted tracks beyond the first as fragments
    for gt_id, predicted_ids in gt_track_to_predicted_tracks.items():
        if len(predicted_ids) > 1:
            report.num_fragmentations += len(predicted_ids) - 1

    return report
