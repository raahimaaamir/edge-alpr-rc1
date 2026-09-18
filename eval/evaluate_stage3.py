"""
evaluate_stage3.py

Produces the numbers the supervisor asked for in the next report:
  - single-frame vs. fused exact-match accuracy
  - track coverage (fraction of tracks that produced an "ok" result vs.
    abstained)
  - track-association failures (missed detections, ID switches,
    fragmentation, false-positive tracks — from tracker_eval)
  - number of frames used per result
  - processing time

This only requires ground truth at the TRACK level (one true plate string
per gt_track_id) — exactly what's available for UFPR's real multi-frame
tracks. Run it after a real VideoAlprPipeline.process_video() call once
video data + models are available; wire it into the report generation step.
"""

from dataclasses import dataclass, field
from typing import Optional

from track_types import Track, TrackResult
from tracker_eval import TrackerEvalReport


@dataclass
class Stage3EvalReport:
    # accuracy
    single_frame_exact_match_acc: Optional[float] = None
    single_frame_n: int = 0
    fused_exact_match_acc: Optional[float] = None
    fused_n: int = 0
    per_method_fused_acc: dict = field(default_factory=dict)  # method_name -> (acc, n)

    # coverage
    num_tracks_total: int = 0
    num_tracks_ok: int = 0
    num_tracks_no_reliable_result: int = 0
    track_coverage: Optional[float] = None  # num_ok / num_total

    # frames used
    mean_frames_used_per_ok_result: Optional[float] = None
    mean_frames_total_per_track: Optional[float] = None

    # timing
    mean_pipeline_ms_per_frame: Optional[float] = None

    # tracker association failures (filled in if a TrackerEvalReport is supplied)
    tracker_eval: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "single_frame_exact_match_acc": self.single_frame_exact_match_acc,
            "single_frame_n": self.single_frame_n,
            "fused_exact_match_acc": self.fused_exact_match_acc,
            "fused_n": self.fused_n,
            "per_method_fused_acc": self.per_method_fused_acc,
            "num_tracks_total": self.num_tracks_total,
            "num_tracks_ok": self.num_tracks_ok,
            "num_tracks_no_reliable_result": self.num_tracks_no_reliable_result,
            "track_coverage": self.track_coverage,
            "mean_frames_used_per_ok_result": self.mean_frames_used_per_ok_result,
            "mean_frames_total_per_track": self.mean_frames_total_per_track,
            "mean_pipeline_ms_per_frame": self.mean_pipeline_ms_per_frame,
            "tracker_eval": self.tracker_eval,
        }


def evaluate_stage3(
    tracks: list,
    track_results: list,
    gt_text_by_track_id: dict,
    total_pipeline_time_ms: float = 0.0,
    num_frames_processed: int = 0,
    tracker_eval_report: Optional[TrackerEvalReport] = None,
    extra_fused_results_by_method: Optional[dict] = None,
) -> Stage3EvalReport:
    """
    tracks: list[Track] from tracker.finalize()
    track_results: list[TrackResult] from the primary fusion method (fuse())
    gt_text_by_track_id: {track_id: ground_truth_plate_text} — only tracks
        present here are scored (tracks with no matching GT are skipped,
        e.g. false-positive tracks; see tracker_eval for those separately)
    extra_fused_results_by_method: optional {method_name: list[TrackResult]}
        to additionally report per-method accuracy (best_single_frame vs.
        majority_vote vs. confidence_weighted_vote vs. per_char_confidence)
    """
    report = Stage3EvalReport()

    # --- single-frame accuracy: every individual "ok" observation, scored
    # against its track's ground truth ---
    sf_correct, sf_total = 0, 0
    for track in tracks:
        gt = gt_text_by_track_id.get(track.track_id)
        if gt is None:
            continue
        for obs in track.observations:
            if obs.status == "ok" and obs.ocr_text:
                sf_total += 1
                if obs.ocr_text == gt:
                    sf_correct += 1
    report.single_frame_n = sf_total
    report.single_frame_exact_match_acc = (sf_correct / sf_total) if sf_total else None

    # --- fused (track-level) accuracy, primary method ---
    fused_correct, fused_total = 0, 0
    frames_used_list = []
    for tr in track_results:
        gt = gt_text_by_track_id.get(tr.track_id)
        if gt is None:
            continue
        if tr.status == "ok":
            fused_total += 1
            frames_used_list.append(tr.num_frames_used)
            if tr.text == gt:
                fused_correct += 1
    report.fused_n = fused_total
    report.fused_exact_match_acc = (fused_correct / fused_total) if fused_total else None
    report.mean_frames_used_per_ok_result = (
        sum(frames_used_list) / len(frames_used_list) if frames_used_list else None
    )

    # --- optional per-method comparison ---
    if extra_fused_results_by_method:
        for method_name, results in extra_fused_results_by_method.items():
            m_correct, m_total = 0, 0
            for tr in results:
                gt = gt_text_by_track_id.get(tr.track_id)
                if gt is None or tr.status != "ok":
                    continue
                m_total += 1
                if tr.text == gt:
                    m_correct += 1
            report.per_method_fused_acc[method_name] = {
                "accuracy": (m_correct / m_total) if m_total else None,
                "n": m_total,
            }

    # --- coverage ---
    scored_tracks = [t for t in tracks if t.track_id in gt_text_by_track_id]
    report.num_tracks_total = len(scored_tracks)
    report.num_tracks_ok = sum(
        1 for tr in track_results if tr.track_id in gt_text_by_track_id and tr.status == "ok"
    )
    report.num_tracks_no_reliable_result = report.num_tracks_total - report.num_tracks_ok
    report.track_coverage = (
        report.num_tracks_ok / report.num_tracks_total if report.num_tracks_total else None
    )
    report.mean_frames_total_per_track = (
        sum(t.num_observations for t in scored_tracks) / len(scored_tracks) if scored_tracks else None
    )

    # --- timing ---
    report.mean_pipeline_ms_per_frame = (
        total_pipeline_time_ms / num_frames_processed if num_frames_processed else None
    )

    # --- tracker association failures ---
    if tracker_eval_report is not None:
        report.tracker_eval = tracker_eval_report.to_dict()

    return report


def print_report(report: Stage3EvalReport) -> None:
    d = report.to_dict()
    print("=== Stage 3 evaluation ===")
    print(f"Single-frame exact-match accuracy: {d['single_frame_exact_match_acc']} (n={d['single_frame_n']})")
    print(f"Fused (track-level) exact-match accuracy: {d['fused_exact_match_acc']} (n={d['fused_n']})")
    if d["per_method_fused_acc"]:
        print("Per fusion method:")
        for method, stats in d["per_method_fused_acc"].items():
            print(f"  {method}: acc={stats['accuracy']}, n={stats['n']}")
    print(f"Track coverage: {d['track_coverage']} ({d['num_tracks_ok']}/{d['num_tracks_total']} ok, "
          f"{d['num_tracks_no_reliable_result']} abstained)")
    print(f"Mean frames used per ok result: {d['mean_frames_used_per_ok_result']}")
    print(f"Mean frames total per track: {d['mean_frames_total_per_track']}")
    print(f"Mean pipeline time per frame (ms): {d['mean_pipeline_ms_per_frame']}")
    if d["tracker_eval"]:
        print("Tracker association failures:")
        for k, v in d["tracker_eval"].items():
            if k != "per_track_purity":
                print(f"  {k}: {v}")
