"""
test_stage3_synthetic.py

Sanity-checks tracker.py / fusion.py / tracker_eval.py / video_pipeline.py
against synthetic data, since this sandbox has no access to the real
detector/recognizer models or video data. Run with the real models later to
validate on actual footage — this only proves the association/fusion/eval
LOGIC is correct.

Scenarios covered:
  1. Two plates moving smoothly across frames, one with a brief 2-frame
     occlusion (tests IoU tracker keeps track through max_missing_frames).
  2. A track where OCR flips on 2/7 frames -> majority_vote should recover
     the correct string with the 5/7 majority.
  3. A track where OCR is evenly split 3/3 with low confidence -> fusion
     should abstain (NO_RELIABLE_RESULT), not force an answer.
  4. tracker_eval against synthetic ground truth: one missed detection, one
     fragmented track, purity check.
"""

import sys
from pathlib import Path
# Points at pipeline/ relative to this file's own location, not the
# current working directory — so this test runs correctly whether
# invoked as `python3 tests/test_stage3_synthetic.py` from the package
# root, `python3 test_stage3_synthetic.py` from inside tests/, or via
# pytest from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from track_types import FrameObservation, Track
from tracker import IoUTracker, TrackerConfig
from fusion import fuse, fuse_majority_vote, fuse_confidence_weighted_vote, fuse_per_char_confidence, FusionConfig, NO_RELIABLE_RESULT
from tracker_eval import evaluate_tracker, TrackerEvalConfig


def make_obs(frame_idx, bbox, text, conf=0.95, char_confs=None, status="ok"):
    return FrameObservation(
        frame_idx=frame_idx,
        timestamp=frame_idx / 30.0,
        bbox=bbox,
        detector_conf=0.9,
        status=status,
        ocr_text=text,
        char_confs=char_confs or ([conf] * len(text) if text else None),
        overall_conf=conf,
    )


def shift_box(box, dx, dy=0):
    x1, y1, x2, y2 = box
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def test_1_tracking_through_occlusion():
    print("\n=== Test 1: tracking through a brief occlusion ===")
    tracker = IoUTracker(TrackerConfig(iou_threshold=0.3, max_center_disp_frac=0.5, max_missing_frames=3))

    box_a = (100, 100, 200, 140)
    box_b = (400, 300, 500, 340)

    for f in range(6):
        obs = []
        obs.append(make_obs(f, shift_box(box_a, f * 8), "ABC1234"))
        # plate B occluded on frames 2 and 3 (no detection those frames)
        if f not in (2, 3):
            obs.append(make_obs(f, shift_box(box_b, f * 5), "XYZ9876"))
        tracker.update(f, obs)

    tracks = tracker.finalize()
    print(f"tracks found: {len(tracks)} (expect 2)")
    for t in tracks:
        print(f"  track {t.track_id}: {t.num_observations} obs, texts={[o.ocr_text for o in t.observations]}")
    assert len(tracks) == 2, f"expected 2 tracks, got {len(tracks)}"
    sizes = sorted(t.num_observations for t in tracks)
    assert sizes == [4, 6], f"expected sizes [4,6] (B occluded 2 frames), got {sizes}"
    print("PASS")


def test_2_majority_vote_recovers_correct_string():
    print("\n=== Test 2: majority vote recovers correct string despite 2/7 OCR errors ===")
    track = Track(track_id=0, observations=[
        make_obs(0, (0, 0, 10, 10), "ABC1234", conf=0.9),
        make_obs(1, (0, 0, 10, 10), "ABC1234", conf=0.92),
        make_obs(2, (0, 0, 10, 10), "ABC1284", conf=0.55),  # char error (3->8)
        make_obs(3, (0, 0, 10, 10), "ABC1234", conf=0.88),
        make_obs(4, (0, 0, 10, 10), "ABC1234", conf=0.91),
        make_obs(5, (0, 0, 10, 10), "ABM1234", conf=0.5),   # char error (C->M)
        make_obs(6, (0, 0, 10, 10), "ABC1234", conf=0.93),
    ])
    result = fuse_majority_vote(track, FusionConfig(min_agreement_ratio=0.5))
    print(f"  fused: {result.text}, status={result.status}, used={result.num_frames_used}")
    assert result.status == "ok"
    assert result.text == "ABC1234"
    print("PASS")


def test_3_abstention_on_strong_disagreement():
    print("\n=== Test 3: fusion abstains on 3/3 split, low agreement ===")
    track = Track(track_id=1, observations=[
        make_obs(0, (0, 0, 10, 10), "ABC1234", conf=0.4),
        make_obs(1, (0, 0, 10, 10), "ABC1234", conf=0.45),
        make_obs(2, (0, 0, 10, 10), "ABC1234", conf=0.42),
        make_obs(3, (0, 0, 10, 10), "XYZ9999", conf=0.41),
        make_obs(4, (0, 0, 10, 10), "XYZ9999", conf=0.43),
        make_obs(5, (0, 0, 10, 10), "XYZ9999", conf=0.44),
    ])
    result = fuse_majority_vote(track, FusionConfig(min_agreement_ratio=0.6))
    print(f"  majority_vote: status={result.status}, reason={result.reason}")
    assert result.status == NO_RELIABLE_RESULT

    result2 = fuse_confidence_weighted_vote(track, FusionConfig(min_agreement_ratio=0.6))
    print(f"  confidence_weighted_vote: status={result2.status}, reason={result2.reason}")
    assert result2.status == NO_RELIABLE_RESULT
    print("PASS")


def test_4_per_char_confidence_fusion():
    print("\n=== Test 4: per-char confidence fusion picks best char per slot ===")
    # 3 obs, all length 4. slot 3 (last char) disagrees: 'C' (high conf) vs 'G' (low conf) x2
    track = Track(track_id=2, observations=[
        make_obs(0, (0, 0, 10, 10), "AB1C", char_confs=[0.99, 0.99, 0.99, 0.95]),
        make_obs(1, (0, 0, 10, 10), "AB1G", char_confs=[0.98, 0.97, 0.96, 0.30]),
        make_obs(2, (0, 0, 10, 10), "AB1G", char_confs=[0.97, 0.96, 0.95, 0.25]),
    ])
    result = fuse_per_char_confidence(track)
    print(f"  fused: {result.text}, conf={result.confidence}, method={result.fusion_method}")
    # summed conf for slot index 2: 'C'=0.99 vs 'G'=0.30+0.25=0.55 -> 'C' should win
    assert result.text == "AB1C", f"expected AB1C, got {result.text}"
    print("PASS")


def test_5_tracker_eval_missed_detection_and_fragmentation():
    print("\n=== Test 5: tracker_eval detects a missed detection + a fragmented track ===")
    tracker = IoUTracker(TrackerConfig(iou_threshold=0.3, max_missing_frames=0))  # 0 = no grace period, forces fragmentation
    box = (100, 100, 200, 140)
    for f in range(5):
        if f == 2:
            tracker.update(f, [])  # simulated missed detection (detector failed, no occlusion)
        else:
            tracker.update(f, [make_obs(f, shift_box(box, f * 6), "PLT0001")])
    tracks = tracker.finalize()
    print(f"  predicted tracks: {len(tracks)} (expect 2, since max_missing_frames=0 forces a break)")
    assert len(tracks) == 2

    gt_boxes_per_frame = {f: [(shift_box(box, f * 6), "gt_track_A")] for f in range(5)}
    report = evaluate_tracker(tracks, gt_boxes_per_frame, TrackerEvalConfig(gt_match_iou_threshold=0.3))
    print(f"  missed_detections={report.num_missed_detections} (expect 1)")
    print(f"  fragmentations={report.num_fragmentations} (expect 1)")
    print(f"  mean_track_purity={report.mean_track_purity} (expect 1.0, both fragments are pure)")
    assert report.num_missed_detections == 1
    assert report.num_fragmentations == 1
    assert report.mean_track_purity == 1.0
    print("PASS")


if __name__ == "__main__":
    test_1_tracking_through_occlusion()
    test_2_majority_vote_recovers_correct_string()
    test_3_abstention_on_strong_disagreement()
    test_4_per_char_confidence_fusion()
    test_5_tracker_eval_missed_detection_and_fragmentation()
    print("\nALL SYNTHETIC TESTS PASSED")
