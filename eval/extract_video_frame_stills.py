"""
extract_video_frame_stills.py

Extracts ONE representative still frame from each of the 7 local videos
(video-01.mp4 ... video-07.mp4), for the final 13-local-plates comparison
(run_v1_1_local13_check.py needs full SCENE images, not video files).

Frame selection: samples N evenly-spaced candidate frames per video and
picks the one with the highest Laplacian-variance sharpness score (a
standard, detector-independent blur metric — higher variance means more
high-frequency detail, i.e. a sharper, more in-focus frame). This is a
deliberately simple, unbiased selection rule: it doesn't look at plate
text or run any recognition/detection model, so it can't be accused of
cherry-picking a frame that happens to favor one recognizer over another.

Run from inside the alpr-train container:
    cd /workspace/home/alpr-week5
    python3 pipeline/extract_video_frame_stills.py
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_VIDEOS_DIR = "/workspace/home/alpr-week5/data/raw/week1_videos"
DEFAULT_OUTPUT_DIR = "/workspace/home/alpr-week5/data/raw/week9_length_test"


def sharpness_score(frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_best_frame(video_path: Path, n_candidates: int = 12):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None, None

    # skip the first and last 10% (often has camera motion or nothing in
    # frame yet) — sample evenly spaced candidates from the middle 80%
    lo, hi = int(total_frames * 0.1), int(total_frames * 0.9)
    candidate_indices = np.linspace(lo, hi, num=min(n_candidates, max(1, hi - lo)), dtype=int)

    best_frame, best_score, best_idx = None, -1.0, None
    for idx in candidate_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        score = sharpness_score(frame)
        if score > best_score:
            best_score, best_frame, best_idx = score, frame, int(idx)

    cap.release()
    return best_frame, best_idx


def dump_all_candidate_frames(video_path: Path, output_dir: Path, n_candidates: int = 30) -> list:
    """Saves N evenly-spaced frames across the FULL video (no 10%/90%
    trim, unlike extract_best_frame) as individually numbered images, so
    a person can look through them and pick the clearest one themselves.
    Use this for a video where the automatic sharpness pick looks poor —
    e.g. one where the vehicle is moving away and the clearest moment
    might be right at the start, which the trimmed range would skip.
    Returns the list of (frame_idx, output_path) written."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    candidate_indices = np.linspace(0, total_frames - 1, num=min(n_candidates, total_frames), dtype=int)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for idx in candidate_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        out_path = output_dir / f"{video_path.stem}_frame{int(idx):04d}.jpg"
        cv2.imwrite(str(out_path), frame)
        written.append((int(idx), out_path))
    cap.release()
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", type=str, default=DEFAULT_VIDEOS_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-candidates", type=int, default=12)
    parser.add_argument("--dump-candidates-for", type=str, default=None,
                         help="Video key (e.g. 'video-07') to dump many full-range candidate "
                              "frames for, instead of doing the normal auto-pick. Writes "
                              "numbered frames to <output-dir>/candidates_<key>/ for manual review.")
    parser.add_argument("--dump-n-candidates", type=int, default=30)
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)

    if args.dump_candidates_for:
        video_path = videos_dir / f"{args.dump_candidates_for}.mp4"
        candidates_dir = output_dir / f"candidates_{args.dump_candidates_for}"
        written = dump_all_candidate_frames(video_path, candidates_dir, n_candidates=args.dump_n_candidates)
        print(f"Wrote {len(written)} candidate frames to {candidates_dir}")
        print("Look through them, pick the clearest, then run:")
        print(f"  cp {candidates_dir}/<chosen_frame>.jpg {output_dir}/{args.dump_candidates_for}.jpg")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(videos_dir.glob("video-*.mp4"))
    print(f"Found {len(video_paths)} video(s) in {videos_dir}")

    for video_path in video_paths:
        key = video_path.stem  # e.g. 'video-01'
        frame, idx = extract_best_frame(video_path, n_candidates=args.n_candidates)
        if frame is None:
            print(f"  {key}: FAILED to extract a frame")
            continue
        out_path = output_dir / f"{key}.jpg"
        cv2.imwrite(str(out_path), frame)
        print(f"  {key}: saved frame #{idx} -> {out_path}")

    print(f"\nDone. Stills written to {output_dir}")


if __name__ == "__main__":
    main()
