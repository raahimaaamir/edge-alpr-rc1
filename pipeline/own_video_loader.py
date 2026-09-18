"""
own_video_loader.py

Week 8, step 4: loads frames from the user's own recorded .mp4 videos
(multi-vehicle scenes UFPR can't provide, since every UFPR video contains
exactly one physical plate) into the same (frame_idx, timestamp, image)
shape every other loader in this project produces — so it plugs directly
into the existing tracker/pipeline code unchanged.

No ground-truth annotations exist for these videos (unlike UFPR/RodoSol),
so this is deliberately paired with a review driver that saves crops for
manual inspection rather than computing automatic accuracy — exactly the
"manually reviewed subset" the plan calls for, not a full annotation
project.
"""

from pathlib import Path
from typing import Optional

import cv2


def extract_frames(video_path, stride: int = 1) -> list:
    """Returns list of (frame_idx, timestamp_seconds, image_bgr).
    frame_idx is the ORIGINAL frame number in the video (not renumbered
    after striding), consistent with how every other loader in this
    project handles stride/subsetting."""
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cv2 failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = []
    frame_idx = 0
    while True:
        ret, image = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            timestamp = (frame_idx / fps) if fps > 0 else None
            frames.append((frame_idx, timestamp, image))
        frame_idx += 1
    cap.release()
    return frames


def list_videos(folder) -> list:
    """Returns sorted list of .mp4 paths in folder."""
    folder = Path(folder)
    return sorted(folder.glob("*.mp4"))
