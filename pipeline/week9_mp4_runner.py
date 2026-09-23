"""
week9_mp4_runner.py

Week 9, step 3: a thin wrapper around ALPRSystem for the common case of
"I have an MP4 file, give me the plate result(s)." This is NOT part of
the core inference path — it exists only to demonstrate that the core
(process_frame / finalize_stream) works correctly when fed incrementally,
and to provide a convenient one-call entry point for that common case.

The core class itself has no idea this frame came from a file — it would
work identically fed from a live camera loop or a network stream, one
frame at a time. This wrapper happens to read a whole file up front (via
own_video_loader.extract_frames, unchanged from Week 8) because that's the
simplest way to get an MP4's frames, and because having them all in memory
lets it trivially supply full-resolution candidate frames to the rescue
path. A true live-stream caller would instead keep a small bounded buffer
of recent frames (or skip rescue for that stream by passing no provider)
rather than holding an entire video in memory.
"""

from pathlib import Path
from typing import Optional
import sys

# alpr_system.py -> week9_rescue_policy.py -> tiled_detection_rescue.py
# imports `pipeline.result_types` (package-style) — this needs the repo
# root on sys.path BEFORE the alpr_system import below runs, not just
# inside the `if __name__ == "__main__":` guard further down (which is
# too late for this specific import chain — this was a real bug: running
# `python3 week9_mp4_runner.py` failed with `ModuleNotFoundError: No
# module named 'pipeline'` before this fix, caught by actually running
# every command in the README rather than assuming it worked).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpr_system import ALPRSystem, SystemConfig
from own_video_loader import extract_frames


def run_on_mp4(system: ALPRSystem, video_path, stream_id: Optional[str] = None,
                extract_stride: int = 1, enable_rescue_frames: bool = True) -> list:
    """Runs one MP4 file through the streaming API end to end and returns
    the final decision(s) — normally one, but a video could in principle
    contain multiple distinct plate tracks.

    stream_id: defaults to the file's stem (e.g. 'video-01' for video-01.mp4).
    extract_stride: how densely to READ frames from disk (1 = every frame).
        This is separate from SystemConfig.stride, which controls how many
        of the frames handed to process_frame are actually RUN through
        detection — extract_stride=1 lets the system's own configured
        stride make that decision, which is the normal setup; a caller
        could set extract_stride > 1 to also skip frames at the file-read
        level (e.g. for a very long video), but that's an unusual case.
    enable_rescue_frames: if True, every frame read from the file is kept
        in memory and offered to the rescue path at finalize time (this is
        what "has the whole file" naturally allows). Set False to simulate
        the tighter-memory streaming case even though this wrapper has the
        file available.
    """
    stream_id = stream_id or Path(video_path).stem
    frames = extract_frames(video_path, stride=extract_stride)

    for frame_idx, timestamp, image in frames:
        system.process_frame(stream_id, image, timestamp=timestamp, frame_idx=frame_idx)

    if enable_rescue_frames:
        return system.finalize_stream(stream_id, candidate_frame_provider=lambda track: frames)
    else:
        return system.finalize_stream(stream_id, candidate_frame_provider=None)


if __name__ == "__main__":
    import argparse
    import json

    from alpr_pipeline import AlprPipeline

    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()

    pipeline = AlprPipeline()
    config = SystemConfig(stride=args.stride)
    system = ALPRSystem(pipeline, config)

    results = run_on_mp4(system, args.video_path)
    print(json.dumps(results, indent=2))
