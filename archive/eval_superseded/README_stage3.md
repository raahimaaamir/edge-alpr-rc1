# Stage 3 — Video Tracking / Plate Association

Builds on the frozen single-frame pipeline (`detector.py`, `recognizer.py`,
`quality.py`, `alpr_pipeline.py`) without modifying it. Nothing here changes
Recognizer V1 or the single-frame `PlateResult` schema.

## Files

| File | Purpose |
|---|---|
| `track_types.py` | `FrameObservation` (one frame's single-frame-pipeline output, attached to a track), `Track` (a sequence of observations), `TrackResult` (final fused output, or `NO_RELIABLE_RESULT`). All JSON-safe via `.to_dict()`. |
| `tracker.py` | `BaseTracker` interface + `IoUTracker`: greedy IoU / centre-displacement matching with a `max_missing_frames` grace period. No motion model — the simplest thing that could work, per the plan to only reach for ByteTrack if this proves insufficient. |
| `fusion.py` | Four fusion baselines behind one `fuse(track, method, config)` entry point: `best_single_frame`, `majority_vote`, `confidence_weighted_vote`, `per_char_confidence`. Every method ends in the same abstention gate — if the winning candidate doesn't clear `min_agreement_ratio`, the result is `NO_RELIABLE_RESULT`, not a forced guess. |
| `video_pipeline.py` | `VideoAlprPipeline`: wires an existing single-frame pipeline + a tracker + a fusion method into one `process_video(frames) -> (track_results, tracks, stats)` call. Thin by design — swapping the tracker or fusion method doesn't touch this file. |
| `tracker_eval.py` | Evaluates the tracker *in isolation* from OCR, given per-frame ground-truth boxes with track identity (e.g. UFPR's real tracks). Reports missed detections, ID switches, fragmentation, false-positive tracks, and per-track purity — so failures can be attributed to tracking vs. OCR/fusion. |
| `evaluate_stage3.py` | Produces exactly the numbers asked for in the next report: single-frame vs. fused exact-match accuracy (overall and per fusion method), track coverage, frames used per result, processing time, plus the tracker-association breakdown. |
| `stage3_config.py` | Every threshold in one place (`TrackerConfig`, `FusionConfig`, fusion method choice, detector confidence threshold) — nothing hard-coded inline. `load_config()` returns sane defaults; `dump_config()` serializes a run's config to JSON for reproducibility. |
| `test_stage3_synthetic.py` | Logic tests against synthetic data (tracking through occlusion, majority vote recovering from OCR errors, abstention on strong disagreement, per-char fusion, tracker-eval missed-detection/fragmentation) — verified passing, since this environment has no access to the real models or video data. |
| `test_video_pipeline_integration.py` | End-to-end wiring test with a mock single-frame pipeline standing in for `AlprPipeline`. Verified passing. |

## How it fits together

```
video / frame sequence
  -> AlprPipeline.process() per frame        [existing, frozen]
  -> FrameObservation.from_plate_result()     [track_types.py]
  -> IoUTracker.update() per frame            [tracker.py]
  -> IoUTracker.finalize() -> list[Track]     [tracker.py]
  -> fuse(track, method=...) per track        [fusion.py]
  -> list[TrackResult]                        (one plate string, or abstain, per track)
```

`VideoAlprPipeline.process_video()` in `video_pipeline.py` runs all of this
for you; you only need to supply an iterable of `(frame_idx, timestamp,
image)` and your existing `AlprPipeline` instance.

## What's NOT done yet (explicitly deferred)

- **Real-data validation.** Every test here uses synthetic detections/OCR
  strings. The logic is verified correct; the actual tracker
  IoU/displacement thresholds and fusion `min_agreement_ratio` are untuned
  defaults (see `stage3_config.py`) and need to be run against real
  video/frame sequences — starting with UFPR's real multi-frame tracks,
  since that's the only ground-truth track data available, then the
  self-collected video dataset as target-domain eval per the supervisor's
  direction.
- **Multi-plate-per-frame support.** `video_pipeline.py` has a `multi_plate`
  flag and the tracker already handles multiple detections per frame, but
  this hasn't been exercised against a real multi-plate scenario yet.
- **ByteTrack comparison.** Only pursue if the IoU baseline's
  `tracker_eval.py` numbers show it isn't robust enough, per the plan.
- **Detector fine-tuning** — still explicitly deferred past this stage.

## Suggested next session

1. Wire real UFPR video/frame sequences through `VideoAlprPipeline` using
   the actual `AlprPipeline`.
2. Build the ground-truth-track-boxes structure `tracker_eval.py` expects
   (`{frame_idx: [(bbox, gt_track_id), ...]}`) from UFPR's existing track
   annotations.
3. Run `evaluate_stage3.py` for real numbers: single-frame vs. fused
   accuracy per fusion method, track coverage, tracker failure breakdown,
   frames-used-per-result, timing.
4. Sweep `stage3_config.py` thresholds against those real numbers before
   locking anything down for the report.
