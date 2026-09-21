# Representative Examples

13 real images — the project's own collected data (6 photos, 7 video
frames), **not** UFPR-ALPR data, which cannot be redistributed under its
license agreement. This is the only real-world imagery in this
repository safe to ship alongside the code.

## Running

```bash
python3 examples/run_examples.py
```

This runs each image through the **single-image entry point**
(`AlprPipeline.process()`) and compares the result against
`ground_truth.json`.

## Important: what this does and doesn't test

This uses the single-image pipeline only — no multi-frame tracking,
fusion, or the RC1 reliability layer (temporal agreement, confidence,
plate-profile check). Those only apply to video tracks, and these are
individual example images, not tracks. So:

- A per-image mismatch here does **not** necessarily mean the full
  video pipeline would get the same vehicle wrong — the video path
  pools multiple frames per vehicle before deciding, and can recover
  from a single bad frame the way this single-image check cannot.
- The authoritative RC1 accuracy numbers are the ones in
  `model_card_recognizer_v1_1.md`'s "Final independent check: 13 local
  plates" section, which used the full video pipeline.

## Expected result (recorded from an actual run, 2026-09-19)

9/13 correct. The 4 mismatches are consistent with limitations already
documented in the model card, not new findings:

- `plate04` (ground truth `NEL248`): no result — this is the same
  detector-level miss already documented (the plate was never detected
  by either recognizer version in the original full-pipeline check).
- `video-07` (ground truth `BCP7506`): mismatched — the same known
  hard frame (vehicle reversing away from the camera) documented in
  the model card's final check.
- `plate02`, `plate03`: character-level substitution errors, consistent
  with the "residual character-level substitutions" limitation already
  documented (T/1-type confusions).

If a fresh run on your machine produces materially different results
than these, that's worth investigating — it could mean the recognizer
artifact doesn't match its recorded hash, or something in the
environment differs from what's documented in the README.
