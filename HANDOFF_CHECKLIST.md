# Handoff Checklist

What was tested in a genuinely clean environment before this handoff,
and the actual results — not a checklist of intentions, a record of
what was actually run and what actually happened.

## Environment

Fresh `git clone` into a brand-new directory (`~/clean-test-final`,
deleted after this checklist was written — nothing about the real
package depends on that specific path), a brand-new Python virtual
environment (`python3 -m venv venv`), nothing pre-installed, dependencies
installed strictly from the pinned `requirements-runtime.txt`.

## What was run, and what happened

**1. Clone + pinned install**
```
git clone https://github.com/raahimaaamir/edge-alpr-rc1.git
pip install -r requirements-runtime.txt
```
Result: clean install, no errors, no version conflicts.

**2. Frozen artifact hash verification**
```
sha256sum output/v1_1_finetune_balanced/onnx/best.onnx output/full_run1/onnx/best.onnx
```
Result: both hashes matched `README.md`'s recorded table exactly —
`c9c4f619...` (V1.1) and `5e03d7fd...` (V1).

**3. All four regression/integration test files**
```
python3 tests/test_stage3_synthetic.py
python3 tests/test_video_pipeline_integration.py
python3 tests/test_multi_plate.py
python3 tests/test_rc1_decision_path.py
```
Result: all four passed every sub-test — synthetic tracking/fusion
(5 sub-tests), video-pipeline integration, multi-plate detection
(4 sub-tests, including the real fix from item 1 of the post-handoff
pass), and the full RC1 decision path (8 sub-tests: agreement,
confidence, profile validation both ways, rescue both ways, zero-
detection rescue, output-schema consistency).

**4. Representative examples**
```
python3 examples/run_examples.py
```
Result: 9/13 correct — byte-identical to every previous run of this
same check, including the original one inside the NGC container. The
4 mismatches (`plate02`, `plate03`, `plate04`, `video-07`) are the
same 4 every time, and match documented limitations in
`model_card_recognizer_v1_1.md`, not new findings.

**5. `frozen_config.json` regeneration**
```
cd pipeline && python3 dump_frozen_config.py
```
Result: regenerated correctly from the fresh clone — repository-relative
paths, correct git commit SHA, both recognizer hashes matching exactly,
and the detector's own hash (`888397b9...`) matching the value recorded
in every prior run on a different machine — the actual, verified
confirmation that pinning `open-image-models==0.6.0` really does
reproduce the identical detector artifact, not just an assertion that
it should.

## What this confirms

- The repository works correctly after being cloned to a brand-new
  location, with nothing carried over from the original development
  machine.
- The pinned dependency versions are sufficient — no missing package,
  no version conflict.
- Every regression test, including the new ones added during the
  post-handoff engineering pass (multi-plate, RC1 decision path),
  passes from a genuinely clean start.
- The recognizer, and independently the detector, are both exactly
  reproducible — confirmed by hash, not assumed.

## What this does NOT cover

- The video entry point (`week9_mp4_runner.py`) was verified via
  `--help` (confirming its CLI argument fix and import-order fix both
  work) but not run end-to-end against a real video file, since no
  `.mp4` was available in the clean environment. The underlying
  `ALPRSystem`/`decide_rc1` path it wraps is covered by
  `test_rc1_decision_path.py`.
- The historical `eval/`/`training/` scripts were not re-run (they
  require the licensed UFPR-ALPR/RodoSol-ALPR raw data, which per
  their license agreements is not bundled in this repository and was
  not available in the clean-test environment either).
- Training was not re-run (the clean environment intentionally used
  only `requirements-runtime.txt`, not `requirements-training.txt`,
  since training reproducibility is separately — and more cautiously —
  documented in that file's own header).
