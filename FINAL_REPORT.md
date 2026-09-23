# Edge Video ALPR — Final Technical Report

**Release Candidate 1 (RC1), frozen 2026-09-14.**
Repository: https://github.com/raahimaaamir/edge-alpr-rc1

## 1. Objective

Build an automatic license plate recognition (ALPR) pipeline for video
streams, suitable for eventual edge/mobile deployment (ONNX runtime,
CoreML-compatible export path). The system needed to go beyond
single-frame OCR: detect and track vehicles across a video, read the
plate from multiple frames, and produce one reliable answer per
vehicle — including the ability to explicitly decline to answer when
the evidence doesn't support a confident result, rather than guessing.

## 2. System architecture

Two components: a single-frame pipeline (detect → crop → quality check
→ recognize) and a video-level layer built on top of it (track → select
frames → fuse → apply a reliability check → accept or abstain, with a
rescue path for detection failures). Full diagrams in
`ARCHITECTURE.md`. Both are exposed as documented entry points
(`AlprPipeline` for images, `ALPRSystem`/`week9_mp4_runner.py` for
video) in `README.md`.

## 3. Detector and initial recognizer baseline

The detector is a pretrained YOLOv9-tiny license-plate model (via the
`open-image-models` package), used as-is throughout the project —
detector improvement was out of scope; effort focused on the
recognizer and the reliability layer around it.

Recognizer V1 is a CCT-S (Compact Convolutional Transformer, small)
model, trained on the UFPR-ALPR and RodoSol-ALPR datasets (Brazilian
plates). This became the frozen baseline recognizer for the rest of
the project, with its own model config
(`weights/cct_s_v2_global_plate_config.yaml`) and ONNX export
(`output/full_run1/onnx/best.onnx`, retained as the RC1 fallback,
SHA256 `5e03d7fd...`).

## 4. Video pipeline: tracking, frame selection, fusion

Built incrementally and evaluated at each stage before moving on:

- **Tracking**: an IoU-based tracker (`tracker.py`) matches detections
  across frames into tracks, tolerant of brief occlusion
  (`max_missing_frames`).
- **Frame selection**: rather than using every sampled frame's OCR
  reading equally, a composite score (detector confidence, recognition
  confidence, blur/sharpness) selects the Top-K most trustworthy
  observations per track before voting.
- **Fusion**: majority vote across the Top-K selected observations was
  compared against confidence-weighted voting and per-character
  confidence fusion; majority vote was adopted as the frozen fusion
  method (`fusion_method: majority_vote` in `frozen_config.json`).
- **Stride**: video frames are sampled at stride=2 (every other frame),
  balancing coverage against redundant computation — frames are highly
  correlated at native frame rate.

## 5. Computational footprint

Measured on the 22 UFPR validation tracks, under the final frozen RC1
decision rule (not an earlier, superseded rule — re-measured
specifically to keep this number honest after the decision layer
changed):

| Metric | Value |
|---|---|
| Mean detector latency | ~23.0 ms/frame |
| Mean recognizer latency | ~16.6 ms/frame |
| Mean tracking/fusion/decision overhead | ~0.07 ms (negligible) |
| Mean total processing time per encounter | ~592 ms |
| Rescue trigger frequency | 4.5% of encounters (1/22) |
| Peak memory | ~544 MB |
| Recognizer model size (ONNX) | 5.04 MB (V1 and V1.1 — identical) |
| Detector model size (ONNX) | 7.41 MB |

**The peak memory figure is a development-environment measurement
only** — not representative of deployed/mobile performance. Device-level
benchmarking on actual target hardware remains necessary before any
mobile deployment estimate can be made (see Section 11).

## 6. The length-bias discovery

Manual review of own-collected footage found a systematic bias in V1:
6-character plates were consistently over-read as 7 characters via a
spurious inserted character. All 6 detected 6-character plates in that
initial check were misread; all 5 seven-character plates were read
correctly. This was reported and a controlled fine-tune (V1.1) was
approved before any change was made.

## 7. Recognizer V1.1: dataset and training

- **20,300 total samples**: 15,000 synthetic (exactly balanced 6-char/
  7-char), 300 manually-verified real US plates (Kaggle, state-based
  review prioritization — V1's own predicted length was uninformative
  for finding 6-char examples: it predicted length 7 for ~99.9% of a
  5,861-image real-plate sample regardless of true length), 5,000
  samples retained from V1's original training data (all 7-char,
  specifically to guard against forgetting).
- 17 real examples with lengths other than 6 or 7 (mostly 5-character
  formats, a few 8-character vanity plates) were excluded from
  train/val entirely and evaluated only as an exploratory-only set.
- Train/val split stratified by (source, length). Training set
  additionally balanced by length via oversampling the minority
  (6-char) class — no 7-character rows were removed, keeping the
  retained legacy data intact for the forgetting check.
- Fine-tuned from V1's weights, learning rate reduced 10x for
  conservative adaptation, early stopping on `val_plate_acc`.

## 8. V1.1 validation results

Recognizer-only evaluation (direct calls on pre-cropped validation
images), by source and length group:

| Group | n | V1 exact | V1.1 exact | V1 ins/del rate | V1.1 ins/del rate |
|---|---|---|---|---|---|
| synthetic, 6-char | 750 | 0.000 | 1.000 | 1.000 | 0.000 |
| synthetic, 7-char | 750 | 0.561 | 1.000 | 0.021 | 0.000 |
| kaggle_verified, 6-char | 18 | 0.000 | 0.944 | 1.000 | 0.000 |
| kaggle_verified, 7-char | 9 | 0.556 | 1.000 | 0.000 | 0.000 |
| existing_retained, 7-char (forgetting check) | 500 | 0.982 | 0.996 | 0.000 | 0.000 |

The `existing_retained` row is the true forgetting check (V1's actual
training distribution); V1.1 shows no degradation there. V1's weaker
numbers on synthetic/Kaggle 7-char data reflect that V1 was never
strong on those distributions to begin with, not forgetting.

ONNX export was verified faithful to the trained checkpoint: a direct
Keras-vs-ONNX decoded-text comparison found 100% agreement on the full
2,027-image validation set (not just a sample).

## 9. System-level regression (V1 vs V1.1)

With the decision layer held identical (only the recognizer swapped),
on the 22 UFPR validation tracks:

| Metric | V1 | V1.1 |
|---|---|---|
| Exact accuracy | 0.9545 | 0.9545 |
| Coverage | 0.9545 | 1.0000 |
| Incorrect-accepted rate | 0.0000 | 0.0455 |

One encounter (track 0080) changed from a correct abstention (V1) to a
confident wrong answer (V1.1) — exact accuracy unchanged, but this
became the seed of the reliability-layer investigation described next.

Final independent check (13 own-collected local plates, run once, full
video pipeline): total insertions 6 (V1) → 0 (V1.1); exact match 5/13
→ 9/13.

## 10. Reliability-by-design: from a recognizer question to a decision-layer question

Track 0080's diagnostic package (crops, per-slot model probabilities,
full vote table) showed the recognizer's own top-3 probabilities never
seriously supported "J" as an alternative to "1" at the disputed
position — a genuine recognizer limitation, not a decision-layer bug.
This reframed the problem: rather than patch this one case, the
question became whether the *decision rule itself* (plain "≥2 of 3
agree") was too permissive in general.

Four rules were compared on the 22 validation tracks, sharing identical
Top-K selection:

| Rule | Precision | Coverage | Incorrect accepts |
|---|---|---|---|
| A — current (agreement only) | 0.9545 | 1.0000 | 1 (track 0080) |
| B — agreement + confidence | 1.0000 | 0.9545 | 0 |
| E — agreement + structural plate-profile validity | 1.0000 | 0.9545 | 0 |
| F — agreement + confidence + profile | 1.0000 | 0.9545 | 0 |

B, E, and F were identical on this dataset. **F was frozen as RC1's
decision rule** — not because this dataset demonstrates an additive
benefit over B or E alone, but because it performs the independent
checks by design (temporal agreement, confidence, and format validity
are three different questions), at negligible extra cost. A dataset
that exercises the checks' disagreement region could reveal a real
difference this one didn't.

The plate-profile validator (`plate_profile.py`) is structural only —
length and per-position character class — and never pads, truncates,
or substitutes; it's skipped entirely (never treated as a rejection)
when the jurisdiction is unknown. The 0.90 confidence threshold is
documented explicitly as a validation-selected conservative heuristic,
not a calibrated probability.

## 11. Rescue-path consistency fix

Rescue's trigger logic (only detection-related failures — zero or
insufficient observations — are ever rescuable) was correct from the
start. But its *final acceptance* previously used a separate, weaker
rule (agreement only, no confidence or profile check) — meaning rescue
had its own, easier way to authorize a plate than the normal path.
This was corrected: rescue-generated observations now pass through the
exact same `decide_rc1` as everything else. Verified: no regression on
any previously-passing case; genuine detection-failure recovery still
works (track 0064, confirmed in two separate real-data runs); rescue
candidates can be rejected by the confidence/profile checks (confirmed
via constructed test data, since real rescue candidates in this
dataset happened to be high-confidence); identical output schema
regardless of whether rescue fired.

## 12. Final RC1 verification

Test-set regression (23 UFPR test tracks — touched only for freezing
moments, per this project's own discipline, the second time in the
whole project): 21/23 exact accuracy and coverage, 0 incorrect accepts,
rescue triggered 0 times. Two abstentions investigated directly:

- **Track 0104**: the recognizer alternates between dropping the
  leading and trailing character across observations (`APL0100` vs
  `PL0100` vs `APL010`) — a plate-boundary/length ambiguity, distinct
  from the character-confusion pattern seen in track 0080. The Top-3
  disagreed three ways; the agreement gate correctly abstained.
- **Track 0107**: every observation across the track had confidence
  between 0.27 and 0.56 — no frame was ever confident. Two of the
  Top-3 agreed, clearing the agreement bar, but at mean confidence
  0.459, correctly rejected by the confidence check. A genuinely hard
  image, not a design gap.

Neither indicates a defect in RC1; both are recorded as known patterns
in `model_card_recognizer_v1_1.md`, not patched case-by-case.

## 13. Known limitations and failure cases

Full detail in `model_card_recognizer_v1_1.md`. Summary:

1. **Length coverage**: strong evidence for 6- and 7-character plates
   specifically; only 17 real other-length examples exist, too few to
   claim general support across all plate formats.
2. **Residual character-level substitutions** (e.g. T/1): a separate,
   smaller issue from the length bias V1.1 was built to fix; not
   triggering another training cycle at this stage.
3. **Track 0080 — 1/U/I/J confusion pattern**: recorded as a known
   recognizer limitation, not patched.
4. **Track 0104 — plate-boundary/length ambiguity**: a second, distinct
   pattern (see Section 12), recorded not patched.
5. **Track 0107 — persistent low confidence**: a hard-image case,
   correctly abstained on by design.
6. **No data-derived visual-quality operating gate yet**: plate size,
   sharpness, and exposure vs. OCR correctness came back non-monotonic
   on the available validation data — it doesn't span enough genuinely
   degraded examples to derive a defensible threshold. Per explicit
   instruction, no threshold was invented to fill this gap. All four
   quantities (plus a continuous edge-margin measurement) are recorded
   as telemetry on every observation, not enforced as a gate.
7. **Plate-profile reranking exists but isn't wired into the live
   path**: implemented and tested (`plate_profile.rerank_with_profile`),
   but requires per-slot model probabilities that aren't currently
   threaded through the tracker/observation layer.
8. **Jurisdiction is not auto-detected**: the profile check requires
   the caller to supply a known profile; there is no automatic
   jurisdiction inference.
9. **Footprint numbers are development-environment measurements only**
   — not representative of mobile/edge deployment performance.

## 14. Implemented, deferred, and recommended next steps

**Implemented**: detection, tracking, multi-frame fusion, the RC1
reliability layer (agreement + confidence + optional plate-format
validation), rescue for detection-related failures (with corrected
consistency), the V1.1 recognizer fine-tune, full telemetry recording,
a portable handoff package with tests, examples, and documentation.

**Deferred, deliberately**:
- A data-derived visual-quality operating gate — needs a broader
  validation set spanning genuinely degraded (small/blurred/clipped/
  low-light) examples.
- Plate-profile reranking wired into the live tracking path.
- Jurisdiction auto-detection.

**Recommended next steps**:
- Collect or synthesize a validation set with genuinely degraded plates
  to derive the quality gate properly, rather than guessing a
  threshold.
- If the plate-boundary ambiguity pattern (track 0104) recurs on more
  data, investigate whether it's a detector cropping issue or a
  recognizer decoding issue before deciding which layer should address
  it — don't patch blind.
- Device-level (mobile/edge) latency and memory benchmarking on actual
  target hardware, since the current numbers are development-machine
  measurements only.
- If reranking is wired into the live path in a future change, it
  should go through the same kind of independent verification this
  report's Sections 10–12 describe — comparison against the current
  rule on validation data, then a one-time test-set check, not an
  assumption that it helps.

## 15. Reproducing this work

See `README.md` for environment setup, one-command examples, and the
full reproduction commands for every result in this report (validation
comparisons, the rule comparison, the rescue-fix verification, the
test-set regression, and the footprint measurement). All raw results
are preserved in `eval/results/*.json` for independent inspection.

## 16. Post-handoff engineering pass

After independent review, the supervisor requested a second pass
focused entirely on engineering cleanup and reproducibility — no new
OCR research, and explicitly, no new thresholds or heuristics unless a
fix revealed an actual defect. Nine items, addressed in priority order:

1. **Multi-plate detection**: `AlprPipeline.process()` only ever used
   the single highest-confidence detection per frame. Added
   `process_all()` (returns every detection, `process()` left
   completely unchanged) and wired it through `video_pipeline.py` and
   `ALPRSystem.SystemConfig.multi_plate`. This surfaced a genuine,
   previously-unexercised bug: `video_pipeline.py`'s existing
   `multi_plate` flag assumed `process()` would return a list, which it
   never did — fixed as part of this work. New regression test
   (`test_multi_plate.py`) confirms two plates in one frame produce two
   correctly-tracked, independently-read vehicles end to end.

2. **Portable `frozen_config.json`**: previously recorded absolute,
   machine-specific paths. `dump_frozen_config.py` now records paths
   relative to the repository root, plus the exact git commit SHA and
   tag this configuration was frozen against, and the detector's own
   identifier, package version, and SHA-256 (located via the same
   cache-directory convention `open-image-models` itself uses, rather
   than a hardcoded filename). A real crash bug — computing a hash on a
   detector that hadn't been downloaded yet — was caught and fixed
   before shipping.

3. **Pinned runtime dependencies**: `requirements-runtime.txt`, pinned
   to the exact versions already verified in a clean-environment test
   (Python 3.12.3; `onnxruntime==1.30.0`, `opencv-python==5.0.0.93`,
   `numpy==2.5.3`, `pyyaml==6.0.3`, `open-image-models==0.6.0`).
   `requirements-training.txt` added too, with an explicit note that
   unlike the runtime pins, it is not independently clean-environment
   verified. Confirmed with a full second clean-environment test —
   fresh clone, fresh venv, pinned install — reproducing byte-identical
   results to every prior run.

4. **Comprehensive RC1 decision-path tests**: prior tests covered the
   tracker/fusion utilities but not `decide_rc1`, the plate-profile
   check, or the rescue policy directly. `test_rc1_decision_path.py`
   now tests all of it against the real code: correct agreement,
   low-confidence rejection, profile-based rejection, profile-skip on
   unknown jurisdiction, rescue succeeding on its own merits, rescue
   correctly declining to override a failure it can't independently
   justify, the zero-detection case, and output-schema consistency
   across every status.

5. **README/code consistency**: literally running every command in the
   README surfaced two real bugs, not just wording problems — the
   documented video-CLI argument format didn't match the actual
   argparse setup (a positional argument, documented as a `--flag`),
   and `week9_mp4_runner.py` had a genuine import-ordering bug: its
   `sys.path` fix lived inside the `if __name__ == "__main__":` guard,
   too late for a top-level import earlier in the same file that
   needed it. Also documented `ALPRSystem.process_image()`, which
   existed in the code but was entirely missing from the README, and
   added an explicit table clarifying which entry points do and don't
   apply RC1's reliability layer.

6. **Quality-gate clarification**: the crop-size check
   (`quality.assess_quality`'s `min_dimension_ok`) is now explicitly
   documented as a basic sanity guard, not a validated OCR-quality
   threshold, and is genuinely configurable (`AlprPipeline`'s
   constructor, and `SystemConfig.quality_min_width`/`quality_min_height`,
   previously recorded but not live, now actually applied). Separately,
   `edge_margin_px` telemetry — promised in the output schema — was
   silently `None` from both `ALPRSystem.process_frame()` and the
   rescue path; only `video_pipeline.py` ever computed it. Both fixed,
   verified against exact expected numeric values.

7. **Detector reproducibility**: documented exactly how to reproduce
   the same detector artifact — the `open-image-models==0.6.0` pin is
   the actual reproducibility mechanism (PyPI versions are immutable),
   with `frozen_config.json`'s recorded hash as the way to confirm it,
   not just trust it.

8. **`THIRD_PARTY_NOTICES.md`**: datasets (UFPR-ALPR, RodoSol-ALPR —
   both non-redistributable per their license agreements, confirmed
   directly against their source pages), models (`open-image-models`
   and `fast-plate-ocr`, both MIT, same author; the underlying YOLOv9
   architecture's own citation), and package licenses. One genuine gap
   flagged explicitly rather than guessed: the exact Kaggle dataset
   used for V1.1's real US-plate samples isn't recorded anywhere in
   this project's existing documentation.

9. **Final packaging**: `ufpr_video_loader.py`/`rodosol_loader.py`
   moved from `pipeline/` to `eval/` — their only callers already live
   there, and their hardcoded paths had been quietly contradicting the
   README's claim that `pipeline/` is fully portable. Confirmed no
   tracked secrets/credentials anywhere in the repository or its
   history. RC1 tag moved to this final commit (see `HANDOFF_CHECKLIST.md`
   for the complete final clean-environment verification).

Every fix in this section was verified against the real code before
shipping, not just asserted — either through a targeted unit test, a
new persisted regression test, or (for the README/CLI fixes) literally
running the documented command and confirming the actual output.
