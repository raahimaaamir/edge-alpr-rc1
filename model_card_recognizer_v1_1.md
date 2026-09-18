# Model Card — Recognizer V1.1

**Status:** FROZEN, 2026-09-12. Replaces Recognizer V1 as the default recognizer
in the ALPR pipeline, per supervisor approval (Week 9). **RC1 (recognizer +
decision layer + rescue policy, together) technically finalized 2026-09-14** —
see "Decision layer — Release Candidate 1" below for the complete frozen
architecture, including the rescue-path consistency fix confirmed in that
section's final update.

## Identity

| | |
|---|---|
| Model | Recognizer V1.1 |
| Base | Fine-tuned from Recognizer V1 (frozen 2026-08-26) |
| Architecture | CCT-S (unchanged from V1) |
| Format | ONNX, opset 17, float32 input, dynamic batch |
| Canonical artifact path | `output/v1_1_finetune_balanced/onnx/best.onnx` |
| Source checkpoint | `output/v1_1_finetune_balanced/2026-09-11_03-29-49/best.keras` |
| SHA256 (ONNX) | `c9c4f6196c6c1a0a10e9d06d74ee091df95cb00a896c1f213241ac1a179feb28` |
| Config | `weights/cct_s_v2_global_plate_config.yaml` (unchanged from V1) |

V1's own frozen artifact is unmodified and remains at
`output/full_run1/onnx/best.onnx` for any explicit side-by-side comparison.

## Why this model exists

Manual review of own-collected footage (Week 9) found a systematic bias in V1:
6-character plates were consistently over-read as 7 characters via a spurious
inserted character. All 6 detected 6-character plates in that check were
misread; all 5 seven-character plates were read correctly. V1.1 is a
conservative, targeted fine-tune to correct this one specific failure mode.

## Training data

- 20,300 total samples: 15,000 synthetic (exactly balanced 6-char/7-char),
  300 manually-verified real US plates (Kaggle), 5,000 retained samples from
  V1's original UFPR/RodoSol training data (all 7-char, retained specifically
  to guard against forgetting).
- 17 real examples with lengths other than 6 or 7 (mostly Rhode Island's
  5-character format, a few 8-character vanity plates) were excluded from
  train/val entirely and evaluated only as an exploratory-only set.
- Train/val split stratified by (source, length). Training set additionally
  balanced by length via oversampling the minority (6-char) class — no
  7-character rows were removed, so the retained legacy data stayed intact.

## Training procedure

Fine-tuned from V1's weights, same general procedure and architecture as V1's
original training, learning rate reduced 10x (1e-4) for conservative
adaptation, early stopping on `val_plate_acc` (patience=3). Stopped early at
epoch 9/15; best checkpoint at epoch 6 (`val_plate_acc = 0.9985`).

## Validation results (by source and length group)

Recognizer-only evaluation on pre-cropped validation images.

| Group | n | V1 exact | V1.1 exact | V1 char acc | V1.1 char acc | V1 ins/del rate | V1.1 ins/del rate |
|---|---|---|---|---|---|---|---|
| synthetic, 6-char | 750 | 0.000 | 1.000 | 0.719 | 1.000 | 1.000 | 0.000 |
| synthetic, 7-char | 750 | 0.561 | 1.000 | 0.910 | 1.000 | 0.021 | 0.000 |
| kaggle_verified, 6-char | 18 | 0.000 | 0.944 | 0.972 | 0.991 | 1.000 | 0.000 |
| kaggle_verified, 7-char | 9 | 0.556 | 1.000 | 0.889 | 1.000 | 0.000 | 0.000 |
| existing_retained, 7-char (forgetting check) | 500 | 0.982 | 0.996 | 0.997 | 0.999 | 0.000 | 0.000 |

Exploratory (17 other-length, not used for model selection): V1 exact 0.0 /
char acc 0.881; V1.1 exact 0.294 / char acc 0.966.

## ONNX export and parity

Export flagged a numeric deviation on the `plate` output beyond default
tolerance (`region` output matched). Direct Keras-vs-ONNX decoded-text
comparison found:
- 50-sample check: 100% agreement (50/50)
- **Full validation set: 100% agreement (2,027/2,027)**

The flagged deviation is floating-point noise that does not change any
decoding decision. The ONNX artifact is confirmed faithful to the trained
checkpoint.

## System-level regression (V1 vs V1.1, identical frozen video pipeline)

Detector, tracking, Top-3 selection, majority voting, rescue, and decision
rule held fixed — only the recognizer differs. Run on the 22 UFPR
validation tracks.

| Metric | V1 | V1.1 |
|---|---|---|
| Exact accuracy | 0.9545 | 0.9545 |
| Coverage | 0.9545 | 1.0000 |
| Incorrect-accepted rate | 0.0000 | 0.0455 |
| NO_RELIABLE_RESULT rate | 0.0455 | 0.0000 |
| Runtime (22 tracks) | 43.15s | 40.66s |

One encounter differs (track 0080, gt=`APJ3829`): V1 correctly abstained
(`no_reliable_result`); V1.1 answered but incorrectly (`AP1382`). This
trades one abstention for one wrong answer — exact accuracy is unchanged
either way. This track has been a known near-miss since Week 7's
majority-vote testing and is unrelated to the 6-character bias V1.1 targets.

## Final independent check: 13 local plates

Run once, after V1.1 was selected from validation results — not used to
tune anything. Full detect+crop+recognize pipeline, own-collected data
(6 photos + 7 video frames).

- **Total insertions: 6 (V1) -> 0 (V1.1).**
- Exact match: 5/13 (38.5%) -> 9/13 (69.2%).
- 4 plates wrong under V1 due to an inserted character are now exactly
  correct under V1.1, including the original plate (`38P697`) that started
  this investigation.
- All 5 previously-correct 7-character plates remain correct.

## Decision layer — Release Candidate 1 (frozen)

Track 0080's diagnostic package showed V1.1's own top-3 probabilities
never seriously support "J" as an alternative to "1" at the disputed
position (present in only one of three examined frames' top-3, at 1.5%
probability). This is a genuine recognizer limitation, not something the
decision layer should paper over with a case-specific fix — so the
response was a general, minimal reliability layer rather than a
correction for this one plate.

**Decision path** (`decision_rule_rc1.py`, replacing `decide_v2`):
```
adequate observations -> temporal agreement -> confidence check
-> applicable plate-profile validation -> ACCEPT / NO_RELIABLE_RESULT
```

**Rule comparison, 22 UFPR validation tracks** (`run_acceptance_rule_comparison.py`):

| Rule | Precision | Coverage | Incorrect accepts | Abstentions |
|---|---|---|---|---|
| A — current (agreement only) | 0.9545 | 1.0000 | 1 (track 0080) | 0 |
| B — agreement + confidence | 1.0000 | 0.9545 | 0 | 1 |
| E — agreement + structural profile | 1.0000 | 0.9545 | 0 | 1 |
| **F — agreement + confidence + profile** | **1.0000** | **0.9545** | **0** | **1** |

**B, E, and F produced IDENTICAL results on this validation set** — the
same single abstention (track 0080) in all three cases. F was chosen
NOT because this dataset demonstrates an additive accuracy benefit from
combining the confidence and profile checks, but because it performs
both independent reliability checks at negligible extra cost, matching
the intended architecture. A validation set that exercises the checks'
disagreement region (confident-but-invalid, or valid-but-low-confidence)
could reveal a real difference this dataset does not.

**Confidence threshold:** 0.90, a **validation-selected conservative
heuristic, not a calibrated probability** — chosen as a single round
default before running against real data, then confirmed (not tuned)
against the validation set. It is a function parameter
(`min_winning_confidence`), not a hard-coded constant.

**Plate-profile validation** (`plate_profile.py`) is structural only
(length + per-position character class) and applies ONLY when the caller
supplies a profile for a known jurisdiction/plate class. When the
jurisdiction is unknown, the check is skipped entirely — a result is
never abstained on merely because its jurisdiction couldn't be
determined. The validator never pads, truncates, or substitutes
characters to force validity. An optional reranking capability
(`rerank_with_profile()`) exists and is tested, but is not wired into the
live decision path — see that module's docstring for why.

**Rescue-path consistency (finalization update, 2026-09-14):** rescue's
trigger logic (only detection-related failures — zero/insufficient
observations — are ever rescuable) is unchanged. Its final acceptance,
however, previously used a separate, weaker rule (`decide_v2`, agreement
only, no confidence or profile check). This has been corrected: rescue-
generated observations now pass through the exact same `decide_rc1` as
the normal path. Rescue generates additional evidence; it no longer has
a separate way to authorize a plate. Verified: no regression on any
previously-passing case, genuine detection-failure recovery still works
(track 0064), rescue candidates can be rejected by the confidence/profile
checks (confirmed with constructed test data), and the output schema is
identical whether or not rescue fired. See `rc1_rescue_fix_verification.md`
for the full verification record.

**Image-quality telemetry** (plate size, sharpness, exposure, and a new
continuous edge-margin metric — `FrameObservation.edge_margin_px`,
replacing an earlier binary edge-touch check that came back completely
uninformative) is recorded on every observation but is **not used as a
hard gate**. See Limitations below.

## Computational footprint

Measured on the 22 UFPR validation tracks, with `decision_rule_rc1.decide_rc1`
as the active decision rule (RC1, matching the actual frozen system —
not an earlier measurement against the superseded `decide_v2`).

| Metric | Value |
|---|---|
| Mean detector latency | ~23.0 ms/frame |
| Mean recognizer latency | ~16.6 ms/frame |
| Mean tracking/fusion/decision overhead | ~0.07 ms (negligible next to detect/recognize) |
| Mean total processing time per encounter | ~592 ms |
| Rescue trigger frequency | 4.5% of encounters (1/22) |
| Mean rescue latency when triggered | ~265 ms |
| **Peak memory** | **~544 MB** |
| Recognizer model size (ONNX) | 5.04 MB (V1 and V1.1 — identical, no architecture bloat from fine-tuning) |
| Detector model size (ONNX) | 7.41 MB |

**The peak memory figure (~544 MB) is a development-environment
measurement only** — taken on the training/development machine described
in this project's environment notes, running inside the full development
stack (TensorFlow/CUDA runtime, ONNX Runtime, and this script's own
Python overhead all loaded simultaneously). It is **not** representative
of deployed inference memory, and must **not** be used to infer or budget
for mobile/edge device performance. Device-level memory and latency
measurements on actual target hardware remain necessary before any
mobile deployment estimate can be made.

## Known limitations (documented per supervisor's instruction)

1. **Length coverage.** Evidence is strong for 6- and 7-character plates.
   The 17 other-length examples (5-char, 8-char) are too few to claim
   general support across all plate lengths — V1.1's improvement on that
   exploratory set (exact match 0.0 -> 0.294) is encouraging but not a
   validated capability.
2. **Residual character-level substitutions.** Two plates in the final
   13-plate check (`DJPT60`, `BPD845`) are now the correct length under
   V1.1 but still contain character-level substitution errors (e.g. a
   T/1 confusion). This is a separate, smaller character-recognition issue
   from the length bias V1.1 was built to fix, and per the supervisor's
   explicit instruction, does not trigger another training cycle at this
   stage.
3. **Known recognizer confusion pattern: 1/U/I/J family.** Diagnosed on
   track 0080 (UFPR validation): the recognizer's own top-3 probabilities
   at a disputed character position never seriously supported `J` as an
   alternative to `1` (present in only one of three examined frames' top-3,
   at 1.5% probability; absent from the other two entirely) — the actual
   confusion is within a `1`/`U`/`I` family. Recorded as a known failure
   pattern per the supervisor's instruction; no case-specific correction
   rule has been added for it, since doing so risks patching one failure
   while introducing others. See `track0080_findings_memo.md` for the
   full diagnostic evidence.
4. **Known recognizer pattern: plate-boundary/sequence-length ambiguity
   (track 0104, UFPR test set).** Ground truth `APL0100`. Across 14
   observations, three readings compete: `APL0100` (5x, correct),
   `PL0100` (4x, missing the leading character), `APL010` (4x, missing
   the trailing character) — the recognizer alternates between dropping
   the leading or trailing character, not confusing characters within
   the string. The Top-3 selected observations disagreed three ways, and
   the agreement gate correctly abstained rather than pick one. Recorded
   as a known pattern, not corrected — a useful example of why the
   reliability layer exists, not evidence RC1 needs a patch.
5. **Persistent low-confidence / hard-visual-input case (track 0107,
   UFPR test set).** Ground truth `MJO0862`. All 15 observations across
   the track had overall confidence between 0.27 and 0.56 — no frame was
   ever confident. Two of the Top-3 agreed on a candidate, clearing the
   agreement bar, but at mean confidence 0.459, correctly rejected by the
   confidence check rather than accepted. No boundary or specific
   character pattern here — a genuinely hard-to-read plate throughout,
   correctly abstained on.
6. **No data-derived visual-quality operating gate yet.** An attempt was
   made (`run_quality_gate_derivation.py`) to derive a conservative
   size/sharpness operating threshold from the 22-track validation set.
   Accuracy vs. plate size and vs. sharpness both came back non-monotonic
   across nearly the full observed range, and the apparent "good"
   thresholds at the extremes were artifacts of very small sample counts
   (n=5 or fewer), not a real accuracy cliff. This validation set does
   not contain enough genuinely degraded (small, blurred, clipped,
   low-light, glare-heavy) examples to derive a defensible threshold —
   consistent with the supervisor's own observation that track 0080's
   crops were "reasonably sized and readable" despite the OCR failure.
   Per explicit instruction, no threshold was invented to fill this gap.
   Plate size, sharpness, exposure, and a continuous edge-margin
   (crop-completeness) metric are recorded as telemetry on every
   observation (`FrameObservation`) so this can be derived properly once
   a broader validation set containing genuinely degraded examples is
   available. This remains important future work, not a gap in the RC1
   decision layer itself.

## Rollback

V1's frozen artifact remains unmodified at `output/full_run1/onnx/best.onnx`
and is passed explicitly via `PlateRecognizer(model_path=V1_FROZEN_MODEL_PATH)`
(constant defined in `recognizer.py`). Reverting the default is a one-line
change back to `recognizer.py`'s `DEFAULT_MODEL_PATH`.
