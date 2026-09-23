# Edge Video ALPR — RC1

A license-plate recognition pipeline for video streams: detect → track →
select frames → fuse OCR readings → apply a reliability check → accept or
abstain. This is Release Candidate 1 (RC1), technically frozen
2026-09-14. This README is written for someone with no prior context on
the project — if anything here doesn't work as described, that's a bug
in the package, not a gap in your setup.

See `ARCHITECTURE.md` for the full pipeline diagram.

## What this system does

Given a video (or a single image), it finds license plates, tracks
vehicles across frames, reads the plate text from multiple frames per
vehicle, and combines those readings into one final answer per vehicle —
or explicitly declines to answer (`NO_RELIABLE_RESULT`) when the evidence
doesn't clear a defined reliability bar, rather than guessing. See
`model_card_recognizer_v1_1.md` for the full story of why it's built this
way, including three real failure cases (tracks 0080, 0104, 0107) that
shaped the design.

## Environment setup

Tested inside an NVIDIA NGC TensorFlow container
(`nvcr.io/nvidia/tensorflow:25.02-tf2-py3`, TensorFlow 2.17.0), which
provides CUDA/cuDNN already configured. If you're not using that
container, you need Python 3.10+ and:

```bash
pip install -r requirements-runtime.txt --break-system-packages
```

Or, to avoid needing that flag at all (the approach this project's own
clean-environment tests used):
```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements-runtime.txt
```

(TensorFlow is only needed if you re-train or re-export a recognizer —
running the frozen pipeline needs only `onnxruntime`. If you do need to
train, see `requirements-training.txt` — read its own header first, it's
less rigorously verified than the runtime pins.)

No manual path configuration is required — every script auto-detects
this package's own location. If you move `pipeline/` to a different
spot relative to `output/` and `weights/`, set `ALPR_ROOT` to this
package's root directory before running anything:
```bash
export ALPR_ROOT=/path/to/this/package
```

## Which entry point should I use?

There are three ways to get a plate reading, and it matters which one
you pick — only one of them applies RC1's reliability layer (temporal
agreement across frames, confidence check, plate-profile validation).
The other two give you a raw, single-frame OCR reading with none of
that — not wrong, just a different, weaker guarantee, and it's
important not to mix them up:

| Call | Reliability layer? | What you get |
|---|---|---|
| `AlprPipeline.process(image)` | **No** | One `PlateResult` from the single highest-confidence detection in that one image |
| `AlprPipeline.process_all(image)` | **No** | A `list[PlateResult]`, one per detection in that one image (multi-plate) |
| `ALPRSystem.process_image(image)` | **No** | Same as `process()` above, just returned as a plain dict — a thin convenience wrapper, nothing more |
| `ALPRSystem.process_frame()` + `finalize_track()`/`finalize_stream()`, or `week9_mp4_runner.py` (which wraps exactly this) | **Yes** | Pools every sampled frame of a tracked vehicle, then applies agreement + confidence + optional plate-profile checks before accepting or abstaining |

If you're reading a single photo where there's no video to track across,
the single-frame calls are the right tool and there's nothing missing —
there's no "vehicle over time" for a reliability layer to pool evidence
from in the first place. The distinction matters when you have a video
or a live stream: use the streaming path if you want RC1's actual
accept-or-abstain guarantee; use the single-frame calls only if you
specifically want one frame's raw reading (e.g. for debugging, or
building your own custom aggregation).

## Quick start

**Single image (no reliability layer — see table above):**
```python
import sys
sys.path.insert(0, "pipeline")
from pipeline.alpr_pipeline import AlprPipeline
import cv2

pipeline = AlprPipeline()  # loads frozen V1.1 recognizer + detector automatically
image = cv2.imread("path/to/image.jpg")
result = pipeline.process(image)
print(result.status, result.plate_text, result.recognition_confidence)
```

**Video / streaming, one command (full reliability layer):**
```bash
cd pipeline
python3 week9_mp4_runner.py /path/to/video.mp4 --stride 2
```
`video_path` is positional (not a flag); `--stride` is optional and
defaults to 2. This runs the full detect → track → fuse →
reliability-check path and prints one structured result per vehicle
encounter.

**Or via the component API** (for streaming frame-by-frame, e.g. from a
live camera feed rather than a file — also full reliability layer):
```python
from pipeline.alpr_pipeline import AlprPipeline
from alpr_system import ALPRSystem, SystemConfig

pipeline = AlprPipeline()
system = ALPRSystem(pipeline, SystemConfig())
for frame_idx, frame in enumerate(frames):
    system.process_frame("stream1", frame, frame_idx=frame_idx)
results = system.finalize_stream("stream1", candidate_frame_provider=lambda track: all_frames)
```

**`ALPRSystem` also exposes `process_image(image)`** as a convenience —
it's just `pipeline.process(image).to_dict()` under the hood, so it has
the same no-reliability-layer semantics as the direct call above; it
exists so callers already holding an `ALPRSystem` instance don't need a
separate `AlprPipeline` reference for one-off single-image reads.

## Structured output

Every result is a small, consistent structure regardless of whether it
came from a single image, a full video, or a rescued detection failure:

| Field | Meaning |
|---|---|
| `status` | `"ok"` (accepted), `"no_reliable_result"` (abstained — agreement, confidence, or plate-profile check failed), or `"complete_detection_failure"` (the plate was never detected across any examined frame) |
| `text` | The plate string, or `None` if not `"ok"` |
| `agreement_count` | How many of the selected observations agreed on `text` |
| `reason` | Human-readable explanation of why the decision landed where it did — always populated on non-`"ok"` results |
| `failure_kind` | One of `zero_observations`, `insufficient_observations` (both detection-related, rescue-eligible), `disagreement` (agreement/confidence/profile failure — never rescued), or `None` on success |
| `num_selected`, `num_usable_total` | How many observations were selected vs. how many were available in total |

Per-frame telemetry (not a hard gate — see Known Limitations) is also
recorded on every observation: plate crop size, sharpness (`blur_score`),
brightness/exposure, and a continuous edge-margin measurement (how close
the detected plate sits to the frame boundary).

## Frozen artifacts

| | Path | SHA256 |
|---|---|---|
| Recognizer (default) | `output/v1_1_finetune_balanced/onnx/best.onnx` | `c9c4f6196c6c1a0a10e9d06d74ee091df95cb00a896c1f213241ac1a179feb28` |
| Recognizer (V1 fallback) | `output/full_run1/onnx/best.onnx` | `5e03d7fdbc6120262a3fabb9a6c39a299426338f4e538a51d75f378a8f7308e9` |

Verify after cloning:
```bash
sha256sum output/v1_1_finetune_balanced/onnx/best.onnx output/full_run1/onnx/best.onnx
```
Both should match the table above exactly. The detector (YOLOv9-tiny
license-plate model, via the `open-image-models` package) is fetched and
cached automatically on first run — it isn't bundled in this repository.

`frozen_config.json` is the single authoritative record of the complete
runtime configuration (tracker, fusion, decision rule, rescue policy,
detector identifier/hash, recognizer paths/hashes, and the exact git
commit/tag this configuration was frozen against). Regenerate it any
time from `pipeline/dump_frozen_config.py` — it's derived from the
actual code defaults, never maintained by hand. Its paths are
repository-relative, so it stays correct if this repository is moved.

To roll back to V1: `PlateRecognizer(model_path=recognizer.V1_FROZEN_MODEL_PATH)`.

## Dataset

This project uses the **UFPR-ALPR** dataset (Laroca et al., IJCNN 2018) —
4,500 annotated images from 150 vehicles, Paraná, Brazil. It is not
bundled in this repository (it requires a signed license agreement with
its authors). To obtain it:
- Dataset page: https://github.com/raysonlaroca/ufpr-alpr-dataset
- Request process: fill out the license agreement in that repository and
  email it to the dataset's maintainers (contact listed there)

This repository ships only our own derived artifacts, which don't
require the raw dataset to inspect: `data/manifests/` (train/val/test
split assignments), `training/kaggle_candidates.csv` and
`training/week9_local13_ground_truth.json` (real US-plate examples used
in V1.1's fine-tune), and all `eval/results/*.json` (every benchmark
number already computed against the real data, for provenance).

## Running the tests

```bash
python3 tests/test_stage3_synthetic.py
python3 tests/test_video_pipeline_integration.py
python3 tests/test_multi_plate.py
python3 tests/test_rc1_decision_path.py
```
All four use synthetic/constructed data — no real dataset or model files
needed. Each should print `PASS` for every sub-test and end with an
`ALL ... PASSED` line.

## Repository structure

```
pipeline/       live, frozen system — the actual deployed code
tests/          regression/integration tests (synthetic data, no models needed)
examples/       13 representative real images (own-collected, not UFPR) with a runner script
training/       V1.1 dataset construction + fine-tuning scripts
eval/           benchmark scripts, Week 5 through RC1 (historical — see note below)
eval/results/   every benchmark's raw output, for provenance
diagnostics/    track-0080-style deep-dive investigation scripts + findings
archive/        superseded code and debug artifacts, kept for history
weights/        shared base model architecture/config (both recognizer versions use this)
output/         frozen ONNX artifacts (see Frozen artifacts above)
data/           dataset manifests (see Dataset above)
```

**Note on `eval/` and `training/`:** these are historical scripts from
the project's development, each written against the original
development machine's absolute paths. They're included for provenance
and reproducibility reference, not as polished re-runnable tools — to
re-run one, set `ALPR_ROOT` (see above) or edit that script's own
`DEFAULT_CONFIG_PATH`-style constant near its top. The live system
(`pipeline/`) and the tests (`tests/`) don't have this limitation — both
are fully portable out of the box.

## Known limitations and failure cases

Full detail — including tracks 0080 (a genuine recognizer confusion
pattern, `1`/`U`/`I`/`J`), 0104 (a plate-boundary/length ambiguity), and
0107 (a persistent low-confidence case correctly abstained on) — is in
`model_card_recognizer_v1_1.md`. Summary: strong on 6- and 7-character
plates specifically; no data-derived visual-quality gate yet (recorded
as telemetry, not enforced — the validation set doesn't contain enough
genuinely degraded examples to derive one defensibly); plate-format
reranking exists as a tested capability but isn't wired into the live
path.

## What's implemented, what's deferred, and what's next

**Implemented:** detection (including multi-plate — see `process_all()`
above), tracking, multi-frame fusion, the RC1 reliability layer
(agreement + confidence + optional plate-format validation), rescue for
detection-related failures, the V1.1 recognizer fine-tune, full
telemetry recording.

**Deferred (deliberately, not forgotten):** a data-derived visual-quality
operating gate (needs a broader validation set with genuinely degraded
examples); plate-profile reranking wired into the live tracking path
(needs per-slot model probabilities threaded through the observation
layer, currently only computed in the diagnostic scripts); jurisdiction
auto-detection (the profile check currently requires the caller to
supply a known profile — there's no automatic jurisdiction inference).

**Recommended next steps:** collect or synthesize a validation set
spanning genuinely small/blurred/clipped/low-light plates to derive the
quality gate properly; if the plate-boundary ambiguity pattern (seen on
track 0104) recurs, investigate whether it's a detector cropping issue
or a recognizer decoding issue before deciding which layer should
address it; device-level (mobile/edge) latency and memory benchmarking —
the current footprint numbers are development-environment measurements
only, not representative of deployed performance.
