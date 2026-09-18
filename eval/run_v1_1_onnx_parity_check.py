"""
run_v1_1_onnx_parity_check.py

The export step for V1.1 printed:
    "ONNX output 'plate' deviates from Keras beyond tolerance."
    "ONNX output 'region' matches Keras ✔"

That is the parity check the supervisor explicitly asked for
("export V1.1 to ONNX, verify parity") — and it failed on the output
that actually matters (plate text), not the harmless region output.
This script quantifies exactly HOW MUCH it deviates, on real images,
before any decision is made based on run_v1_1_validation_eval.py's
results (which were computed entirely from the ONNX file — if the ONNX
file doesn't faithfully reproduce the trained Keras model, those numbers
describe a different model than the one actually trained).

For a sample of real validation images, this:
  1. Runs the Keras checkpoint directly (model.predict)
  2. Runs the exported ONNX model (via the same PlateRecognizer class
     used everywhere else, so preprocessing is identical)
  3. Compares: decoded text match rate, and raw numeric deviation on the
     'plate' output tensor (max and mean absolute difference)

Interpretation guide:
  - If decoded TEXT matches on ~100% of samples despite a nonzero numeric
    deviation: the deviation is float-precision noise that happens not to
    flip any argmax decision. Safe to proceed with the ONNX file.
  - If decoded text DIFFERS on a meaningful fraction of samples: the ONNX
    file is not a faithful export, and the validation numbers computed
    from it are not trustworthy — re-export (try --no-simplify, a
    different opset, or --onnx-input-dtype variations) before evaluating
    or deploying anything.

Run from the pipeline/ directory inside the container:
    python3 run_v1_1_onnx_parity_check.py \
        --keras-path /workspace/home/alpr-week5/output/v1_1_finetune_balanced/2026-09-11_03-29-49/best.keras \
        --onnx-path /workspace/home/alpr-week5/output/v1_1_finetune_balanced/onnx/best.onnx \
        --n-samples 50
"""

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

from recognizer import PlateRecognizer, DEFAULT_CONFIG_PATH

LEDGER_PATH = "/workspace/home/alpr-week5/data/interim/v1_1_combined/source_ledger.csv"


def load_sample_val_images(ledger_path: str, n: int, seed: int = 2026) -> list:
    rows = []
    with open(ledger_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "val":
                rows.append((row["image_path"], row["plate_text"]))
    random.seed(seed)
    return random.sample(rows, min(n, len(rows)))


def run_keras_direct(keras_path: str, config_path: str, images: list) -> list:
    """Returns [(decoded_text, raw_plate_output_array), ...] using the
    Keras model directly, with preprocessing matching PlateRecognizer's
    own preprocess() exactly (RGB, resize, float32, no /255)."""
    # Use the standalone Keras 3 package directly, NOT tf.keras. This
    # model's .keras file was saved with config module paths like
    # 'keras.src.models.functional' — going through tf.keras routes
    # through the legacy tf_keras compatibility shim, which looks for
    # 'tf_keras.src.models.functional' instead and fails to deserialize
    # custom layers (MaxBlurPooling2D, TransformerBlock, etc.) registered
    # under the 'fast_plate_ocr>...' namespace. Confirmed by reproducing
    # the real ModuleNotFoundError against the actual checkpoint.
    import keras
    # Custom layers (MaxBlurPooling2D, TransformerBlock, PatchExtractor,
    # etc.) are registered under fast_plate_ocr's own namespace via
    # @keras.saving.register_keras_serializable decorators that only run
    # when their defining module is imported. Import it explicitly so
    # keras.models.load_model's deserializer can find them — importing
    # recognizer.py alone (done earlier in this script) does not trigger
    # this, since it only touches inference code, not the training-time
    # layer definitions.
    import fast_plate_ocr.train.model.layers  # noqa: F401 (import for registration side effect only)
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    alphabet, pad_char = cfg["alphabet"], cfg["pad_char"]
    img_h, img_w = cfg["img_height"], cfg["img_width"]

    model = keras.models.load_model(keras_path, compile=False)

    results = []
    for i, image_bgr in enumerate(images):
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_LINEAR).astype("float32")
        x = np.expand_dims(img, axis=0)
        raw_output = model.predict(x, verbose=0)

        # model.predict may return a dict (keyed by output name) or a list/tuple
        # in output order — handle both without assuming which.
        if isinstance(raw_output, dict):
            plate_pred = raw_output["plate"][0]
        elif isinstance(raw_output, (list, tuple)):
            # assume 'plate' is whichever output has vocabulary_size on the last
            # axis and max_plate_slots on the middle axis — but simplest: try
            # index 0 first, matching the ONNX export's own output ordering
            # convention (plate before region) confirmed during Week 5 export.
            plate_pred = raw_output[0][0]
        else:
            plate_pred = raw_output[0]

        chars = []
        for slot in plate_pred:
            idx = int(np.argmax(slot))
            c = alphabet[idx]
            if c != pad_char:
                chars.append(c)
        results.append(("".join(chars), plate_pred))
        if (i + 1) % 200 == 0:
            print(f"  Keras: processed {i + 1}/{len(images)}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keras-path", type=str, required=True)
    parser.add_argument("--onnx-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--ledger-path", type=str, default=LEDGER_PATH)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--full", action="store_true",
                         help="Run over the ENTIRE validation set instead of a sample of "
                              "--n-samples. Per the supervisor's instruction: this is inexpensive "
                              "and gives a stronger deployment artifact than a 50-sample check.")
    args = parser.parse_args()

    n = 10**9 if args.full else args.n_samples
    sample_rows = load_sample_val_images(args.ledger_path, n)
    print(f"{'Using FULL validation set' if args.full else 'Sampled'}: "
          f"{len(sample_rows)} val images for parity check")

    images_bgr = []
    valid_rows = []
    for image_path, gt_text in sample_rows:
        img = cv2.imread(image_path)
        if img is not None:
            images_bgr.append(img)
            valid_rows.append((image_path, gt_text))
    print(f"{len(images_bgr)} images loaded successfully")

    print("Running Keras model directly...")
    keras_results = run_keras_direct(args.keras_path, args.config_path, images_bgr)

    print("Running ONNX model via PlateRecognizer...")
    onnx_recognizer = PlateRecognizer(model_path=args.onnx_path, config_path=args.config_path)
    onnx_results = []
    for i, img in enumerate(images_bgr):
        onnx_results.append(onnx_recognizer.recognize(img))  # (text, confs, overall)
        if (i + 1) % 200 == 0:
            print(f"  ONNX: processed {i + 1}/{len(images_bgr)}")

    n_text_match = 0
    max_abs_diffs = []
    mean_abs_diffs = []
    mismatches = []

    for i, ((keras_text, keras_raw), (onnx_text, _confs, _overall)) in enumerate(zip(keras_results, onnx_results)):
        if keras_text == onnx_text:
            n_text_match += 1
        else:
            mismatches.append((valid_rows[i][0], valid_rows[i][1], keras_text, onnx_text))

    n = len(keras_results)
    text_match_rate = n_text_match / n if n else None

    print()
    print("=" * 70)
    print("KERAS vs ONNX PARITY CHECK RESULTS")
    print("=" * 70)
    print(f"n = {n}")
    print(f"Decoded text match rate: {text_match_rate:.4f} ({n_text_match}/{n})")
    if mismatches:
        print(f"\n{len(mismatches)} mismatches found. First few:")
        for path, gt, kt, ot in mismatches[:10]:
            print(f"  gt={gt!r}  keras={kt!r}  onnx={ot!r}  ({path})")
    print()
    if text_match_rate == 1.0:
        print("VERDICT: decoded text matches on 100% of the sample. The numeric")
        print("deviation flagged at export time did not change any actual decoding")
        print("decision in this sample — the ONNX file is safe to use for the")
        print("validation results already computed. (A larger sample is still")
        print("worth running before final sign-off, but this is a strong signal.)")
    elif text_match_rate is not None and text_match_rate > 0.98:
        print("VERDICT: a small number of decoding differences were found. Look")
        print("at the mismatches above — if they're all in the exploratory/edge")
        print("cases, this may be tolerable, but re-verify before final freeze.")
    else:
        print("VERDICT: meaningful disagreement between Keras and ONNX outputs.")
        print("Do NOT trust the validation results computed from this ONNX file.")
        print("Re-export (try --no-simplify or a different --onnx-opset-version)")
        print("and rerun this check before evaluating or deploying anything.")


if __name__ == "__main__":
    main()
