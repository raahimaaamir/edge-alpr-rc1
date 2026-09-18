"""
Plate recognizer component — wraps the frozen Recognizer V1.1 ONNX model.

FROZEN 2026-09-12: Recognizer V1.1 replaces V1 as the default, per the
supervisor's approval. V1.1 is a conservative fine-tune from V1 (see
Week 9 report) that removes a systematic insertion bias on 6-character
plates (V1 over-read them as 7 characters), verified to preserve
7-character performance (system-level regression check: exact accuracy
0.9545 -> 0.9545 unchanged, coverage 0.9545 -> 1.0000, one encounter
traded an abstention for a wrong answer — see model card for the full
writeup) and confirmed via ONNX/Keras parity on the complete validation
set (2,027/2,027 decoded-text match). V1's own frozen artifact remains
on disk, unmodified, at V1_FROZEN_MODEL_PATH below, for any script that
needs an explicit side-by-side comparison against it.

Implements exactly the spec in recognizer_v1_spec.md (architecture and
I/O contract unchanged between V1 and V1.1):
- input: [N, 64, 128, 3] float32, RGB, raw 0-255 (no manual /255 normalization)
- output 'plate': [N, 10, 37] per-slot char probabilities; 'region' output is ignored
- decode: argmax per slot, drop pad char '_', concatenate
- confidence: per-char = argmax probability; overall = min over output chars
"""
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort
import yaml

# Package root — this file lives at <root>/pipeline/recognizer.py, so its
# parent's parent is <root>. Auto-detected so the package works immediately
# after being copied/cloned anywhere, with no manual setup step required.
# ALPR_ROOT env var overrides this if set, for anyone using a different
# layout (e.g. running pipeline/ from a location other than its packaged
# position relative to output/ and weights/).
_PACKAGE_ROOT = Path(os.environ.get("ALPR_ROOT", Path(__file__).resolve().parent.parent))

# V1.1 — frozen default as of 2026-09-12. See model_card_recognizer_v1_1.md
# for the hash and full freeze record.
DEFAULT_MODEL_PATH = str(_PACKAGE_ROOT / "output" / "v1_1_finetune_balanced" / "onnx" / "best.onnx")
DEFAULT_CONFIG_PATH = str(_PACKAGE_ROOT / "weights" / "cct_s_v2_global_plate_config.yaml")

# V1's own frozen artifact, kept on disk unmodified — pass this explicitly
# to PlateRecognizer(model_path=V1_FROZEN_MODEL_PATH) for any script that
# needs to compare against the previous frozen version directly.
V1_FROZEN_MODEL_PATH = str(_PACKAGE_ROOT / "output" / "full_run1" / "onnx" / "best.onnx")


class PlateRecognizer:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, config_path: str = DEFAULT_CONFIG_PATH):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.alphabet: str = cfg["alphabet"]
        self.pad_char: str = cfg["pad_char"]
        self.img_h: int = cfg["img_height"]
        self.img_w: int = cfg["img_width"]

        self.session = ort.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name
        outputs = [o.name for o in self.session.get_outputs()]
        self.plate_output_idx = outputs.index("plate")
        self.model_path = str(model_path)

    def preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        return img.astype("float32")  # do NOT divide by 255 — model has internal Rescaling

    def _decode(self, pred_array: np.ndarray):
        chars: List[str] = []
        confs: List[float] = []
        for slot in pred_array:
            idx = int(np.argmax(slot))
            c = self.alphabet[idx]
            if c != self.pad_char:
                chars.append(c)
                confs.append(float(slot[idx]))
        return "".join(chars), confs

    def recognize(self, crop_bgr: np.ndarray):
        """
        Returns (plate_text, per_char_confidence, overall_confidence).
        overall_confidence is None if the plate decoded to an empty string.
        """
        x = np.expand_dims(self.preprocess(crop_bgr), axis=0)
        outputs = self.session.run(None, {self.input_name: x})
        plate_pred = outputs[self.plate_output_idx][0]
        text, confs = self._decode(plate_pred)
        overall = min(confs) if confs else None
        return text, confs, overall
