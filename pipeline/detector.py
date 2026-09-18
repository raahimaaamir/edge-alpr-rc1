"""
Plate detector component.

KNOWN LIMITATION (per supervisor's Week 3 audit request): the pretrained
YOLO-v9 models distributed via `open-image-models` are ONNX inference-only —
this package does not provide a trainable checkpoint, training config, or
reproducible training source for these specific exported models. They are
used here as-is, unmodified, as a functional placeholder. When a trainable,
ONNX-exportable, permissively-licensed detector is selected (e.g. YOLOX,
Apache-2.0) and fine-tuned on project data, swap it in by implementing the
same `detect(image) -> list[(BoundingBox, confidence)]` interface below —
nothing else in the pipeline needs to change.
"""
from typing import List, Tuple

import numpy as np

from pipeline.result_types import BoundingBox

DEFAULT_MODEL = "yolo-v9-t-384-license-plate-end2end"


class PlateDetector:
    def __init__(self, model: str = DEFAULT_MODEL, conf_thresh: float = 0.25):
        from open_image_models import create_detector
        self._detector = create_detector(model, conf_thresh=conf_thresh)
        self.model_name = model
        self.conf_thresh = conf_thresh

    def detect(self, image: np.ndarray) -> List[Tuple[BoundingBox, float]]:
        """
        image: BGR numpy array (as loaded by cv2.imread).
        Returns list of (BoundingBox, confidence), sorted by confidence descending.
        """
        raw = self._detector.predict(image)
        out = []
        for det in raw:
            box = det.bounding_box
            bbox = BoundingBox(x1=float(box.x1), y1=float(box.y1), x2=float(box.x2), y2=float(box.y2))
            out.append((bbox, float(det.confidence)))
        out.sort(key=lambda t: t[1], reverse=True)
        return out
