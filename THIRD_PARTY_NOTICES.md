# Third-Party Notices

This file lists the datasets, models, and open-source packages this
project depends on, their sources, and the license/usage terms and
redistribution limitations identified during this project. It states
what the source licenses say, not legal conclusions beyond that — if
anything here matters for a real deployment decision, consult the
original source directly rather than relying solely on this summary.

## Datasets

### UFPR-ALPR
- **Source**: https://github.com/raysonlaroca/ufpr-alpr-dataset
- **Citation**: R. Laroca et al., "A Robust Real-Time Automatic License
  Plate Recognition Based on the YOLO Detector," IJCNN 2018.
- **Usage terms**: requires filling out a signed license agreement and
  emailing it to the dataset's maintainers (contact listed on the
  source page above); approval is not automatic or immediate.
- **Redistribution**: **not permitted**. The agreement explicitly
  states the requester will not make any part of the dataset available
  to a third party. This project respects that: no UFPR-ALPR images,
  crops, or derived pixel data are included in this repository — only
  our own derived non-image artifacts (manifests, benchmark result
  JSONs) that don't reconstruct the source images.

### RodoSol-ALPR
- **Source**: https://github.com/raysonlaroca/rodosol-alpr-dataset
- **Citation**: R. Laroca, E. V. Cardoso, D. R. Lucio, V. Estevam, and
  D. Menotti, "On the Cross-dataset Generalization in License Plate
  Recognition," VISAPP 2022.
- **Usage terms**: same license-agreement process and maintainer
  contact as UFPR-ALPR above (both datasets are maintained by the same
  research group).
- **Redistribution**: treated with the same restriction as UFPR-ALPR
  above — not independently re-confirmed word-for-word for this
  specific dataset's agreement text, but assumed at least as
  restrictive given the identical process and maintainers. No
  RodoSol-ALPR data is included in this repository.
- **Used for**: V1 recognizer training, per the project's own training
  records (see `FINAL_REPORT.md` Section 3).

### Kaggle dataset (real US license plates)
- **Used for**: ~300 manually-verified real US-plate samples in V1.1's
  training data (see `FINAL_REPORT.md` Section 7).
- **NEEDS CONFIRMATION**: the exact Kaggle dataset name/URL and its
  license terms are not recorded anywhere in this project's existing
  documentation that this notice file was built from. Whoever
  maintains this repository next should identify the specific dataset
  used (check `training/build_v1_1_training_manifest.py` and
  `training/kaggle_candidates.csv` for any source hints) and add its
  terms here before treating this project's provenance trail as
  complete.

## Models

### Detector: YOLOv9 (via `open-image-models`)
- **Package source**: https://github.com/ankandrew/open-image-models
  (PyPI: `open-image-models`, MIT license)
- **Model architecture citation**: C.-Y. Wang and H.-Y. M. Liao,
  "YOLOv9: Learning What You Want to Learn Using Programmable Gradient
  Information," arXiv:2402.13616, 2024.
- **Specific checkpoint**: `yolo-v9-t-384-license-plate-end2end` — an
  unmodified, off-the-shelf pretrained checkpoint distributed by the
  `open-image-models` package itself. Its own training data and
  license are not separately published beyond what that package's own
  documentation states; this project did not fine-tune or modify it in
  any way (see `detector.py`'s module docstring and `README.md`'s
  "Detector reproducibility" section for the full reproducibility
  details).
- **Redistribution**: not bundled in this repository at all — fetched
  and cached by the `open-image-models` package itself on first use.

### Recognizer base architecture: CCT (via `fast-plate-ocr`)
- **Package source**: https://github.com/ankandrew/fast-plate-ocr
  (PyPI: `fast-plate-ocr`, MIT license) — same author as
  `open-image-models`.
- **Used for**: the CCT-S (Compact Convolutional Transformer, small)
  base architecture this project's V1 and V1.1 recognizers were trained
  from, and the `fast-plate-ocr train` CLI used to run that training
  (see `README.md`'s environment setup and `FINAL_REPORT.md` Section 3
  — note the CLI command name; an earlier, incorrect invocation of this
  tool as a Python module was caught and corrected during this
  project).
- **This project's own trained weights** (`output/v1_1_finetune_balanced/`,
  `output/full_run1/`, and the shared base config under `weights/`) are
  this project's own artifacts, fine-tuned from that base architecture
  on UFPR-ALPR and RodoSol-ALPR data (see the dataset entries above for
  those datasets' own restrictions) plus this project's own
  synthetic/Kaggle/retained training data.

## Open-source packages

| Package | License | Used for |
|---|---|---|
| `onnxruntime` | MIT | Running the frozen detector/recognizer ONNX models |
| `opencv-python` (OpenCV) | Apache-2.0 | Image loading, cropping, quality assessment |
| `numpy` | BSD-3-Clause | Array operations throughout |
| `pyyaml` | MIT | Reading the recognizer's plate-config YAML |
| `open-image-models` | MIT | Detector model loading and inference (see above) |
| `fast-plate-ocr` | MIT | Recognizer training/fine-tuning only, not runtime inference (see above) |
| `tensorflow` | Apache-2.0 | Training/fine-tuning only, not runtime inference |

None of the above impose copyleft or attribution requirements beyond
what's already satisfied by this file and the citations in
`FINAL_REPORT.md` — all are permissive licenses (MIT, BSD, Apache-2.0).
This is a summary for provenance purposes, not a legal compliance
audit; if this project is ever redistributed commercially or at scale,
each package's actual LICENSE file should be reviewed directly rather
than relying solely on this table.
