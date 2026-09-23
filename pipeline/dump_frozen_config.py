"""
dump_frozen_config.py

Writes the COMPLETE frozen system configuration to one JSON file —
tracking, fusion, decision/acceptance, rescue, recognizer identity, and
stride — so the release candidate has a single source of truth for
"what exactly is this system running," rather than defaults scattered
across several modules.

PORTABLE BY CONSTRUCTION: every path recorded here is relative to the
package root (the directory containing pipeline/, output/, weights/) —
never an absolute, machine-specific path. This file is correct
regardless of where the repository is cloned or moved to. Also records
the exact git commit and tag this configuration was frozen against, so
"what state was RC1 actually in" is always independently verifiable,
not just asserted.

Uses the existing stage3_config.config_to_dict() for the tracker/fusion
portion (unchanged, already the project's own convention) and adds the
pieces stage3_config.py doesn't cover: the video-pipeline stride, the
Top-K/decision AcceptanceConfig, the rescue policy, and which recognizer
artifact (path + hash) this configuration is frozen against.

Run from the pipeline/ directory:
    python3 dump_frozen_config.py
"""

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

# tiled_detection_rescue.py (a dependency of week9_rescue_policy) imports
# `pipeline.result_types` — put the directory ABOVE pipeline/ on sys.path
# so that resolves correctly, same fix as the other Week 9 scripts.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PACKAGE_ROOT))

from stage3_config import load_config, config_to_dict
from week9_decision_rule import AcceptanceConfig
from week9_rescue_policy import RescuePolicyConfig
from recognizer import DEFAULT_MODEL_PATH, DEFAULT_CONFIG_PATH, V1_FROZEN_MODEL_PATH
from decision_rule_rc1 import MIN_WINNING_CONFIDENCE
from detector import DEFAULT_MODEL as DEFAULT_DETECTOR_MODEL

STRIDE = 2  # matches run_week8_final_test_evaluation.py / run_v1_vs_v1_1_system_regression.py
OUTPUT_PATH = "frozen_config.json"


def find_detector_onnx(model_identifier: str) -> str:
    """Locates the cached detector ONNX file that `open-image-models`
    downloads on first use. Doesn't hardcode the exact filename (it
    doesn't exactly match the model identifier's own naming — the
    identifier says "plate", the cached file says "plates") since that's
    the library's own internal detail, not something to hardcode and
    risk silently breaking if it ever changes; glob-searches the
    standard cache directory instead."""
    cache_dir = Path.home() / ".cache" / "open-image-models" / model_identifier
    if not cache_dir.is_dir():
        return None
    onnx_files = list(cache_dir.glob("*.onnx"))
    return str(onnx_files[0]) if onnx_files else None


def get_package_version(package_name: str) -> str:
    try:
        from importlib.metadata import version
        return version(package_name)
    except Exception:
        return None


def sha256_of_file(path: str) -> str:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_repo_relative(absolute_path: str) -> str:
    """Converts an absolute path (as recognizer.py resolves it at
    runtime, via ALPR_ROOT auto-detection) to a path relative to the
    package root, so the recorded value is portable — correct
    regardless of where this repository lives on disk."""
    try:
        return str(Path(absolute_path).resolve().relative_to(_PACKAGE_ROOT))
    except ValueError:
        # only happens if the path genuinely isn't under the package
        # root (e.g. a custom ALPR_ROOT override pointing elsewhere) —
        # record it as-is rather than raise, since that's still
        # meaningful information, just not portable in this case.
        return absolute_path


def _run_git(args: list) -> str:
    """Returns git command output, or None if git isn't available, this
    isn't a git repository, or the specific ref doesn't exist (e.g. no
    tags yet) — never raises, since this is metadata, not a hard
    dependency of the frozen config itself."""
    try:
        result = subprocess.run(
            ["git"] + args, cwd=str(_PACKAGE_ROOT),
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def build_frozen_config() -> dict:
    video_config = load_config()
    acceptance_config = AcceptanceConfig()
    rescue_config = RescuePolicyConfig()

    git_commit_sha = _run_git(["rev-parse", "HEAD"])
    git_tag = _run_git(["describe", "--tags", "--exact-match"])  # None if HEAD isn't exactly on a tag
    git_describe = _run_git(["describe", "--tags", "--always"])  # falls back to short commit hash if no tags at all

    return {
        "frozen_date": "2026-09-12",
        "release_candidate": "RC1",
        "git": {
            "commit_sha": git_commit_sha,
            "tag_exact": git_tag,
            "describe": git_describe,
            "note": "tag_exact is None if HEAD is not exactly on a tagged commit — "
                    "tag this commit (e.g. `git tag -a RC1 -m \"Release Candidate 1\"`) "
                    "and rerun this script to capture it.",
        },
        "stride": STRIDE,
        "tracking_and_fusion": config_to_dict(video_config),
        "acceptance": asdict(acceptance_config),
        "decision_rule": {
            "module": "decision_rule_rc1.decide_rc1",
            "supersedes": "week9_decision_rule.decide_v2",
            "path": ["adequate observations", "temporal agreement", "confidence check",
                     "applicable plate-profile validation", "ACCEPT / NO_RELIABLE_RESULT"],
            "min_winning_confidence": MIN_WINNING_CONFIDENCE,
            "min_winning_confidence_note": "validation-selected conservative heuristic, not a calibrated probability",
            "plate_profile_note": "optional; skipped entirely when jurisdiction unknown, never a forced rejection",
            "rule_comparison_note": "rules B (confidence-only), E (profile-only), and F (both, this rule) "
                                     "produced IDENTICAL results on the 22-track validation set — F was chosen "
                                     "for independent-check architecture, not a demonstrated accuracy benefit "
                                     "on this dataset; see model_card_recognizer_v1_1.md",
        },
        "rescue": {
            "max_candidate_frames": rescue_config.max_candidate_frames,
            "tiled_config": asdict(rescue_config.tiled_config),
            "rescue_acceptance": asdict(rescue_config.rescue_acceptance),
        },
        "detector": {
            "model_identifier": DEFAULT_DETECTOR_MODEL,
            "package": "open-image-models",
            "package_version": get_package_version("open-image-models"),
            "conf_thresh": 0.25,
            "onnx_path": (lambda p: str(Path(p).resolve()) if p else None)(find_detector_onnx(DEFAULT_DETECTOR_MODEL)),
            "onnx_sha256": sha256_of_file(find_detector_onnx(DEFAULT_DETECTOR_MODEL)),
            "source_note": "Fetched and cached automatically by the open-image-models package on "
                           "first use (requires internet access that one time; cached locally "
                           "thereafter, at ~/.cache/open-image-models/). Not bundled in this "
                           "repository. An unmodified, off-the-shelf pretrained model, not a "
                           "trainable checkpoint — see detector.py's module docstring.",
            "onnx_path_note": "the recorded path is this machine's local cache location, NOT "
                              "repository-relative — the detector artifact lives outside the repo "
                              "by design (see source_note); this path will differ on every machine.",
        },
        "recognizer": {
            "default_model_path": to_repo_relative(DEFAULT_MODEL_PATH),
            "default_model_sha256": sha256_of_file(DEFAULT_MODEL_PATH),
            "config_path": to_repo_relative(DEFAULT_CONFIG_PATH),
            "v1_fallback_model_path": to_repo_relative(V1_FROZEN_MODEL_PATH),
            "v1_fallback_model_sha256": sha256_of_file(V1_FROZEN_MODEL_PATH),
            "path_note": "paths are relative to the repository root (this file's grandparent "
                         "directory) — portable regardless of where the repo is cloned/moved.",
        },
    }


def main():
    config = build_frozen_config()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote {OUTPUT_PATH}")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
