"""
dump_frozen_config.py

Writes the COMPLETE frozen system configuration to one JSON file —
tracking, fusion, decision/acceptance, rescue, recognizer identity, and
stride — so the release candidate has a single source of truth for
"what exactly is this system running," rather than defaults scattered
across several modules.

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
import sys
from dataclasses import asdict
from pathlib import Path

# tiled_detection_rescue.py (a dependency of week9_rescue_policy) imports
# `pipeline.result_types` — put the directory ABOVE pipeline/ on sys.path
# so that resolves correctly, same fix as the other Week 9 scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage3_config import load_config, config_to_dict
from week9_decision_rule import AcceptanceConfig
from week9_rescue_policy import RescuePolicyConfig
from recognizer import DEFAULT_MODEL_PATH, DEFAULT_CONFIG_PATH, V1_FROZEN_MODEL_PATH
from decision_rule_rc1 import MIN_WINNING_CONFIDENCE

STRIDE = 2  # matches run_week8_final_test_evaluation.py / run_v1_vs_v1_1_system_regression.py
OUTPUT_PATH = "frozen_config.json"


def sha256_of_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_frozen_config() -> dict:
    video_config = load_config()
    acceptance_config = AcceptanceConfig()
    rescue_config = RescuePolicyConfig()

    return {
        "frozen_date": "2026-09-12",
        "release_candidate": "RC1",
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
        "recognizer": {
            "default_model_path": DEFAULT_MODEL_PATH,
            "default_model_sha256": sha256_of_file(DEFAULT_MODEL_PATH),
            "config_path": DEFAULT_CONFIG_PATH,
            "v1_fallback_model_path": V1_FROZEN_MODEL_PATH,
            "v1_fallback_model_sha256": sha256_of_file(V1_FROZEN_MODEL_PATH),
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
