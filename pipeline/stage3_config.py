"""
stage3_config.py

Single place for every stage-3 threshold, so nothing is hard-coded inline in
tracker.py / fusion.py / video_pipeline.py. Load a preset with
`load_config()` or build a VideoPipelineConfig by hand for experiments.

Values below are reasonable starting points, not tuned — first thing to
sweep once real video data is available.
"""

from dataclasses import asdict
import json

from tracker import TrackerConfig
from fusion import FusionConfig
from video_pipeline import VideoPipelineConfig
from tracker_eval import TrackerEvalConfig

DEFAULT_TRACKER_CONFIG = TrackerConfig(
    iou_threshold=0.3,
    max_center_disp_frac=0.5,
    max_missing_frames=5,
    require_both_gates=False,
)

DEFAULT_FUSION_CONFIG = FusionConfig(
    min_agreement_ratio=0.5,
    min_frames_used=1,
    min_overall_conf=0.0,
)

DEFAULT_TRACKER_EVAL_CONFIG = TrackerEvalConfig(
    gt_match_iou_threshold=0.5,
)

DEFAULT_FUSION_METHOD = "majority_vote"
DEFAULT_DETECTOR_CONF_THRESH = 0.25


def load_config() -> VideoPipelineConfig:
    return VideoPipelineConfig(
        tracker=DEFAULT_TRACKER_CONFIG,
        fusion=DEFAULT_FUSION_CONFIG,
        fusion_method=DEFAULT_FUSION_METHOD,
        detector_conf_thresh=DEFAULT_DETECTOR_CONF_THRESH,
    )


def config_to_dict(config: VideoPipelineConfig) -> dict:
    return {
        "tracker": asdict(config.tracker),
        "fusion": asdict(config.fusion),
        "fusion_method": config.fusion_method,
        "detector_conf_thresh": config.detector_conf_thresh,
    }


def dump_config(config: VideoPipelineConfig, path: str) -> None:
    with open(path, "w") as f:
        json.dump(config_to_dict(config), f, indent=2)
