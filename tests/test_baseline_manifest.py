import json
from pathlib import Path

from core.settings import AppSettings


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_configs_are_identical_and_loadable():
    primary_path = ROOT / "config" / "app_config.json"
    alias_path = ROOT / "config" / "app_config_7_85x10.json"

    assert primary_path.read_bytes() == alias_path.read_bytes()

    settings = AppSettings.load(primary_path)
    assert settings.area_width == 7.85
    assert settings.area_height == 10.0
    assert settings.grid_size == 1.0
    assert settings.boundary_margin_m == 0.15
    assert settings.calibration_path == "config/calibration_7_85x10.json"
    assert settings.source.endswith("20260711_164845.mp4")


def test_manifest_matches_config_and_calibration():
    manifest_path = ROOT / "config" / "baseline_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    settings = AppSettings.load(ROOT / manifest["primary_config"])
    calibration = json.loads(
        (ROOT / manifest["primary_calibration"]).read_text(
            encoding="utf-8"
        )
    )

    assert manifest["baseline_id"] == (
        "crowdflow-7_85x10-yolo11-20260803"
    )
    assert settings.source == manifest["default_video"]
    assert settings.model_path == manifest["primary_model"]
    assert calibration["dst_points"] == [
        [0.0, 0.0],
        [7.85, 0.0],
        [0.0, 10.0],
        [7.85, 10.0],
    ]
