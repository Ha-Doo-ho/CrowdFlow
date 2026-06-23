import json

import pytest

from core.settings import AppSettings


def test_load_settings_from_json(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({
            "source": "data/test_video1.mp4",
            "model_path": "weights/model.pt",
            "model_type": "yolo",
            "imgsz": 640,
            "analyze_every": 2,
        }),
        encoding="utf-8",
    )

    settings = AppSettings.load(config_path)

    assert settings.source == "data/test_video1.mp4"
    assert settings.model_type == "yolo"
    assert settings.imgsz == 640
    assert settings.analyze_every == 2
    assert settings.grid_size == 2.0


def test_rejects_invalid_confidence(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"detection_confidence": 1.2}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        AppSettings.load(config_path)
