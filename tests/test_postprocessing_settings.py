import json

import pytest

from core.settings import AppSettings


def test_default_postprocessing_settings_match_experiment_rule():
    settings = AppSettings()

    assert settings.cross_class_dedup_enabled is True
    assert settings.dedup_iou_threshold == 0.65
    assert settings.boundary_stabilization_enabled is True
    assert settings.boundary_history_size == 3
    assert settings.boundary_majority_count == 2


def test_rejects_majority_larger_than_history(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({
            "boundary_history_size": 3,
            "boundary_majority_count": 4,
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        AppSettings.load(config_path)
