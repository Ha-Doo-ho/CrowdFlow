import sqlite3

import numpy as np
import pytest

from frontend.data_logger import DataLogger


def sample_result():
    return {
        "timestamp": "2026-06-15T12:00:00",
        "grid": np.zeros((5, 5)),
        "level": np.ones((5, 5), dtype=int),
        "count": np.zeros((5, 5), dtype=int),
        "avg_conf": np.ones((5, 5)),
        "max_density": 0.0,
        "ignored_count": 2,
        "alerts": [],
        "metadata": {
            "frame_number": 3,
            "inference_ms": 25.5,
            "model_name": "model.pt",
            "model_type": "yolo",
            "calibration_mode": "full_frame_fallback",
        },
    }


def test_logger_persists_diagnostic_fields(tmp_path):
    db_path = tmp_path / "crowdflow.db"
    logger = DataLogger(str(db_path), verbose=False)

    logger.log_frame(sample_result())
    stats = logger.get_stats()
    logger.close()

    assert stats["total_frames"] == 1
    assert stats["ignored_detections"] == 2
    assert stats["avg_inference_ms"] == 25.5

    connection = sqlite3.connect(db_path)
    row = connection.execute(
        "SELECT frame_number, model_name, calibration_mode FROM frames"
    ).fetchone()
    connection.close()

    assert row == (3, "model.pt", "full_frame_fallback")


def test_experiment_metrics(tmp_path):
    logger = DataLogger(str(tmp_path / "crowdflow.db"), verbose=False)
    logger.log_experiment(1, actual_count=10, estimated_count=8, area_m2=20)
    logger.log_experiment(2, actual_count=10, estimated_count=12, area_m2=20)

    metrics = logger.get_experiment_metrics()
    logger.close()

    assert metrics["sample_count"] == 2
    assert metrics["count_mae"] == 2.0
    assert metrics["count_rmse"] == 2.0
    assert metrics["density_mae"] == pytest.approx(0.1)
