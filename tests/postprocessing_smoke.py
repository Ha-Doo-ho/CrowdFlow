"""Dependency-light smoke checks for environments without pytest."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.boundary_stabilizer import BoundaryStateStabilizer
from core.detection_postprocessor import suppress_cross_class_duplicates
from core.grid_calculator import GridCalculator
from core.settings import AppSettings


def detection(class_id, confidence):
    return {
        "x1": 10,
        "y1": 10,
        "x2": 50,
        "y2": 90,
        "conf": confidence,
        "class_id": class_id,
        "class_name": "pedestrian" if class_id == 0 else "people",
    }


def mapping(status, foot_x):
    counted = status == "counted"
    return {
        "status": status,
        "bbox": {"x1": 100, "y1": 100, "x2": 160, "y2": 260},
        "conf": 0.9,
        "class_id": 0,
        "class_name": "pedestrian",
        "foot_pixel": [130.0, 260.0],
        "foot_m": [foot_x, 5.0],
        "row": 5 if counted else None,
        "col": 0 if counted else None,
    }


def main():
    filtered, stats = suppress_cross_class_duplicates([
        detection(0, 0.7),
        detection(1, 0.9),
    ])
    assert len(filtered) == 1
    assert filtered[0]["class_id"] == 1
    assert stats["duplicate_boxes_removed"] == 1

    stabilizer = BoundaryStateStabilizer(
        area_width=10,
        area_height=10,
        history_size=3,
        majority_count=2,
        boundary_band_m=0.5,
    )
    stabilizer.update([mapping("counted", 0.1)])
    second, _ = stabilizer.update([mapping("ignored", -0.1)])
    third, _ = stabilizer.update([mapping("ignored", -0.1)])
    assert second[0]["status"] == "counted"
    assert third[0]["status"] == "ignored"

    calculator = GridCalculator(
        grid_size=1.0,
        area_width=10.0,
        area_height=10.0,
    )
    identity = [[0, 0], [10, 0], [0, 10], [10, 10]]
    calculator.set_homography(identity, identity)
    detections = [{
        "x1": 1.0,
        "y1": 0.0,
        "x2": 2.0,
        "y2": 1.5,
        "conf": 0.9,
    }]
    direct = calculator.calculate(detections)
    split = calculator.calculate_from_mappings(
        calculator.map_detections(detections)
    )
    assert np.array_equal(direct["count"], split["count"])

    settings = AppSettings.load("config/app_config_7_85x10.json")
    assert settings.boundary_history_size == 3
    assert settings.boundary_majority_count == 2
    assert settings.dedup_iou_threshold == 0.65
    print("postprocessing smoke checks passed")


if __name__ == "__main__":
    main()
