import numpy as np

from core.grid_calculator import GridCalculator


IDENTITY_POINTS = [
    [0, 0],
    [10, 0],
    [0, 10],
    [10, 10],
]


def make_detection(x, y):
    return {
        "x1": x - 0.5,
        "y1": y - 2.0,
        "x2": x + 0.5,
        "y2": y,
        "conf": 0.9,
        "class_id": 0,
        "class_name": "pedestrian",
    }


def test_split_mapping_pipeline_matches_legacy_calculate():
    calculator = GridCalculator(
        grid_size=1.0,
        area_width=10.0,
        area_height=10.0,
    )
    calculator.set_homography(IDENTITY_POINTS, IDENTITY_POINTS)
    detections = [
        make_detection(1.5, 1.5),
        make_detection(4.5, 7.5),
    ]

    direct = calculator.calculate(detections)
    mappings = calculator.map_detections(detections)
    split = calculator.calculate_from_mappings(mappings)

    assert np.array_equal(split["count"], direct["count"])
    assert np.array_equal(split["grid"], direct["grid"])
    assert split["ignored_count"] == direct["ignored_count"]
