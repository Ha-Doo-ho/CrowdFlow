import pytest

from core.grid_calculator import GridCalculator


IDENTITY_POINTS = [
    [0, 0],
    [10, 0],
    [0, 10],
    [10, 10],
]


def make_detection(x, y, confidence=0.9):
    return {
        "x1": x,
        "y1": y - 1,
        "x2": x,
        "y2": y,
        "conf": confidence,
    }


def test_requires_homography_before_calculation():
    calculator = GridCalculator()

    with pytest.raises(RuntimeError):
        calculator.calculate([make_detection(1, 1)])


def test_exact_bottom_right_boundary_is_in_last_cell():
    calculator = GridCalculator()
    calculator.set_homography(IDENTITY_POINTS, IDENTITY_POINTS)

    result = calculator.calculate([make_detection(10, 10)])

    assert result["count"][4, 4] == 1
    assert result["ignored_count"] == 0


def test_outside_detection_is_counted_as_ignored():
    calculator = GridCalculator(boundary_margin_m=0.0)
    calculator.set_homography(IDENTITY_POINTS, IDENTITY_POINTS)

    result = calculator.calculate([make_detection(11, 5)])

    assert result["count"].sum() == 0
    assert result["ignored_count"] == 1


def test_small_boundary_margin_keeps_near_edge_detection():
    calculator = GridCalculator(boundary_margin_m=0.1)
    calculator.set_homography(IDENTITY_POINTS, IDENTITY_POINTS)

    result = calculator.calculate([make_detection(10.05, 5)])

    assert result["count"].sum() == 1
    assert result["ignored_count"] == 0


def test_uses_alternate_foot_point_when_bbox_bottom_is_just_outside():
    calculator = GridCalculator(
        grid_size=1.0,
        area_width=10.0,
        area_height=10.0,
        boundary_margin_m=0.0,
    )
    calculator.set_homography(IDENTITY_POINTS, IDENTITY_POINTS)

    detection = {
        "x1": 4.5,
        "y1": 8.0,
        "x2": 5.5,
        "y2": 10.05,
        "conf": 0.9,
    }
    result = calculator.calculate([detection])

    assert result["count"].sum() == 1
    assert result["ignored_count"] == 0


def test_density_level_uses_cell_area():
    calculator = GridCalculator()
    calculator.set_homography(IDENTITY_POINTS, IDENTITY_POINTS)
    detections = [make_detection(1, 1) for _ in range(8)]

    result = calculator.calculate(detections)

    assert result["grid"][0, 0] == 2.0
    assert result["level"][0, 0] == 2


def test_low_confidence_creates_observation_warning():
    calculator = GridCalculator(conf_threshold=0.4, low_confidence_min_level=3)
    calculator.set_homography(IDENTITY_POINTS, IDENTITY_POINTS)

    result = calculator.calculate([make_detection(1, 1, confidence=0.2)])

    assert result["alerts"][0]["type"] == "low_confidence"
    assert result["level"][0, 0] == 3


def test_partial_last_cell_is_allowed_and_uses_actual_area():
    src_points = [[0, 0], [1.5, 0], [0, 1], [1.5, 1]]
    calculator = GridCalculator(
        grid_size=1.0,
        area_width=1.5,
        area_height=1.0,
        boundary_margin_m=0.0,
    )
    calculator.set_homography(src_points, src_points)

    result = calculator.calculate([make_detection(1.25, 0.5)])

    assert result["count"].shape == (1, 2)
    assert result["cell_widths"] == [1.0, 0.5]
    assert result["count"][0, 1] == 1
    assert result["grid"][0, 1] == pytest.approx(2.0)
