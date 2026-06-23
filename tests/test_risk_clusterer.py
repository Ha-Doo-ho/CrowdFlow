import numpy as np

from core.risk_clusterer import find_risk_clusters


def make_result(grid, count=None, alerts=None):
    grid = np.asarray(grid, dtype=float)
    if count is None:
        count = np.zeros_like(grid, dtype=int)
    return {
        "grid": grid,
        "count": np.asarray(count, dtype=int),
        "grid_size": 2.0,
        "area_width": 10.0,
        "area_height": 10.0,
        "alerts": alerts or [],
    }


def test_no_risk_cells_returns_empty_clusters():
    result = make_result([[1.0, 2.0], [3.0, 0.0]])

    assert find_risk_clusters(result) == []


def test_single_risk_cell_creates_one_cluster():
    result = make_result(
        [[1.0, 4.0], [0.0, 2.0]],
        count=[[1, 16], [0, 2]],
    )

    clusters = find_risk_clusters(result)

    assert len(clusters) == 1
    assert clusters[0]["id"] == 1
    assert clusters[0]["cells"] == [[0, 1]]
    assert clusters[0]["bbox_m"] == {"x1": 2.0, "y1": 0.0, "x2": 4.0, "y2": 2.0}
    assert clusters[0]["total_count"] == 16
    assert clusters[0]["max_density"] == 4.0
    assert clusters[0]["max_level"] == 3


def test_four_direction_connected_cells_are_one_cluster():
    result = make_result(
        [[4.0, 4.5, 1.0], [1.0, 6.0, 0.0], [0.0, 0.0, 4.0]],
        count=[[16, 18, 0], [0, 24, 0], [0, 0, 16]],
    )

    clusters = find_risk_clusters(result)

    assert len(clusters) == 2
    assert clusters[0]["cells"] == [[0, 0], [0, 1], [1, 1]]
    assert clusters[0]["max_level"] == 4
    assert clusters[0]["total_count"] == 58
    assert clusters[1]["cells"] == [[2, 2]]


def test_diagonal_cells_are_separate_clusters():
    result = make_result(
        [[4.0, 0.0], [0.0, 4.0]],
        count=[[16, 0], [0, 16]],
    )

    clusters = find_risk_clusters(result)

    assert len(clusters) == 2
    assert clusters[0]["cells"] == [[0, 0]]
    assert clusters[1]["cells"] == [[1, 1]]


def test_low_confidence_alert_does_not_create_risk_cluster():
    result = make_result(
        [[1.0, 2.0], [3.0, 0.5]],
        count=[[4, 8], [12, 2]],
        alerts=[{"type": "low_confidence", "row": 0, "col": 0}],
    )

    assert find_risk_clusters(result) == []
