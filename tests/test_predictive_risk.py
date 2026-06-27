import numpy as np

from core.predictive_risk import PredictiveRiskTracker


def make_result(grid):
    grid = np.asarray(grid, dtype=float)
    return {
        "grid": grid,
        "count": np.rint(grid * 4).astype(int),
        "grid_size": 2.0,
        "area_width": grid.shape[1] * 2.0,
        "area_height": grid.shape[0] * 2.0,
    }


def test_prediction_waits_until_history_is_long_enough():
    tracker = PredictiveRiskTracker(
        horizon_seconds=10,
        history_seconds=15,
        min_history_seconds=3,
    )

    prediction = tracker.update(make_result([[1.0, 1.0]]), now_seconds=0)

    assert prediction["has_enough_history"] is False
    assert prediction["soon_risk_cells"] == []
    assert prediction["clusters"] == []


def test_increasing_density_predicts_future_risk_cell():
    tracker = PredictiveRiskTracker(
        horizon_seconds=10,
        history_seconds=15,
        min_history_seconds=3,
    )

    tracker.update(make_result([[1.0, 1.0]]), now_seconds=0)
    prediction = tracker.update(make_result([[3.0, 1.0]]), now_seconds=10)

    assert prediction["has_enough_history"] is True
    assert prediction["predicted_grid"][0, 0] == 5.0
    assert prediction["soon_risk_cells"][0]["row"] == 0
    assert prediction["soon_risk_cells"][0]["col"] == 0
    assert prediction["soon_risk_cells"][0]["predicted_level"] == 3
    assert len(prediction["clusters"]) == 1


def test_cumulative_risk_score_tracks_repeated_risk_cells():
    tracker = PredictiveRiskTracker(
        cumulative_half_life_seconds=1000,
        cumulative_threshold=30,
    )

    tracker.update(make_result([[4.0, 1.0]]), now_seconds=0)
    prediction = tracker.update(make_result([[4.0, 1.0]]), now_seconds=5)

    assert prediction["cumulative_score"][0, 0] >= 30
    assert prediction["cumulative_cells"][0]["row"] == 0
    assert prediction["cumulative_cells"][0]["col"] == 0
