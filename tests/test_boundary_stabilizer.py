from core.boundary_stabilizer import BoundaryStateStabilizer


def make_mapping(status, foot_x=0.1, foot_y=5.0):
    counted = status == "counted"
    return {
        "status": status,
        "bbox": {
            "x1": 100,
            "y1": 100,
            "x2": 160,
            "y2": 260,
        },
        "conf": 0.9,
        "class_id": 0,
        "class_name": "pedestrian",
        "foot_pixel": [130.0, 260.0],
        "foot_m": [foot_x, foot_y],
        "row": 5 if counted else None,
        "col": 0 if counted else None,
    }


def make_stabilizer():
    return BoundaryStateStabilizer(
        area_width=10.0,
        area_height=10.0,
        history_size=3,
        majority_count=2,
        boundary_band_m=0.5,
        max_missing_updates=2,
    )


def test_single_ignored_flicker_does_not_change_counted_status():
    stabilizer = make_stabilizer()

    first, _ = stabilizer.update([make_mapping("counted")])
    second, stats = stabilizer.update([make_mapping("ignored", foot_x=-0.1)])

    assert first[0]["status"] == "counted"
    assert second[0]["status"] == "counted"
    assert second[0]["row"] == 5
    assert second[0]["col"] == 0
    assert second[0]["stabilized"] is True
    assert stats["ignored_to_counted"] == 1


def test_two_of_three_ignored_votes_switch_status():
    stabilizer = make_stabilizer()

    stabilizer.update([make_mapping("counted")])
    stabilizer.update([make_mapping("ignored", foot_x=-0.1)])
    third, stats = stabilizer.update(
        [make_mapping("ignored", foot_x=-0.1)]
    )

    assert third[0]["status"] == "ignored"
    assert third[0]["row"] is None
    assert third[0]["col"] is None
    assert third[0]["stability_votes"] == {
        "samples": 3,
        "counted": 1,
        "ignored": 2,
    }
    assert stats["boundary_status_overrides"] == 0


def test_two_of_three_counted_votes_switch_from_ignored():
    stabilizer = make_stabilizer()

    stabilizer.update([make_mapping("ignored", foot_x=-0.1)])
    second, _ = stabilizer.update([make_mapping("counted")])
    third, _ = stabilizer.update([make_mapping("counted")])

    assert second[0]["status"] == "ignored"
    assert second[0]["stabilized"] is True
    assert third[0]["status"] == "counted"
    assert third[0]["row"] == 5
    assert third[0]["col"] == 0


def test_far_from_boundary_is_not_tracked_or_modified():
    stabilizer = make_stabilizer()
    mapping = make_mapping("counted", foot_x=5.0, foot_y=5.0)

    result, stats = stabilizer.update([mapping])

    assert result[0]["status"] == "counted"
    assert "stabilized" not in result[0]
    assert stats["boundary_candidates"] == 0
