import pytest

from core.detection_postprocessor import (
    box_iou,
    suppress_cross_class_duplicates,
)


def make_detection(
    class_id,
    confidence,
    box=(10, 10, 50, 90),
):
    return {
        "x1": box[0],
        "y1": box[1],
        "x2": box[2],
        "y2": box[3],
        "conf": confidence,
        "class_id": class_id,
        "class_name": "pedestrian" if class_id == 0 else "people",
    }


def test_iou_of_identical_boxes_is_one():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_removes_lower_confidence_box_from_different_class():
    detections = [
        make_detection(0, 0.71),
        make_detection(1, 0.88),
    ]

    filtered, stats = suppress_cross_class_duplicates(
        detections,
        iou_threshold=0.65,
    )

    assert len(filtered) == 1
    assert filtered[0]["class_id"] == 1
    assert stats["raw_detection_count"] == 2
    assert stats["duplicate_boxes_removed"] == 1


def test_keeps_overlapping_boxes_from_same_class():
    detections = [
        make_detection(0, 0.90),
        make_detection(0, 0.80),
    ]

    filtered, stats = suppress_cross_class_duplicates(detections)

    assert filtered == detections
    assert stats["duplicate_boxes_removed"] == 0


def test_keeps_different_class_boxes_below_iou_threshold():
    detections = [
        make_detection(0, 0.90, box=(0, 0, 20, 40)),
        make_detection(1, 0.80, box=(15, 0, 35, 40)),
    ]

    filtered, stats = suppress_cross_class_duplicates(
        detections,
        iou_threshold=0.65,
    )

    assert filtered == detections
    assert stats["duplicate_boxes_removed"] == 0


def test_rejects_invalid_iou_threshold():
    with pytest.raises(ValueError):
        suppress_cross_class_duplicates([], iou_threshold=1.1)
