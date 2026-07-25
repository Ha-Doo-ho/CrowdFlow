import numpy as np


def box_iou(box_a, box_b):
    """Return the intersection-over-union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = _normalized_box(box_a)
    bx1, by1, bx2, by2 = _normalized_box(box_b)

    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def suppress_cross_class_duplicates(detections, iou_threshold=0.65):
    """
    Remove near-identical human boxes emitted under different class IDs.

    Boxes with the same class are left to the model's own NMS. This avoids
    deleting two real people who overlap in a dense crowd.
    """
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in the range 0~1.")

    raw_count = len(detections)
    if raw_count < 2:
        return list(detections), _build_stats(raw_count, raw_count)

    boxes = np.asarray(
        [
            _normalized_box(
                (det["x1"], det["y1"], det["x2"], det["y2"])
            )
            for det in detections
        ],
        dtype=float,
    )
    scores = np.asarray(
        [float(det.get("conf", 0.0)) for det in detections],
        dtype=float,
    )
    class_keys = [_class_key(det) for det in detections]

    order = np.argsort(-scores, kind="stable")
    suppressed = np.zeros(raw_count, dtype=bool)
    kept_indices = []

    for order_position, current_index in enumerate(order):
        if suppressed[current_index]:
            continue

        kept_indices.append(int(current_index))
        remaining = order[order_position + 1:]
        if remaining.size == 0:
            continue

        different_class = np.asarray(
            [
                class_keys[int(candidate)] != class_keys[int(current_index)]
                for candidate in remaining
            ],
            dtype=bool,
        )
        candidates = remaining[different_class & ~suppressed[remaining]]
        if candidates.size == 0:
            continue

        current_box = boxes[current_index]
        candidate_boxes = boxes[candidates]

        intersection_x1 = np.maximum(current_box[0], candidate_boxes[:, 0])
        intersection_y1 = np.maximum(current_box[1], candidate_boxes[:, 1])
        intersection_x2 = np.minimum(current_box[2], candidate_boxes[:, 2])
        intersection_y2 = np.minimum(current_box[3], candidate_boxes[:, 3])
        intersection = (
            np.maximum(0.0, intersection_x2 - intersection_x1)
            * np.maximum(0.0, intersection_y2 - intersection_y1)
        )

        current_area = (
            max(0.0, current_box[2] - current_box[0])
            * max(0.0, current_box[3] - current_box[1])
        )
        candidate_areas = (
            np.maximum(0.0, candidate_boxes[:, 2] - candidate_boxes[:, 0])
            * np.maximum(0.0, candidate_boxes[:, 3] - candidate_boxes[:, 1])
        )
        union = current_area + candidate_areas - intersection
        ious = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        suppressed[candidates[ious >= iou_threshold]] = True

    kept_indices.sort()
    filtered = [detections[index] for index in kept_indices]
    return filtered, _build_stats(raw_count, len(filtered))


def _normalized_box(box):
    if isinstance(box, dict):
        x1, y1, x2, y2 = (
            box["x1"],
            box["y1"],
            box["x2"],
            box["y2"],
        )
    else:
        x1, y1, x2, y2 = box
    return (
        min(float(x1), float(x2)),
        min(float(y1), float(y2)),
        max(float(x1), float(x2)),
        max(float(y1), float(y2)),
    )


def _class_key(detection):
    class_id = detection.get("class_id")
    if class_id is not None:
        return ("id", int(class_id))
    return ("name", str(detection.get("class_name", "")).strip().lower())


def _build_stats(raw_count, kept_count):
    return {
        "raw_detection_count": int(raw_count),
        "deduplicated_detection_count": int(kept_count),
        "duplicate_boxes_removed": int(raw_count - kept_count),
    }
