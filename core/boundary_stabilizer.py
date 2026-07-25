from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
import math

from core.detection_postprocessor import box_iou


@dataclass
class _BoundaryTrack:
    track_id: int
    bbox: dict
    confirmed_status: str
    last_seen_update: int
    history: deque = field(default_factory=deque)
    last_counted_cell: tuple[int, int] | None = None


class BoundaryStateStabilizer:
    """Stabilize counted/ignored decisions only near the measured boundary."""

    def __init__(
        self,
        area_width,
        area_height,
        history_size=3,
        majority_count=2,
        boundary_band_m=0.5,
        max_missing_updates=2,
        match_iou_threshold=0.15,
        match_center_distance=1.5,
    ):
        if area_width <= 0 or area_height <= 0:
            raise ValueError("area size must be greater than 0.")
        if history_size < 1:
            raise ValueError("history_size must be at least 1.")
        if not 1 <= majority_count <= history_size:
            raise ValueError("majority_count must be within history_size.")
        if boundary_band_m < 0:
            raise ValueError("boundary_band_m must be non-negative.")
        if max_missing_updates < 0:
            raise ValueError("max_missing_updates must be non-negative.")

        self.area_width = float(area_width)
        self.area_height = float(area_height)
        self.history_size = int(history_size)
        self.majority_count = int(majority_count)
        self.boundary_band_m = float(boundary_band_m)
        self.max_missing_updates = int(max_missing_updates)
        self.match_iou_threshold = float(match_iou_threshold)
        self.match_center_distance = float(match_center_distance)

        self._tracks = {}
        self._next_track_id = 1
        self._update_index = 0

    def update(self, mapped_detections):
        self._update_index += 1
        stabilized = [deepcopy(item) for item in mapped_detections]
        candidate_indices = [
            index
            for index, item in enumerate(stabilized)
            if self._is_near_boundary(item)
        ]

        active_tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if self._update_index - track.last_seen_update
            <= self.max_missing_updates + 1
        }
        assignments = self._match_candidates(
            stabilized,
            candidate_indices,
            active_tracks,
        )

        override_count = 0
        counted_to_ignored = 0
        ignored_to_counted = 0

        for detection_index in candidate_indices:
            item = stabilized[detection_index]
            raw_status = item.get("status", "ignored")
            track_id = assignments.get(detection_index)
            if track_id is None:
                track = self._new_track(item, raw_status)
                track_id = track.track_id
            else:
                track = self._tracks[track_id]

            track.bbox = deepcopy(item.get("bbox", {}))
            track.last_seen_update = self._update_index
            track.history.append(raw_status)

            if raw_status == "counted":
                row = item.get("row")
                col = item.get("col")
                if row is not None and col is not None:
                    track.last_counted_cell = (int(row), int(col))

            if len(track.history) >= self.history_size:
                counted_votes = sum(
                    status == "counted" for status in track.history
                )
                track.confirmed_status = (
                    "counted"
                    if counted_votes >= self.majority_count
                    else "ignored"
                )

            item["raw_status"] = raw_status
            item["stability_track_id"] = track_id
            item["stability_votes"] = {
                "samples": len(track.history),
                "counted": sum(
                    status == "counted" for status in track.history
                ),
                "ignored": sum(
                    status == "ignored" for status in track.history
                ),
            }

            confirmed_status = track.confirmed_status
            if confirmed_status != raw_status:
                override_count += 1
                if raw_status == "counted":
                    counted_to_ignored += 1
                else:
                    ignored_to_counted += 1

            self._apply_confirmed_status(item, track)

        self._remove_stale_tracks()
        return stabilized, {
            "boundary_candidates": len(candidate_indices),
            "boundary_active_tracks": len(self._tracks),
            "boundary_status_overrides": override_count,
            "counted_to_ignored": counted_to_ignored,
            "ignored_to_counted": ignored_to_counted,
        }

    def reset(self):
        self._tracks.clear()
        self._next_track_id = 1
        self._update_index = 0

    def _new_track(self, item, raw_status):
        track_id = self._next_track_id
        self._next_track_id += 1
        history = deque(maxlen=self.history_size)
        track = _BoundaryTrack(
            track_id=track_id,
            bbox=deepcopy(item.get("bbox", {})),
            confirmed_status=raw_status,
            last_seen_update=self._update_index,
            history=history,
        )
        self._tracks[track_id] = track
        return track

    def _is_near_boundary(self, item):
        foot_m = item.get("foot_m")
        if not isinstance(foot_m, (list, tuple)) or len(foot_m) != 2:
            return False

        x = float(foot_m[0])
        y = float(foot_m[1])
        band = self.boundary_band_m
        if not (
            -band <= x <= self.area_width + band
            and -band <= y <= self.area_height + band
        ):
            return False

        nearest_edge = min(
            abs(x),
            abs(self.area_width - x),
            abs(y),
            abs(self.area_height - y),
        )
        return nearest_edge <= band

    def _match_candidates(self, mappings, candidate_indices, tracks):
        possible_matches = []
        for detection_index in candidate_indices:
            detection_bbox = mappings[detection_index].get("bbox", {})
            for track_id, track in tracks.items():
                iou = box_iou(detection_bbox, track.bbox)
                center_distance = self._normalized_center_distance(
                    detection_bbox,
                    track.bbox,
                )
                if (
                    iou < self.match_iou_threshold
                    and center_distance > self.match_center_distance
                ):
                    continue
                score = iou * 2.0 + max(
                    0.0,
                    1.0 - center_distance / self.match_center_distance,
                )
                possible_matches.append(
                    (score, detection_index, track_id)
                )

        possible_matches.sort(reverse=True)
        assignments = {}
        assigned_tracks = set()
        for _, detection_index, track_id in possible_matches:
            if detection_index in assignments or track_id in assigned_tracks:
                continue
            assignments[detection_index] = track_id
            assigned_tracks.add(track_id)
        return assignments

    def _normalized_center_distance(self, box_a, box_b):
        ax, ay, diagonal_a = self._box_center_and_diagonal(box_a)
        bx, by, diagonal_b = self._box_center_and_diagonal(box_b)
        scale = max(diagonal_a, diagonal_b, 1.0)
        return math.hypot(ax - bx, ay - by) / scale

    @staticmethod
    def _box_center_and_diagonal(box):
        x1 = float(box.get("x1", 0.0))
        y1 = float(box.get("y1", 0.0))
        x2 = float(box.get("x2", x1))
        y2 = float(box.get("y2", y1))
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            math.hypot(width, height),
        )

    @staticmethod
    def _apply_confirmed_status(item, track):
        item["status"] = track.confirmed_status
        item["stabilized"] = (
            item.get("raw_status") != track.confirmed_status
        )
        if track.confirmed_status == "counted":
            if item.get("row") is None or item.get("col") is None:
                if track.last_counted_cell is not None:
                    item["row"], item["col"] = track.last_counted_cell
        else:
            item["row"] = None
            item["col"] = None

    def _remove_stale_tracks(self):
        stale_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if self._update_index - track.last_seen_update
            > self.max_missing_updates
        ]
        for track_id in stale_ids:
            del self._tracks[track_id]
