from datetime import datetime
import math

import cv2
import numpy as np


class GridCalculator:
    EPSILON = 1e-6
    FOOT_POINT_OFFSETS = (
        (0.50, 1.00),
        (0.40, 1.00),
        (0.60, 1.00),
        (0.50, 0.95),
        (0.35, 0.95),
        (0.65, 0.95),
    )

    def __init__(
        self,
        grid_size=2.0,
        area_width=10.0,
        area_height=10.0,
        conf_threshold=0.4,
        low_confidence_min_level=3,
        boundary_margin_m=0.05,
    ):
        self.grid_size = grid_size
        self.area_width = area_width
        self.area_height = area_height
        self.conf_threshold = conf_threshold
        self.low_confidence_min_level = low_confidence_min_level
        self.boundary_margin_m = boundary_margin_m
        self.H = None
        self.src_points = None
        self.dst_points = None

        self._validate_configuration()
        self.x_edges = self._build_axis_edges(self.area_width)
        self.y_edges = self._build_axis_edges(self.area_height)
        self.cell_widths = np.diff(self.x_edges)
        self.cell_heights = np.diff(self.y_edges)
        self.cell_areas = np.outer(self.cell_heights, self.cell_widths)
        self.cols = len(self.cell_widths)
        self.rows = len(self.cell_heights)

    def _validate_configuration(self):
        if self.grid_size <= 0 or self.area_width <= 0 or self.area_height <= 0:
            raise ValueError("grid_size and area size must be greater than 0.")
        if not 0 <= self.conf_threshold <= 1:
            raise ValueError("conf_threshold must be in the range 0~1.")
        if self.boundary_margin_m < 0:
            raise ValueError(
                "boundary_margin_m must be greater than or equal to 0."
            )
        if self.low_confidence_min_level is not None:
            if not 1 <= self.low_confidence_min_level <= 5:
                raise ValueError(
                    "low_confidence_min_level must be 1~5 or None."
                )

    def _build_axis_edges(self, length):
        cell_count = max(1, math.ceil(length / self.grid_size))
        edges = np.arange(cell_count + 1, dtype=float) * self.grid_size
        edges[-1] = length
        return edges

    def set_homography(self, src_pts, dst_pts):
        """Set a perspective transform from pixels to floor coordinates."""
        src_pts = np.asarray(src_pts, dtype=np.float32)
        dst_pts = np.asarray(dst_pts, dtype=np.float32)
        if src_pts.shape != (4, 2) or dst_pts.shape != (4, 2):
            raise ValueError(
                "src_pts and dst_pts must each contain four 2D points."
            )

        self.H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        if (
            not np.all(np.isfinite(self.H))
            or abs(np.linalg.det(self.H)) < self.EPSILON
        ):
            raise ValueError(
                "Invalid homography: check duplicate or collinear points."
            )
        self.src_points = src_pts.copy()
        self.dst_points = dst_pts.copy()

    def _pixel_to_real(self, cx, cy):
        if self.H is None:
            raise RuntimeError("Homography is not set.")
        point = np.array([[[cx, cy]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.H)
        return (
            float(transformed[0][0][0]),
            float(transformed[0][0][1]),
        )

    def _real_to_grid_index(self, real_x, real_y):
        margin = self.boundary_margin_m
        in_x = -margin <= real_x <= self.area_width + margin
        in_y = -margin <= real_y <= self.area_height + margin
        if not (in_x and in_y):
            return None

        clipped_x = min(
            max(real_x, 0.0),
            self.area_width - self.EPSILON,
        )
        clipped_y = min(
            max(real_y, 0.0),
            self.area_height - self.EPSILON,
        )
        col = int(clipped_x // self.grid_size)
        row = int(clipped_y // self.grid_size)
        return row, col

    def _candidate_foot_points(self, detection):
        x1 = float(detection["x1"])
        y1 = float(detection["y1"])
        x2 = float(detection["x2"])
        y2 = float(detection["y2"])
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return [((x1 + x2) / 2, y2)]

        return [
            (
                x1 + width * x_ratio,
                y1 + height * y_ratio,
            )
            for x_ratio, y_ratio in self.FOOT_POINT_OFFSETS
        ]

    def _map_detection(self, detection):
        first_candidate = None
        for pixel_x, pixel_y in self._candidate_foot_points(detection):
            real_x, real_y = self._pixel_to_real(pixel_x, pixel_y)
            if first_candidate is None:
                first_candidate = (
                    pixel_x,
                    pixel_y,
                    real_x,
                    real_y,
                )

            grid_index = self._real_to_grid_index(real_x, real_y)
            if grid_index is not None:
                row, col = grid_index
                return self._build_mapping(
                    detection,
                    "counted",
                    pixel_x,
                    pixel_y,
                    real_x,
                    real_y,
                    row,
                    col,
                )

        pixel_x, pixel_y, real_x, real_y = first_candidate
        return self._build_mapping(
            detection,
            "ignored",
            pixel_x,
            pixel_y,
            real_x,
            real_y,
            None,
            None,
        )

    def _build_mapping(
        self,
        detection,
        status,
        pixel_x,
        pixel_y,
        real_x,
        real_y,
        row,
        col,
    ):
        return {
            "status": status,
            "bbox": {
                "x1": int(detection["x1"]),
                "y1": int(detection["y1"]),
                "x2": int(detection["x2"]),
                "y2": int(detection["y2"]),
            },
            "conf": float(detection.get("conf", 0.0)),
            "class_id": detection.get("class_id"),
            "class_name": detection.get("class_name"),
            "foot_pixel": [
                round(float(pixel_x), 2),
                round(float(pixel_y), 2),
            ],
            "foot_m": [
                round(float(real_x), 3),
                round(float(real_y), 3),
            ],
            "row": row,
            "col": col,
        }

    def map_detections(self, detections):
        """Map detections without aggregating them into the density grid."""
        return [self._map_detection(detection) for detection in detections]

    def calculate_from_mappings(self, mapped_detections):
        """Aggregate mapped, optionally stabilized detections."""
        count_grid = np.zeros((self.rows, self.cols), dtype=int)
        conf_sum = np.zeros((self.rows, self.cols), dtype=float)
        conf_count = np.zeros((self.rows, self.cols), dtype=int)
        ignored_count = 0

        for mapping in mapped_detections:
            row = mapping.get("row")
            col = mapping.get("col")
            if (
                mapping.get("status") != "counted"
                or row is None
                or col is None
            ):
                ignored_count += 1
                continue

            row = int(row)
            col = int(col)
            confidence = float(mapping.get("conf", 0.0))
            count_grid[row, col] += 1
            conf_sum[row, col] += confidence
            conf_count[row, col] += 1

        density_grid = count_grid.astype(float) / self.cell_areas
        level_grid = self._density_levels(density_grid)

        with np.errstate(divide="ignore", invalid="ignore"):
            avg_conf = np.where(
                conf_count > 0,
                conf_sum / conf_count,
                1.0,
            )

        alerts = self._build_confidence_alerts(
            count_grid,
            conf_count,
            avg_conf,
            level_grid,
        )

        return {
            "grid": density_grid,
            "level": level_grid,
            "count": count_grid,
            "avg_conf": avg_conf,
            "max_density": float(density_grid.max()),
            "max_level": int(level_grid.max()),
            "ignored_count": ignored_count,
            "mapped_detections": mapped_detections,
            "grid_size": self.grid_size,
            "area_width": self.area_width,
            "area_height": self.area_height,
            "x_edges": self.x_edges.tolist(),
            "y_edges": self.y_edges.tolist(),
            "cell_widths": self.cell_widths.tolist(),
            "cell_heights": self.cell_heights.tolist(),
            "cell_areas": self.cell_areas,
            "boundary_margin_m": self.boundary_margin_m,
            "timestamp": datetime.now().isoformat(),
            "alerts": alerts,
        }

    def calculate(self, detections):
        mappings = self.map_detections(detections)
        return self.calculate_from_mappings(mappings)

    @staticmethod
    def _density_levels(density_grid):
        level_grid = np.zeros_like(density_grid, dtype=int)
        level_grid[density_grid < 2.0] = 1
        level_grid[
            (density_grid >= 2.0) & (density_grid < 4.0)
        ] = 2
        level_grid[
            (density_grid >= 4.0) & (density_grid < 6.0)
        ] = 3
        level_grid[
            (density_grid >= 6.0) & (density_grid < 8.0)
        ] = 4
        level_grid[density_grid >= 8.0] = 5
        return level_grid

    def _build_confidence_alerts(
        self,
        count_grid,
        conf_count,
        avg_conf,
        level_grid,
    ):
        alerts = []
        for row in range(self.rows):
            for col in range(self.cols):
                if (
                    conf_count[row, col] > 0
                    and avg_conf[row, col] < self.conf_threshold
                ):
                    alerts.append({
                        "type": "low_confidence",
                        "row": row,
                        "col": col,
                        "avg_conf": float(avg_conf[row, col]),
                        "count": int(count_grid[row, col]),
                        "message": (
                            f"[{row},{col}] detection confidence is low "
                            f"({avg_conf[row, col]:.2f}); check occlusion, "
                            "distance, or shake"
                        ),
                    })
                    if self.low_confidence_min_level is not None:
                        level_grid[row, col] = max(
                            level_grid[row, col],
                            self.low_confidence_min_level,
                        )
        return alerts
