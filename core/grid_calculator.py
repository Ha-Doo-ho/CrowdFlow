from datetime import datetime
import math

import cv2
import numpy as np


class GridCalculator:
    EPSILON = 1e-6
    FOOT_POINT_OFFSETS = (
        (0.50, 1.00),  # bottom center
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
            raise ValueError("boundary_margin_m must be greater than or equal to 0.")
        if self.low_confidence_min_level is not None:
            if not 1 <= self.low_confidence_min_level <= 5:
                raise ValueError("low_confidence_min_level must be 1~5 or None.")

    def _build_axis_edges(self, length):
        cell_count = max(1, math.ceil(length / self.grid_size))
        edges = np.arange(cell_count + 1, dtype=float) * self.grid_size
        edges[-1] = length
        return edges

    def set_homography(self, src_pts, dst_pts):
        """Set a perspective transform from image pixels to measured floor coordinates."""
        src_pts = np.asarray(src_pts, dtype=np.float32)
        dst_pts = np.asarray(dst_pts, dtype=np.float32)
        if src_pts.shape != (4, 2) or dst_pts.shape != (4, 2):
            raise ValueError("src_pts and dst_pts must each contain four 2D points.")

        self.H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        if not np.all(np.isfinite(self.H)) or abs(np.linalg.det(self.H)) < self.EPSILON:
            raise ValueError("Invalid homography: check duplicate or nearly collinear points.")
        self.src_points = src_pts.copy()
        self.dst_points = dst_pts.copy()

    def _pixel_to_real(self, cx, cy):
        if self.H is None:
            raise RuntimeError("Homography is not set.")
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H)
        return float(transformed[0][0][0]), float(transformed[0][0][1])

    def _real_to_grid_index(self, real_x, real_y):
        margin = self.boundary_margin_m
        in_x = -margin <= real_x <= self.area_width + margin
        in_y = -margin <= real_y <= self.area_height + margin
        if not (in_x and in_y):
            return None

        clipped_x = min(max(real_x, 0.0), self.area_width - self.EPSILON)
        clipped_y = min(max(real_y, 0.0), self.area_height - self.EPSILON)
        col = int(clipped_x // self.grid_size)
        row = int(clipped_y // self.grid_size)
        return row, col

    def _candidate_foot_points(self, det):
        x1 = float(det["x1"])
        y1 = float(det["y1"])
        x2 = float(det["x2"])
        y2 = float(det["y2"])
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return [((x1 + x2) / 2, y2)]

        return [
            (x1 + width * x_ratio, y1 + height * y_ratio)
            for x_ratio, y_ratio in self.FOOT_POINT_OFFSETS
        ]

    def _map_detection(self, det):
        first_candidate = None
        for pixel_x, pixel_y in self._candidate_foot_points(det):
            real_x, real_y = self._pixel_to_real(pixel_x, pixel_y)
            if first_candidate is None:
                first_candidate = (pixel_x, pixel_y, real_x, real_y)

            grid_index = self._real_to_grid_index(real_x, real_y)
            if grid_index is not None:
                row, col = grid_index
                return row, col, self._build_mapping(
                    det, "counted", pixel_x, pixel_y, real_x, real_y, row, col
                )

        pixel_x, pixel_y, real_x, real_y = first_candidate
        return None, None, self._build_mapping(
            det, "ignored", pixel_x, pixel_y, real_x, real_y, None, None
        )

    def _build_mapping(self, det, status, pixel_x, pixel_y, real_x, real_y, row, col):
        return {
            "status": status,
            "bbox": {
                "x1": int(det["x1"]),
                "y1": int(det["y1"]),
                "x2": int(det["x2"]),
                "y2": int(det["y2"]),
            },
            "conf": float(det.get("conf", 0.0)),
            "class_id": det.get("class_id"),
            "class_name": det.get("class_name"),
            "foot_pixel": [round(float(pixel_x), 2), round(float(pixel_y), 2)],
            "foot_m": [round(float(real_x), 3), round(float(real_y), 3)],
            "row": row,
            "col": col,
        }

    def calculate(self, detections):
        count_grid = np.zeros((self.rows, self.cols), dtype=int)
        conf_sum = np.zeros((self.rows, self.cols), dtype=float)
        conf_count = np.zeros((self.rows, self.cols), dtype=int)
        ignored_count = 0
        mapped_detections = []

        for det in detections:
            row, col, mapping = self._map_detection(det)
            mapped_detections.append(mapping)

            if row is None or col is None:
                ignored_count += 1
                continue

            conf = float(det["conf"])
            count_grid[row, col] += 1
            conf_sum[row, col] += conf
            conf_count[row, col] += 1

        density_grid = count_grid.astype(float) / self.cell_areas

        level_grid = np.zeros_like(count_grid)
        level_grid[density_grid < 2.0] = 1
        level_grid[(density_grid >= 2.0) & (density_grid < 4.0)] = 2
        level_grid[(density_grid >= 4.0) & (density_grid < 6.0)] = 3
        level_grid[(density_grid >= 6.0) & (density_grid < 8.0)] = 4
        level_grid[density_grid >= 8.0] = 5

        alerts = []
        with np.errstate(divide="ignore", invalid="ignore"):
            avg_conf = np.where(conf_count > 0, conf_sum / conf_count, 1.0)

        for r in range(self.rows):
            for c in range(self.cols):
                if conf_count[r, c] > 0 and avg_conf[r, c] < self.conf_threshold:
                    alerts.append({
                        "type": "low_confidence",
                        "row": r,
                        "col": c,
                        "avg_conf": float(avg_conf[r, c]),
                        "count": int(count_grid[r, c]),
                        "message": (
                            f"[{r},{c}] detection confidence is low "
                            f"({avg_conf[r, c]:.2f}); check occlusion, distance, or shake"
                        ),
                    })
                    if self.low_confidence_min_level is not None:
                        level_grid[r, c] = max(
                            level_grid[r, c],
                            self.low_confidence_min_level,
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
