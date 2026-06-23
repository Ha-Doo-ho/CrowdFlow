# grid_calculator.py (하두호 담당)
import numpy as np
import cv2
from datetime import datetime


class GridCalculator:
    EPSILON = 1e-6

    def __init__(
        self,
        grid_size=2.0,
        area_width=10.0,
        area_height=10.0,
        conf_threshold=0.4,
        low_confidence_min_level=3,
    ):
        self.grid_size = grid_size
        self.area_width = area_width
        self.area_height = area_height
        self.conf_threshold = conf_threshold
        self.low_confidence_min_level = low_confidence_min_level
        self.H = None  # Homography 매트릭스 (캘리브레이션 후 설정)

        self._validate_configuration()
        self.cols = round(area_width / grid_size)
        self.rows = round(area_height / grid_size)

    def _validate_configuration(self):
        if self.grid_size <= 0 or self.area_width <= 0 or self.area_height <= 0:
            raise ValueError("grid_size와 영역 크기는 0보다 커야 합니다.")
        if not 0 <= self.conf_threshold <= 1:
            raise ValueError("conf_threshold는 0~1 범위여야 합니다.")
        if self.low_confidence_min_level is not None:
            if not 1 <= self.low_confidence_min_level <= 5:
                raise ValueError("low_confidence_min_level은 1~5 또는 None이어야 합니다.")

        width_cells = self.area_width / self.grid_size
        height_cells = self.area_height / self.grid_size
        if not np.isclose(width_cells, round(width_cells)):
            raise ValueError("area_width는 grid_size로 정확히 나누어져야 합니다.")
        if not np.isclose(height_cells, round(height_cells)):
            raise ValueError("area_height는 grid_size로 정확히 나누어져야 합니다.")

    def set_homography(self, src_pts, dst_pts):
        """4개의 픽셀 좌표와 실제 좌표로 투시 변환 행렬을 계산한다."""
        src_pts = np.asarray(src_pts, dtype=np.float32)
        dst_pts = np.asarray(dst_pts, dtype=np.float32)
        if src_pts.shape != (4, 2) or dst_pts.shape != (4, 2):
            raise ValueError("src_pts와 dst_pts는 각각 4개의 2차원 좌표여야 합니다.")

        self.H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        if not np.all(np.isfinite(self.H)) or abs(np.linalg.det(self.H)) < self.EPSILON:
            raise ValueError("투시 변환 계산 실패: 중복되거나 일직선인 좌표가 있는지 확인하세요.")

    def _pixel_to_real(self, cx, cy):
        """픽셀 좌표 → 실제 좌표(m) 변환"""
        if self.H is None:
            raise RuntimeError("Homography가 설정되지 않았습니다.")
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H)
        return transformed[0][0][0], transformed[0][0][1] #x, y임.

    def _real_to_grid_index(self, real_x, real_y):
        """실제 좌표(m)를 격자 인덱스로 변환. 영역 밖이면 None을 반환한다."""
        in_x = -self.EPSILON <= real_x <= self.area_width + self.EPSILON
        in_y = -self.EPSILON <= real_y <= self.area_height + self.EPSILON
        if not (in_x and in_y):
            return None

        clipped_x = min(max(real_x, 0.0), self.area_width - self.EPSILON)
        clipped_y = min(max(real_y, 0.0), self.area_height - self.EPSILON)
        col = int(clipped_x // self.grid_size)
        row = int(clipped_y // self.grid_size)
        return row, col

    def calculate(self, detections):
        """
        입력: list[dict] — [{"x1","y1","x2","y2","conf"}, ...]
        출력: dict — {"grid": np.array, "level": np.array,
                      "max_density": float, "timestamp": str,
                      "alerts": list}
        """

        # 격자별 데이터 저장
        count_grid = np.zeros((self.rows, self.cols), dtype=int)
        conf_sum = np.zeros((self.rows, self.cols), dtype=float)
        conf_count = np.zeros((self.rows, self.cols), dtype=int)
        ignored_count = 0

        for det in detections:
            # bbox 하단 중심 = 발 위치 (가장 정확한 지면 접점)
            cx = (det["x1"] + det["x2"]) / 2
            cy = det["y2"]  # 하단 y좌표
            conf = det["conf"]

            # 픽셀 → 실제 좌표 변환
            real_x, real_y = self._pixel_to_real(cx, cy)

            # 격자 인덱스 계산
            grid_index = self._real_to_grid_index(real_x, real_y)

            # 범위 체크
            if grid_index is None:
                ignored_count += 1
                continue

            row, col = grid_index
            count_grid[row, col] += 1
            conf_sum[row, col] += conf
            conf_count[row, col] += 1

        # 밀집도 계산: 인원수 / 셀 면적
        cell_area = self.grid_size ** 2
        density_grid = count_grid.astype(float) / cell_area

        # Level 판정
        level_grid = np.zeros_like(count_grid)
        level_grid[density_grid < 2.0] = 1  # 안전
        level_grid[(density_grid >= 2.0) & (density_grid < 4.0)] = 2  # 주의
        level_grid[(density_grid >= 4.0) & (density_grid < 6.0)] = 3  # 경고
        level_grid[(density_grid >= 6.0) & (density_grid < 8.0)] = 4  # 위험
        level_grid[density_grid >= 8.0] = 5  # 긴급

        alerts = []
        with np.errstate(divide='ignore', invalid='ignore'):
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
                            f"[{r},{c}] 탐지 신뢰도 저하 "
                            f"({avg_conf[r, c]:.2f}) — 가림·거리·흔들림 확인 필요"
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
            "grid_size": self.grid_size,
            "area_width": self.area_width,
            "area_height": self.area_height,
            "timestamp": datetime.now().isoformat(),
            "alerts": alerts
        }
