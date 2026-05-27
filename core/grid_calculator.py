# grid_calculator.py (하두호 담당)
import numpy as np
import cv2
from datetime import datetime

class GridCalculator:
    EPSILON = 1e-6

    def __init__(self, grid_size=2.0, area_width=10.0, area_height=10.0, #단위: (m)
                 conf_threshold=0.4):
        self.grid_size = grid_size
        self.area_width = area_width
        self.area_height = area_height
        self.conf_threshold = conf_threshold
        self.H = None  # Homography 매트릭스 (캘리브레이션 후 설정)

        # 격자 크기 계산
        self.cols = int(area_width / grid_size)  # 가로 셀 수
        self.rows = int(area_height / grid_size)  # 세로 셀 수

    def set_homography(self, src_pts, dst_pts):
        """캘리브레이션: 픽셀 좌표 4점 + 실제 좌표 4점으로 H 산출"""
        self.H, _ = cv2.findHomography(src_pts, dst_pts)
        if self.H is None:
            raise ValueError("Homography 계산 실패: src_pts와 dst_pts를 확인하세요.")

    def _pixel_to_real(self, cx, cy):
        """픽셀 좌표 → 실제 좌표(m) 변환"""
        if self.H is None:
            return cx, cy  # H 미설정 시 그대로 반환 (테스트용)
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H)
        return transformed[0][0][0], transformed[0][0][1]

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
    #예를 들어 실제 좌표가 정확히 x=10.0, y=10.0이면 5x5격자에서 인덱스가 5가 된다. 그런데 실제 유효 인덱스는 0~4라서,
    #경계에 있는 사람이 계산에서 사라질 수 있다.

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

        # ★ 핵심 창의성: Confidence 저하 기반 고밀집 추정
        alerts = []
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_conf = np.where(conf_count > 0, conf_sum / conf_count, 1.0)

        for r in range(self.rows):
            for c in range(self.cols):
                if conf_count[r, c] > 0 and avg_conf[r, c] < self.conf_threshold:
                    alerts.append({
                        "row": r, "col": c,
                        "avg_conf": float(avg_conf[r, c]),
                        "count": int(count_grid[r, c]),
                        "message": f"[{r},{c}] 고밀집 추정 — AI 신뢰도 급락 ({avg_conf[r, c]:.2f})"
                    })
                    # 탐지 수가 적더라도 Level 상향
                    if level_grid[r, c] < 3:
                        level_grid[r, c] = 3  # 최소 경고 Level

        return {
            "grid": density_grid,
            "level": level_grid,
            "count": count_grid,
            "avg_conf": avg_conf,
            "max_density": float(density_grid.max()),
            "ignored_count": ignored_count,
            "timestamp": datetime.now().isoformat(),
            "alerts": alerts
        }
