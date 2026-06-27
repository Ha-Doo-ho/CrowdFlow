from collections import deque
import time

import numpy as np

from core.risk_clusterer import density_to_level, find_risk_clusters


class PredictiveRiskTracker:
    def __init__(
        self,
        horizon_seconds=10.0,
        history_seconds=15.0,
        min_history_seconds=3.0,
        cumulative_half_life_seconds=30.0,
        cumulative_increment=20.0,
        cumulative_threshold=30.0,
        risk_density_threshold=4.0,
        growth_epsilon=0.03,
    ):
        self.horizon_seconds = float(horizon_seconds)
        self.history_seconds = float(history_seconds)
        self.min_history_seconds = float(min_history_seconds)
        self.cumulative_half_life_seconds = float(cumulative_half_life_seconds)
        self.cumulative_increment = float(cumulative_increment)
        self.cumulative_threshold = float(cumulative_threshold)
        self.risk_density_threshold = float(risk_density_threshold)
        self.growth_epsilon = float(growth_epsilon)

        self._validate()
        self.history = deque()
        self.cumulative_score = None
        self.last_update_time = None

    def _validate(self):
        if self.horizon_seconds <= 0:
            raise ValueError("horizon_seconds는 0보다 커야 합니다.")
        if self.history_seconds <= 0:
            raise ValueError("history_seconds는 0보다 커야 합니다.")
        if self.min_history_seconds <= 0:
            raise ValueError("min_history_seconds는 0보다 커야 합니다.")
        if self.cumulative_half_life_seconds <= 0:
            raise ValueError("cumulative_half_life_seconds는 0보다 커야 합니다.")
        if self.cumulative_increment <= 0:
            raise ValueError("cumulative_increment는 0보다 커야 합니다.")
        if not 0 <= self.cumulative_threshold <= 100:
            raise ValueError("cumulative_threshold는 0~100 범위여야 합니다.")
        if self.risk_density_threshold <= 0:
            raise ValueError("risk_density_threshold는 0보다 커야 합니다.")
        if self.growth_epsilon < 0:
            raise ValueError("growth_epsilon은 0 이상이어야 합니다.")

    def update(self, grid_result, now_seconds=None):
        now_seconds = time.monotonic() if now_seconds is None else float(now_seconds)
        density = np.asarray(grid_result["grid"], dtype=float)
        count = np.asarray(grid_result["count"], dtype=int)
        grid_size = float(grid_result.get("grid_size", 2.0))
        area_width = float(grid_result.get("area_width", density.shape[1] * grid_size))
        area_height = float(grid_result.get("area_height", density.shape[0] * grid_size))

        self._ensure_shape(density.shape)
        self._update_cumulative_score(density, now_seconds)
        self._append_history(now_seconds, density)

        growth_grid, has_enough_history = self._calculate_growth(density, now_seconds)
        positive_growth_grid = np.maximum(growth_grid, 0.0)
        predicted_grid = density + positive_growth_grid * self.horizon_seconds
        predicted_grid = np.maximum(predicted_grid, density)
        predicted_level = np.vectorize(density_to_level)(predicted_grid).astype(int)

        if has_enough_history:
            predicted_count = np.rint(predicted_grid * (grid_size ** 2)).astype(int)
            clusters = self._future_clusters(
                predicted_grid,
                predicted_count,
                density,
                growth_grid,
                grid_size,
                area_width,
                area_height,
            )
            soon_risk_cells = self._soon_risk_cells(density, predicted_grid, growth_grid)
        else:
            clusters = []
            soon_risk_cells = []

        cumulative_cells = self._cumulative_cells()

        return {
            "enabled": True,
            "horizon_seconds": self.horizon_seconds,
            "history_seconds": self.history_seconds,
            "min_history_seconds": self.min_history_seconds,
            "has_enough_history": has_enough_history,
            "predicted_grid": predicted_grid,
            "predicted_level": predicted_level,
            "growth_grid": growth_grid,
            "cumulative_score": self.cumulative_score.copy(),
            "max_predicted_density": float(predicted_grid.max()),
            "max_growth_per_second": float(positive_growth_grid.max()),
            "max_cumulative_score": float(self.cumulative_score.max()),
            "soon_risk_cells": soon_risk_cells,
            "cumulative_cells": cumulative_cells,
            "clusters": clusters,
        }

    def _ensure_shape(self, shape):
        if self.cumulative_score is None or self.cumulative_score.shape != shape:
            self.cumulative_score = np.zeros(shape, dtype=float)
            self.history.clear()
            self.last_update_time = None

    def _update_cumulative_score(self, density, now_seconds):
        if self.last_update_time is None:
            elapsed = 0.0
        else:
            elapsed = max(0.0, now_seconds - self.last_update_time)

        decay = 0.5 ** (elapsed / self.cumulative_half_life_seconds)
        current_risk = density >= self.risk_density_threshold
        self.cumulative_score *= decay
        self.cumulative_score += current_risk.astype(float) * self.cumulative_increment
        self.cumulative_score = np.clip(self.cumulative_score, 0.0, 100.0)
        self.last_update_time = now_seconds

    def _append_history(self, now_seconds, density):
        self.history.append((now_seconds, density.copy()))
        while self.history and now_seconds - self.history[0][0] > self.history_seconds:
            self.history.popleft()

    def _calculate_growth(self, density, now_seconds):
        if len(self.history) < 2:
            return np.zeros_like(density, dtype=float), False

        oldest_time, oldest_density = self.history[0]
        elapsed = now_seconds - oldest_time
        if elapsed < self.min_history_seconds:
            return np.zeros_like(density, dtype=float), False

        growth_grid = (density - oldest_density) / elapsed
        growth_grid[np.abs(growth_grid) < self.growth_epsilon] = 0.0
        return growth_grid, True

    def _future_clusters(
        self,
        predicted_grid,
        predicted_count,
        current_density,
        growth_grid,
        grid_size,
        area_width,
        area_height,
    ):
        cluster_result = {
            "grid": predicted_grid,
            "count": predicted_count,
            "grid_size": grid_size,
            "area_width": area_width,
            "area_height": area_height,
        }
        clusters = find_risk_clusters(cluster_result, min_density=self.risk_density_threshold)
        filtered = []
        for cluster in clusters:
            cells = cluster.get("cells", [])
            has_future_signal = any(
                (
                    current_density[row, col] < self.risk_density_threshold
                    and predicted_grid[row, col] >= self.risk_density_threshold
                )
                or growth_grid[row, col] > self.growth_epsilon
                for row, col in cells
            )
            if has_future_signal:
                filtered.append(cluster)
        return filtered

    def _soon_risk_cells(self, current_density, predicted_grid, growth_grid):
        cells = []
        rows, cols = predicted_grid.shape
        for row in range(rows):
            for col in range(cols):
                current = float(current_density[row, col])
                predicted = float(predicted_grid[row, col])
                growth = float(growth_grid[row, col])
                if current >= self.risk_density_threshold:
                    continue
                if predicted < self.risk_density_threshold:
                    continue
                if growth <= self.growth_epsilon:
                    continue
                cells.append({
                    "row": int(row),
                    "col": int(col),
                    "current_density": current,
                    "predicted_density": predicted,
                    "growth_per_second": growth,
                    "current_level": int(density_to_level(current)),
                    "predicted_level": int(density_to_level(predicted)),
                })

        cells.sort(
            key=lambda item: (
                item["predicted_level"],
                item["predicted_density"],
                item["growth_per_second"],
            ),
            reverse=True,
        )
        return cells

    def _cumulative_cells(self):
        cells = []
        rows, cols = self.cumulative_score.shape
        for row in range(rows):
            for col in range(cols):
                score = float(self.cumulative_score[row, col])
                if score < self.cumulative_threshold:
                    continue
                cells.append({
                    "row": int(row),
                    "col": int(col),
                    "score": score,
                })

        cells.sort(key=lambda item: item["score"], reverse=True)
        return cells
