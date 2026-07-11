import numpy as np


def density_to_level(density):
    if density < 2.0:
        return 1
    if density < 4.0:
        return 2
    if density < 6.0:
        return 3
    if density < 8.0:
        return 4
    return 5


def find_risk_clusters(grid_result, min_density=4.0):
    density = np.asarray(grid_result["grid"], dtype=float)
    count = np.asarray(grid_result["count"], dtype=int)
    grid_size = float(grid_result.get("grid_size", 2.0))
    area_width = float(grid_result.get("area_width", density.shape[1] * grid_size))
    area_height = float(grid_result.get("area_height", density.shape[0] * grid_size))
    x_edges = np.asarray(
        grid_result.get("x_edges", _uniform_edges(density.shape[1], grid_size, area_width)),
        dtype=float,
    )
    y_edges = np.asarray(
        grid_result.get("y_edges", _uniform_edges(density.shape[0], grid_size, area_height)),
        dtype=float,
    )
    cell_areas = np.asarray(
        grid_result.get("cell_areas", _cell_areas_from_edges(x_edges, y_edges)),
        dtype=float,
    )

    rows, cols = density.shape
    candidates = density >= min_density
    visited = np.zeros((rows, cols), dtype=bool)
    clusters = []

    for start_row in range(rows):
        for start_col in range(cols):
            if visited[start_row, start_col] or not candidates[start_row, start_col]:
                continue

            stack = [(start_row, start_col)]
            visited[start_row, start_col] = True
            cells = []

            while stack:
                row, col = stack.pop()
                cells.append((row, col))

                for next_row, next_col in (
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                ):
                    if not (0 <= next_row < rows and 0 <= next_col < cols):
                        continue
                    if visited[next_row, next_col] or not candidates[next_row, next_col]:
                        continue
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))

            clusters.append(_build_cluster(cells, density, count, x_edges, y_edges,
                                           cell_areas))

    clusters.sort(
        key=lambda item: (
            item["max_level"],
            item["max_density"],
            item["total_count"],
            item["area_m2"],
        ),
        reverse=True,
    )

    for index, cluster in enumerate(clusters, start=1):
        cluster["id"] = index

    return clusters


def _uniform_edges(cell_count, grid_size, area_length):
    edges = np.arange(cell_count + 1, dtype=float) * grid_size
    if len(edges) and area_length < edges[-1]:
        edges[-1] = area_length
    return edges


def _cell_areas_from_edges(x_edges, y_edges):
    return np.outer(np.diff(y_edges), np.diff(x_edges))


def _build_cluster(cells, density, count, x_edges, y_edges, cell_areas):
    rows = [row for row, _ in cells]
    cols = [col for _, col in cells]
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    max_density = max(float(density[row, col]) for row, col in cells)

    return {
        "id": 0,
        "cells": [[int(row), int(col)] for row, col in sorted(cells)],
        "bbox_m": {
            "x1": float(x_edges[min_col]),
            "y1": float(y_edges[min_row]),
            "x2": float(x_edges[max_col + 1]),
            "y2": float(y_edges[max_row + 1]),
        },
        "total_count": int(sum(int(count[row, col]) for row, col in cells)),
        "max_density": max_density,
        "max_level": int(density_to_level(max_density)),
        "area_m2": float(sum(float(cell_areas[row, col]) for row, col in cells)),
    }
