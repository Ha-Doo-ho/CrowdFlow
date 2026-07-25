import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np

from core.predictive_risk import PredictiveRiskTracker
from core.result_publisher import save_latest_result, to_jsonable
from core.risk_clusterer import density_to_level, find_risk_clusters


SCENARIOS = {
    "safe": "Sparse safe state with no risk cells.",
    "warning": "Moderate crowding below the risk-cluster threshold.",
    "risk_cluster": "Adjacent risk cells that should be grouped into one cluster.",
    "emergency": "High-density level 5 cells for emergency visualization.",
    "growth": "Current density is not dangerous yet, but 10-second prediction becomes risky.",
}


def build_edges(area_length, grid_size):
    edges = [0.0]
    while edges[-1] + grid_size < area_length:
        edges.append(round(edges[-1] + grid_size, 6))
    if edges[-1] != area_length:
        edges.append(float(area_length))
    return edges


def cell_areas(x_edges, y_edges):
    return np.outer(np.diff(y_edges), np.diff(x_edges))


def build_count_grid(name, rows, cols):
    count = np.zeros((rows, cols), dtype=int)

    if name == "safe":
        count[2, 2] = 1
        count[7, 5] = 1
    elif name == "warning":
        count[3, 3] = 2
        count[3, 4] = 3
        count[4, 3] = 2
        count[6, 2] = 3
    elif name == "risk_cluster":
        count[3, 3] = 5
        count[3, 4] = 6
        count[4, 3] = 4
        count[4, 4] = 7
        count[7, 1] = 2
    elif name == "emergency":
        count[2, 4] = 8
        count[2, 5] = 9
        count[3, 4] = 7
        count[3, 5] = 10
        count[6, 2] = 5
    elif name == "growth":
        count[4, 4] = 3
        count[4, 5] = 3
        count[5, 4] = 2
        count[7, 2] = 1
    else:
        raise ValueError(f"Unknown scenario: {name}")

    return count


def build_result(name, area_width, area_height, grid_size):
    x_edges = build_edges(area_width, grid_size)
    y_edges = build_edges(area_height, grid_size)
    rows = len(y_edges) - 1
    cols = len(x_edges) - 1
    areas = cell_areas(x_edges, y_edges)
    count = build_count_grid(name, rows, cols)
    density = count / areas
    level = np.vectorize(density_to_level)(density).astype(int)
    avg_conf = np.where(count > 0, 0.88, 1.0)

    result = {
        "grid": density,
        "level": level,
        "count": count,
        "avg_conf": avg_conf,
        "max_density": float(density.max()),
        "max_level": int(level.max()),
        "ignored_count": 0,
        "mapped_detections": [],
        "grid_size": float(grid_size),
        "area_width": float(area_width),
        "area_height": float(area_height),
        "x_edges": x_edges,
        "y_edges": y_edges,
        "cell_widths": np.diff(x_edges),
        "cell_heights": np.diff(y_edges),
        "cell_areas": areas,
        "boundary_margin_m": 0.15,
        "timestamp": datetime.now().isoformat(),
        "alerts": [],
        "metadata": {
            "frame_number": 0,
            "source": f"JSON scenario: {name}",
            "model_name": "scenario_json",
            "model_type": "scenario",
            "imgsz": 0,
            "detection_confidence": 1.0,
            "person_classes": [0, 1],
            "inference_ms": 0.0,
            "calibration_mode": "scenario_json",
        },
    }
    result["risk_clusters"] = find_risk_clusters(result)
    result["predicted_risk"] = build_predicted_risk(name, result)
    return result


def build_predicted_risk(name, result):
    tracker = PredictiveRiskTracker(
        horizon_seconds=10.0,
        history_seconds=15.0,
        min_history_seconds=3.0,
        cumulative_threshold=30.0,
    )

    current = np.asarray(result["grid"], dtype=float)
    if name == "growth":
        previous = np.maximum(current - 2.0, 0.0)
    else:
        previous = current.copy()

    previous_result = deepcopy(result)
    previous_result["grid"] = previous
    previous_result["count"] = np.rint(previous * np.asarray(result["cell_areas"])).astype(int)

    tracker.update(previous_result, now_seconds=0.0)
    predicted = tracker.update(result, now_seconds=5.0)
    return predicted


def write_result(result, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_latest_result(result, str(output_path))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate JSON-based CrowdFlow dashboard risk scenarios."
    )
    parser.add_argument(
        "--scenario",
        choices=[*SCENARIOS.keys(), "all"],
        default="risk_cluster",
    )
    parser.add_argument("--area-width", type=float, default=7.85)
    parser.add_argument("--area-height", type=float, default=10.0)
    parser.add_argument("--grid-size", type=float, default=1.0)
    parser.add_argument("--out-dir", default="data/scenario_results")
    parser.add_argument("--latest", default="data/latest_result.json")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Also write the selected scenario to data/latest_result.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    scenario_names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    out_dir = Path(args.out_dir)

    for name in scenario_names:
        result = build_result(name, args.area_width, args.area_height, args.grid_size)
        scenario_path = out_dir / f"latest_result_{name}.json"
        write_result(result, scenario_path)
        print(f"[Scenario] saved: {scenario_path}")
        print(
            f"           max_density={result['max_density']:.2f}, "
            f"max_level={result['max_level']}, "
            f"risk_clusters={len(result['risk_clusters'])}"
        )

        if args.publish and len(scenario_names) == 1:
            write_result(result, Path(args.latest))
            print(f"[Scenario] published to: {args.latest}")

    if args.publish and len(scenario_names) != 1:
        print("[Scenario] --publish is skipped when --scenario all is used.")


if __name__ == "__main__":
    main()
