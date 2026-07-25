import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from core.boundary_stabilizer import BoundaryStateStabilizer
from core.detector import CrowdDetector
from core.grid_calculator import GridCalculator
from core.predictive_risk import PredictiveRiskTracker
from core.result_publisher import save_latest_result
from core.risk_clusterer import find_risk_clusters
from core.settings import AppSettings
from frontend.data_logger import DataLogger
from hardware.drone_controller import DroneCamera


ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT, "config", "app_config.json")
DATA_DIR = os.path.join(ROOT, "data")
LATEST_RESULT_PATH = os.path.join(DATA_DIR, "latest_result.json")
DB_PATH = os.path.join(DATA_DIR, "crowdflow.db")
WINDOW_NAME = "CrowdFlow AI Detection"


def parse_args():
    parser = argparse.ArgumentParser(description="Run CrowdFlow analysis.")
    parser.add_argument(
        "--config",
        default=CONFIG_PATH,
        help="Path to app config JSON. Defaults to config/app_config.json.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help=(
            "Optional video source override. Use this to test another video "
            "without editing JSON."
        ),
    )
    return parser.parse_args()


def publish_result(grid_result, logger):
    save_latest_result(grid_result, LATEST_RESULT_PATH)
    logger.log_frame(grid_result)


def resolve_path(path_value):
    path = Path(path_value)
    return str(path if path.is_absolute() else Path(ROOT) / path)


def resolve_source(source):
    if isinstance(source, int) or source == "tello":
        return source
    return resolve_path(source)


def configure_homography(calculator, frame, calibration_path):
    height, width = frame.shape[:2]

    if calibration_path:
        resolved_path = resolve_path(calibration_path)
        with open(resolved_path, "r", encoding="utf-8") as calibration_file:
            calibration = json.load(calibration_file)
        src_points = calibration.get("src_points", [])
        dst_points = calibration.get("dst_points", [])
        if len(src_points) != 4 or len(dst_points) != 4:
            raise ValueError(
                "Calibration file must contain exactly four src_points "
                f"and four dst_points: {resolved_path}"
            )
        calculator.set_homography(src_points, dst_points)
        print(f"[Calibration] measured coordinates: {resolved_path}")
        return "measured"

    src = np.array(
        [[0, 0], [width, 0], [0, height], [width, height]],
        dtype=np.float32,
    )
    dst = np.array(
        [
            [0, 0],
            [calculator.area_width, 0],
            [0, calculator.area_height],
            [calculator.area_width, calculator.area_height],
        ],
        dtype=np.float32,
    )
    calculator.set_homography(src, dst)
    print(
        "[Calibration] temporary full-frame mapping. "
        "Do not treat density as measured."
    )
    return "full_frame_fallback"


def analyze_frame(
    detector,
    calculator,
    frame,
    frame_number,
    source_name,
    calibration_mode,
    predictive_tracker=None,
    boundary_stabilizer=None,
):
    started_at = time.perf_counter()
    detections, raw_result = detector.detect(frame)
    inference_ms = (time.perf_counter() - started_at) * 1000

    raw_mappings = calculator.map_detections(detections)
    raw_counted = sum(
        mapping.get("status") == "counted"
        for mapping in raw_mappings
    )
    raw_ignored = len(raw_mappings) - raw_counted

    if boundary_stabilizer is not None:
        mappings, boundary_stats = boundary_stabilizer.update(raw_mappings)
    else:
        mappings = raw_mappings
        boundary_stats = {
            "boundary_candidates": 0,
            "boundary_active_tracks": 0,
            "boundary_status_overrides": 0,
            "counted_to_ignored": 0,
            "ignored_to_counted": 0,
        }

    grid_result = calculator.calculate_from_mappings(mappings)
    grid_result["risk_clusters"] = find_risk_clusters(grid_result)
    if predictive_tracker is not None:
        grid_result["predicted_risk"] = predictive_tracker.update(grid_result)
    else:
        grid_result["predicted_risk"] = {"enabled": False}

    detector_stats = dict(detector.last_postprocess_stats)
    processing_diagnostics = {
        **detector_stats,
        "raw_counted": int(raw_counted),
        "raw_ignored": int(raw_ignored),
        "stabilized_counted": int(grid_result["count"].sum()),
        "stabilized_ignored": int(grid_result["ignored_count"]),
        **boundary_stats,
    }
    grid_result["processing_diagnostics"] = processing_diagnostics

    grid_result["metadata"] = {
        "frame_number": frame_number,
        "source": source_name,
        "model_name": Path(detector.model_path).name,
        "model_type": detector.model_type,
        "imgsz": detector.imgsz,
        "detection_confidence": detector.conf,
        "person_classes": detector.person_classes,
        "inference_ms": round(inference_ms, 2),
        "calibration_mode": calibration_mode,
        "cross_class_dedup_enabled": (
            detector.cross_class_dedup_enabled
        ),
        "dedup_iou_threshold": detector.dedup_iou_threshold,
        "boundary_stabilization_enabled": (
            boundary_stabilizer is not None
        ),
    }
    return grid_result, raw_result


def draw_calibration_boundary(annotated, src_points):
    if src_points is None:
        return

    points = np.asarray(src_points, dtype=np.int32)
    if points.shape != (4, 2):
        return

    polygon = np.array(
        [points[0], points[1], points[3], points[2]],
        dtype=np.int32,
    )
    cv2.polylines(
        annotated,
        [polygon],
        isClosed=True,
        color=(0, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    labels = ("LT", "RT", "LB", "RB")
    for label, (x, y) in zip(labels, points):
        cv2.circle(
            annotated,
            (int(x), int(y)),
            5,
            (0, 255, 255),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def draw_detection_debug(
    frame,
    grid_result,
    calibration_src_points=None,
):
    annotated = frame.copy()
    draw_calibration_boundary(annotated, calibration_src_points)

    metadata = grid_result.get("metadata", {}) if grid_result else {}
    frame_number = metadata.get("frame_number")
    if frame_number is not None:
        cv2.putText(
            annotated,
            f"last analysis frame: {frame_number}",
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    diagnostics = (
        grid_result.get("processing_diagnostics", {})
        if grid_result
        else {}
    )
    if diagnostics:
        cv2.putText(
            annotated,
            (
                "duplicates removed: "
                f"{diagnostics.get('duplicate_boxes_removed', 0)} | "
                "boundary overrides: "
                f"{diagnostics.get('boundary_status_overrides', 0)}"
            ),
            (15, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    mapped_detections = grid_result.get("mapped_detections", [])
    if not mapped_detections:
        return annotated

    for item in mapped_detections:
        bbox = item.get("bbox", {})
        x1 = int(bbox.get("x1", 0))
        y1 = int(bbox.get("y1", 0))
        x2 = int(bbox.get("x2", 0))
        y2 = int(bbox.get("y2", 0))

        counted = item.get("status") == "counted"
        color = (0, 220, 0) if counted else (150, 150, 150)
        label = "OUT"
        if counted:
            label = f"IN [{item.get('row')},{item.get('col')}]"
        else:
            foot_m = item.get("foot_m")
            if isinstance(foot_m, list) and len(foot_m) == 2:
                label = (
                    f"OUT ({foot_m[0]:.1f},{foot_m[1]:.1f}m)"
                )
        if item.get("stabilized"):
            label += " [STABLE]"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        foot = item.get("foot_pixel", [])
        if len(foot) == 2:
            foot_x = int(round(float(foot[0])))
            foot_y = int(round(float(foot[1])))
            cv2.circle(annotated, (foot_x, foot_y), 4, color, -1)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def build_predictive_tracker(settings):
    if not settings.predictive_risk_enabled:
        return None
    return PredictiveRiskTracker(
        horizon_seconds=settings.prediction_horizon_seconds,
        history_seconds=settings.prediction_history_seconds,
        min_history_seconds=settings.prediction_min_history_seconds,
        cumulative_half_life_seconds=(
            settings.cumulative_risk_half_life_seconds
        ),
        cumulative_threshold=settings.cumulative_risk_threshold,
    )


def build_boundary_stabilizer(settings):
    if not settings.boundary_stabilization_enabled:
        return None
    return BoundaryStateStabilizer(
        area_width=settings.area_width,
        area_height=settings.area_height,
        history_size=settings.boundary_history_size,
        majority_count=settings.boundary_majority_count,
        boundary_band_m=settings.boundary_stability_band_m,
        max_missing_updates=settings.boundary_track_max_missing,
    )


def print_frame_summary(grid_result):
    metadata = grid_result["metadata"]
    diagnostics = grid_result.get("processing_diagnostics", {})
    total = int(grid_result["count"].sum())
    max_density = float(grid_result["max_density"])
    ignored = int(grid_result.get("ignored_count", 0))
    inference_ms = float(metadata["inference_ms"])
    frame_number = int(metadata["frame_number"])

    print(
        f"[Frame {frame_number:>5}] "
        f"counted: {total:>3} | "
        f"max density: {max_density:.2f}/m2 | "
        f"ignored: {ignored} | "
        "duplicates removed: "
        f"{diagnostics.get('duplicate_boxes_removed', 0)} | "
        "boundary overrides: "
        f"{diagnostics.get('boundary_status_overrides', 0)} | "
        f"inference: {inference_ms:.1f}ms",
        end="",
    )
    alerts = grid_result.get("alerts", [])
    if alerts:
        print(f" | warnings: {len(alerts)}")
        for alert in alerts:
            print(f"    -> {alert['message']}")
    else:
        print()


def main(config_path=CONFIG_PATH, source_override=None):
    settings = AppSettings.load(config_path)
    source_value = (
        source_override
        if source_override is not None
        else settings.source
    )
    source = resolve_source(source_value)
    model_path = resolve_path(settings.model_path)

    drone = None
    logger = None

    try:
        print(f"[Config] loaded: {config_path}")
        print(f"[Config] source: {source_value}")
        print(f"[Config] calibration_path: {settings.calibration_path}")
        print(
            "[Config] area/grid: "
            f"{settings.area_width}m x {settings.area_height}m, "
            f"grid {settings.grid_size}m"
        )
        print(
            "[Config] postprocess: "
            f"cross-class dedup={settings.cross_class_dedup_enabled} "
            f"(IoU {settings.dedup_iou_threshold}), "
            "boundary stabilization="
            f"{settings.boundary_stabilization_enabled} "
            f"({settings.boundary_majority_count}/"
            f"{settings.boundary_history_size})"
        )

        drone = DroneCamera(source)
        detector = CrowdDetector(
            model_path=model_path,
            imgsz=settings.imgsz,
            conf=settings.detection_confidence,
            person_classes=settings.person_classes,
            model_type=settings.model_type,
            cross_class_dedup_enabled=(
                settings.cross_class_dedup_enabled
            ),
            dedup_iou_threshold=settings.dedup_iou_threshold,
        )
        calculator = GridCalculator(
            grid_size=settings.grid_size,
            area_width=settings.area_width,
            area_height=settings.area_height,
            conf_threshold=settings.low_confidence_threshold,
            low_confidence_min_level=(
                settings.low_confidence_min_level
            ),
            boundary_margin_m=settings.boundary_margin_m,
        )
        predictive_tracker = build_predictive_tracker(settings)
        boundary_stabilizer = build_boundary_stabilizer(settings)
        logger = DataLogger(DB_PATH)

        first_frame = drone.get_frame()
        if first_frame is None:
            print(
                "[Video] Could not read the first frame. "
                "Check source path or stream."
            )
            return

        height, width = first_frame.shape[:2]
        print(f"[Video] resolution: {width} x {height}")
        calibration_mode = configure_homography(
            calculator,
            first_frame,
            settings.calibration_path,
        )

        print("=" * 50)
        print("CrowdFlow running. Press q in the video window to quit.")
        print("=" * 50)

        frame_count = 1
        grid_result, _ = analyze_frame(
            detector,
            calculator,
            first_frame,
            frame_count,
            str(source_value),
            calibration_mode,
            predictive_tracker,
            boundary_stabilizer,
        )
        publish_result(grid_result, logger)
        print_frame_summary(grid_result)
        cv2.imshow(
            WINDOW_NAME,
            draw_detection_debug(
                first_frame,
                grid_result,
                calculator.src_points,
            ),
        )

        while True:
            frame = drone.get_frame()
            if frame is None:
                break

            frame_count += 1
            if frame_count % settings.analyze_every == 0:
                grid_result, _ = analyze_frame(
                    detector,
                    calculator,
                    frame,
                    frame_count,
                    str(source_value),
                    calibration_mode,
                    predictive_tracker,
                    boundary_stabilizer,
                )
                publish_result(grid_result, logger)
                print_frame_summary(grid_result)
                cv2.imshow(
                    WINDOW_NAME,
                    draw_detection_debug(
                        frame,
                        grid_result,
                        calculator.src_points,
                    ),
                )
            else:
                cv2.imshow(
                    WINDOW_NAME,
                    draw_detection_debug(
                        frame,
                        grid_result,
                        calculator.src_points,
                    ),
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if drone is not None:
            drone.release()
        if logger is not None:
            logger.close()
        cv2.destroyAllWindows()
        print("\nCrowdFlow stopped.")


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments.config, arguments.source)
