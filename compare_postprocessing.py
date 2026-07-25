import argparse
import csv
import json
from pathlib import Path
import time

import cv2

from core.boundary_stabilizer import BoundaryStateStabilizer
from core.detection_postprocessor import suppress_cross_class_duplicates
from core.detector import CrowdDetector
from core.grid_calculator import GridCalculator
from core.settings import AppSettings


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw detections with duplicate suppression and "
            "boundary stabilization using one model inference per frame."
        ),
    )
    parser.add_argument(
        "--config",
        default="config/app_config_7_85x10.json",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Offline video path.",
    )
    parser.add_argument(
        "--output",
        default="output/postprocessing_comparison.csv",
    )
    parser.add_argument(
        "--max-analyzed-frames",
        type=int,
        default=0,
        help="0 processes the full video.",
    )
    return parser.parse_args()


def resolve_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def build_calculator(settings, calibration_path):
    calculator = GridCalculator(
        grid_size=settings.grid_size,
        area_width=settings.area_width,
        area_height=settings.area_height,
        conf_threshold=settings.low_confidence_threshold,
        low_confidence_min_level=settings.low_confidence_min_level,
        boundary_margin_m=settings.boundary_margin_m,
    )
    with resolve_path(calibration_path).open(
        "r",
        encoding="utf-8",
    ) as calibration_file:
        calibration = json.load(calibration_file)
    calculator.set_homography(
        calibration["src_points"],
        calibration["dst_points"],
    )
    return calculator


def build_stabilizer(settings):
    return BoundaryStateStabilizer(
        area_width=settings.area_width,
        area_height=settings.area_height,
        history_size=settings.boundary_history_size,
        majority_count=settings.boundary_majority_count,
        boundary_band_m=settings.boundary_stability_band_m,
        max_missing_updates=settings.boundary_track_max_missing,
    )


def compare(config_path, source_path, output_path, max_analyzed_frames=0):
    settings = AppSettings.load(resolve_path(config_path))
    if not settings.calibration_path:
        raise ValueError(
            "A measured calibration_path is required for comparison."
        )

    source = resolve_path(source_path)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise IOError(f"Could not open video: {source}")

    detector = CrowdDetector(
        model_path=str(resolve_path(settings.model_path)),
        imgsz=settings.imgsz,
        conf=settings.detection_confidence,
        person_classes=settings.person_classes,
        model_type=settings.model_type,
        cross_class_dedup_enabled=False,
        dedup_iou_threshold=settings.dedup_iou_threshold,
    )
    calculator = build_calculator(
        settings,
        settings.calibration_path,
    )
    stabilizer = build_stabilizer(settings)

    rows = []
    frame_index = -1
    analyzed_frames = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_index += 1
            if frame_index % settings.analyze_every != 0:
                continue

            started_at = time.perf_counter()
            raw_detections, _ = detector.detect(frame)
            inference_ms = (time.perf_counter() - started_at) * 1000

            baseline = calculator.calculate(raw_detections)
            deduplicated, dedup_stats = (
                suppress_cross_class_duplicates(
                    raw_detections,
                    iou_threshold=settings.dedup_iou_threshold,
                )
            )
            raw_mappings = calculator.map_detections(deduplicated)
            stabilized_mappings, stability_stats = stabilizer.update(
                raw_mappings
            )
            improved = calculator.calculate_from_mappings(
                stabilized_mappings
            )

            rows.append({
                "frame_index": frame_index,
                "video_time_seconds": round(
                    frame_index
                    / max(capture.get(cv2.CAP_PROP_FPS), 1.0),
                    3,
                ),
                "inference_ms": round(inference_ms, 3),
                "raw_detection_count": len(raw_detections),
                "duplicate_boxes_removed": dedup_stats[
                    "duplicate_boxes_removed"
                ],
                "before_counted": int(baseline["count"].sum()),
                "after_counted": int(improved["count"].sum()),
                "before_ignored": int(baseline["ignored_count"]),
                "after_ignored": int(improved["ignored_count"]),
                "before_max_density": round(
                    float(baseline["max_density"]),
                    4,
                ),
                "after_max_density": round(
                    float(improved["max_density"]),
                    4,
                ),
                "boundary_status_overrides": stability_stats[
                    "boundary_status_overrides"
                ],
            })
            analyzed_frames += 1
            if (
                max_analyzed_frames > 0
                and analyzed_frames >= max_analyzed_frames
            ):
                break
    finally:
        capture.release()

    output = resolve_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output.open("w", encoding="utf-8-sig", newline="") as csv_file:
        if fieldnames:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    summary = build_summary(rows, settings, source)
    summary_path = output.with_name(f"{output.stem}_summary.json")
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)

    print(f"[Comparison] frames: {len(rows)}")
    print(f"[Comparison] CSV: {output}")
    print(f"[Comparison] summary: {summary_path}")
    return rows, summary


def build_summary(rows, settings, source):
    frame_count = len(rows)

    def mean(key):
        if not rows:
            return 0.0
        return sum(float(row[key]) for row in rows) / frame_count

    return {
        "source": str(source),
        "analyzed_frames": frame_count,
        "analyze_every": settings.analyze_every,
        "dedup_iou_threshold": settings.dedup_iou_threshold,
        "boundary_vote_rule": (
            f"{settings.boundary_majority_count}/"
            f"{settings.boundary_history_size}"
        ),
        "boundary_stability_band_m": (
            settings.boundary_stability_band_m
        ),
        "duplicate_boxes_removed_total": sum(
            int(row["duplicate_boxes_removed"]) for row in rows
        ),
        "frames_with_duplicates": sum(
            int(row["duplicate_boxes_removed"]) > 0 for row in rows
        ),
        "frames_with_boundary_override": sum(
            int(row["boundary_status_overrides"]) > 0 for row in rows
        ),
        "frames_with_count_change": sum(
            int(row["before_counted"]) != int(row["after_counted"])
            for row in rows
        ),
        "before": {
            "mean_counted": round(mean("before_counted"), 4),
            "mean_ignored": round(mean("before_ignored"), 4),
            "mean_max_density": round(
                mean("before_max_density"),
                4,
            ),
        },
        "after": {
            "mean_counted": round(mean("after_counted"), 4),
            "mean_ignored": round(mean("after_ignored"), 4),
            "mean_max_density": round(
                mean("after_max_density"),
                4,
            ),
        },
    }


if __name__ == "__main__":
    arguments = parse_args()
    compare(
        arguments.config,
        arguments.source,
        arguments.output,
        arguments.max_analyzed_frames,
    )
