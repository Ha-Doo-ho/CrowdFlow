import os
import time
from pathlib import Path

import cv2
import json
import numpy as np

from core.detector import CrowdDetector
from core.grid_calculator import GridCalculator
from core.result_publisher import save_latest_result
from core.risk_clusterer import find_risk_clusters
from core.settings import AppSettings
from frontend.data_logger import DataLogger
from hardware.drone_controller import DroneCamera

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT, "config", "app_config.json")
DATA_DIR = os.path.join(ROOT, 'data')
LATEST_RESULT_PATH = os.path.join(DATA_DIR, 'latest_result.json')
DB_PATH = os.path.join(DATA_DIR, 'crowdflow.db')


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
        calculator.set_homography(
            calibration["src_points"],
            calibration["dst_points"],
        )
        print(f"[Calibration] 실측 좌표 사용: {resolved_path}")
        return "measured"

    # 캘리브레이션 전 개발용 fallback이다. 실제 밀집도 검증에는 사용하지 않는다.
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
    print("[Calibration] 임시 전체 프레임 매핑 사용 — 실측 결과로 해석하면 안 됩니다.")
    return "full_frame_fallback"


def analyze_frame(
    detector,
    calculator,
    frame,
    frame_number,
    source_name,
    calibration_mode,
):
    started_at = time.perf_counter()
    detections, raw_result = detector.detect(frame)
    inference_ms = (time.perf_counter() - started_at) * 1000

    grid_result = calculator.calculate(detections)
    grid_result["risk_clusters"] = find_risk_clusters(grid_result)
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
    }
    return grid_result, raw_result


def main():
    settings = AppSettings.load(CONFIG_PATH)
    source = resolve_source(settings.source)
    model_path = resolve_path(settings.model_path)

    drone = None
    logger = None

    try:
        drone = DroneCamera(source)
        detector = CrowdDetector(
            model_path=model_path,
            imgsz=settings.imgsz,
            conf=settings.detection_confidence,
            person_classes=settings.person_classes,
            model_type=settings.model_type,
        )
        calculator = GridCalculator(
            grid_size=settings.grid_size,
            area_width=settings.area_width,
            area_height=settings.area_height,
            conf_threshold=settings.low_confidence_threshold,
            low_confidence_min_level=settings.low_confidence_min_level,
        )
        logger = DataLogger(DB_PATH)

        test_frame = drone.get_frame()
        if test_frame is None:
            print("첫 프레임을 가져오지 못했습니다. 영상 경로나 스트리밍 상태를 확인하세요.")
            return

        h, w = test_frame.shape[:2]
        print(f"영상 해상도: {w} x {h}")
        calibration_mode = configure_homography(
            calculator,
            test_frame,
            settings.calibration_path,
        )

        print("=" * 50)
        print("CrowdFlow 관제 시스템 실행중... (종료: q키)")
        print("=" * 50)

        frame_count = 1

        grid_result, raw_result = analyze_frame(
            detector,
            calculator,
            test_frame,
            frame_count,
            str(settings.source),
            calibration_mode,
        )
        publish_result(grid_result, logger)
        total = int(grid_result['count'].sum())
        max_d = grid_result['max_density']
        inference_ms = grid_result["metadata"]["inference_ms"]
        print(f"[Frame {frame_count:>5}] 탐지: {total:>3}명 | "
              f"최대 밀집도: {max_d:.2f}인/m² | 추론: {inference_ms:.1f}ms")
        annotated = raw_result.plot()
        cv2.imshow("CrowdFlow AI Detection", annotated)

        while True:
            frame = drone.get_frame()
            if frame is None:
                break

            frame_count += 1

            if frame_count % settings.analyze_every == 0:
                grid_result, raw_result = analyze_frame(
                    detector,
                    calculator,
                    frame,
                    frame_count,
                    str(settings.source),
                    calibration_mode,
                )
                publish_result(grid_result, logger)

                total = int(grid_result['count'].sum())
                max_d = grid_result['max_density']
                ignored = grid_result["ignored_count"]
                inference_ms = grid_result["metadata"]["inference_ms"]
                print(f"[Frame {frame_count:>5}] "
                      f"탐지: {total:>3}명 | "
                      f"최대 밀집도: {max_d:.2f}인/m² | "
                      f"제외: {ignored}건 | "
                      f"추론: {inference_ms:.1f}ms", end="")

                if grid_result['alerts']:
                    print(f" | ⚠️ 경고 {len(grid_result['alerts'])}건")
                    for alert in grid_result['alerts']:
                        print(f"    → {alert['message']}")
                else:
                    print()  # 줄바꿈

            last_annotated = raw_result.plot()

            cv2.imshow("CrowdFlow AI Detection", last_annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if drone is not None:
            drone.release()
        if logger is not None:
            logger.close()
        cv2.destroyAllWindows()
        print("\n시스템 종료")


if __name__ == '__main__':
    main()
