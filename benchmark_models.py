import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from core.detector import CrowdDetector
from core.result_publisher import save_latest_result


def parse_model_spec(value):
    try:
        model_type, model_path = value.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "모델은 yolo:경로 또는 rtdetr:경로 형식이어야 합니다."
        ) from exc

    if model_type not in {"yolo", "rtdetr"}:
        raise argparse.ArgumentTypeError("모델 유형은 yolo 또는 rtdetr이어야 합니다.")
    return model_type, model_path


def percentile_95(values):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return ordered[index]


def benchmark_model(
    video_path,
    model_type,
    model_path,
    imgsz,
    confidence,
    stride,
    max_frames,
):
    detector = CrowdDetector(
        model_path=model_path,
        model_type=model_type,
        imgsz=imgsz,
        conf=confidence,
    )
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise IOError(f"영상을 열 수 없습니다: {video_path}")

    processed_frames = 0
    source_frames = 0
    inference_times = []
    detection_counts = []

    try:
        while processed_frames < max_frames:
            success, frame = capture.read()
            if not success:
                break

            source_frames += 1
            if (source_frames - 1) % stride != 0:
                continue

            started_at = time.perf_counter()
            detections, _ = detector.detect(frame)
            inference_ms = (time.perf_counter() - started_at) * 1000

            inference_times.append(inference_ms)
            detection_counts.append(len(detections))
            processed_frames += 1
    finally:
        capture.release()

    if not inference_times:
        raise RuntimeError("벤치마크할 프레임을 읽지 못했습니다.")

    avg_inference_ms = sum(inference_times) / len(inference_times)
    return {
        "model_name": Path(model_path).name,
        "model_type": model_type,
        "imgsz": imgsz,
        "confidence": confidence,
        "processed_frames": processed_frames,
        "stride": stride,
        "avg_inference_ms": round(avg_inference_ms, 2),
        "p95_inference_ms": round(percentile_95(inference_times), 2),
        "estimated_inference_fps": round(1000 / avg_inference_ms, 2),
        "avg_detections_per_frame": round(
            sum(detection_counts) / len(detection_counts),
            2,
        ),
        "min_detections": min(detection_counts),
        "max_detections": max(detection_counts),
        "person_classes": detector.person_classes,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="같은 영상에서 YOLO/RT-DETR 추론 성능을 비교합니다."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_spec,
        required=True,
        help="예: --model yolo:weights/yolo11l_crowdflow.pt",
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument(
        "--output",
        default="data/model_benchmark.json",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.stride <= 0 or args.max_frames <= 0:
        raise ValueError("stride와 max-frames는 1 이상이어야 합니다.")

    results = []
    for model_type, model_path in args.model:
        print(f"[Benchmark] {model_type}: {model_path}")
        result = benchmark_model(
            video_path=args.video,
            model_type=model_type,
            model_path=model_path,
            imgsz=args.imgsz,
            confidence=args.confidence,
            stride=args.stride,
            max_frames=args.max_frames,
        )
        results.append(result)
        print(
            f"  평균 {result['avg_inference_ms']:.1f}ms | "
            f"추론 FPS {result['estimated_inference_fps']:.1f} | "
            f"평균 탐지 {result['avg_detections_per_frame']:.1f}명"
        )

    report = {
        "created_at": datetime.now().isoformat(),
        "video": args.video,
        "note": (
            "이 결과는 속도와 탐지량 비교이며 정확도 평가가 아닙니다. "
            "정확도는 ground truth로 별도 평가해야 합니다."
        ),
        "results": results,
    }
    save_latest_result(report, args.output)
    print(f"[Benchmark] 결과 저장: {args.output}")


if __name__ == "__main__":
    main()
