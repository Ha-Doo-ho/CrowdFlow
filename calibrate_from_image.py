import argparse
import json
from pathlib import Path

import cv2
import numpy as np


POINT_LABELS = [
    "left_top",
    "right_top",
    "left_bottom",
    "right_bottom",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Click four ground reference points and create config/calibration.json"
    )
    parser.add_argument("image", help="Calibration frame image path")
    parser.add_argument("--area-width", type=float, default=1.0)
    parser.add_argument("--area-height", type=float, default=3.0)
    parser.add_argument("--out", default="config/calibration.json")
    parser.add_argument("--preview", default="output/calibration_preview.jpg")
    return parser.parse_args()


def draw_points(image, points):
    canvas = image.copy()
    for index, (x, y) in enumerate(points):
        cv2.circle(canvas, (int(x), int(y)), 7, (0, 255, 0), -1)
        cv2.putText(
            canvas,
            f"{index + 1}:{POINT_LABELS[index]}",
            (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if len(points) >= 2:
        if len(points) == 4:
            lines = [(points[0], points[1]), (points[1], points[3]),
                     (points[3], points[2]), (points[2], points[0])]
        else:
            lines = list(zip(points, points[1:]))
        for start, end in lines:
            cv2.line(
                canvas,
                tuple(map(int, start)),
                tuple(map(int, end)),
                (0, 255, 0),
                2,
            )
    return canvas


def collect_points(image):
    points = []
    window_name = "Calibration Point Picker"

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or len(points) >= 4:
            return
        points.append([float(x), float(y)])
        label = POINT_LABELS[len(points) - 1]
        print(f"[Calibration] {len(points)}. {label}: ({x}, {y})")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    print("[Calibration] Click X centers in this order:")
    print("  1) left_top")
    print("  2) right_top")
    print("  3) left_bottom")
    print("  4) right_bottom")
    print("[Keys] u: undo | r: reset | s/enter: save after 4 points | q/esc: quit")

    while True:
        canvas = draw_points(image, points)
        if len(points) < 4:
            next_label = POINT_LABELS[len(points)]
            cv2.putText(
                canvas,
                f"Click {len(points) + 1}/4: {next_label}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                canvas,
                "Press s or Enter to save",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            raise SystemExit("Calibration cancelled.")
        if key == ord("u") and points:
            removed = points.pop()
            print(f"[Calibration] Undo: {removed}")
        elif key == ord("r"):
            points.clear()
            print("[Calibration] Reset points.")
        elif key in (ord("s"), 13) and len(points) == 4:
            cv2.destroyWindow(window_name)
            return points


def save_calibration(points, area_width, area_height, output_path):
    calibration = {
        "src_points": points,
        "dst_points": [
            [0.0, 0.0],
            [float(area_width), 0.0],
            [0.0, float(area_height)],
            [float(area_width), float(area_height)],
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(calibration, output_file, ensure_ascii=False, indent=2)
    print(f"[Calibration] Saved: {output_path}")

    archive_path = next_indexed_path(output_path)
    with archive_path.open("w", encoding="utf-8") as archive_file:
        json.dump(calibration, archive_file, ensure_ascii=False, indent=2)
    print(f"[Calibration] Archive saved: {archive_path}")
    return calibration


def next_indexed_path(path):
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find available indexed path for: {path}")


def save_preview(image, calibration, preview_path):
    src = np.asarray(calibration["src_points"], dtype=np.float32)
    dst = np.asarray(calibration["dst_points"], dtype=np.float32)

    scale = 300
    preview_size = (
        max(1, int(round(dst[:, 0].max() * scale))),
        max(1, int(round(dst[:, 1].max() * scale))),
    )
    dst_pixels = dst * scale
    matrix = cv2.getPerspectiveTransform(src, dst_pixels)
    warped = cv2.warpPerspective(image, matrix, preview_size)

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), warped)
    print(f"[Calibration] Preview saved: {preview_path}")

    archive_path = next_indexed_path(preview_path)
    cv2.imwrite(str(archive_path), warped)
    print(f"[Calibration] Preview archive saved: {archive_path}")


def main():
    args = parse_args()
    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
    if args.area_width <= 0 or args.area_height <= 0:
        raise ValueError("area width/height must be positive.")

    points = collect_points(image)
    calibration = save_calibration(
        points,
        args.area_width,
        args.area_height,
        Path(args.out),
    )
    save_preview(image, calibration, Path(args.preview))


if __name__ == "__main__":
    main()
