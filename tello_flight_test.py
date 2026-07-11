import argparse
import socket
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from djitellopy import Tello


WINDOW_NAME = "Tello Flight Test"
TELLO_IP = "192.168.10.1"
TELEMETRY_INTERVAL_SECONDS = 2.0
HOVER_KEEPALIVE_INTERVAL_SECONDS = 0.5
DEFAULT_FRAME_DIR = "data/calibration_frames"
DEFAULT_RECORDING_DIR = "data/tello_recordings"
DEFAULT_RECORDING_FPS = 30.0


def get_route_local_ip(target_ip):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((target_ip, Tello.CONTROL_UDP_PORT))
        return sock.getsockname()[0]
    finally:
        sock.close()


def is_real_video_frame(frame):
    return frame is not None and frame.size > 0 and np.any(frame)


def tello_rgb_to_bgr(frame):
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def get_height_cm(tello):
    state = tello.get_current_state()
    if not state:
        return None
    height = state.get("h")
    return int(height) if height is not None else None


def format_height(height):
    if height in (None, "?"):
        return "?cm"
    return f"{int(height)}cm ({int(height) / 100:.2f}m)"


def print_telemetry(tello):
    state = tello.get_current_state()
    if not state:
        print("[Tello] Waiting for state data...")
        return None

    battery = state.get("bat", "?")
    height = state.get("h", "?")
    flight_time = state.get("time", "?")
    temp_low = state.get("templ", "?")
    temp_high = state.get("temph", "?")
    print(
        "[Tello] State | "
        f"battery: {battery}% | "
        f"height: {format_height(height)} | "
        f"flight time: {flight_time}s | "
        f"temperature: {temp_low}~{temp_high}C"
    )
    return state


def key_to_target_height_cm(key):
    if ord("1") <= key <= ord("9"):
        return (key - ord("0")) * 100
    if key == ord("0"):
        return 1000
    return None


def move_vertical_in_chunks(tello, direction, distance_cm):
    remaining = int(abs(distance_cm))
    while remaining > 0:
        step = min(remaining, 500)
        if step < 20:
            break
        if direction == "up":
            tello.move_up(step)
        else:
            tello.move_down(step)
        remaining -= step


def move_to_target_height(tello, target_cm, max_height_cm):
    if target_cm > max_height_cm:
        print(
            "[Tello] Target height blocked | "
            f"requested: {target_cm}cm, limit: {max_height_cm}cm"
        )
        return

    current_cm = get_height_cm(tello)
    if current_cm is None:
        print("[Tello] Current height is not available yet.")
        return

    delta = target_cm - current_cm
    if abs(delta) < 20:
        print(
            "[Tello] Already close to target height | "
            f"current: {format_height(current_cm)}, target: {format_height(target_cm)}"
        )
        return

    direction = "up" if delta > 0 else "down"
    print(
        f"[Tello] Move to {format_height(target_cm)} | "
        f"current: {format_height(current_cm)}, delta: {delta}cm"
    )
    move_vertical_in_chunks(tello, direction, delta)
    print_telemetry(tello)


def save_frame(frame, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"calibration_frame_{timestamp}.jpg"
    cv2.imwrite(str(output_path), frame)
    print(f"[Tello] Calibration frame saved: {output_path}")


def make_recording_path(record_dir, prefix):
    Path(record_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(record_dir) / f"{prefix}_{timestamp}.mp4"


def create_video_writer(output_path, frame, fps):
    if fps <= 0:
        raise ValueError("--record-fps must be greater than 0.")

    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")
    return writer


def draw_recording_indicator(frame, output_path):
    display = frame.copy()
    cv2.circle(display, (24, 28), 9, (0, 0, 255), -1)
    cv2.putText(
        display,
        f"REC {output_path.name}",
        (42, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return display


def build_parser():
    parser = argparse.ArgumentParser(
        description="Tello flight, hover, calibration frame capture, and recording test"
    )
    parser.add_argument("--min-battery", type=int, default=30)
    parser.add_argument("--step-cm", type=int, default=100)
    parser.add_argument("--move-cm", type=int, default=50)
    parser.add_argument("--max-height-cm", type=int, default=1000)
    parser.add_argument("--keepalive-interval", type=float, default=HOVER_KEEPALIVE_INTERVAL_SECONDS)
    parser.add_argument("--output-dir", default=DEFAULT_FRAME_DIR)
    parser.add_argument("--record-dir", default=DEFAULT_RECORDING_DIR)
    parser.add_argument("--record-fps", type=float, default=DEFAULT_RECORDING_FPS)
    parser.add_argument("--record-prefix", default="tello_flight_recording")
    return parser


def validate_args(args):
    if not 20 <= args.step_cm <= 500:
        raise ValueError("--step-cm must be in the Tello SDK range: 20~500cm.")
    if not 20 <= args.move_cm <= 500:
        raise ValueError("--move-cm must be in the Tello SDK range: 20~500cm.")
    if args.keepalive_interval <= 0:
        raise ValueError("--keepalive-interval must be greater than 0.")
    if args.max_height_cm < args.step_cm:
        raise ValueError("--max-height-cm must be greater than or equal to --step-cm.")
    if args.record_fps <= 0:
        raise ValueError("--record-fps must be greater than 0.")


def print_controls(args):
    print("[Controls]")
    print("  t: takeoff")
    print("  l: land")
    print("  q: quit")
    print("  r: start/stop recording")
    print("  c: save current calibration frame")
    print("  h: print telemetry")
    print("  u / j: move up / down")
    print("  w / s: move forward / back")
    print("  a / d: move left / right")
    print("  1~9: move to 1m~9m")
    print("  0: move to 10m")
    print(f"[Limits] max height: {args.max_height_cm}cm")
    print(f"[Step] vertical: {args.step_cm}cm | horizontal: {args.move_cm}cm")


def main():
    args = build_parser().parse_args()
    validate_args(args)

    local_ip = get_route_local_ip(TELLO_IP)
    print(f"[Network] selected local IP for Tello: {local_ip}")
    if not local_ip.startswith("192.168.10."):
        raise RuntimeError(
            "The selected network interface does not look like Tello Wi-Fi. "
            "Connect to TELLO-XXXXXX Wi-Fi and try again."
        )

    tello = Tello()
    stream_started = False
    is_flying = False
    last_frame = None
    video_writer = None
    recording_path = None

    try:
        print("[Tello] Connecting...")
        tello.connect()
        battery = tello.get_battery()
        print(f"[Tello] Connected | battery: {battery}%")
        if battery < args.min_battery:
            raise RuntimeError(
                f"Battery too low: {battery}% < {args.min_battery}%. "
                "Charge before flight."
            )

        tello.streamon()
        stream_started = True
        frame_reader = tello.get_frame_read()
        print_controls(args)

        next_telemetry_at = time.time()
        next_hover_keepalive_at = time.time()
        while True:
            frame = frame_reader.frame
            if is_real_video_frame(frame):
                frame = tello_rgb_to_bgr(frame)
                last_frame = frame

                if video_writer is not None:
                    video_writer.write(frame)
                    cv2.imshow(WINDOW_NAME, draw_recording_indicator(frame, recording_path))
                else:
                    cv2.imshow(WINDOW_NAME, frame)

            now = time.time()
            if now >= next_telemetry_at:
                print_telemetry(tello)
                next_telemetry_at = now + TELEMETRY_INTERVAL_SECONDS

            if is_flying and now >= next_hover_keepalive_at:
                tello.send_rc_control(0, 0, 0, 0)
                next_hover_keepalive_at = now + args.keepalive_interval

            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue

            if key == ord("t"):
                if is_flying:
                    print("[Tello] Already flying.")
                    continue
                print("[Tello] Takeoff")
                tello.takeoff()
                is_flying = True
                next_hover_keepalive_at = time.time()
                print_telemetry(tello)

            elif key == ord("r"):
                if video_writer is None:

                    if last_frame is None:
                        print("[Tello] No video frame yet. Wait before recording.")
                        continue
                    recording_path = make_recording_path(args.record_dir, args.record_prefix)
                    video_writer = create_video_writer(recording_path, last_frame, args.record_fps)
                    print(f"[Tello] Recording started: {recording_path}")
                else:
                    video_writer.release()
                    video_writer = None
                    print(f"[Tello] Recording saved: {recording_path}")
                    recording_path = None

            elif key == ord("u"):
                if not is_flying:
                    print("[Tello] Take off first with t.")
                    continue
                height = get_height_cm(tello)
                if height is not None and height + args.step_cm > args.max_height_cm:
                    print(
                        "[Tello] Up command blocked | "
                        f"current: {height}cm, requested: {height + args.step_cm}cm, "
                        f"limit: {args.max_height_cm}cm"
                    )
                    continue
                print(f"[Tello] Move up {args.step_cm}cm")
                tello.move_up(args.step_cm)
                print_telemetry(tello)

            elif key == ord("j"):
                if not is_flying:
                    print("[Tello] Take off first with t.")
                    continue
                print(f"[Tello] Move down {args.step_cm}cm")
                tello.move_down(args.step_cm)
                print_telemetry(tello)

            elif key_to_target_height_cm(key) is not None:
                if not is_flying:
                    print("[Tello] Take off first with t.")
                    continue
                move_to_target_height(tello, key_to_target_height_cm(key), args.max_height_cm)

            elif key == ord("w"):
                if not is_flying:
                    print("[Tello] Take off first with t.")
                    continue
                print(f"[Tello] Move forward {args.move_cm}cm")
                tello.move_forward(args.move_cm)
                print_telemetry(tello)

            elif key == ord("s"):
                if not is_flying:
                    print("[Tello] Take off first with t.")
                    continue
                print(f"[Tello] Move back {args.move_cm}cm")
                tello.move_back(args.move_cm)
                print_telemetry(tello)

            elif key == ord("a"):
                if not is_flying:
                    print("[Tello] Take off first with t.")
                    continue
                print(f"[Tello] Move left {args.move_cm}cm")
                tello.move_left(args.move_cm)
                print_telemetry(tello)

            elif key == ord("d"):
                if not is_flying:
                    print("[Tello] Take off first with t.")
                    continue
                print(f"[Tello] Move right {args.move_cm}cm")
                tello.move_right(args.move_cm)
                print_telemetry(tello)

            elif key == ord("c"):
                if last_frame is None:
                    print("[Tello] No video frame yet.")
                    continue
                save_frame(last_frame, Path(args.output_dir))

            elif key == ord("h"):
                print_telemetry(tello)

            elif key == ord("l"):
                if not is_flying:
                    print("[Tello] Not flying.")
                    continue
                print("[Tello] Land")
                tello.land()
                is_flying = False

            elif key == ord("q"):
                break

    except Exception as exc:
        print(f"[Tello] Flight test failed: {type(exc).__name__}: {exc}")
        raise
    finally:
        if video_writer is not None:
            try:
                video_writer.release()
                print(f"[Tello] Recording saved: {recording_path}")
            except Exception:
                pass
        if is_flying:
            try:
                print("[Tello] Landing before exit")
                tello.land()
            except Exception:
                pass
        if stream_started:
            try:
                tello.streamoff()
            except Exception:
                pass
        try:
            tello.end()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[Tello] Disconnected")


if __name__ == "__main__":
    main()
