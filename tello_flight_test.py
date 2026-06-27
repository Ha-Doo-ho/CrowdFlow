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
TELEMETRY_INTERVAL_SECONDS = 2
HOVER_KEEPALIVE_INTERVAL_SECONDS = 0.5


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
    """DJITelloPy 프레임(RGB)을 OpenCV/탐지 파이프라인 기준(BGR)으로 맞춘다."""
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
        print("[Tello] 상태 데이터 대기 중...")
        return None

    battery = state.get("bat", "?")
    height = state.get("h", "?")
    flight_time = state.get("time", "?")
    temp_low = state.get("templ", "?")
    temp_high = state.get("temph", "?")
    print(
        "[Tello] 상태 | "
        f"배터리: {battery}% | "
        f"높이: {format_height(height)} | "
        f"비행시간: {flight_time}s | "
        f"온도: {temp_low}~{temp_high}°C"
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
            "[Tello] 목표 고도 제한: "
            f"{target_cm}cm 요청, 현재 제한 {max_height_cm}cm"
        )
        return

    current_cm = get_height_cm(tello)
    if current_cm is None:
        print("[Tello] 현재 고도를 아직 확인하지 못했습니다.")
        return

    delta = target_cm - current_cm
    if abs(delta) < 20:
        print(
            "[Tello] 목표 고도에 충분히 가깝습니다. "
            f"현재 {format_height(current_cm)}, 목표 {format_height(target_cm)}"
        )
        return

    direction = "up" if delta > 0 else "down"
    action = "상승" if delta > 0 else "하강"
    print(
        f"[Tello] 목표 고도 {format_height(target_cm)}로 이동 | "
        f"현재 {format_height(current_cm)}에서 {abs(delta)}cm {action}"
    )
    move_vertical_in_chunks(tello, direction, delta)
    print_telemetry(tello)


def save_frame(frame, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"calibration_frame_{timestamp}.jpg"
    cv2.imwrite(str(output_path), frame)
    print(f"[Tello] 캘리브레이션 프레임 저장: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Tello 이륙, 높이 확인, 캘리브레이션 프레임 저장 테스트"
    )
    parser.add_argument("--min-battery", type=int, default=30)
    parser.add_argument("--step-cm", type=int, default=100)
    parser.add_argument("--move-cm", type=int, default=50)
    parser.add_argument("--max-height-cm", type=int, default=1000)
    parser.add_argument("--keepalive-interval", type=float, default=HOVER_KEEPALIVE_INTERVAL_SECONDS)
    parser.add_argument(
        "--output-dir",
        default="data/calibration_frames",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if not 20 <= args.step_cm <= 500:
        raise ValueError("--step-cm은 Tello SDK 이동 명령 범위인 20~500cm 안에서 지정하세요.")
    if not 20 <= args.move_cm <= 500:
        raise ValueError("--move-cm은 Tello SDK 이동 명령 범위인 20~500cm 안에서 지정하세요.")
    if args.keepalive_interval <= 0:
        raise ValueError("--keepalive-interval은 0보다 커야 합니다.")
    if args.max_height_cm < args.step_cm:
        raise ValueError("--max-height-cm은 --step-cm보다 크거나 같아야 합니다.")

    local_ip = get_route_local_ip(TELLO_IP)
    print(f"[Network] Tello 통신에 선택된 로컬 IP: {local_ip}")
    if not local_ip.startswith("192.168.10."):
        raise RuntimeError(
            "Tello Wi-Fi가 아닌 네트워크 어댑터가 선택되었습니다. "
            "TELLO-XXXXXX Wi-Fi 연결을 확인하세요."
        )

    tello = Tello()
    stream_started = False
    is_flying = False
    last_frame = None

    try:
        print("[Tello] 연결 중...")
        tello.connect()
        battery = tello.get_battery()
        print(f"[Tello] 연결 성공 | 배터리: {battery}%")
        if battery < args.min_battery:
            raise RuntimeError(
                f"배터리 부족: {battery}% < {args.min_battery}% "
                "충전 후 다시 시도하세요."
            )

        tello.streamon()
        stream_started = True
        frame_reader = tello.get_frame_read()

        print("[조작]")
        print("  t: 이륙")
        print("  u: 상승")
        print("  j: 하강")
        print("  1~9: 해당 m 고도로 이동")
        print("  0: 10m 고도로 이동")
        print("  w: 전진")
        print("  s: 후진")
        print("  a: 좌측 이동")
        print("  d: 우측 이동")
        print("  c: 현재 프레임 저장")
        print("  h: 현재 높이 출력")
        print("  l: 착륙")
        print("  q: 종료")
        print(f"[안전 제한] 상승은 {args.max_height_cm}cm 이하에서만 허용합니다.")
        print(f"[이동 단위] 상하 {args.step_cm}cm | 전후좌우 {args.move_cm}cm")

        next_telemetry_at = time.time()
        next_hover_keepalive_at = time.time()
        while True:
            frame = frame_reader.frame
            if is_real_video_frame(frame):
                frame = tello_rgb_to_bgr(frame)
                last_frame = frame
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
                    print("[Tello] 이미 비행 중입니다.")
                    continue
                print("[Tello] 이륙")
                tello.takeoff()
                is_flying = True
                next_hover_keepalive_at = time.time()
                print_telemetry(tello)

            elif key == ord("u"):
                if not is_flying:
                    print("[Tello] 먼저 t 키로 이륙하세요.")
                    continue
                height = get_height_cm(tello)
                if height is not None and height + args.step_cm > args.max_height_cm:
                    print(
                        "[Tello] 상승 제한: "
                        f"현재 {height}cm, 요청 후 {height + args.step_cm}cm, "
                        f"제한 {args.max_height_cm}cm"
                    )
                    continue
                print(f"[Tello] {args.step_cm}cm 상승")
                tello.move_up(args.step_cm)
                print_telemetry(tello)

            elif key == ord("j"):
                if not is_flying:
                    print("[Tello] 먼저 t 키로 이륙하세요.")
                    continue
                print(f"[Tello] {args.step_cm}cm 하강")
                tello.move_down(args.step_cm)
                print_telemetry(tello)

            elif key_to_target_height_cm(key) is not None:
                if not is_flying:
                    print("[Tello] 먼저 t 키로 이륙하세요.")
                    continue
                target_cm = key_to_target_height_cm(key)
                move_to_target_height(tello, target_cm, args.max_height_cm)

            elif key == ord("w"):
                if not is_flying:
                    print("[Tello] 먼저 t 키로 이륙하세요.")
                    continue
                print(f"[Tello] {args.move_cm}cm 전진")
                tello.move_forward(args.move_cm)
                print_telemetry(tello)

            elif key == ord("s"):
                if not is_flying:
                    print("[Tello] 먼저 t 키로 이륙하세요.")
                    continue
                print(f"[Tello] {args.move_cm}cm 후진")
                tello.move_back(args.move_cm)
                print_telemetry(tello)

            elif key == ord("a"):
                if not is_flying:
                    print("[Tello] 먼저 t 키로 이륙하세요.")
                    continue
                print(f"[Tello] {args.move_cm}cm 좌측 이동")
                tello.move_left(args.move_cm)
                print_telemetry(tello)

            elif key == ord("d"):
                if not is_flying:
                    print("[Tello] 먼저 t 키로 이륙하세요.")
                    continue
                print(f"[Tello] {args.move_cm}cm 우측 이동")
                tello.move_right(args.move_cm)
                print_telemetry(tello)

            elif key == ord("c"):
                if last_frame is None:
                    print("[Tello] 저장할 영상 프레임이 아직 없습니다.")
                    continue
                save_frame(last_frame, Path(args.output_dir))

            elif key == ord("h"):
                print_telemetry(tello)

            elif key == ord("l"):
                if not is_flying:
                    print("[Tello] 현재 비행 중이 아닙니다.")
                    continue
                print("[Tello] 착륙")
                tello.land()
                is_flying = False

            elif key == ord("q"):
                break

    except Exception as exc:
        print(f"[Tello] 테스트 실패: {type(exc).__name__}: {exc}")
        raise
    finally:
        if is_flying:
            try:
                print("[Tello] 종료 전 안전 착륙")
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
        print("[Tello] 연결 종료")


if __name__ == "__main__":
    main()
