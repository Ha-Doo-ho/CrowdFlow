import argparse
import socket
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from djitellopy import Tello


WINDOW_NAME = "Tello Video Test"
FRAME_WAIT_SECONDS = 10
TELEMETRY_INTERVAL_SECONDS = 5
TELLO_IP = "192.168.10.1"
DEFAULT_RECORDING_DIR = "data/tello_recordings"
DEFAULT_RECORDING_FPS = 30.0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Tello video stream test with optional recording"
    )
    parser.add_argument("--record-dir", default=DEFAULT_RECORDING_DIR)
    parser.add_argument("--record-fps", type=float, default=DEFAULT_RECORDING_FPS)
    parser.add_argument("--record-prefix", default="tello_recording")
    return parser


def get_route_local_ip(target_ip):
    """Windows가 target_ip로 송신할 때 선택하는 로컬 어댑터 IP를 확인한다."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((target_ip, Tello.CONTROL_UDP_PORT))
        return sock.getsockname()[0]
    finally:
        sock.close()


def is_real_video_frame(frame):
    """DJITelloPy의 초기 400x300 검은 프레임을 실제 영상으로 오인하지 않는다."""
    return (
        frame is not None
        and frame.size > 0
        and np.any(frame)
    )


def tello_rgb_to_bgr(frame):
    """DJITelloPy 프레임(RGB)을 OpenCV/탐지 파이프라인 기준(BGR)으로 맞춘다."""
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def make_recording_path(record_dir, prefix):
    Path(record_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(record_dir) / f"{prefix}_{timestamp}.mp4"


def create_video_writer(output_path, frame, fps):
    if fps <= 0:
        raise ValueError("--record-fps는 0보다 커야 합니다.")

    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"녹화 파일을 열 수 없습니다: {output_path}")
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


def print_telemetry(tello):
    state = tello.get_current_state()
    if not state:
        print("[Tello] 상태 데이터 대기 중...")
        return

    battery = state.get("bat", "?")
    temp_low = state.get("templ", "?")
    temp_high = state.get("temph", "?")
    height = state.get("h", "?")
    flight_time = state.get("time", "?")

    print(
        "[Tello] 상태 | "
        f"배터리: {battery}% | "
        f"온도: {temp_low}~{temp_high}°C | "
        f"높이: {height}cm | "
        f"비행시간: {flight_time}s"
    )


def main():
    args = build_parser().parse_args()
    local_ip = get_route_local_ip(TELLO_IP)
    print(f"[Network] Tello 통신에 선택된 로컬 IP: {local_ip}")
    if not local_ip.startswith("192.168.10."):
        raise RuntimeError(
            "Tello가 아닌 다른 네트워크 어댑터가 선택되었습니다. "
            "TELLO-XXXXXX Wi-Fi 연결과 Windows 라우팅 설정을 확인하세요."
        )

    tello = Tello()
    stream_started = False
    video_writer = None
    recording_path = None
    last_frame = None

    try:
        print("[Tello] 연결 중...")
        tello.connect()

        battery = tello.get_battery()
        print(f"[Tello] 연결 성공 | 배터리: {battery}%")

        tello.streamon()
        stream_started = True
        frame_reader = tello.get_frame_read()

        print("[Tello] 영상 대기 중...")
        print("  r: 녹화 시작/정지")
        print("  q: 종료")
        deadline = time.time() + FRAME_WAIT_SECONDS
        received_frame = False
        next_telemetry_at = time.time()

        while True:
            frame = frame_reader.frame

            if is_real_video_frame(frame):
                frame = tello_rgb_to_bgr(frame)
                last_frame = frame
                if not received_frame:
                    height, width = frame.shape[:2]
                    print(f"[Tello] 영상 수신 성공 | 해상도: {width}x{height}")
                    received_frame = True

                if video_writer is not None:
                    video_writer.write(frame)
                    cv2.imshow(WINDOW_NAME, draw_recording_indicator(frame, recording_path))
                else:
                    cv2.imshow(WINDOW_NAME, frame)
            elif not received_frame and time.time() >= deadline:
                raise TimeoutError(
                    f"{FRAME_WAIT_SECONDS}초 동안 Tello 영상을 받지 못했습니다."
                )

            now = time.time()
            if now >= next_telemetry_at:
                print_telemetry(tello)
                next_telemetry_at = now + TELEMETRY_INTERVAL_SECONDS

            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                if video_writer is None:
                    if last_frame is None:
                        print("[Tello] 녹화를 시작할 실제 영상 프레임이 아직 없습니다.")
                        continue
                    recording_path = make_recording_path(args.record_dir, args.record_prefix)
                    video_writer = create_video_writer(
                        recording_path,
                        last_frame,
                        args.record_fps,
                    )
                    print(f"[Tello] 녹화 시작: {recording_path}")
                else:
                    video_writer.release()
                    video_writer = None
                    print(f"[Tello] 녹화 저장 완료: {recording_path}")
                    recording_path = None
            elif key == ord("q"):
                break

    except Exception as exc:
        print(f"[Tello] 테스트 실패: {type(exc).__name__}: {exc}")
        raise
    finally:
        if video_writer is not None:
            try:
                video_writer.release()
                print(f"[Tello] 녹화 저장 완료: {recording_path}")
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
