import socket
import time

import cv2
import numpy as np
from djitellopy import Tello


WINDOW_NAME = "Tello Video Test"
FRAME_WAIT_SECONDS = 10
TELEMETRY_INTERVAL_SECONDS = 5
TELLO_IP = "192.168.10.1"


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
    local_ip = get_route_local_ip(TELLO_IP)
    print(f"[Network] Tello 통신에 선택된 로컬 IP: {local_ip}")
    if not local_ip.startswith("192.168.10."):
        raise RuntimeError(
            "Tello가 아닌 다른 네트워크 어댑터가 선택되었습니다. "
            "TELLO-XXXXXX Wi-Fi 연결과 Windows 라우팅 설정을 확인하세요."
        )

    tello = Tello()
    stream_started = False

    try:
        print("[Tello] 연결 중...")
        tello.connect()

        battery = tello.get_battery()
        print(f"[Tello] 연결 성공 | 배터리: {battery}%")

        tello.streamon()
        stream_started = True
        frame_reader = tello.get_frame_read()

        print("[Tello] 영상 대기 중... 종료하려면 q 키를 누르세요.")
        deadline = time.time() + FRAME_WAIT_SECONDS
        received_frame = False
        next_telemetry_at = time.time()

        while True:
            frame = frame_reader.frame

            if is_real_video_frame(frame):
                if not received_frame:
                    height, width = frame.shape[:2]
                    print(f"[Tello] 영상 수신 성공 | 해상도: {width}x{height}")
                    received_frame = True

                cv2.imshow(WINDOW_NAME, frame)
            elif not received_frame and time.time() >= deadline:
                raise TimeoutError(
                    f"{FRAME_WAIT_SECONDS}초 동안 Tello 영상을 받지 못했습니다."
                )

            now = time.time()
            if now >= next_telemetry_at:
                print_telemetry(tello)
                next_telemetry_at = now + TELEMETRY_INTERVAL_SECONDS

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as exc:
        print(f"[Tello] 테스트 실패: {type(exc).__name__}: {exc}")
        raise
    finally:
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
