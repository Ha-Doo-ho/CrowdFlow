# drone_controller.py (손성 담당)
# 모드 전환: source='tello' → 실시간 드론 / source='파일경로' → 테스트 영상
import cv2
import time


class DroneCamera:
    def __init__(self, source='test_video.mp4'):
        """
        source='tello'   → DJI Tello EDU 실시간 스트리밍
        source='파일경로' → 테스트 영상 파일 (개발용)
        source=0         → 노트북 웹캠 (테스트용)
        """
        self.source = source
        self.mode = None
        self.tello = None
        self.frame_reader = None
        self.cap = None

        if source == 'tello':
            self._connect_tello()
        else:
            self._connect_video(source)

    # ─── Tello 연결 ───────────────────────────────
    def _connect_tello(self):
        try:
            from djitellopy import Tello
            self.tello = Tello()
            self.tello.connect()

            battery = self.tello.get_battery()
            print(f"[드론] 연결 성공 | 배터리: {battery}%")

            if battery < 20:
                print(f"[드론] ⚠️ 배터리 부족! ({battery}%) — 충전 후 사용하세요")

            self.tello.streamon()
            self.frame_reader = self.tello.get_frame_read()
            self.mode = 'tello'

            # 스트리밍 안정화 대기 (첫 프레임이 None일 수 있음)
            print("[드론] 스트리밍 안정화 대기 중...")
            for _ in range(30):  # 최대 3초 대기
                frame = self.frame_reader.frame
                if frame is not None and frame.size > 0:
                    h, w = frame.shape[:2]
                    print(f"[드론] 스트리밍 시작 | 해상도: {w}x{h}")
                    break
                time.sleep(0.1)
            else:
                print("[드론] ⚠️ 스트리밍 시작 지연 — 계속 시도합니다")

        except ImportError:
            print("[드론] djitellopy 미설치 → pip install djitellopy")
            print("[드론] 테스트 영상 모드로 전환합니다")
            self._connect_video('test_video.mp4')
        except Exception as e:
            print(f"[드론] 연결 실패: {e}")
            print("[드론] 테스트 영상 모드로 전환합니다")
            self._connect_video('test_video.mp4')

    # ─── 테스트 영상/웹캠 연결 ─────────────────────
    def _connect_video(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise IOError(f"영상 소스를 열 수 없음: {source}")
        self.mode = 'video'

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"[영상] 파일 로드: {source} | {w}x{h} @ {fps}FPS")

    # ─── 프레임 1장 가져오기 (핵심 함수) ─────────────
    def get_frame(self):
        """
        출력: BGR numpy array (H, W, 3) 또는 None
        ★ while문 없음 — main.py가 호출할 때마다 최신 1장만 반환
        """
        if self.mode == 'tello':
            frame = self.frame_reader.frame
            # Tello는 항상 최신 프레임을 frame_reader.frame에 보관
            # → main.py의 while 루프에서 호출할 때마다 그 순간의 최신 사진
            if frame is None or frame.size == 0:
                return None
            return frame

        elif self.mode == 'video':
            success, frame = self.cap.read()
            return frame if success else None

        return None

    # ─── 배터리 확인 (Tello 전용) ──────────────────
    def get_battery(self):
        """Tello 배터리 잔량 반환. 영상 모드면 -1"""
        if self.mode == 'tello' and self.tello:
            try:
                return self.tello.get_battery()
            except Exception:
                return -1
        return -1

    # ─── 비상 착륙 (Tello 전용) ────────────────────
    def emergency_land(self):
        """긴급 상황 시 즉시 착륙"""
        if self.mode == 'tello' and self.tello:
            try:
                print("[드론] ⚠️ 비상 착륙 실행!")
                self.tello.land()
            except Exception as e:
                print(f"[드론] 비상 착륙 실패: {e}")

    # ─── 자원 해제 ─────────────────────────────────
    def release(self):
        """프로그램 종료 시 호출"""
        if self.mode == 'tello' and self.tello:
            try:
                self.tello.streamoff()
                print("[드론] 스트리밍 종료")
            except Exception:
                pass
        elif self.mode == 'video' and self.cap:
            self.cap.release()
            print("[영상] 파일 닫기 완료")


# class DroneCamera:
#     def __init__(self, source='test_video.mp4'):
#         self.cap = cv2.VideoCapture(source) #비디오 파일이나 웹캠을 열 때 self.cap = cv2.VideoCapture(0)과 같이 정의.
#         if not self.cap.isOpened():
#             raise IOError(f"Cannot open video source: {source}")
#
#     def get_frame(self): #main.py에 있는 두호가 짠 코드에 사진 한 장 달라고 호출할 때마다 실행되는 메서드.
#         """
#         출력: BGR numpy array (H, W, 3) 또는 None
#         """
#         success, frame = self.cap.read() #캡처 사진 한장씩 읽음. frame: 읽어서 나온 진짜 사진(BGR NumPy 배열) 데이터, Success: 성공적으로 페이지를 넘겼는가? boolean.
#         return frame if success else None
#
#     def release(self):
#         self.cap.release()
