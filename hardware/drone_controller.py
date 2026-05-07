# drone_controller.py (손성 담당)
import cv2


class DroneCamera:
    def __init__(self, source='test_video.mp4'):
        self.cap = cv2.VideoCapture(source) #비디오 파일이나 웹캠을 열 때 self.cap = cv2.VideoCapture(0)과 같이 정의.
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")

    def get_frame(self): #main.py에 있는 두호가 짠 코드에 사진 한 장 달라고 호출할 때마다 실행되는 메서드.
        """
        출력: BGR numpy array (H, W, 3) 또는 None
        """
        success, frame = self.cap.read() #캡처 사진 한장씩 읽음. frame: 읽어서 나온 진짜 사진(BGR NumPy 배열) 데이터, Success: 성공적으로 페이지를 넘겼는가? boolean.
        return frame if success else None

    def release(self):
        self.cap.release()
