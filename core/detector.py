# detector.py (홍유택 담당)
from ultralytics import YOLO


class CrowdDetector:
    def __init__(self, model_path='yolov8s.pt', imgsz=1280, conf=0.1):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf

    def detect(self, frame):
        """
        입력: BGR numpy array (H, W, 3)
        출력: list[dict] — [{"x1","y1","x2","y2","conf"}, ...] 리스트 내부에 딕셔너리 추가.
        """
        #results_debug = self.model(frame, classes=0, imgsz=self.imgsz, conf=self.conf, verbose=False)
        #all_confs = [box.conf[0].item() for box in results_debug[0].boxes]
        # if all_confs:
        #     print(f"    [DEBUG] 원시 탐지 {len(all_confs)}건 | "
        #           f"conf 범위: {min(all_confs):.3f} ~ {max(all_confs):.3f}")
        # else:
        #     print(f"    [DEBUG] conf=0.01에서도 탐지 0건 → 영상 문제 가능성")

        # 제안서에 imgsz=1280을 명시한 이유가 있습니다. Tello 720p 영상에서 사람이 37pixel 수준의 소형 객체이므로, 기본값 640으로는 탐지율이 급격히 떨어집니다.
        # 단, 테스트 단계에서는 640으로 두고 속도를 우선 확보한 뒤, Fine-tuning 후 1280으로 올리는 전략도 괜찮습니다.
        results = self.model(frame, classes=0, imgsz=self.imgsz, conf=self.conf,verbose=False)
        detections = [] # 1. 빈 껍데기 리스트 [ ] 제작
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            # ... (좌표 추출) ...
            # 2. 리스트 안에 딕셔너리 { } 를 하나씩 추가
            detections.append({
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
                "conf": conf
            })
        return detections, results[0] # 3. 완성된 [ {}, {}, {} ] 리턴
