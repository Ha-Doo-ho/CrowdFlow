# detector.py (홍유택 담당)
from ultralytics import RTDETR
from ultralytics import YOLO

class CrowdDetector:
    HUMAN_CLASS_NAMES = {"person", "pedestrian", "people"}

    def __init__(self, model_path='yolov8s.pt', imgsz=1280, conf=0.1, person_classes=None):
        """
        기존 self.model = RTDETR(model_path)를 삭제함.
        근거: if로 모델을 고르기 전에 이미 무조건 한 번 RTDETR(model_path)를 실행한다는 점에 있다.
        즉 main.py (line 21)에서 이렇게 YOLO 계열 가중치를 넘기고 있어도, CrowdDetector는 self.model = RTDETR(model_path)를 실행함.
        """
        self.imgsz = imgsz #해상도
        self.conf = conf #신뢰도 임계값 (0.1) --> 0.1보다 낮으면 정말 사람이 많아서 탐지를 못한 것으로 결정

        if 'rtdetr' in model_path.lower():
            self.model = RTDETR(model_path)
        else:
            self.model = YOLO(model_path)
        self.person_classes = self._resolve_person_classes(person_classes)

    def _resolve_person_classes(self, person_classes):
        #일반 YOLO는 0번이 사람. VisDrone데이터셋은 0번은 보행자, 1번이 사람. 무조건 0번이 사람이라는 위험을 막기 위한 방어로직.
        if person_classes is not None:
            if isinstance(person_classes, int): #사용자가 0처럼 숫자 하나만(int)만 넘겼다면
                return [person_classes] #에러가 나지 않게 [0]처럼 리스트로 포장함.
            return list(person_classes) #[0, 1] 리스트로 잘 넘겼다면, 원본 그대로 돌려준다.

        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            name_items = names.items()
        else:
            name_items = enumerate(names)

        matched = [
            int(class_id)
            for class_id, class_name in name_items
            if str(class_name).lower().strip() in self.HUMAN_CLASS_NAMES
        ]
        return matched or [0]

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
        predict_args = { #predict_args: 추론할 때 넣을 옵션들을 딕셔너리로 묶음. predict_args 문법을 사용하여 옵션들을 한 번에 풀어헤쳐서 모델에 던짐.
            "imgsz": self.imgsz,
            "conf": self.conf,
            "verbose": False,
        }
        if self.person_classes:
            predict_args["classes"] = self.person_classes

        results = self.model(frame, **predict_args)
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
