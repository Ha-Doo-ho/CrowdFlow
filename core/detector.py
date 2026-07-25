from pathlib import Path

from ultralytics import RTDETR
from ultralytics import YOLO

from core.detection_postprocessor import suppress_cross_class_duplicates


class CrowdDetector:
    HUMAN_CLASS_NAMES = {"person", "pedestrian", "people"}

    def __init__(
        self,
        model_path="yolov8s.pt",
        imgsz=1280,
        conf=0.1,
        person_classes=None,
        model_type="auto",
        cross_class_dedup_enabled=True,
        dedup_iou_threshold=0.65,
    ):
        self.model_path = str(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.model_type = self._resolve_model_type(model_type)
        self.cross_class_dedup_enabled = bool(cross_class_dedup_enabled)
        self.dedup_iou_threshold = float(dedup_iou_threshold)
        if not 0 <= self.dedup_iou_threshold <= 1:
            raise ValueError("dedup_iou_threshold must be in the range 0~1.")

        self.last_postprocess_stats = {
            "raw_detection_count": 0,
            "deduplicated_detection_count": 0,
            "duplicate_boxes_removed": 0,
        }

        if self.model_type == "rtdetr":
            self.model = RTDETR(model_path)
        else:
            self.model = YOLO(model_path)

        self.person_classes = self._resolve_person_classes(person_classes)

    def _resolve_model_type(self, model_type):
        if model_type not in {"auto", "yolo", "rtdetr"}:
            raise ValueError("model_type must be auto, yolo, or rtdetr.")
        if model_type != "auto":
            return model_type

        model_name = Path(self.model_path).stem.lower()
        return "rtdetr" if "rtdetr" in model_name else "yolo"

    def _resolve_person_classes(self, person_classes):
        if person_classes is not None:
            if isinstance(person_classes, int):
                return [person_classes]
            return list(person_classes)

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

    def _class_name(self, class_id):
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def detect(self, frame):
        """
        Input: BGR numpy array with shape (H, W, 3).
        Output: filtered human detections and the original Ultralytics result.
        """
        if frame is None or frame.size == 0:
            raise ValueError("The detection frame is empty.")

        predict_args = {
            "imgsz": self.imgsz,
            "conf": self.conf,
            "verbose": False,
        }
        if self.person_classes:
            predict_args["classes"] = self.person_classes

        results = self.model(frame, **predict_args)
        raw_detections = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            raw_detections.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "conf": confidence,
                "class_id": class_id,
                "class_name": self._class_name(class_id),
            })

        if self.cross_class_dedup_enabled:
            detections, stats = suppress_cross_class_duplicates(
                raw_detections,
                iou_threshold=self.dedup_iou_threshold,
            )
        else:
            detections = raw_detections
            stats = {
                "raw_detection_count": len(raw_detections),
                "deduplicated_detection_count": len(raw_detections),
                "duplicate_boxes_removed": 0,
            }

        self.last_postprocess_stats = stats
        return detections, results[0]
