import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppSettings:
    source: str | int = "data/test_video2.mp4"
    model_path: str = "weights/yolo11l_crowdflow.pt"
    model_type: str = "auto"
    imgsz: int = 1280
    detection_confidence: float = 0.1
    person_classes: list[int] | None = None
    analyze_every: int = 3
    grid_size: float = 2.0
    area_width: float = 10.0
    area_height: float = 10.0
    low_confidence_threshold: float = 0.4
    low_confidence_min_level: int | None = 3
    calibration_path: str | None = None
    predictive_risk_enabled: bool = True
    prediction_horizon_seconds: float = 10.0
    prediction_history_seconds: float = 15.0
    prediction_min_history_seconds: float = 3.0
    cumulative_risk_half_life_seconds: float = 30.0
    cumulative_risk_threshold: float = 30.0

    @classmethod
    def load(cls, path):
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with config_path.open("r", encoding="utf-8") as config_file:
            data = json.load(config_file)

        settings = cls(**data)
        settings.validate()
        return settings

    def validate(self):
        if self.model_type not in {"auto", "yolo", "rtdetr"}:
            raise ValueError("model_type은 auto, yolo, rtdetr 중 하나여야 합니다.")
        if self.imgsz <= 0:
            raise ValueError("imgsz는 0보다 커야 합니다.")
        if not 0 <= self.detection_confidence <= 1:
            raise ValueError("detection_confidence는 0~1 범위여야 합니다.")
        if self.analyze_every <= 0:
            raise ValueError("analyze_every는 1 이상이어야 합니다.")
        if self.grid_size <= 0 or self.area_width <= 0 or self.area_height <= 0:
            raise ValueError("격자와 영역 크기는 0보다 커야 합니다.")
        if not 0 <= self.low_confidence_threshold <= 1:
            raise ValueError("low_confidence_threshold는 0~1 범위여야 합니다.")
        if self.low_confidence_min_level is not None:
            if not 1 <= self.low_confidence_min_level <= 5:
                raise ValueError("low_confidence_min_level은 1~5 또는 null이어야 합니다.")
        if self.prediction_horizon_seconds <= 0:
            raise ValueError("prediction_horizon_seconds는 0보다 커야 합니다.")
        if self.prediction_history_seconds <= 0:
            raise ValueError("prediction_history_seconds는 0보다 커야 합니다.")
        if self.prediction_min_history_seconds <= 0:
            raise ValueError("prediction_min_history_seconds는 0보다 커야 합니다.")
        if self.prediction_min_history_seconds > self.prediction_history_seconds:
            raise ValueError("prediction_min_history_seconds는 prediction_history_seconds보다 작거나 같아야 합니다.")
        if self.cumulative_risk_half_life_seconds <= 0:
            raise ValueError("cumulative_risk_half_life_seconds는 0보다 커야 합니다.")
        if not 0 <= self.cumulative_risk_threshold <= 100:
            raise ValueError("cumulative_risk_threshold는 0~100 범위여야 합니다.")

        if self.person_classes is not None:
            if not all(isinstance(class_id, int) and class_id >= 0
                       for class_id in self.person_classes):
                raise ValueError("person_classes에는 0 이상의 정수만 사용할 수 있습니다.")
