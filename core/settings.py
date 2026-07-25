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
    boundary_margin_m: float = 0.05
    low_confidence_threshold: float = 0.4
    low_confidence_min_level: int | None = 3
    calibration_path: str | None = None

    cross_class_dedup_enabled: bool = True
    dedup_iou_threshold: float = 0.65
    boundary_stabilization_enabled: bool = True
    boundary_history_size: int = 3
    boundary_majority_count: int = 2
    boundary_stability_band_m: float = 0.5
    boundary_track_max_missing: int = 2

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
            settings = cls()
            settings.validate()
            return settings

        with config_path.open("r", encoding="utf-8") as config_file:
            data = json.load(config_file)

        settings = cls(**data)
        settings.validate()
        return settings

    def validate(self):
        if self.model_type not in {"auto", "yolo", "rtdetr"}:
            raise ValueError("model_type must be auto, yolo, or rtdetr.")
        if self.imgsz <= 0:
            raise ValueError("imgsz must be greater than 0.")
        if not 0 <= self.detection_confidence <= 1:
            raise ValueError(
                "detection_confidence must be in the range 0~1."
            )
        if self.analyze_every <= 0:
            raise ValueError("analyze_every must be at least 1.")
        if (
            self.grid_size <= 0
            or self.area_width <= 0
            or self.area_height <= 0
        ):
            raise ValueError("grid and area sizes must be greater than 0.")
        if self.boundary_margin_m < 0:
            raise ValueError(
                "boundary_margin_m must be greater than or equal to 0."
            )
        if not 0 <= self.low_confidence_threshold <= 1:
            raise ValueError(
                "low_confidence_threshold must be in the range 0~1."
            )
        if self.low_confidence_min_level is not None:
            if not 1 <= self.low_confidence_min_level <= 5:
                raise ValueError(
                    "low_confidence_min_level must be 1~5 or null."
                )

        if not 0 <= self.dedup_iou_threshold <= 1:
            raise ValueError(
                "dedup_iou_threshold must be in the range 0~1."
            )
        if self.boundary_history_size < 1:
            raise ValueError("boundary_history_size must be at least 1.")
        if not (
            1
            <= self.boundary_majority_count
            <= self.boundary_history_size
        ):
            raise ValueError(
                "boundary_majority_count must be within "
                "boundary_history_size."
            )
        if self.boundary_stability_band_m < 0:
            raise ValueError(
                "boundary_stability_band_m must be non-negative."
            )
        if self.boundary_track_max_missing < 0:
            raise ValueError(
                "boundary_track_max_missing must be non-negative."
            )

        if self.prediction_horizon_seconds <= 0:
            raise ValueError(
                "prediction_horizon_seconds must be greater than 0."
            )
        if self.prediction_history_seconds <= 0:
            raise ValueError(
                "prediction_history_seconds must be greater than 0."
            )
        if self.prediction_min_history_seconds <= 0:
            raise ValueError(
                "prediction_min_history_seconds must be greater than 0."
            )
        if (
            self.prediction_min_history_seconds
            > self.prediction_history_seconds
        ):
            raise ValueError(
                "prediction_min_history_seconds must not exceed "
                "prediction_history_seconds."
            )
        if self.cumulative_risk_half_life_seconds <= 0:
            raise ValueError(
                "cumulative_risk_half_life_seconds must be greater than 0."
            )
        if not 0 <= self.cumulative_risk_threshold <= 100:
            raise ValueError(
                "cumulative_risk_threshold must be in the range 0~100."
            )

        if self.person_classes is not None:
            if not all(
                isinstance(class_id, int) and class_id >= 0
                for class_id in self.person_classes
            ):
                raise ValueError(
                    "person_classes must contain non-negative integers."
                )
