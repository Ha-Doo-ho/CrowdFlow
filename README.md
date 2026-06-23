# CrowdFlow

드론 조감 영상에서 사람을 탐지하고, 지면 좌표로 변환해 격자별 밀집도와
위험 수준을 계산하는 관제 보조 시스템입니다.

## 처리 흐름

```text
영상/Tello -> YOLO11 또는 RT-DETR -> 사람 bbox
-> Homography -> 2m x 2m 격자 -> JSON/SQLite -> Streamlit
```

현재 기본 설정은 다음과 같습니다.

- 입력: `data/test_video2.mp4`
- 모델: `weights/yolo11l_crowdflow.pt`
- 추론 크기: `1280`
- 탐지 confidence: `0.1`
- 작업 영역: `10m x 10m`
- 격자: `2m x 2m`
- 분석 주기: 3프레임마다 1회

모든 실행값은 [config/app_config.json](config/app_config.json)에서 변경합니다.

## 설치

Python 3.10~3.12 환경을 권장합니다.

```powershell
python -m pip install -r requirements.txt
```

## 실행

분석 프로그램:

```powershell
python main.py
```

대시보드:

```powershell
streamlit run frontend/dashboard.py
```

Tello 영상 연결만 확인:

```powershell
python tello_video_test.py
```

## 모델 속도 비교

기존 테스트 영상에서 YOLO11과 RT-DETR을 같은 조건으로 비교할 수 있습니다.

```powershell
python benchmark_models.py `
  --video data/test_video2.mp4 `
  --model yolo:weights/yolo11l_crowdflow.pt `
  --model rtdetr:weights/rtdetr_l.pt `
  --imgsz 1280 `
  --confidence 0.1 `
  --stride 10 `
  --max-frames 100
```

결과는 `data/model_benchmark.json`에 저장됩니다. 이 비교는 추론시간,
추론 FPS, 탐지량을 비교할 뿐 정확도 평가는 아닙니다. 정확도는 실제
인원수 또는 bbox ground truth가 있는 검증셋으로 별도 평가해야 합니다.

## 모델 변경

`config/app_config.json`에서 모델 경로와 유형을 함께 변경합니다.

```json
{
  "model_path": "weights/rtdetr_l.pt",
  "model_type": "rtdetr"
}
```

VisDrone 기반 커스텀 모델은 일반적으로 `pedestrian`, `people` 클래스를
사용합니다. `person_classes`가 `null`이면 모델의 클래스 이름에서
`person`, `pedestrian`, `people`을 자동으로 찾습니다.

## 캘리브레이션 상태

현재 `calibration_path`가 `null`이므로 영상 전체를 10m x 10m로 매핑하는
개발용 fallback을 사용합니다. 이 결과는 실제 밀집도 검증값으로 해석하면
안 됩니다.

실측 좌표를 확보한 뒤 `config/calibration.example.json`을 복사하여 실제
픽셀/지면 좌표로 수정하고 다음처럼 지정합니다.

```json
{
  "calibration_path": "config/calibration.json"
}
```

점 순서는 양쪽에서 동일해야 합니다.

```text
좌상 -> 우상 -> 좌하 -> 우하
```

600 x 400 캘리브레이션 보드는 렌즈 왜곡과 카메라 내부 파라미터 측정에
사용할 수 있습니다. 전체 지면 Homography에는 별도의 지면 기준점 4개가
필요합니다.

## 낮은 confidence 경고

평균 confidence가 `0.4` 미만인 셀은 관측 신뢰도 저하 경고를 생성합니다.
이는 고밀집을 확정하는 기준이 아닙니다. 가림, 거리, 흔들림, 조명 등도
confidence를 낮출 수 있으므로 실제 실험으로 검증해야 합니다.

기존 동작과의 호환을 위해 기본 설정은 해당 셀을 최소 Level 3으로
표시합니다. 상향을 끄려면 다음처럼 설정합니다.

```json
{
  "low_confidence_min_level": null
}
```

## 테스트

```powershell
pytest -q
```

테스트 범위:

- 설정 파일 검증
- NumPy 결과의 JSON 저장
- Homography 미설정 방지
- 10m 경계의 마지막 셀 포함
- 영역 밖 탐지 집계
- 격자 밀집도/Level 계산
- SQLite 진단 데이터 및 실험 MAE/RMSE

## 현재 남은 핵심 작업

1. 실제 지면 기준점으로 Homography 측정
2. Tello 영상에서 YOLO11/RT-DETR 성능 비교
3. 실제 인원수 기반 반복 실험
4. MAE/RMSE와 실패 사례 정리
5. 낮은 confidence와 실제 혼잡도의 상관관계 검증
