# CrowdFlow

드론 영상에서 사람을 탐지하고, Homography로 실측 좌표에 투영한 뒤 격자별
밀집도, 위험 군집, 10초 예측 위험을 계산해 OpenCV와 Streamlit으로 표시하는
캡스톤디자인 프로젝트입니다.

## 최종 재현 베이스라인

2026-08-03 기준 최종 실험 베이스라인은 아래 한 가지입니다.

| 항목 | 동결 값 |
|---|---|
| 베이스라인 ID | `crowdflow-7_85x10-yolo11-20260803` |
| 기본 설정 | `config/app_config.json` |
| 동일 별칭 | `config/app_config_7_85x10.json` |
| 실측 영역 | 가로 `7.85m` x 세로 `10.0m` |
| 캘리브레이션 | `config/calibration_7_85x10.json` |
| 기본 모델 | `weights/yolo11l_crowdflow.pt` |
| 기본 대표 영상 | `tello_flight_recording_20260711_164845.mp4` |
| 격자 | `1.0m` |
| 경계 여유 | `0.15m` |
| 탐지 신뢰도 | `0.35` |
| 분석 주기 | 30프레임마다 1회, 30 FPS 영상에서 약 1초 간격 |
| 중복 제거 | 클래스 간 IoU `0.65` 이상이면 높은 신뢰도 박스 유지 |
| 경계 안정화 | 최근 3회 중 2회 다수결, 경계 밴드 `0.5m` |
| 예측 | 최근 이력 기반 10초 선형 예측, 실측 예측 정확도와는 구분 |

기계 판독용 전체 정보와 SHA-256은
[`config/baseline_manifest.json`](config/baseline_manifest.json)에 있습니다.
`calibration_7_85x10_001.json`처럼 번호가 붙은 파일은 과거 후보이며, 최종 실행에는
번호가 없는 기준 파일만 사용합니다.

## 처리 흐름

```text
Tello/MP4 영상
-> YOLO11 또는 RT-DETR 사람 탐지
-> pedestrian/people 클래스 간 중복 제거
-> bbox 발끝점 Homography 변환
-> 측정 영역 IN/OUT 및 경계 안정화
-> 1m 격자별 인원/밀집도
-> 위험 셀 군집 및 10초 예측
-> latest_result.json + SQLite
-> OpenCV 디버그 화면 + Streamlit 대시보드
```

## 1. 설치

검증된 환경은 Python `3.13.12`입니다. 새 가상환경을 만든 뒤 아래 명령을
사용합니다.

```powershell
python -m pip install -r requirements-baseline.txt
```

최소 버전 범위로 설치하려면 `requirements.txt`를 사용할 수 있지만, 팀 간 결과
비교에는 고정 버전 파일을 권장합니다.

## 2. Git에 없는 대용량 파일 배치

모델 `*.pt`와 영상 `*.mp4`는 Git에 포함되지 않습니다. 팀 공유 저장소에서 받은
파일을 원래 이름 그대로 다음 위치에 둡니다.

```text
weights/yolo11l_crowdflow.pt
weights/rtdetr_l.pt
data/tello_recordings/tello_flight_recording_20260711_123722.mp4
data/tello_recordings/tello_flight_recording_20260711_155226.mp4
data/tello_recordings/tello_flight_recording_20260711_164845.mp4
```

파일 크기와 SHA-256은 [`weights/README.md`](weights/README.md)와
[`data/tello_recordings/README.md`](data/tello_recordings/README.md)를 확인합니다.

## 3. 베이스라인 검증

최초 전달 후에는 모델과 영상까지 SHA-256을 전부 확인합니다. 큰 파일을 읽으므로
약 1분 정도 걸릴 수 있습니다.

```powershell
python verify_baseline.py
```

이미 검증한 컴퓨터에서 설정·경로만 빠르게 확인할 때는 다음을 사용합니다.

```powershell
python verify_baseline.py --skip-hash
```

`[PASS] crowdflow-7_85x10-yolo11-20260803`가 출력되어야 합니다.

## 4. 기본 실행

PowerShell 1에서 대시보드를 실행합니다.

```powershell
streamlit run frontend/dashboard.py
```

PowerShell 2에서 기본 대표 영상을 분석합니다.

```powershell
python main.py
```

위 명령은 다음 명령과 동일합니다.

```powershell
python main.py --config config/app_config_7_85x10.json
```

OpenCV 창에서 `q`를 누르면 분석이 종료됩니다. 대시보드는 `main.py`가 갱신하는
`data/latest_result.json`을 표시합니다.

## 5. 대표 영상 세트

설정과 캘리브레이션은 바꾸지 않고 `--source`만 바꿔 동일 조건으로 비교합니다.

| 영상 | 검증 목적 | 기존 관찰 |
|---|---|---|
| `123722` | 중앙·상하단·영역 밖 단일 인원 IN/OUT | 중앙/하단/영역 밖 성공, 상단은 흔들림 영향으로 부분 성공 |
| `155226` | 3명 분산·동일 격자·2 IN + 1 OUT | 분산 배치와 2 IN + 1 OUT 성공, 동일 격자 밀집도는 부분 성공 |
| `164845` | 2명에서 3명 변화, 모델 및 후처리 회귀 비교 | YOLO11 기본 대표 영상, RT-DETR 반응 속도 비교에 사용 |

단일 인원 위치 검증:

```powershell
python main.py --config config/app_config.json `
  --source data/tello_recordings/tello_flight_recording_20260711_123722.mp4
```

3명 밀집 검증:

```powershell
python main.py --config config/app_config.json `
  --source data/tello_recordings/tello_flight_recording_20260711_155226.mp4
```

인원 변화 및 후처리 회귀 검증:

```powershell
python main.py --config config/app_config.json `
  --source data/tello_recordings/tello_flight_recording_20260711_164845.mp4
```

각 실행에서 기록할 최소 항목은 영상명, 설정명, 캘리브레이션명, 프레임 번호,
실제 인원, `counted`, `ignored`, 최대 밀집도, 중복 제거 수, 경계 보정 수입니다.

## 6. 캘리브레이션 규칙

최종 기준점 클릭 순서는 다음과 같습니다.

```text
1. left_top
2. right_top
3. left_bottom
4. right_bottom
```

최종 원본 좌표는 `960x720` 프레임의 `(291,426)`, `(684,427)`, `(76,692)`,
`(890,700)`이며, 각각 `(0,0)`, `(7.85,0)`, `(0,10)`, `(7.85,10)`m로
매핑됩니다.

캘리브레이션 사진과 분석 영상은 같은 드론 위치, 고도, 카메라 각도에서 촬영해야
합니다. 현재 Homography는 프레임마다 자동으로 카메라 흔들림을 보정하지 않으므로,
큰 피치·요 변화가 있으면 경계에서 IN/OUT 오차가 발생할 수 있습니다.

새 촬영 조건을 시험할 때만 새 캘리브레이션을 생성하고, 최종 기준 파일을 바로
덮어쓰지 않습니다.

```powershell
python calibrate_from_image.py data/calibration_frames/<frame>.jpg `
  --area-width 7.85 --area-height 10 `
  --out config/calibration_7_85x10_candidate.json `
  --preview output/calibration_preview_7_85x10_candidate.jpg
```

## 7. 모델 비교

운영 기본 모델은 YOLO11입니다. RT-DETR은 인원 변화 반응을 비교하는 보조
모델이며, 동일 영상·해상도·confidence·stride 조건에서만 비교합니다.

```powershell
python benchmark_models.py `
  --video data/tello_recordings/tello_flight_recording_20260711_164845.mp4 `
  --model yolo:weights/yolo11l_crowdflow.pt `
  --model rtdetr:weights/rtdetr_l.pt `
  --imgsz 640 `
  --confidence 0.35 `
  --stride 30
```

## 8. 자동 테스트

```powershell
pytest -q
```

테스트는 설정 검증, Homography/격자 계산, 위험 군집, 예측 위험, JSON 발행,
클래스 간 중복 제거, 경계 상태 안정화를 포함합니다. 자동 테스트 통과는 실제
군중 정확도를 의미하지 않으며, 대표 영상 결과와 함께 해석해야 합니다.

## 9. 보조·과거 설정

- `config/app_config_indoor_1x3.json`: 실내 1m x 3m 시험 보존본
- `config/app_config_6x4.json`: 야외 4m x 6m 시험 보존본
- `config/app_config_10x10.json`: 초기 10m x 10m 후보, 최종 기준 아님

자세한 구분은 [`config/experiment_configs.md`](config/experiment_configs.md)를
참고합니다. 포스터와 최종 보고서의 정량 결과는 반드시 최종 7.85m x 10m
베이스라인 결과를 중심으로 작성합니다.

## 현재 한계

- Tello의 사선 카메라와 바람에 의한 자세 변화는 고정 Homography 오차를 만듭니다.
- 3명 세로 배열이나 밀집 배치에서는 가림으로 미탐이 발생할 수 있습니다.
- AI 합성 군중 영상은 스트레스 테스트용이며 실제 현장 정확도 근거로 사용하지 않습니다.
- 10초 예측은 증가 추세 표시 기능으로 검증되었으며, 실제 미래 예측 정확도는 추가
  시계열 실험이 필요합니다.
