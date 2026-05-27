# main.py (하두호 담당 — 전체 연결)
import os
import cv2
import numpy as np
import json
from frontend.data_logger import DataLogger
from core.detector import CrowdDetector
from hardware.drone_controller import DroneCamera
from core.grid_calculator import GridCalculator

# 프로젝트 루트 경로 (어디서 실행하든 정확한 경로 보장)
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
LATEST_RESULT_PATH = os.path.join(DATA_DIR, 'latest_result.json')
DB_PATH = os.path.join(DATA_DIR, 'crowdflow.db')


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def save_latest_result(grid_result, json_path=LATEST_RESULT_PATH):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    tmp_path = f"{json_path}.tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(_to_jsonable(grid_result), f, ensure_ascii=False)
    os.replace(tmp_path, json_path)


def publish_result(grid_result, logger):
    save_latest_result(grid_result) #data/latest_result.json 저장
    logger.log_frame(grid_result)   #data/crowdFlow.db (SQLite 기록)
#대시보드가 최신 결과를 읽을 수 있음.
#실험 후 DB 기반 평가 가능.

def main():
    # 1. 모듈 초기화
    video_path = os.path.join(ROOT, 'data/test_video2.mp4')
    model_path = os.path.join(ROOT, 'weights', 'yolo11l_crowdflow.pt')

    drone = DroneCamera(video_path)

    # Tello 실시간 모드 (드론 가져오면 이것만 바꾸면 됨)
    # drone = DroneCamera('tello')

    # 웹캠 모드 (테스트용)
    # drone = DroneCamera(0)

    logger = None

    try:
        detector = CrowdDetector(model_path)
        calculator = GridCalculator()
        logger = DataLogger(DB_PATH)

        # 임시 Homography: 영상 전체를 10m×10m로 매핑
        test_frame = drone.get_frame()
        if test_frame is None:
            print("첫 프레임을 가져오지 못했습니다. 영상 경로나 스트리밍 상태를 확인하세요.")
            return

        h, w = test_frame.shape[:2]
        print(f"영상 해상도: {w} x {h}")

        # 2. (선택) 캘리브레이션 — 실제 콘 좌표 확보 후 교체
        src = np.array([[0,0],[w,0],[0,h],[w,h]], dtype=np.float32)
        dst = np.array([[0,0],[10,0],[0,10],[10,10]], dtype=np.float32)
        calculator.set_homography(src, dst)

        print("=" * 50)
        print("CrowdFlow 관제 시스템 실행중... (종료: q키)")
        print("=" * 50)

        frame_count = 1 #첫 프레임은 이미 읽었으므로 1부터
        ANALYZE_EVERY = 3 # 3프레임당 1회 분석 (GPU 부하 관리)

        #첫 프레임 분석
        detections, raw_result = detector.detect(test_frame)
        grid_result = calculator.calculate(detections)
        publish_result(grid_result, logger)
        total = int(grid_result['count'].sum())
        max_d = grid_result['max_density']
        print(f"[Frame {frame_count:>5}] 탐지: {total:>3}명 | "
              f"최대 밀집도: {max_d:.2f}인/m²")
        annotated = raw_result.plot()
        cv2.imshow("CrowdFlow AI Detection", annotated)

        while True:
            #1. 드론에게 지금 사진 1장만 달라고 요청 (손성 모듈 호출)
            frame = drone.get_frame()
            if frame is None:
                break

            frame_count += 1

            if frame_count % ANALYZE_EVERY == 0:
                # 2. AI 분석 수행
                # 유택이 훈련시킨 모델을 통해 사진에서 사람을 찾음.
                detections, raw_result = detector.detect(frame)
                grid_result = calculator.calculate(detections) # 3. 내가 만든 모듈을 호출. grid_calculator.py로 Grid로 위험도 호출
                publish_result(grid_result, logger)

                # 콘솔 출력 (개발 중 확인용)
                total = int(grid_result['count'].sum())
                max_d = grid_result['max_density']
                print(f"[Frame {frame_count:>5}] "
                      f"탐지: {total:>3}명 | "
                      f"최대 밀집도: {max_d:.2f}인/m²", end="")

                if grid_result['alerts']:
                    print(f" | ⚠️ 경고 {len(grid_result['alerts'])}건")
                    for alert in grid_result['alerts']:
                        print(f"    → {alert['message']}")
                else:
                    print()  # 줄바꿈

            last_annotated = raw_result.plot()

            cv2.imshow("CrowdFlow AI Detection", last_annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        drone.release()
        if logger is not None:
            logger.close()
        cv2.destroyAllWindows()
        print("\n시스템 종료")


if __name__ == '__main__':
    main()
