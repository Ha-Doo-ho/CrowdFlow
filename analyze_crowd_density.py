import numpy as np
def analyze_crowd_density(img):
    # 앞선 test_perspective_transform.py에서 사람들의 픽셀 위치를 실제 지면 좌표인 (X,Y) 미터(m)단위로 변환하는 데 성공했습니다.
    # 이제 이데이터와 YOLO가 뱉어낸 Confidence(확신도) 점수를 결합하여 위험 상황을 판별하는 알고리즘을 짤 차례
    # 이 과정을 NumPy 배열 연산만으로 처리한다.

    # 1. 맵 설정: 10m x 10m 구역을 2m 간격으로 쪼갬 (총 5x5 = 25게 격자)
    GRID_SIZE = 2.0

    # 2. YOLO 및 호모그래피 변환을 통해 얻어낸 가상의 데이터
    # 데이터 구조: [X좌표(m), Y좌표(m), Confidence 점수]
    detections = np.array([
        [1.5, 1.2, 0.88],  # 저밀집 구역 --> 안전하게 잘 탐지됨 Conf 높음
        [1.8, 1.5, 0.90],  # 안전
        [7.5, 7.2, 0.35],  # 고밀집 구역 --> 사람들이 엉켜있어 AI가 헷갈려함! Conf 낮음
        [7.6, 7.5, 0.31],  # 고밀집 구역
        [7.4, 7.3, 0.28]  # 고밀집 구역
    ])

    # 3. 각 격자별 데이터를 저장할 딕셔너리 생성
    # key: (격자_X_인덱스, 격자_Y_인덱스), value: [Confidence 점수들]
    grid_data = {}

    # 4. 탐지된 사람들을 각각의 2m x 2m 격자 방(Cell)에 배정
    for person in detections:
        x, y, conf = person
        grid_x_idx, grid_y_idx = int(x // GRID_SIZE), int(y // GRID_SIZE)

        cell_key = (grid_x_idx, grid_y_idx)
        if cell_key not in grid_data:
            grid_data[cell_key] = []
        grid_data[cell_key].append(conf)

    # 5. [핵심 창의성] 각 격자별 위험도 판정
    CONFIDENCE_THRESHOLD = 0.4 # 이 이하로 떨어지면 겹침 발생(위험)으로 간주

    print("=== CrowdFlow 밀집도 분석 결과 ===")
    for cell, conf_list in grid_data.items():
        count = len(conf_list)
        avg_conf = sum(conf_list) / count

        # 격자 좌표를 미터 단위로 표시
        area_str = f"[X:{cell[0] * 2} ~ {cell[0] * 2 + 2}m, Y:{cell[1] * 2} ~ {cell[1] * 2 + 2}m]"

        # 역발상 로직 적용: 탐지된 인원수가 적더라도, 평균 Confidence가 낮으면 위험 경고!
        if avg_conf < CONFIDENCE_THRESHOLD:
            print(f"🚨 {area_str}: [경고! 고밀집 추정 구역] AI 신뢰도 급락 ({avg_conf:.2f}) -> 겹침(Occlusion) 발생!")
        else:
            print(f"✅ {area_str}: 안전 구역 - 인원수 {count}명 (평균 신뢰도 {avg_conf:.2f})")
if __name__ == "__main__":
    analyze_crowd_density()