import cv2
import numpy as np

#Vision 처리 : 사람들의 픽셀 위치를 실제 지면 좌표인 (X, Y) 미터(m) 단위로 변환하는 코드
def test_perspective_transform():
    #1. 화면 픽셀 좌표 (드론 카메라로 찍은 라바콘 4개의 위치라고 가정)
    src_pts = np.array([
        [320, 480],
        [960, 480],
        [180, 600],
        [1100, 600]
    ], dtype=np.float32)

    #2. 실제 지면 좌표 (우리가 원하는 10m x 10m 정사각형 지도 상의 위치)
    dst_pts = np.array([
        [0, 0],  #0m,  0m
        [10, 0], #10m, 0m
        [0, 10], #0m, 10m
        [10, 10] #10m, 10m
    ], dtype=np.float32)

    # 3. 변환의 핵심 : Homography 행렬 H 산출
    # 이 행렬 하나만 구하면 이후 모든 사람의 좌표를 변환할 수 있다.
    # H는 당연하게도 호모그래피 행렬
    # mask: 강건한 방법을 사용할 때 입력된 점들 중 어떤점이 정상데이터이고 어떤 점이 이상치(아웃라이어)인지 표시해 주는 마스크
    H, mask = cv2.findHomography(src_pts, dst_pts)

    # 4. YOLO가 탐지한 사람의 '발끝' 좌표라고 가정 (픽셀)
    # u(가로픽셀):640, v(세로픽셀):550
    # OpenCV의 cv2.perspectiveTransform 함수는 점 한 개가 아니라, 수백 수천 개의 점을 한꺼번에 묶어서 변환하도록 설계되어 있다.
    # 그래서 데이터를 넘겨줄 때 (점의 전체 개수, 1, 2차원 좌표) 형태의 다차원 배열 구조로 겹겹이 포장해서 줘야 한다. 데이터를 포장하는 껍데기가 [[[]]]라고 생각하면 된다.
    person_pixel_coord = np.array([[[640, 550]]], dtype=np.float32)

    # 5. 픽셀 좌표를 실제 지면 좌표(m)으로 변환
    person_real_coord = cv2.perspectiveTransform(person_pixel_coord, H)