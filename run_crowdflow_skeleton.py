import cv2 #드론 영상 프레임을 수신하고 화면에 띄우는 역할을 합니다.
from ultralytics import YOLO #YOLOv8 모델을 사용하기 위한 라이브러리를 불러옵니다.

def run_crowdflow_skeleton():
    #모델 로드 : 제안서 상세 설계에 맞춰 yolov8L 모델 다운로드 및 로드
    print("Running CrowdFlow skeleton")
    model = YOLO('yolov8l.pt') #정확성과 속도를 조절한 large모델 가중치 파일(.pt)를 메모리에 로드함. 사용할 모델은 yolov8l이다.

    # 2. 비디오 소스 연결
    cap = cv2.VideoCapture('test_video.mp4') #비디오 소스와 연결하는 객체 cap을 생성함. 괄호내부에 0을 넣으면 내 컴퓨터의 웹캠.

    if not cap.isOpened(): #웹캠이나 드론 등 비디오 소스가 정상적으로 연결되지 않았을 경우 프로그램이 뻗지 않도록 예외처리.
        raise IOError("Cannot open webcam")

    print("관제 시스템 프로토타입 실행중 ...(종료 'q'키)")

    while True: #영상 처리를 위한 무한루프. 동영상은 연속된 정지 이미지(프레임)의 모음. 루프를 돌며 이미지를 계속 뽑아냄.
        success, frame = cap.read() # 비디오 소스에서 프레임(이미지) 1장을 읽어옴. 그 프레임 한장을 읽는데, 성공 여부는 success(True/False), 실제 이미지 데이터가 NumPy배열 형태인 frame에 저장
        if not success: # 더 이상 읽어올 프레임이 없거나 끊기면 루프를 탈출한다.
            break

        # 3. 객체 탐지 추론: classes=0 옵션으로 '사람'만 탐지
        # 상세 설계 반영: 원래 해상도를 유지하려면 imgsz 인자를 활용할 수 있다.
        results = model(frame, classes=0) # 읽어온 이미지(프레임)를 YOLO모델에 통과시켜 추론한다. 이때 classes=0옵션을 주어 COCO데이터셋 중 '사람'만 탐지하도록 강제함.
                                          # result에는 탐지된 객체의 좌표[0]와 Confidence 점수[1]가 results 변수에 담김.

        # 4.결과 시각화: Ultralytics 내장 함수 plot()을 사용하여 바운딩 박스와 Confidence를 프레임에 그림.
        annotated_frame = results[0].plot() #results에 담긴 좌표를 바탕으로 원본 이미지 위에 네모 박스와 라벨을 그려 넣은 새로운 이미지(annotated_frame)를 제작

        # ---------------------------------------------------------
        # [과제 핵심 아이디어 테스트 공간]
        # 향후 이 부분에 결과 객체(results[0].boxes.conf)에서 Confidence 값들을
        # 추출하여 평균을 내고 밀집도를 판정하는 로직이 추가될 것입니다.
        # ---------------------------------------------------------

        # 5. 화면 출력
        cv2.imshow("CrowdFlow AI Detection", annotated_frame) #박스가 그려진 최종 이미지를 "CrowdFlow AI Detection"이라는 이름의 윈도우 창에 띄웁니다.

        # 'q'키를 누르면 루프 탈출 및 시스템 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

#이 스크립트가 직접 실행될 때만 run_crowdflow_skeleton() 함수를 호출
if __name__ == '__main__':
    run_crowdflow_skeleton()