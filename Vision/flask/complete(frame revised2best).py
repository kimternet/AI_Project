import cv2
from ultralytics import YOLO
import pygame
import os
import time

# 학습된 모델 로드
model = YOLO('D:/Temp/YOLO4/best.pt')

# 내장 카메라 사용
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# 임계값 설정
threshold = 0.7

# pygame 초기화
pygame.mixer.init()

# 음성 파일 경로 설정
voice_path = 'D:/Temp/YOLO4/voice'

# 마지막으로 감지된 시간과 레이블 저장
last_detected = None

# 음성을 재생할 최소 시간 간격 (초)
cooldown_time = 3

# 마지막으로 음성이 재생된 시간
last_played_time = 0

# 첫 번째 프레임인지 확인
first_frame = True

# 감지된 레이블에 따른 음성 파일 재생 함수
def play_voice(filename):
    file_path = os.path.join(voice_path, filename)
    if os.path.exists(file_path):
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pass

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    # YOLO 모델로 예측 수행
    results = model.predict(frame)

    # 예측 결과를 프레임에 표시
    annotated_frame = frame.copy()
    detected_labels = set()

    for result in results:
        boxes = result.boxes.xyxy.numpy()  # 예측된 bounding box 좌표
        scores = result.boxes.conf.numpy()  # 예측된 confidence scores
        class_ids = result.boxes.cls.numpy()  # 예측된 클래스 ID
        labels = model.names  # 클래스 이름 (result.names가 아닌 model.names로 접근)

        for box, score, class_id in zip(boxes, scores, class_ids):
            if score >= threshold:
                x1, y1, x2, y2 = map(int, box)
                label = labels[int(class_id)]
                detected_labels.add(label)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated_frame, f'{label} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Am 코드만 감지 현재 시간
    current_time = time.time()
    
    if first_frame:
        first_frame = False
        last_detected = 'Am' if 'Am' in detected_labels else 'not_Am'
        time.sleep(1) # 1초 지연
    else:
        if 'Am' in detected_labels:
            # 'Am'가 감지되었고, 마지막으로 감지된 것이 'Am'가 아니었을 때만 음성 재생
            if last_detected != 'Am' and current_time - last_played_time > cooldown_time:
                play_voice("Amvoice.mp3")
                last_played_time = current_time
                last_detected = 'Am'
        else:
            # Am가 감지되지 않았고, 마지막으로 감지된 것이 'Am'일 때만 음성 재생
            if last_detected != 'not_Am' and current_time - last_played_time > cooldown_time:
                play_voice('Am_fingering.mp3')
                last_played_time = current_time
                last_detected = 'not_Am'

    # 프레임 표시
    cv2.imshow('YOLO Detection', annotated_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()