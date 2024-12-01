from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import pygame
import os
import time

app = Flask(__name__, template_folder='accessibility-web')

# 학습된 모델 로드
model = YOLO('D:/YOLO4/best.pt')

# 음성 파일 경로 설정
voice_path = 'D:/YOLO4/voice'

# 마지막으로 감지된 시간과 레이블 저장
last_played = {
    'Am': 0,
    'Not_Am': 0
}
# 음성을 재생할 최소 시간 간격 (초)
cooldown_time = 10

# pygame 초기화
pygame.mixer.init()

# 감지된 레이블에 따른 음성 파일 재생 함수
def play_voice(label):
    current_time = time.time()
    if current_time - last_played[label] > cooldown_time:
        voice_files = {
            'Am': 'Amvoice.mp3',
            'Not_Am': 'Am_fingering.mp3'  # Am가 아닌 경우에 재생할 음성 파일
        }
        if label in voice_files:
            file_path = os.path.join(voice_path, voice_files[label])
            if os.path.exists(file_path):
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                last_played[label] = current_time

@app.route('/')
@app.route('/index5')
def index5():
    return render_template('index5.html')

@app.route('/index6')
def index6():
    return render_template('index6.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Error: Could not open video stream."

    threshold = 0.6

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            results = model.predict(frame)
        except Exception as e:
            print(f"Prediction failed: {e}")
            break

        annotated_frame = frame.copy()
        detected_labels = set()

        for result in results:
            try:
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
            except Exception as e:
                print(f"Error processing results: {e}")

        if 'Am' in detected_labels:
            play_voice('Am')
        elif len(detected_labels) > 0:  # 다른 레이블이 감지된 경우
            play_voice('Not_Am')

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    pygame.mixer.quit()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
