import cv2
from flask import Flask, request, jsonify, Response, render_template
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# YOLOv8 모델 로드
model_path = os.path.join('C:/Users/ksanz/Desktop/YOLO', 'best.pt')
model = YOLO(model_path)

# 내장 카메라 사용
cap = cv2.VideoCapture(0)

# 임계값 설정
threshold = 0.75

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read()))

    results = model(img)

    predictions = []
    for result in results:
        boxes = result.boxes.xyxy
        classes = result.boxes.cls
        names = result.names

        for i, box in enumerate(boxes):
            predictions.append({
                'xmin': box[0].item(),
                'ymin': box[1].item(),
                'xmax': box[2].item(),
                'ymax': box[3].item(),
                'name': names[int(classes[i].item())]
            })

    return jsonify(predictions)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            continue
        
        results = model.predict(frame)

        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes.xyxy.numpy()
            scores = result.boxes.conf.numpy()
            class_ids = result.boxes.cls.numpy()
            labels = model.names
            
            for box, score, class_id in zip(boxes, scores, class_ids):
                if score >= threshold:
                    x1, y1, x2, y2 = map(int, box)
                    label = labels[int(class_id)]
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(annotated_frame, f'{label} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            print("Failed to encode frame")
            continue
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index4.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
