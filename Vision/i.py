from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2

app = Flask(__name__)

# YOLO 모델 로드
model = YOLO('C:/Users/3gkim/OneDrive/Desktop/YOLO/best.pt')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # YOLO 모델로 예측 수행
    results = model(img)

    predictions = []
    for result in results:
        for box in result.boxes:
            if box.conf.numpy()[0] >= 0.5:  # 임계값 설정 (0.5 이상인 경우에만)
                predictions.append({
                    'name': result.names[int(box.cls.numpy()[0])],
                    'xmin': int(box.xyxy.numpy()[0][0]),
                    'ymin': int(box.xyxy.numpy()[0][1]),
                    'xmax': int(box.xyxy.numpy()[0][2]),
                    'ymax': int(box.xyxy.numpy()[0][3]),
                    'confidence': float(box.conf.numpy()[0])
                })
    
    # 예측 결과를 출력하여 디버깅
    print(predictions)
    
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
