from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# 모델과 스케일러 로드
model = tf.keras.models.load_model('/home/tone/models/final_model.keras')
scaler = joblib.load('/home/tone/models/scaler.pkl')

def preprocess_input(hr, resp, spo2, temp):
    input_array = np.array([[hr, resp, spo2, temp]])
    input_scaled = scaler.transform(input_array)
    
    # 입력 데이터의 차원을 LSTM이 기대하는 3차원으로 변경( 1, time_steps, features)
    
    input_scaled = np.expand_dims(input_scaled, axis=1) # (1, 1, 4)로 변환
    input_scaled = np.repeat(input_scaled, 5, axis=1) # ( 1, 5, 4)로 변환하여 time_steps을 맞춤
    return input_scaled

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    hr = float(request.form['hr'])
    resp = float(request.form['resp'])
    spo2 = float(request.form['spo2'])
    temp = float(request.form['temp'])

    processed_data = preprocess_input(hr, resp, spo2, temp)
    prediction = model.predict(processed_data)
    prediction_class = int(prediction[0][0] > 0.5)  # 0: Normal, 1: Abnormal

    return jsonify({'prediction': prediction_class})

if __name__ == "__main__":
    app.run(debug=True)
