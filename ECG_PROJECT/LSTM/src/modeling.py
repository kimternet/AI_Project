import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import classification_report

# 3.1 타겟 변수 생성 및 데이터 준비
def prepare_data_for_lstm(data, time_steps=5):
    # 라벨 인코딩
    label_encoder = LabelEncoder()
    data['OUTPUT'] = label_encoder.fit_transform(data['OUTPUT'])

    # 비정상 상태 데이터를 사용해 타겟 변수를 생성 (1: Abnormal, 0: Normal)
    data['TARGET'] = data['OUTPUT']

    # 입력 변수 선택
    features = ['HR(BPM)', 'RESP(BPM)', 'SpO2(%)', 'TEMP (*C)']

    # 시계열 데이터 준비 함수 정의
    def create_sequences(X, y, time_steps):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:i + time_steps])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)

    # X (생체 신호)와 y (타겟)를 정의
    X = data[features].values
    y = data['TARGET'].values

    # 시계열 데이터 생성
    X_seq, y_seq = create_sequences(X, y, time_steps)

    # 데이터셋을 훈련과 테스트로 분할
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

# 3.2 LSTM 모델 정의 및 학습
def build_and_train_lstm(X_train, y_train, X_test, y_test):
    # LSTM 모델 정의
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    # 모델 컴파일
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # EarlyStopping과 ModelCheckpoint 설정
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('../models/best_model.keras', save_best_only=True, monitor='val_loss', mode='min')

    # GPU 사용 여부 확인
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU is available and will be used for training.")
        print(f"Available GPU: {gpus}")
    else:
        print("GPU is not available. Training will use CPU.")

    # 모델 학습
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

    # 테스트 데이터로 모델 평가
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy:.4f}')

    return model

# 3.3 예측 및 성능 평가
def evaluate_model(model, X_test, y_test):
    # 예측 결과 분석
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print(classification_report(y_test, y_pred))

# 메인 함수
if __name__ == "__main__":
    from preprocessing import load_and_preprocess_data

    data_path = '../data/Human_vital_signs_R.csv'
    data_filled = load_and_preprocess_data(data_path)

    X_train, X_test, y_train, y_test = prepare_data_for_lstm(data_filled)
    model = build_and_train_lstm(X_train, y_train, X_test, y_test)
    evaluate_model(model, X_test, y_test)
