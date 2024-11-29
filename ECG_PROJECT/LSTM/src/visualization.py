import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2.1 비정상 상태의 생체 신호 분포 시각화
def visualize_abnormal_vital_signs(data):
    # 비정상 상태 데이터만 필터링
    abnormal_data = data[data['OUTPUT'] == 'Abnormal']

    # 히스토그램을 통해 비정상 상태의 생체 신호 분포 시각화
    plt.figure(figsize=(12, 10))

    # 심박수 분포
    plt.subplot(2, 2, 1)
    plt.hist(abnormal_data['HR(BPM)'], bins=20, color='blue', edgecolor='black')
    plt.title('Distribution of HR(BPM) in Abnormal Data')
    plt.xlabel('HR(BPM)')
    plt.ylabel('Frequency')

    # 호흡수 분포
    plt.subplot(2, 2, 2)
    plt.hist(abnormal_data['RESP(BPM)'], bins=20, color='green', edgecolor='black')
    plt.title('Distribution of RESP(BPM) in Abnormal Data')
    plt.xlabel('RESP(BPM)')
    plt.ylabel('Frequency')

    # 산소포화도 분포
    plt.subplot(2, 2, 3)
    plt.hist(abnormal_data['SpO2(%)'], bins=20, color='red', edgecolor='black')
    plt.title('Distribution of SpO2(%) in Abnormal Data')
    plt.xlabel('SpO2(%)')
    plt.ylabel('Frequency')

    # 체온 분포
    plt.subplot(2, 2, 4)
    plt.hist(abnormal_data['TEMP (*C)'], bins=20, color='purple', edgecolor='black')
    plt.title('Distribution of TEMP (*C) in Abnormal Data')
    plt.xlabel('TEMP (*C)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# 2.2 상관관계 분석 시각화
def visualize_correlation_matrix(data):
    # 주요 생체 신호 간의 상관관계 계산
    correlation_matrix = data[['HR(BPM)', 'RESP(BPM)', 'SpO2(%)', 'TEMP (*C)']].corr()

    # 상관관계 행렬 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Vital Signs in Abnormal Data')
    plt.show()

# 메인 함수
if __name__ == "__main__":
    from preprocessing import load_and_preprocess_data

    data_path = '../data/Human_vital_signs_R.csv'
    data_filled = load_and_preprocess_data(data_path)

    visualize_abnormal_vital_signs(data_filled)
    visualize_correlation_matrix(data_filled)