import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# 1.2 데이터 불러오기 및 전처리
def load_and_preprocess_data(filepath):
    # 데이터 불러오기
    data = pd.read_csv(filepath)

    # 불필요한 열 제거
    data = data.drop(columns=['Unnamed: 0'])

    # 각 열의 NaN 값 개수와 비율 확인
    nan_counts = data.isna().sum()
    nan_percentage = (nan_counts / len(data)) * 100
    print("NaN 값 개수:\n", nan_counts)
    print("\nNaN 값 비율 (%):\n", nan_percentage)

    # 숫자형 열만 선택하여 결측치를 평균값으로 대체
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data_filled = data.copy()
    data_filled[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

    # 결측치 대체 후 NaN 값 개수 확인
    nan_counts_after = data_filled.isna().sum()
    print("결측치 대체 후 NaN 값 개수:\n", nan_counts_after)

    return data_filled

# 1.3 클래스 분포 확인 및 시각화
def visualize_class_distribution(data):
    # 각 클래스의 Output 샘플 수 확인
    class_counts = data['OUTPUT'].value_counts()
    print("Class distribution:\n", class_counts)

    # 클래스 분포를 시각화
    class_counts.plot(kind='bar')
    plt.title('Class Distribution (Normal vs. Abnormal)')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(ticks=[0, 1], labels=['Normal', 'Abnormal'])
    plt.show()

# 메인 함수
if __name__ == "__main__":
    data_path = '../data/Human_vital_signs_R.csv'
    data_filled = load_and_preprocess_data(data_path)
    visualize_class_distribution(data_filled)