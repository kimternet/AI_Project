# 1. CCTV & UAV 연계형 AI 응급 대응 시스템 -ECG 데이터 분석 모델 개발

## [문제정의]
2024년 무인이동체 산업엑스포 AI 해커톤 대회에 참가하여 드론에 탑재될 ECG(심전도) 데이터 분석 및 딥러닝 모델 구축을 담당했습니다.
드론 탑재용 모델이라는 제약 조건 하에 높은 성능을 유지하면서도 경량화된 모델 개발이 핵심 과제였습니다.
## [해결 과정]
모델구조 비교 분석
- inception 구조를 사용한 CNN 모델과 일반 1차원 CNN 모델의 성능 비교
- 일반 1차원 CNN이 Inception 구조보다 5.6% 더 높은 성능을 보이나, 48.933MB로 상당한 공간 차지
- 파라미터 수 약 50만~100만개 사이의 모델을 기반으로 경량화 진행을 결정
## [다양한 경량화 기법 실험]
- 가지치기(Pruning): PyTorch에서 pruning 하이퍼파라키터 값을 0.2로 설정하여 테스트했으나, ROC 평균 51.4% 기록.
- 가중치 masking으로 파라미터 수에 영향 없어 실질적 경량화 효과 미미
- 양자화(Quantization): 양자화 전 AUC 93.4%에서 양자화 후 58.2%로 감소. 모델 크기와 연산 FLOPS에서 유의미한 차이 확인 불가
- 지식 증류(Knowledge Distillation): 파라미터 수에 따른 성능 변화 분석 후, Teacher 모델(650만 개 파라미터)과 Student 모델(57만 개 파라미터)
설정
## [결과]
- 지식 증류 기법을 통해 모델 크기를 대폭 줄이면서도 성능을 93.58%까지 향상
- 초기 모델 대비 약 25.1% 성능 향상 달성
- 드론 탑재에 적합한 경량화된 고성능 ECG 분석 모델 구현
- 2024 무인이동체산업엑스포 전국 성인부 대상 수상

---

## 🌟 Features

- **🌐 Language:** `Python`, `HTML`, `JavaScript`, and `CSS` and `Flask`
- **📊 Data Analysis & Processing:** `Numpy` and `Pandas`.
- **🤖 Machine Learning/Deep Learning:** `TensorFlow`, `PyTorch`, and `Scikit-learn`
  
---
# 2. 시각장애인을 위한 AI 악기 교육(Vision_PROJECT)
**기존 점자 악보의 불편함을 개선하기 위해 Computer Vision과 YOLOv8s모델을 활용하여 AI악기 교육을 제안했습니다.**!

**기획, 데이터 수집, 모델 구현 및 성능 개선 담당**! 🚀  

Intel GETI를 통해 데이터 어노테이션 작업을 수행했습니다. 총 데이터 4개의 클래스 각 1300개씩 6200개
총 3차 성능 개선을 시도했습니다. 
1차는 EfficientNet-B0 모델에서 2차는 YOLOv8s 모델을 활용했지만 여전히 AM, F코드의 혼동이 존재, 3차는 추가 데이터 수집 및 데이터 증강을 진행 후 모델의 성능을 3% 개선했습니다.

---

## 🌟 Features

- **🌐 Language:** `Python`, `HTML`, `JavaScript`, and `CSS` and `Flask`
- **📊 Data Analysis & Processing:** `Numpy` and `Pandas`.
- **🤖 Machine Learning/Deep Learning:**  `PyTorch`
  
---

# 3. RangChain, RAG, LLM모델을 활용한 요약ChatBot(LLM 활용_PROJECT)
**6개월간 배운 내용을 PDF로 만들어서 커리큘럼을 요약해주는 ChatBot 구현.**!

**개인프로젝트**! 🚀  

RangChain과 RAG를 활용하여 LLM모델이 답변을 생성하는 요약 ChatBot을 만들었습니다.

---

## 🌟 Features

- **🌐 Language:** `Python`, `HTML`, `JavaScript`, and `CSS` and `Flask`
- **📊 Data Analysis & Processing:** `RangChain` and `RAG`.
- **🤖 Machine Learning/Deep Learning:**  `HuggigFace LLM모델`
  
---
