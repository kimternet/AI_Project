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
# 2. 시각장애인을 위한 AI 악기 교육 시스템 개발(Vision_PROJECT)

## [문제정의]
시각적 요소가 중요한 악기 교육 분야에서 시각장애인의 접근성을 높이기 위한 AI 모델 개발을 목표로 했습니다.
컴퓨터 비전 기술을 활용하여 기존 점자악보의 복잡성으로 인한 접근성을 해결하고, 시각장애인이 악기 운지법을 정확히 파악하고 교정받을 수 있는 시스템 구축이 필요했습니다.

## [해결과정]
### [모델 선정 과정과 시행착오]
- 처음에는 EfficientNet-B0모델을 선택하여 실험했으나, Am코드와 F코드 간 분류 성능이 현저히 저하되는 문제 발견
- 두 코드의 손가락 위치가 유사하여 모델이 특징을 제대로 구분하지 못하는 한계 확인
- 더 정확한 객체 탐지 성능이 필요하다고 판단하여 YOLOv8s 모델로 전환
- YOLOv8s 모델 채택 이유: 경량성, 빠른 추론 속도, 적은 계산 자원 요구량, 그리고 특히 미세한 손가락 포지션 차이를 더 정확히 감지하는 능력

## [데이터 수집과 반복적 개선]
- 초기 데이터셋으로 학습한 결과 전반적으로 높은 성능을 보였으나, 특정 각도와 조명 조건에서 추론 정확도 저하 문제 발견
- 20명의 참가자를 대상으로 더 다양한 각도, 조명 조건, 배경에서 약 6,200장의 이미지 데이터를 다시 수집
- 특히 문제가 되었던 Am와 F 코드 이미지에 중점을 두고 다양한 촬영 조건에서 추가 데이터 확보
- 데이터 증강 기법(augmentation)적용: 회전, 밝기 조절, 확대/축소, 수평 반전 등 다양한 변형을 통해 학습 데이터 다양화
- 각 이미지에 대한 세밀한 라벨링 작업으로 데이터 품질 일관성 확보
## [모델 학습 최적화 전략]
- 첫 번째 학습 시도에서 과적합 문제 발견 후, 드롭아웃 레이어 추가 및 정규화 기법 적용
- 에포크와 배치 사이즈를 작게 설정(에포크:50, 배치 사이즈:16)하여 더 세밀한 학습 진행
- 학습률(learning rate)을 0.001에서 시작하여 점진적으로 감소시키는 스케줄링 적용
- 손실 함수로 Focal Loss를 사용하여 난이도가 높은 샘플에 더 집중할 수 있도록 조정
- 검증 데이터셋을 통한 주기적 성능 모니터링으로 최적의 모델 파라미터 도출
## [결과]
- box(p): 98.8%, recall: 98.7% 달성
- mAP50: 99.1%, mAP50-95: 82.2%의 우수한 객체 탐지 성능 확보
- 시각장애인이 실시간으로 악기 연주 자세를 교정받을 수 있는 효과적인 AI 시스템 개발 완료
- 해당 연구를 졸업 논문으로 제출하여 학술적 가치 인정

## 🌟 Features

- **🌐 Language:** `Python`, `HTML`, `JavaScript`, and `CSS` and `Flask`
- **📊 Data Analysis & Processing:** `Numpy` and `Pandas`.
- **🤖 Machine Learning/Deep Learning:**  `PyTorch`
  
---

# RangChain, RAG, LLM 모델을 활용한 요약 ChatBot 개발
## [문제정의]
2024년 8월 개인 프로젝트로 LangChain, RAG, LLM 기술을 활용한 PDF 요약 챗봇을 개발했습니다.
사용자 질문에 대한 정확한 응답을 생성하면서도, 대용량 언어모델을 경량화하여 효율적으로 활용하는 것이 핵심 과제였습니다. 또한, 문서 기반 질의응답 시스템의 구현과 실제 서비스 적용 가능성을 실험하고자 했습니다.
## [해결 과정]
* Flask 앱 구성
  - Flask를 활용해 웹서버를 구축하고, 템플릿 기반의 사용자 입력 처리 기능을 구현했습니다.
* 양자화된 LLM 로딩
  - Ko-PlatYi-6B 모델을 4비트로 양자화하여 BitsAndBytesConfig로 로드했습니다.
  - bfloat16 계산형식과 함께 사용하여 메모리 사용량과 응답 속도를 최적화했습니다.

* LangChain 기반 체인 구성
  - LangChain의 LLMChain을 통해 사용자 입력을 처리하고 프롬프트 기반의 자연어 응답을 구성했습니다.

## PDF 기반 RAG 구현
* PyPDFLoader를 사용하여 PDF 텍스트를 로드하고, 이를 청크 단위로 분리해 FAISS 인덱스를 구성했습니다.
* 인덱스를 통해 관련 문서를 검색하고 RAG 체인을 통해 요약 응답을 생성하는 구조를 구현했습니다.

## Flask API 엔드포인트 구성
* 사용자의 질문을 입력받고, RAG 체인을 거쳐 응답을 생성 후 JSON 형식으로 반환하는 API를 구축했습니다.

## [결과]

* 사용자 질의에 대해 PDF 기반으로 유의미하고 요약된 답변을 생성하는 챗봇을 완성

* 4비트 양자화된 LLM을 활용해 적은 자원으로도 실시간 반응 가능한 성능 확보

* Flask와 LangChain을 연계한 RAG 아키텍처를 성공적으로 구현

# [활용 기술 및 도구]

Python, Flask, LangChain, FAISS, PyPDFLoader, Ko-PlatYi-6B (4bit quantized), HuggingFace Transformers, BitsAndBytesConfig

## 🌟 Features

- **🌐 Language:** `Python`, `HTML`, `JavaScript`, and `CSS` and `Flask`
- **📊 Data Analysis & Processing:** `RangChain` and `RAG`, BitsAndBytesConfig,FAISS, PyPDFLoader
- **🤖 Machine Learning/Deep Learning:**  `HuggigFace Ko-PlatYi-6B (4bit quantized) 모델`
  
---
