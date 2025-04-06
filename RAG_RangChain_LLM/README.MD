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
