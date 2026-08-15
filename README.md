# Product Anomaly Detection
(Windows 환경에서 제품 이상 탐지 모델을 학습하고 테스트하기 위한 프로젝트)
[전반적인 코드 백업용 branch]

## 파일 구조
```text
Product_Anomaly_Detection/
├── training/
│   ├── Patch_core/
│   │   └── dataset/
│   │       ├── data_selector.py
│   │       ├── patch_core_trainig.py
│   │       └── patch_data_collector.py
│   └── Yolo_v8/
│       └── dataset/
│           ├── rename.py
│           ├── yolo_data_collector.py
│           └── Yolo_training.py
├── Product_anomaly_detection_code.py
├── Product_anomaly_detection_code_ubuntu.py
├── only_yolo.py
├── only_yolo_ubuntu.py
├── check.py
├── code_inventor.py
├── test.py
└── README.md
```

## 코드 구성

### PatchCore
`training/Patch_core/`
(PatchCore 모델 관련 코드)

* `data_selector.py` : 학습 데이터 선택(영상에서 데이터 추출하는 코드)[사용 X]
* `patch_data_collector.py` : PatchCore 학습 데이터 수집(ubuntu에서 astra camera를 사용해서 데이터를 수집하기 위한 코드)[사용 O]
* `patch_core_trainig.py` : PatchCore 모델 학습(안에 주석으로 처리된 코드는 최신버전용, 실사용은 예전 버전)[사용 O]

### YOLO
`training/Yolo_v8/`
(YOLO 모델 관련 코드)
* `rename.py` : 데이터 파일 이름 정리(labeling studio의 경우 작업 후 내보내기한 결과물의 이름이 다름)[사용 O]
* `yolo_data_collector.py` : YOLO 학습 데이터 수집(ubuntu에서 astra camera를 사용해서 데이터를 수집하기 위한 코드)[사용 O]
* `Yolo_training.py` : YOLO 모델 학습[사용 O]

### 실행 코드
* `Product_anomaly_detection_code.py` : Windows 환경의 제품 이상 탐지(전반적인 기능 확인)[사용 O]
* `Product_anomaly_detection_code_ubuntu.py` : Ubuntu 환경의 제품 이상 탐지[사용 O]
* `only_yolo.py` : YOLO만 사용하는 탐지 코드(YOLO 기능 확인)[사용 O]
* `only_yolo_ubuntu.py` : Ubuntu 환경에서 YOLO만 사용하는 탐지 코드[사용 O]

### 기타
* `check.py` : 코드 및 환경 확인
* `code_inventor.py` : 코드 작성 및 테스트용
* `test.py` : 테스트용 코드

## 실행 환경
* Windows version:
* Python version:
* YOLO version:
* PatchCore version:
* OpenCV version:
* numpy  version:
* Toch version:
