# Product Anomaly Detection
(Windows 환경에서 제품 이상 탐지 모델을 학습하고 테스트하기 위한 프로젝트)

## 프로젝트 구조
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

## 구성

### PatchCore
`training/Patch_core/`
(PatchCore 모델 관련 코드)

* `data_selector.py` : 학습 데이터 선택
* `patch_core_trainig.py` : PatchCore 모델 학습
* `patch_data_collector.py` : PatchCore 학습 데이터 수집

### YOLO

`training/Yolo_v8/`

YOLO 기반 객체 탐지 및 학습을 위한 코드입니다.

* `rename.py` : 데이터 파일 이름 정리
* `yolo_data_collector.py` : YOLO 학습 데이터 수집
* `Yolo_training.py` : YOLO 모델 학습

## 실행 코드

* `Product_anomaly_detection_code.py` : Windows 환경의 제품 이상 탐지
* `Product_anomaly_detection_code_ubuntu.py` : Ubuntu 환경의 제품 이상 탐지
* `only_yolo.py` : YOLO만 사용하는 탐지 코드
* `only_yolo_ubuntu.py` : Ubuntu 환경에서 YOLO만 사용하는 탐지 코드

## 기타

* `check.py` : 코드 및 환경 확인
* `code_inventor.py` : 코드 작성 및 테스트용
* `test.py` : 테스트용 코드

## 실행 환경

* Windows
* Python
* YOLO
* PatchCore
* OpenCV
