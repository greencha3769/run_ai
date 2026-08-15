# run_ai
Product Anomaly Detection ROS 2 Package
## 프로젝트 개요
Ubuntu 환경에서 ROS 2를 기반으로 제품의 객체를 탐지하고 이상 여부를 판별하기 위한 프로젝트
ROS 2 환경에서 Astra 카메라 영상을 입력받아 YOLO를 이용하여 제품의 위치를 탐지하고, Patch Core를 이용하여 실시간으로 제품의 이상 탐지를 수행하는 것을 목표로 한다.
## 개발 환경
### Hardware
* Jetson Orin AGX
* Jetson Orin Nano
### OS
* Ubuntu
### JetPack
### ROS 2
* ROS 2 Galactic
### Python
* 3.8.10
### 라이브러리
| 항목 | 버전 |
|---|---|
| PyTorch ||
| torchvision ||
| NumPy ||
| OpenCV ||
| anomalib ||
| Ultralytics ||
## 환경 설정
### 4.1 ROS 2 설치
* Debian packages for ROS 2 Galactic 이용 [ROS 2 Documentation: Galactic](https://docs.ros.org/en/galactic/Installation/Ubuntu-Install-Debians.html)
### 4.2 Python Workspace 생성
### 4.3 PyTorch 설치
### 4.4 Torchvision 설치
### 4.5 OpenCV 설정
### 4.6 Anomalib / PatchCore 설정
### 4.7 YOLO 설정
### 4.8 ros2_astra_camera
## 모델 파일
### YOLO
### PatchCore
## 실행
### ROS 2 workspace 설정
### run_ai 빌드
### YOLO 실행
### PatchCore 실행
### 통합 이상 탐지 실행
## 파일 구조
