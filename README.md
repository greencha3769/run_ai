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
* JetPack 5.1.6 (L4T 35.6.5)
### ROS 2
* ROS 2 Galactic
### Python
* 3.8.10
### 라이브러리
| 항목 | 버전 |
|---|---|
| PyTorch | 2.0.0+nv23.05 |
| torchvision | 0.15.1 |
| NumPy | numpy: 1.24.4 |
| OpenCV | 4.8.0.76 |
| Anomalib | 0.7.0 |
| Ultralytics | 8.0.227 |
## 환경 설정
### 4.1 ROS 2 설치
* Debian packages for ROS 2 Galactic 이용 [ROS 2 Documentation: Galactic](https://docs.ros.org/en/galactic/Installation/Ubuntu-Install-Debians.html)
### 4.2 Python Workspace 생성
```text
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
```
### 4.3기본 개발 패키지 및 도구 설치
```text
sudo apt update
sudo apt install nano
sudo apt install python3-pip
python3 -m pip install --upgrade pip
```
### 4.4 PyTorch 설치
```text
sudo apt install -y libopenblas-dev
export TORCH_INSTALL=https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
python3 -m pip install --no-cache-dir $TORCH_INSTALL
```
### 4.5 Torchvision 설치
```text
sudo apt install -y \
libjpeg-dev \
zlib1g-dev \
libpython3-dev \
libopenblas-dev \
libavcodec-dev \
libavformat-dev \
libswscale-dev
cd ~
git clone --branch v0.15.1 https://github.com/pytorch/vision.git torchvision
cd ~/torchvision
export BUILD_VERSION=0.15.1
python3 setup.py install --user
```
### 4.6 Ultralytics / YOLO 설정
```text
cd ~
git clone --branch v8.0.227 https://github.com/ultralytics/ultralytics.git
cd ~/ultralytics
git status
nano requirements.txt # 관련 종속성에 있어 문제가 될 수 있는 부분 제거 후 저장
python3 -m pip install .
```
### 4.7 Anomalib / PatchCore 설정
```text
python3 -m pip install "huggingface-hub==0.18.0"
python3 -m pip install anomalib==0.7.0
python3 -m pip install scikit-learn==1.3.2
```
### 4.8 OpenCV 설정
```text
python3 -c "import cv2; print(cv2.__version__)"
python3 -m pip show opencv-python # 혹시 다른 버전의 open cv가 설치 되어있는 확인(무작정 실행 X)
python3 -m pip show opencv-python-headless # 혹시 다른 버전의 open cv가 설치 되어있는 확인(무작정 실행 X)
python3 -m pip uninstall -y opencv-python opencv-python-headless # 다른 버전의 open cv가 설치 되어있는 경우(무작정 실행 X)
python3 -m pip install --user opencv-python==4.8.0.76
```
### 4.9 ros2_astra_camera
* ros2_astra_camera package 이용 [orbbec/ros2_astra_camera
](https://github.com/orbbec/ros2_astra_camera)
## 모델 파일
Package 안에 model 파일 생성한 후 제작한 모델을 불러오고 관련 경로를 수정해준다
## 실행
### run_ai 빌드
```text
cd ~/ros2_ws
colcon build --symlink-install
source ~/ros2_ws/install/setup.bash
```
### rviz2 실행
```text
source /opt/ros/galactic/setup.bash 
source ~/ros2_ws/install/setup.bash
rviz2
```
### astra camera 실행
```text
source /opt/ros/galactic/setup.bash 
source ~/ros2_ws/install/setup.bash
ros2 launch astra_camera astra.launch.xml
```
### 통합 이상 탐지 실행
```text
source ~/ros2_ws/install/setup.bash
ros2 run run_ai padcu
```
## Package 구조
```text
run_ai/
├── .gitignore
├── image/
├── model/
├── package.xml
├── resource/
├── run_ai/
│   ├── __init__.py
│   ├── only_yolo_ubuntu.py
│   ├── patch_data_collector.py
│   ├── Product_anomaly_detection_code_ubuntu.py
│   └── yolo_data_collector.py
├── setup.cfg
├── setup.py
└── test/
```
