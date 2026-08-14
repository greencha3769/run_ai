import os
import cv2
import numpy as np
from ultralytics import YOLO

# 주소 
#VIDEO = r"C:\Product_Anomaly_Detection\training\Patch_core\dataset\train\video\test.mp4" # X
VIDEO = r"C:\yolo_project\dataset\videos\KakaoTalk_20260628_144000439.mp4" # O
#VIDEO = r"C:\Product_Anomaly_Detection\training\Patch_core\dataset\train\video\KakaoTalk_20260704_174815180.mp4"
#VIDEO = r"C:\Product_Anomaly_Detection\training\Patch_core\dataset\train\video\KakaoTalk_20260704_175650424.mp4"

MODEL = r"C:\Product_Anomaly_Detection\runs\pepsi_seg\weights\best.pt"
SAVE_FILE = r"C:\Product_Anomaly_Detection\training\Patch_core\dataset\train\good"

# 기호 상수
FRAME = 5 # 저장
RESOLUTION = 0.69945 # 해상도
MARGIN = 2 # 여유

# 변수
frame_count = 0
save_count = 574

# 파일 생성
os.makedirs(SAVE_FILE, exist_ok=True)

model = YOLO(MODEL) # 모델 불러오기
cap = cv2.VideoCapture(VIDEO) # 영상 불러오기

# 해상도 변경 함수
def resize_only(img, resolution):
    return cv2.resize(img, dsize=(0, 0), fx = resolution, fy = resolution, interpolation=cv2.INTER_LINEAR)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    frame_count += 1

    if frame_count % FRAME != 0:
        continue

    results = model(frame, imgsz=640, verbose=False)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        continue

    # bbox 가져오기
    boxes = result.boxes.xyxy.cpu().numpy()

    # 가장 큰 박스 선택
    areas = []
    for x1, y1, x2, y2 in boxes:
        area = (x2 - x1) * (y2 - y1)
        areas.append(area)

    idx = np.argmax(areas) # 함수의 출력값을 최대화하는 입력값(인덱스)을 반환하는 함수
    x1, y1, x2, y2 = boxes[idx].astype(np.int32) # 정수로 변환

    #여유 주기
    x1 -= MARGIN
    x2 += MARGIN
    y1 -= MARGIN
    y2 += MARGIN

    # 이미지 범위 벗어나지 않게 처리 (ai가 필요하다고 했음)
    h, w = frame.shape[:2]

    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    if x2 > w: x2 = w
    if y2 > h: y2 = h

    # crop
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        continue

    crop = resize_only(crop, RESOLUTION)

    save_path = os.path.join(
        SAVE_FILE,
        f"good_{save_count:06d}.png"
    )

    cv2.imwrite(save_path, crop)

    save_count += 1

cap.release()
print(f"완료: {save_count}장 저장")