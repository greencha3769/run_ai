import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque

class Detect_AI():
    def __init__(self):
        # AI Model 경로
        self.YOLO_PATH = r"C:\Product_Anomaly_Detection\runs\pepsi_detect_s\weights\best.pt"
        # device 지정
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        # 기호함수
        self.MARGIN = 2
        self.CONF_TH = 0.6
        #변수
        self.prev_map = None
        # AI Model 초기화
        self.frame_boxes = deque(maxlen=5)
        # ROI 변경 확인 횟수
        self.change_count = 0
        # 허용 오차
        self.CENTER_TH = 15
        self.SIZE_TH = 10
        # 연속 몇 번 확인 후 변경
        self.CHANGE_LIMIT = 4
        # 최근 YOLO 결과
        # 이전 평균 ROI
        # 실제 사용하는 ROI
        self.current_roi = None
        # 연속 이동 횟수
        # 설정값
        # AI Model 초기화
        self.Y_model = YOLO(self.YOLO_PATH)
        self.Y_model.to(self.DEVICE)
        self.cap = cv2.VideoCapture(0)

    def video_callback(self):
        ret, frame = self.cap.read()

        if not ret:
            return None
        return frame

    def detect_objects(self, src):
        import time
        start = time.time()
        results = self.Y_model(src, imgsz=640, verbose=None, device=self.DEVICE)
        print("YOLO time:", time.time()-start)
        result = results[0]
        print("BOX COUNT:", len(result.boxes))
        if len(result.boxes) == 0:
            return None
        # 가장 높은 confidence 선택
        idx = result.boxes.conf.argmax()
    
        conf = float(result.boxes.conf[idx])
    
        if conf < self.CONF_TH:
            return None
        box = result.boxes.xyxy[idx].cpu().numpy().astype(np.float32)
        print("BOX:", box)
        return box

    def crop_video(self, src, box):
        self.frame_boxes.append(box)
    
        if len(self.frame_boxes) < 5:
    
            if self.current_roi is None:
                self.current_roi = box.copy()
    
            box = self.current_roi
    
        else:
            boxes = np.array(self.frame_boxes)
    
            median_box = np.median(boxes, axis=0)
    
            mcx = (median_box[0] + median_box[2]) / 2
            mcy = (median_box[1] + median_box[3]) / 2
    
            mw = median_box[2] - median_box[0]
            mh = median_box[3] - median_box[1]
    
            valid_boxes = []
    
            for b in boxes:
    
                cx = (b[0] + b[2]) / 2
                cy = (b[1] + b[3]) / 2
    
                w = b[2] - b[0]
                h = b[3] - b[1]
    
                if abs(cx - mcx) < self.CENTER_TH and \
                    abs(cy - mcy) < self.CENTER_TH and \
                    abs(w - mw) < self.SIZE_TH and \
                    abs(h - mh) < self.SIZE_TH:
    
                    valid_boxes.append(b)
    
            if len(valid_boxes) == 0:
                avg_box = median_box
            else:
                avg_box = np.mean(valid_boxes, axis=0)
    
            if self.current_roi is None:
    
                self.current_roi = avg_box.copy()
    
            else:
    
                ccx = (self.current_roi[0] + self.current_roi[2]) / 2
                ccy = (self.current_roi[1] + self.current_roi[3]) / 2
    
                cw = self.current_roi[2] - self.current_roi[0]
                ch = self.current_roi[3] - self.current_roi[1]
    
                acx = (avg_box[0] + avg_box[2]) / 2
                acy = (avg_box[1] + avg_box[3]) / 2
    
                aw = avg_box[2] - avg_box[0]
                ah = avg_box[3] - avg_box[1]
    
                center_x_diff = abs(acx - ccx)
                center_y_diff = abs(acy - ccy)
    
                width_diff = abs(aw - cw)
                height_diff = abs(ah - ch)
    
                center_ok = (
                    center_x_diff < self.CENTER_TH and
                    center_y_diff < self.CENTER_TH
                )
    
                size_ok = (
                    width_diff < self.SIZE_TH and
                    height_diff < self.SIZE_TH
                )
    
                if center_ok and size_ok:
                
                    # ROI 유지
                    self.change_count = 0
    
                else:
                
                    # ROI 변경 후보
                    self.change_count += 1
    
                    # 2번 연속이면 ROI 변경
                    if self.change_count >= self.CHANGE_LIMIT:
                    
                        self.current_roi = avg_box.copy()
                        self.change_count = 0
    
            box = self.current_roi
    
        x1, y1, x2, y2 = box
    
        x1 -= self.MARGIN
        y1 -= self.MARGIN
        x2 += self.MARGIN
        y2 += self.MARGIN
    
        H, W = src.shape[:2]
    
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(W, int(x2))
        y2 = min(H, int(y2))
    
        crop_img = src[y1:y2, x1:x2]
    
        if crop_img.size == 0:
            return None
    
        return crop_img
    
    def main(self):
        while True:
            if cv2.waitKey(1) == 27:
                break

            frame = self.video_callback()
            if frame is None:
                continue

            cv2.imshow("frame", frame)

            boxes = self.detect_objects(frame)
            if boxes is None:
                continue

            crop_img = self.crop_video(frame, boxes)

            cv2.imshow("CROP ONLY", crop_img)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_ai = Detect_AI()
    detect_ai.main()