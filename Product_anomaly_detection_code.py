import cv2
import torch
import numpy as np
from ultralytics import YOLO
from anomalib.models.patchcore import Patchcore
from torchvision import transforms

class Detect_AI():
    def __init__(self):
        # AI Model 경로
        self.PATCH_PATH = r"C:\Product_Anomaly_Detection\lightning_logs\version_36\checkpoints\epoch=0-step=628.ckpt"
        self.YOLO_PATH = r"C:\Product_Anomaly_Detection\runs\pepsi_seg\weights\best.pt"
        # device 지정
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        # 기호함수
        self.MARGIN = 2
        self.VIDEO_LINE = 60.0
        self.ANOMALY_LINE = 65
        self.RED = 0.3
        self.MIN_AREA = 100
        self.SMOOTHING = 0.4
        #변수
        self.prev_map = None
        # AI Model 초기화
        self.Y_model = YOLO(self.YOLO_PATH)
        self.P_model = Patchcore(
            input_size=(256,256),
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"],
            pre_trained=True
        )
        checkpoint = torch.load(
            self.PATCH_PATH,
            map_location=self.DEVICE
        )
        self.P_model.load_state_dict(
            checkpoint["state_dict"]
        )
        self.P_model.eval()
        self.P_model.to(self.DEVICE)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

        self.cap = cv2.VideoCapture(0)

    def video_callback(self):
        ret, frame = self.cap.read()

        if not ret:
            return False
        return frame

    def detect_objects(self, frame):
        results = self.Y_model(frame, imgsz=640, verbose=False)
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return False
        boxes = result.boxes.xyxy.cpu().numpy()
        return boxes

    def detect_defect(self, crop_tensor):
        with torch.no_grad():
            anomaly_map, anomaly_score = self.P_model(crop_tensor)

        anomaly_map = anomaly_map.detach().cpu().numpy()
        anomaly_map = np.squeeze(anomaly_map).astype(np.float32)

        print(f"score : {anomaly_score.item():.3f}")

        return anomaly_map, anomaly_score.item()

    def crop_video(self, frame, boxes):
        areas = []
        for x1, y1, x2, y2 in boxes:
            areas.append((x2 - x1) * (y2 - y1))

        idx = np.argmax(areas)
        x1, y1, x2, y2 = boxes[idx].astype(np.int32)

        x1 -= self.MARGIN
        y1 -= self.MARGIN
        x2 += self.MARGIN
        y2 += self.MARGIN

        h, w = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop_img = frame[y1:y2, x1:x2]

        if crop_img.size == 0:
            return False
        crop_tensor = self.transform(crop_img)
        crop_tensor = crop_tensor.unsqueeze(0)
        crop_tensor = crop_tensor.to(self.DEVICE)
        return crop_img, crop_tensor
    
    def result_video(self, anomaly_map, crop_img):
        if self.prev_map is None:
            smoothed_map = anomaly_map
        else:
            smoothed_map = self.SMOOTHING * self.prev_map + (1 - self.SMOOTHING) * anomaly_map

        self.prev_map = smoothed_map.copy()

        # anomaly_map에서 가장 높은 점수
        mask = smoothed_map >= self.VIDEO_LINE
        mask = mask.astype(np.uint8) * 255

        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 작은 영역 제거
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        clean_mask = np.zeros_like(mask)

        for cnt in contours:
            if cv2.contourArea(cnt) >= self.MIN_AREA:
                cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

        mask = clean_mask

        crop_h, crop_w = crop_img.shape[:2]

        mask = cv2.resize(
            mask.astype(np.uint8),
            (crop_w, crop_h),
            interpolation=cv2.INTER_NEAREST
        )

        mask = mask.astype(bool)

        overlay = crop_img.copy()
        overlay[mask] = (0, 0, 255)

        result_overlay = cv2.addWeighted(
            overlay,
            self.RED,
            crop_img,
            1 - self.RED,
            0
        )
        return result_overlay

    def main(self):
        while True:
            if cv2.waitKey(1) == 27:
                break

            frame = self.video_callback()
            if frame is False:
                continue

            cv2.imshow("frame", frame)

            boxes = self.detect_objects(frame)
            if boxes is False:
                continue

            crop_img, crop_tensor = self.crop_video(frame, boxes)
            if crop_tensor is False:
                continue

            cv2.imshow("CROP ONLY", crop_img)

            anomaly_map, anomaly_score = self.detect_defect(crop_tensor)

            # 정상/불량 판단
            if anomaly_score < self.ANOMALY_LINE:
                result_overlay = crop_img          # 정상
            else:
                result_overlay = self.result_video(anomaly_map, crop_img)   # 불량

            cv2.imshow("result_overlay", result_overlay)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_ai = Detect_AI()
    detect_ai.main()