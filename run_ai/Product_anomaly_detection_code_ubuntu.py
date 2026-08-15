import cv2
import torch
import numpy as np
# sklearn 0.22 호환용---
np.float = float
np.int = int
np.bool = bool
np.object = object
#-----------------------
from collections import deque
#deque는 double-ended queue 의 줄임말로, 앞과 뒤에서 즉, 양방향에서 데이터를 처리할 수 있는 queue형 자료구조를 의미
from ultralytics import YOLO
from anomalib.models.patchcore import Patchcore
from torchvision import transforms
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class Detect_AI(Node):
    def __init__(self):
        super().__init__("Detect_AI")
        # qos 설정
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )
        # Subscriber(astra camera topic 받아오기)
        self.videoSubscriber = self.create_subscription(Image,
            "/camera/color/image_raw",
            self.videoSubscriber_callback,
            qos
        )
        #publisher
        # src publisher(기본화면 출력)
        self.src_pub = self.create_publisher(
            Image,
            "/yolo/src_image",
            10
        )
        # crop publisher(yolo roi 출력)
        self.crop_pub = self.create_publisher(
            Image,
            "/yolo/crop_image",
            10
        )
        # result publisher(최종 불량 표출 화면)
        self.result_pub = self.create_publisher(
            Image,
            "/yolo/result_image",
            10
        )
        # 기호함수-----------------------------------------------------------------------------
        # AI Model 위치(주소)
        self.PATCH_PATH = r"/home/rtree/ros2_ws/src/run_ai/model/pepsi_resnet18.ckpt"
        self.YOLO_PATH = r"/home/rtree/ros2_ws/src/run_ai/model/best.pt"
        # device 지정
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        # roi 영역 여백
        self.MARGIN = 2
        # patch core 불량 판정 시 overlay할 부분 결정하는 임계값
        self.VIDEO_LINE = 3.18
        # patch core 불량 판정 결정하는 임계값
        self.ANOMALY_LINE = 3.23
        # overlay 정도 결정
        self.RED = 0.3
        # overlay할 부분의 최소 면적 임계값
        self.MIN_AREA = 100
        # overlay 면적 안정화
        self.SMOOTHING = 0.4
        # yolo 신뢰도 임계값
        self.CONF_TH = 0.6
        # yolo 중심 좌표 위치 변화 허용 오차 임계값
        self.CENTER_TH = 15
        # yolo w,h 길이 변화 허용 오차 임계값
        self.SIZE_TH = 10
        # 연속 몇 번 확인 후 변경
        self.CHANGE_LIMIT = 4
        #변수-------------------------------------------------------------------
        # 이전 박스
        self.prev_map = None
        # 최근 5프레임 저장
        self.frame_boxes = deque(maxlen=5)
        # ROI 변경 확인 횟수
        self.change_count = 0
        # 실제 사용하는 ROI
        self.current_roi = None
        # AI Model 초기화
        self.Y_model = YOLO(self.YOLO_PATH)
        self.Y_model.to(self.DEVICE)
        self.P_model = Patchcore(
            input_size=(256,256),
            backbone="resnet18",
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
        self.bridge = CvBridge()

    def videoSubscriber_callback(self, msg):
        try:
            print("ok")
            src = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            src = cv2.convertScaleAbs(src, alpha=1.5, beta=50)
            print("gg")
            print("run")
            self.run(src)
            src_msg = self.bridge.cv2_to_imgmsg(
                src,
                "bgr8"
            )
            print("src_pub")
            self.src_pub.publish(src_msg)
        except Exception as e:
            print(e)
            #self.get_logger().error(f'Error')
            return      
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            self.destroy_node()
            rclpy.shutdown()

    def detect_objects(self, src):
        import time
        start = time.time() # 현재 시간 저장
        results = self.Y_model(src, imgsz=640, verbose=False, device=self.DEVICE) # yolo 돌림
        print("YOLO time:", time.time()-start) # 현재 시간 빼기 이전 시간
        result = results[0] # YOLO는 추론 결과를 객체들의 리스트 형태로 반환하고, 우리가 내부 데이터(box, confidence 등)에 접근하려면 해당 이미지의 객체를 꺼내야 하기 때문에 사용
        print("BOX COUNT:", len(result.boxes)) # yolo가 찾은 Bounding Box 정보를 불러온 후 len()을 이용하여 자료의 개수 알아내 검출된 객체의 수를 확인
        if len(result.boxes) == 0: # 검출된 객체의 수가 없을 경우 None 값을 반환해 다음으로 진행되는 것을 막음
            return None
        # 가장 높은 confidence 선택
        idx = result.boxes.conf.argmax() # yolo가 찾은 Bounding Box 정보 중 confidence 관한 정보를 불러온 후, argmax()를 이용하여 가장 큰 값의 위치(index)를 반환
        conf = float(result.boxes.conf[idx]) # 신뢰도가 가장 큰 Bounding Box의 confidence 값은 반환 

        if conf < self.CONF_TH: # 신뢰도가 일정 이하인 경우 None 값을 반환해 다음으로 진행되는 것을 막음
            return None
        box = result.boxes.xyxy[idx].cpu().numpy().astype(np.float32) # 가장 높은 신뢰도를 가진 Bounding Box의 좌표를 가져옴
        # result.boxes.xyxy[idx], 가장 높은 신뢰도를 가진 Bounding Box의 좌표를 가져옴
        # .cpu(), YOLO 모델을 GPU에서 실행했기 때문에 결과 Tensor가 GPU 메모리에 존재하는데, NumPy는 GPU Tensor를 직접 처리할 수 없으므로 GPU Tensor를 CPU 메모리의 Tensor로 이동시킴
        # .numpy(), CPU Tensor를 NumPy 배열로 변환
        # .astype(np.float32), 자료형을 지정
        print("BOX:", box)
        return box

    def detect_defect(self, crop_tensor):
        with torch.no_grad():
            anomaly_map, anomaly_score = self.P_model(crop_tensor)

        anomaly_map = anomaly_map.detach().cpu().numpy()
        anomaly_map = np.squeeze(anomaly_map).astype(np.float32)

        print(f"score : {anomaly_score.item():.3f}")

        return anomaly_map, anomaly_score.item()

    def crop_video(self, src, box):
        self.frame_boxes.append(box) # 최근 5프레임 저장

        if len(self.frame_boxes) < 5: # 아직 5프레임이 안 모였으면 현재 ROI 사용
            if self.current_roi is None: # 초반에 current_roi가 아직 없는 경우 사용
                self.current_roi = box.copy()
            box = self.current_roi
        else:
            boxes = np.array(self.frame_boxes) # 최근 5프레임 리스티에서 배열로 변경
            
            median_box = np.median(boxes, axis=0) # Median(중앙값) 계산
            # [x1, y1, x2, y2] 형태에서 axis=0 세로 줄끼리 비교하여 중앙값 계산
            # [x1, y1, x2, y2]
            # [x1, y1, x2, y2]

            mcx = (median_box[0] + median_box[2]) / 2 # Median(중앙값) x 중심 좌표 계산
            mcy = (median_box[1] + median_box[3]) / 2 # Median(중앙값) y 중심 좌표 계산

            mw = median_box[2] - median_box[0] # 길이 계산
            mh = median_box[3] - median_box[1] # 높이 계산

            valid_boxes = []

            for b in boxes: # Median과 가까운 Box만 사용, 튀는 값 제거

                cx = (b[0] + b[2]) / 2 # x 중심 좌표 계산
                cy = (b[1] + b[3]) / 2 # y 중심 좌표 계산

                w = b[2] - b[0] # 길이 계산
                h = b[3] - b[1] # 높이 계산

                if abs(cx - mcx) < self.CENTER_TH and \
                    abs(cy - mcy) < self.CENTER_TH and \
                    abs(w - mw) < self.SIZE_TH and \
                    abs(h - mh) < self.SIZE_TH:

                    valid_boxes.append(b)

            # ------------------------------------------------
            # 5. 평균 Box 계산
            # ------------------------------------------------
            if len(valid_boxes) == 0:
                avg_box = median_box
            else:
                avg_box = np.mean(valid_boxes, axis=0)

            # ------------------------------------------------
            # 6. 현재 ROI가 없으면 최초 설정
            # ------------------------------------------------
            if self.current_roi is None:

                self.current_roi = avg_box.copy()

            else:

                # -------------------------------
                # 현재 ROI
                # -------------------------------
                ccx = (self.current_roi[0] + self.current_roi[2]) / 2
                ccy = (self.current_roi[1] + self.current_roi[3]) / 2

                cw = self.current_roi[2] - self.current_roi[0]
                ch = self.current_roi[3] - self.current_roi[1]

                # -------------------------------
                # 새 평균 ROI
                # -------------------------------
                acx = (avg_box[0] + avg_box[2]) / 2
                acy = (avg_box[1] + avg_box[3]) / 2

                aw = avg_box[2] - avg_box[0]
                ah = avg_box[3] - avg_box[1]

                # -------------------------------
                # 차이 계산
                # -------------------------------
                center_x_diff = abs(acx - ccx)
                center_y_diff = abs(acy - ccy)

                width_diff = abs(aw - cw)
                height_diff = abs(ah - ch)

                # -------------------------------
                # ROI 유지 여부
                # -------------------------------
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

        # ----------------------------------------------------
        # 8. Crop
        # ----------------------------------------------------
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
            return None, None

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

    def run(self, src):
        print("why")
        boxes = self.detect_objects(src)
        print("4")
        if boxes is None:
            return
        print("1")
        crop_img, crop_tensor = self.crop_video(src, boxes)
        if crop_img is None:
            return
        print("2")
        if crop_img is not None:
            crop_msg = self.bridge.cv2_to_imgmsg(
                crop_img,
                "bgr8"
            )

            self.crop_pub.publish(crop_msg)

        anomaly_map, anomaly_score = self.detect_defect(crop_tensor)
        # 정상/불량 판단
        if anomaly_score < self.ANOMALY_LINE:
            result_overlay = crop_img          # 정상
        else:
            result_overlay = self.result_video(anomaly_map, crop_img)   # 불량
        if result_overlay is not None:
            result_msg = self.bridge.cv2_to_imgmsg(
                result_overlay,
                "bgr8"
            )
            self.result_pub.publish(result_msg)

def main(args=None):
    rclpy.init(args=args)
    node = Detect_AI()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
