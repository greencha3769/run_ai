import os
import cv2
import queue
import torch
import numpy as np
# sklearn 0.22 호환용---
np.float = float
np.int = int
np.bool = bool
np.object = object
#-----------------------
from collections import deque
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy)
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

class DataCollector(Node):
    def __init__(self):
        super().__init__("DataCollector")
        self.YOLO_PATH = r""
        self.SAVE_FILE = r"/home/rtree/testai_ws/src/test_ai/image"
        os.makedirs(self.SAVE_FILE, exist_ok=True)
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )
        # callback group 생성
        self.camera_group = MutuallyExclusiveCallbackGroup()
        self.save_group = MutuallyExclusiveCallbackGroup()

        self.videoSubscriber = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.videoSubscriber_callback,
            qos,
            callback_group=self.camera_group
        )
        self.timer = self.create_timer(
            0.1,
            self.save_timer_callback,
            callback_group=self.save_group
        )
        self.bridge = CvBridge()
        self.src_pub = self.create_publisher(
            Image,
            "/yolo/src_image",
            10
        )
        self.crop_pub = self.create_publisher(
            Image,
            "/yolo/crop_image",
            10
        )
        self.save_queue = queue.Queue(maxsize=1)
        self.frame_count = 0
        self.frame_interval = 15
        self.save_count = 0
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        # AI Model 초기화
        self.Y_model = YOLO(self.YOLO_PATH)
        self.Y_model.to(self.DEVICE)
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
        # 실제 사용하는 ROI
        self.current_roi = None


    def videoSubscriber_callback(self, msg):
        try:
            src = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            src = cv2.convertScaleAbs(src, alpha=1.2, beta=-70)
            self.run(src)
            """
            src_msg = self.bridge.cv2_to_imgmsg(
                src,
                "bgr8"
            )
            self.src_pub.publish(src_msg)
            #print("pub")
            """
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

    def save_timer_callback(self):
        try:
            # 저장할 이미지 가져오기
            src = self.save_queue.get_nowait()
        except queue.Empty:
            return

        try:
            save_path = os.path.join(
                self.SAVE_FILE,
                f"good_{self.save_count:04d}.png"
            )
            success = cv2.imwrite(
                save_path,
                src
            )

            if success:
                self.get_logger().info(
                    f"완료: {self.save_count}장 저장"
                )
                self.save_count += 1
            else:
                self.get_logger().error(
                    f"저장 실패: {save_path}"
                )

        except Exception as e:
            self.get_logger().error(f"Save error: {e}")
    
    def run(self, src):
        print("why")
        boxes = self.detect_objects(src)
        print("4")
        if boxes is None:
            return
        print("1")
        crop_img = self.crop_video(src, boxes)
        if crop_img is None:
            return
        print("2")
        if crop_img is not None:
            self.frame_count += 1
            if self.frame_count >= self.frame_interval:
                self.frame_count = 0
                try:
                    # Queue에 넣고 callback 종료
                    self.save_queue.put_nowait(crop_img)
                # 예외 설명 queue가 가득 차있다는 것은 저장이 오래걸려 큐에서 아직 값이 안 빠져나가가 것으로 간주한다
                except queue.Full:
                    # 병목을 줄이고 최신 프레임을 유지하기 위해 프레임 버리고 다시 받음
                    self.save_queue.get_nowait()
                    self.save_queue.put_nowait(crop_img)
                    #self.get_logger().warn("Save queue full - frame skipped")
            crop_msg = self.bridge.cv2_to_imgmsg(
                crop_img,
                "bgr8"
            )
            self.crop_pub.publish(crop_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt (SIGINT)")
    finally:
        node.destroy_node()
        executor.shutdown()

if __name__ == "__main__":
    main()
