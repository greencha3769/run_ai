import cv2
import torch
import numpy as np
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy

class Detect_AI(Node):
    def __init__(self):
        super().__init__("Detect_AI")
        # AI Model 경로
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )
        self.videoSubscriber = self.create_subscription(Image, "/camera/color/image_raw", self.videoSubscriber_callback, qos)
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
        self.bridge = CvBridge()
        self.YOLO_PATH = r"/home/rtree/ros2_ws/src/run_ai/model/best.pt"
        # device 지정
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        # 기호함수
        self.MARGIN = 0
        self.MIN_AREA = 100
        self.SMOOTHING = 0.4
        #변수
        self.prev_map = None
        # AI Model 초기화
        self.Y_model = YOLO(self.YOLO_PATH)
        self.Y_model.to(self.DEVICE)

    def videoSubscriber_callback(self, msg):
        try:
            print("ko")
            src = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            print("shape:", src.shape)
            print("dtype:", src.dtype)
            print("type:", src.type if hasattr(src, "type") else "no")
            src = cv2.convertScaleAbs(
                src,
                alpha=1.2,
                beta=40
            )

            src = np.ascontiguousarray(src, dtype=np.uint8)
            src_msg = self.bridge.cv2_to_imgmsg(
                src,
                "bgr8"
            )
            self.src_pub.publish(src_msg)
            self.run(src)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return      
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            self.destroy_node()
            rclpy.shutdown()

    def detect_objects(self, src):
        import time
        start = time.time()
        print("dldi")
        results = self.Y_model(src, imgsz=640, verbose=False, device=self.DEVICE)
        print("YOLO time:", time.time()-start)
        print(self.Y_model.device)
        result = results[0]
        print("BOX COUNT:", len(result.boxes))
        if len(result.boxes) == 0:
            return False
        boxes = result.boxes.xyxy.cpu().numpy()
        print("BOXES:", boxes)
        return boxes

    def crop_video(self, src, boxes):
        print("video?")
        areas = []
        for x1, y1, x2, y2 in boxes:
            areas.append((x2 - x1) * (y2 - y1))

        idx = np.argmax(areas)
        x1, y1, x2, y2 = boxes[idx].astype(np.int32)

        x1 -= self.MARGIN
        y1 -= self.MARGIN
        x2 += self.MARGIN
        y2 += self.MARGIN

        h, w = src.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop_img = src[y1:y2, x1:x2]

        if crop_img.size == 0:
            print("videoX")
            return False
        
        print("videoO")
        return crop_img
        
    def run(self, src):
        print("why")
        boxes = self.detect_objects(src)
        print("4")
        if boxes is False:
            return
        print("1")
        crop_img = self.crop_video(src, boxes)
        if crop_img is False:
            return
        print("2")
        if crop_img is not None:
            crop_msg = self.bridge.cv2_to_imgmsg(
                crop_img,
                "bgr8"
            )

            self.crop_pub.publish(crop_msg)

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
