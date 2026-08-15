import os
import cv2
import queue
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy)
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

class DataCollector(Node):
    def __init__(self):
        super().__init__("DataCollector")
        self.SAVE_FILE = r"/home/rtree/ros2_ws/src/run_ai/image"
        os.makedirs(self.SAVE_FILE, exist_ok=True)
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )
        # callback group 생성
        self.group = ReentrantCallbackGroup()

        self.videoSubscriber = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.videoSubscriber_callback,
            qos,
            callback_group=self.group
        )
        self.timer = self.create_timer(
            0.1,
            self.save_timer_callback,
            callback_group=self.group
        )
        self.bridge = CvBridge()
        self.src_pub = self.create_publisher(
            Image,
            "/yolo/src_image",
            10
        )
        self.save_queue = queue.Queue(maxsize=1)
        self.frame_count = 0
        self.frame_interval = 15
        self.save_count = 272


    def videoSubscriber_callback(self, msg):
        self.get_logger().info("videoSubscriber_callback")
        try:
            print("1")
            src = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            src = cv2.convertScaleAbs(src, alpha=1.2, beta=-70)
            # 지금은 사용하는 이유 존재 X (카메라의 밝기나 대비를 조정한 경우 사용)
            """
            src_msg = self.bridge.cv2_to_imgmsg(
                src,
                "bgr8"
            )
            self.src_pub.publish(src_msg)
            #print("pub")
            """
            self.frame_count += 1
            if self.frame_count >= self.frame_interval:
                self.frame_count = 0
                try:
                    # Queue에 넣고 callback 종료
                    self.save_queue.put_nowait(src)
                # 예외 설명 queue가 가득 차있다는 것은 저장이 오래걸려 큐에서 아직 값이 안 빠져나가가 것으로 간주한다
                except queue.Full:
                    # 병목을 줄이고 최신 프레임을 유지하기 위해 프레임 버리고 다시 받음
                    self.save_queue.get_nowait()
                    self.save_queue.put_nowait(src)
                    self.get_logger().warn("Save queue full - frame skipped")
        except Exception as e:
            self.get_logger().info(f'{e}')
            return

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
