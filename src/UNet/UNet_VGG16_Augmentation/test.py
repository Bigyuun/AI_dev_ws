# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile
# from std_msgs.msg import String
# from sensor_msgs.msg import Image
# from sensor_msgs.msg import CameraInfo
# from sensor_msgs.msg import Imu
#
# from rclpy.action import ActionServer
# from rclpy.callback_groups import ReentrantCallbackGroup
# from rclpy.node import Node
# from rclpy.qos import QoSDurabilityPolicy
# from rclpy.qos import QoSHistoryPolicy
# from rclpy.qos import QoSProfile
# from rclpy.qos import QoSReliabilityPolicy
import os
import tensorflow as tf
MP = os.path.join("../resource", "model.h5")
M = tf.keras.models.load_model(MP)

import cv2
import numpy as np
# from cv_bridge import CvBridge

# Tensorflow
import os
# import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
# from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import f1_score, jaccard_score
from UNet_train import create_dir, load_dataset



g_depth = 10
g_frame_shape = (480, 640, 3)

class UNetProcessor(Node):
  """_summary_
  author - DY
  Inter(R) RealSense Subscriber Node
  It premise that you use the 'Depth' Camera.
  """

  def __init__(self):
    """_summary_
    ROS2 QoS setup and making subscriber for each topics.
    """
    super().__init__("unet_process_node")
    qos_profiile = QoSProfile(depth=g_depth)
    self.rgb_frame = np.zeros(g_frame_shape)
    self.callback_group = ReentrantCallbackGroup()
    self.model_path = os.path.join("../resource", "model.h5")
    self.model = tf.keras.models.load_model(self.model_path)
    self.model.summary()
    self.rgb_codes = [
        [0, 0, 0],
        [70, 70, 70],
        [160, 160, 160]
        ]

    ## Camera calibration and metadata
    self.color_camera_info_subscriber = self.create_subscription(
      CameraInfo,
      "color/camera_info",
      self.color_camera_info_callback,
      qos_profiile)

    ## color rectified image. RGB format
    self.color_image_rect_raw_subscriber = self.create_subscription(
      Image,
      "color/image_rect_raw",
      self.color_image_rect_raw_callback,
      qos_profiile)

    ## Camera calibration and metadata
    self.depth_camera_info_subscriber = self.create_subscription(
      CameraInfo,
      "depth/camera_info",
      self.depth_camera_info_callback,
      qos_profiile)

    ## Raw image from device. Contains uint16depth in mm.
    self.depth_image_rect_raw_subscriber = self.create_subscription(
      Image,
      "depth/image_rect_raw",
      self.depth_image_rect_raw_callback,
      qos_profiile)

    self.imu_subscriber = self.create_subscription(
      Imu,
      "imu",
      self.imu_callback,
      qos_profiile)

    self.br_rgb = CvBridge()


  def color_camera_info_callback(self, data):
    return

  def color_image_rect_raw_callback(self, data):
    self.get_logger().info("Receiving RGB frame")
    self.rgb_frame = self.br_rgb.imgmsg_to_cv2(data, 'bgr8')
    if self.rgb_frame.shape != g_frame_shape:
      self.rgb_frame = np.resize(self.rgb_frame, g_frame_shape)
    self.rgb_frame = self.rgb_frame/255.0
    self.rgb_frame = np.expand_dims(self.rgb_frame, axis=0)
    self.rgb_frame = self.rgb_frame.astype(np.float32)

    # prediction
    pred = self.model.predict(self.rgb_frame)
    pred = np.argmax(pred, axis=-1)
    pred= pred.astype(np.int32)
    pred = np.expand_dims(pred, axis=-1)
    pred = grayscale_to_rgb(pred, self.rgb_codes)

    cv2.imshow("prediction", pred)
    cv2.waitKey(1)
    return

  def depth_camera_info_callback(self, data):
    return

  def depth_image_rect_raw_callback(self, data):
    return

  def imu_callback(self, data):
    return

  # custom function
  def grayscale_to_rgb(mask, rgb_codes):
    h, w = mask.shape[0], mask.shape[1]
    mask = mask.astype(np.int32)
    output = []

    for i, pixel in enumerate(mask.flatten()):
        output.append(rgb_codes[pixel])

    output = np.reshape(output, (h, w, 3))
    return output


def main(args=None):
  rclpy.init(args=args)
  node = UNetProcessor()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    node.get_logger().info('Keyboard Interrupt (SIGINT)')
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main":
  main()
