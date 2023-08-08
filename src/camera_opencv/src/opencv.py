import pyrealsense2 as rs
import numpy as np
import cv2

pipe = rs.pipeline()
cfg = rs.config()

# D405 has revolutions of 720p on RGB and 640p on depth
cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)

# Start steaming
profile = pipe.start(cfg)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scael : {}".format(depth_scale))



frame_loss = 0
while True:

    # wait for a coherent pair of frames : depth and color
    frame = pipe.wait_for_frames()
    depth_frame = frame.get_depth_frame()
    color_frame = frame.get_color_frame()
    if not depth_frame or not color_frame:
        frame_loss = frame_loss + 1
        print("frame loss : {}".format(frame_loss))
        continue


    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha = 0.5),
                                 cv2.COLORMAP_JET)

    gray_image = cv2.cvtColor(color_image,
                              cv2.COLOR_BGR2GRAY)

    cv2.imshow('rgb', color_image)
    cv2.imshow('depth', depth_cm)

    if cv2.waitKey(1) == ord('q'):
        break

pipe.stop()