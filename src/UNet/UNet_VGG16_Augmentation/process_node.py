import platform
import subprocess
import os
if 'Linux' in platform.system():
    print(platform.system())
    os.system("echo 0 | sudo tee -a /sys/bus/pci/devices/0000\:01\:00.0/numa_node && qwe123")

import time

import pyrealsense2 as rs
import numpy as np
import cv2

import numpy as np
import cv2
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import f1_score, jaccard_score
from UNet_train import create_dir, load_dataset

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

dataset_path = "./dataset"
model_path = os.path.join("files", "model.h5")
model = tf.keras.models.load_model(model_path)

frame_loss = 0
while True:
    '''
    Unet process
    '''
    start_time = time.time()

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

    color_input = color_image/255.0
    color_input = np.expand_dims(color_input, axis=0)
    color_input = color_input.astype(np.float32)

    pred = model.predict(color_input)[0]    # [0] means : [1, w, h, rgb] -> [w,h,rgb]
    pred = np.argmax(pred, axis=-1)
    pred = pred.astype(np.int32)
    print(pred.shape)
    temp_pred = pred
    temp_pred = np.dstack((temp_pred,)*3).astype(np.uint8)

    print(pred.shape[0], pred.shape[1])
    print(temp_pred[0,0])
    print(temp_pred[0,0,:])
    for r in range(pred.shape[0]):
        for c in range(pred.shape[1]):
            if np.array_equal(temp_pred[r,c,:], [1,1,1]):
                temp_pred[r,c,:] = [0,255,0]
            if np.array_equal(temp_pred[r,c,:], [2,2,2]):
                temp_pred[r,c,:] = [0,0,255]


    # pred = pred*255./2



    # np_horizontal = np.hstack((color_image, np.resize(pred,(480,640,3))))
    pred_rgb = np.dstack((pred,)*3).astype(np.uint8)
    pred_rgb[:, :, 1:3] = 0

    input_pred_image = np.concatenate((color_image, pred_rgb), axis=1)

    end_time = time.time()
    fr = 1 / (end_time - start_time)
    fr_str = f'FPS = {fr:.2f}'
    cv2.putText(color_image,
                fr_str,
                (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2)
    cv2.imshow('prediction', pred_rgb)
    # cv2.imshow('UNet', input_pred_image)
    cv2.imshow('depth', depth_cm)
    cv2.imshow('raw_rgb', color_image)


    if cv2.waitKey(1) == ord('q'):
        break

pipe.stop()