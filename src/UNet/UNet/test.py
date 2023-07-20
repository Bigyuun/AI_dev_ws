import os
import sys
import random
import warnings
import glob

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
# from itertools import chain
# from skimage.io import imread, imshow, imread_collection, concatenate_images
# from skimage.transform import resize
# from skimage.morphology import label

import tensorflow as tf
tf.config.run_functions_eagerly(True)
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# from keras.layers.merge import concatenate
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras import backend as K

import UNet_keras

def preprocess_image(image, width, height):
    # resize image
    image = tf.image.resize(image, (width, height))

    # Nomalization
    image = image/255.0

    return image

# 학습 데이터와 라벨을 묶어주는 제너레이터 생성
def train_data_generator():
    while True:
        image_batch = train_image_generator.next()
        label_batch = train_label_generator.next()
        yield image_batch, label_batch

def val_data_generator():
    while True:
        image_batch = val_image_generator.next()
        label_batch = val_label_generator.next()
        yield image_batch, label_batch

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Preparing Data
train_data_dir = './dataset/train/raw'
label_data_dir = './dataset/train/masks'

path_test_raw = './dataset/test/raw'
path_test_mask = './dataset/test/masks'

train_data_list = glob.glob(f"{train_data_dir}/*.png")

n_img = len(train_data_list)
img_input = Image.open(train_data_list[0])
img_width, img_height = img_input.size  # 640 x 480

batch_size = 4
n_epoch = 10

# ImageDataGenerator 생성 및 설정
data_generator = ImageDataGenerator(preprocessing_function=preprocess_image,  # 사용자 정의 전처리 함수
                                    validation_split=0.2,  # 학습 데이터의 일부를 검증 데이터로 분할
                                    rotation_range=20,  # 랜덤하게 이미지를 회전시키는 범위 (도)
                                    width_shift_range=0.2,  # 랜덤하게 이미지를 수평 이동시키는 범위
                                    height_shift_range=0.2,  # 랜덤하게 이미지를 수직 이동시키는 범위
                                    zoom_range=0.2,  # 랜덤하게 이미지를 확대/축소하는 범위
                                    horizontal_flip=True,  # 랜덤하게 이미지를 수평으로 뒤집기
                                    vertical_flip=True,  # 랜덤하게 이미지를 수직으로 뒤집기
                                    )
# 학습 데이터 제너레이터
train_image_generator = data_generator.flow_from_directory(
                                    train_data_dir,
                                    target_size=(img_width, img_height),
                                    batch_size=batch_size,
                                    class_mode='input',  # None으로 설정하여 라벨 없이 이미지만 로드
                                    color_mode='rgb',  # RGB 이미지 사용
                                    subset='training',  # 학습 데이터
                                    )

# 라벨링된 데이터 제너레이터
train_label_generator = data_generator.flow_from_directory(
                                    label_data_dir,
                                    target_size=(img_width, img_height),
                                    batch_size=batch_size,
                                    class_mode='categorical',  # None으로 설정하여 라벨 없이 이미지만 로드
                                    color_mode='grayscale',  # 라벨은 흑백 이미지로 로드
                                    subset='training',  # 학습 데이터
                                    )


val_image_generator = data_generator.flow_from_directory(
                                    train_data_dir,
                                    target_size=(img_width, img_height),
                                    batch_size=batch_size,
                                    class_mode='input',
                                    subset='validation',  # 검증 데이터
                                    color_mode='rgb',  # RGB 이미지 사용
                                    )

val_label_generator = data_generator.flow_from_directory(
                                    label_data_dir,
                                    target_size=(img_width, img_height),
                                    batch_size=batch_size,
                                    class_mode='categorical',
                                    color_mode='grayscale',
                                    subset='validation',  # 검증 데이터
                                    )

model = UNet_keras.unet(n_classes=3, input_size=(img_width, img_height, 3))

model.fit(train_data_generator(),
          epochs=n_epoch,
          steps_per_epoch=len(train_image_generator),
          validation_data=val_data_generator,
          validation_steps=len(val_image_generator)
          )




























