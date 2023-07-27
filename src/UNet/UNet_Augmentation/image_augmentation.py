

import platform
import os
if 'Linux' in platform.system():
    print(platform.system())
    os.system("echo 0 | sudo tee -a /sys/bus/pci/devices/0000\:01\:00.0/numa_node && qwe123")

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image


""" image data generator"""
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    # rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='constant',
    cval=0
)
# raw_images_path = os.path.join('./dataset/train/raw/')
# mask_images_path = os.path.join('./dataset/train/graymasks_normalized/')
raw_images_path = './dataset/train/raw'
mask_images_path = './dataset/train/graymasks_normalized'
print(sorted(glob(os.path.join(raw_images_path, '*.png'))))
#
# raw_datagen = datagen.flow_from_directory(
#     raw_images_path,
#     target_size=(480, 640),
#     batch_size=4,
#     class_mode=None,
#     classes=None,
#     seed=42,
#     # save_format='png',
#     # save_to_dir='./dataset/train/raw_augmented'
# )
# mask_datagen = datagen.flow_from_directory(
#     mask_images_path,
#     target_size=(480, 640),
#     batch_size=4,
#     class_mode=None,
#     classes=None,
#     seed=42,
#     # save_format='png',
#     # save_to_dir='./dataset/train/graymasks_normalized_augmented'
# )

# print(len(raw_datagen))
# print(len(mask_datagen))


# raw 이미지와 mask 이미지 불러오기
raw_images = sorted(glob(os.path.join(raw_images_path, '*.png')))
mask_images = sorted(glob(os.path.join(mask_images_path, '*.png')))

# 이미지 데이터 묶기
raw_datagen = datagen.flow(
    np.array([cv2.cvtColor(cv2.imread(img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for img in raw_images]),
    batch_size=32,
    seed=42,
    shuffle=False,
    save_format='png',
    save_prefix='aug_',
    save_to_dir='./dataset/train/raw_augmented/'
)
mask_datagen = datagen.flow(
    np.array([cv2.cvtColor(cv2.imread(img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for img in mask_images]),
    batch_size=32,
    seed=42,
    shuffle=False,
    save_format='png',
    save_prefix='aug_',
    save_to_dir='./dataset/train/graymasks_normalized_augmented/'
)

# raw_count = 0
# for i, x in tqdm(enumerate(raw_datagen), total=len(raw_datagen)):
#     raw_count += 1
#     if raw_count == 10:
#         break
#
# mask_count = 0
# for j, y in tqdm(enumerate(mask_datagen), total=len(mask_datagen)):
#     mask_count += 1
#     if mask_count == 10:
#         break

raw_count = 0
for batch in tqdm(datagen.flow(
                        np.array([cv2.cvtColor(cv2.imread(img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for img in raw_images]),
                        batch_size=120,
                        seed=42,
                        shuffle=False,
                        save_format='png',
                        save_prefix='aug_',
                        save_to_dir='./dataset/train/raw_augmented/'
                        ), total=len(raw_images)):
    raw_count += 1
    if raw_count == 10:
        break


mask_count = 0
for batch in tqdm(datagen.flow(
                        np.array([cv2.cvtColor(cv2.imread(img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for img in mask_images]),
                        batch_size=120,
                        seed=42,
                        shuffle=False,
                        save_format='png',
                        save_prefix='aug_',
                        save_to_dir='./dataset/train/graymasks_normalized_augmented/'
                        ), total=len(mask_images)):
    mask_count += 1
    if mask_count == 10:
        break

# for i, x in tqdm(enumerate(raw_datagen), total=len(raw_datagen)):
#     for j in range(len(x)):
#         img_raw = Image.fromarray((x[j]*255).astype('uint8'))
#         img_raw_path =


# raw_datagen과 mask_datagen의 길이 확인
print(len(raw_datagen))  # raw 이미지의 개수
print(len(mask_datagen))  # mask 이미지의 개수