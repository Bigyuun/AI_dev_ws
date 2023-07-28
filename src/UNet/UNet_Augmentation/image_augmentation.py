

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
    brightness_range=None,
    rescale=1.0 /255,
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
mask_images_path = './dataset/train/masks'
print(sorted(glob(os.path.join(raw_images_path, '*.png'))))

# raw 이미지와 mask 이미지 불러오기
raw_images = sorted(glob(os.path.join(raw_images_path, '*.png')))
mask_images = sorted(glob(os.path.join(mask_images_path, '*.png')))

r = np.array([cv2.cvtColor(cv2.imread(img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for img in raw_images])
print(r.shape)
print(np.max(r))
print(np.unique(r))

f = np.array([cv2.cvtColor(cv2.imread(img, cv2.IMREAD_GRAYSCALE), cv2.COLOR_BGR2RGB) for img in mask_images])
print(f.shape)
print(np.max(f))
print(np.unique(f))


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
                        save_to_dir='./dataset/train/masks_augmented/'
                        ), total=len(mask_images)):
    mask_count += 1
    if mask_count == 10:
        break

