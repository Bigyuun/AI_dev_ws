

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

global NUM_OF_AUG_FOR_1_IMAGE
NUM_OF_AUG_FOR_1_IMAGE = 20

ENDOVIS_COLORMAP = [
    [0, 0, 0],
    [70, 70, 70],
    [160, 160, 160]
]

def custom_preprocessing(img):

    img = img * (160. / 255.)
    img = np.around(img)
    img = img.astype(np.int32)

    height = img.shape[0]
    width = img.shape[1]
    for h in range(height):
        for w in range(width):
            # print(img[h,w])
            if (img[h, w] != ENDOVIS_COLORMAP).all() :
                img[h, w] = [0, 0, 0]

    return img

rotation_range = 60
width_shift_range = 0.3
height_shift_range = 0.3
brightness_range = [0.6, 1.8]
rescale = 1.0 /255.
shear_range = 0.5
zoom_range = [0.5, 1.5]
horizontal_flip = True
vertical_flip = True
fill_mode = 'constant'
cval = 0

""" image data generator"""
datagen_raw = ImageDataGenerator(
    rotation_range=rotation_range,
    width_shift_range=width_shift_range,
    height_shift_range=height_shift_range,
    # brightness_range=brightness_range,
    # rescale=rescale,
    shear_range=shear_range,
    zoom_range=zoom_range,
    horizontal_flip=horizontal_flip,
    vertical_flip=vertical_flip,
    fill_mode=fill_mode,
    cval=cval,
    # preprocessing_function=custom_preprocessing
)

datagen_mask = ImageDataGenerator(
    rotation_range=rotation_range,
    width_shift_range=width_shift_range,
    height_shift_range=height_shift_range,
    # brightness_range=brightness_range,
    # rescale=rescale,
    shear_range=shear_range,
    zoom_range=zoom_range,
    horizontal_flip=horizontal_flip,
    vertical_flip=vertical_flip,
    fill_mode=fill_mode,
    cval=cval,
    # preprocessing_function=custom_preprocessing
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
for batch in tqdm(datagen_raw.flow(
                        np.array([cv2.cvtColor(cv2.imread(img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for img in raw_images]),
                        batch_size=120,
                        seed=42,
                        shuffle=False,
                        save_format='png',
                        save_prefix='aug_',
                        save_to_dir='./dataset/train/raw_augmented/'
                        ), total=len(raw_images)):
    raw_count += 1
    if raw_count == NUM_OF_AUG_FOR_1_IMAGE:
        break


mask_count = 0
for batch in tqdm(datagen_mask.flow(
                        np.array([cv2.cvtColor(cv2.imread(img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for img in mask_images]),
                        batch_size=120,
                        seed=42,
                        shuffle=False,
                        save_format='png',
                        save_prefix='aug_',
                        save_to_dir='./dataset/train/masks_augmented/'
                        ), total=len(mask_images)):
    mask_count += 1
    if mask_count == NUM_OF_AUG_FOR_1_IMAGE:
        break

