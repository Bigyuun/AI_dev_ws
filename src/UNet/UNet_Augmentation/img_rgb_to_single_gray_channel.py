
"""
references : https://github.com/nikhilroxtomar/RGB-Mask-to-Single-Channel-Mask-for-Multiclass-Segmentation/blob/main/rgb_mask_to_single_channel_mask.py
video : https://www.youtube.com/watch?v=WYCvYLwIltk
pre : class image has multi-class data
comments : convert rgb-images to gray-images(1 channel) for multi-class segmentation
            so, this modul will make 1-class only images for all classes
            클래스 분리해서 각자 저장한다고~
"""

import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

GLOBAL_GRAYMASK_DIR = './dataset/train/masks_augmented_filter0'
GLOBAL_GRAYMASK_NORMALIZE_DIR = './dataset/train/graymasks_normalized_augmented'
""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_mask(rgb_mask, colormap, file_name):
    output_mask = []

    # np.equal : numpy 요소별 비교 (True or False)
    # np.all : 행 비교 (혹은 행렬 자체 비교)
    for i, color in enumerate(colormap):
        cmap = np.all(np.equal(rgb_mask, color), axis=-1)
        # cv2.imwrite(f"{GLOBAL_GRAYMASK_DIR}/{file_name}_{i}.png", cmap*255)
        output_mask.append(cmap)

    output_mask = np.stack(output_mask, axis=-1)

    return output_mask

if __name__ == "__main__":

    """ Create Directory """
    create_dir(GLOBAL_GRAYMASK_DIR)
    create_dir(GLOBAL_GRAYMASK_NORMALIZE_DIR)

    dataset_path = './dataset'
    images = sorted(glob(os.path.join(dataset_path, 'train/raw_augmented', '*.png')))
    masks = sorted(glob(os.path.join(dataset_path, 'train/masks_augmented_filter0', '*.png')))

    print(f"Image(raw) : {len(images)}")
    print(f"Mask(masks) : {len(masks)}")

    # EndoVis 2014 Sub-Challenge's colormap and class names"
    ENDOVIS_COLORMAP = [
        [0, 0, 0],
        [70, 70, 70],
        [160, 160, 160]
    ]
    ENDOVIS_CLASSES = [
        'background',
        'manipulator',
        'shaft'
    ]

    # for display
    for name, color in zip(ENDOVIS_CLASSES, ENDOVIS_COLORMAP):
        print(f"{name} - {color}")

    # Loop over the images and masks
    for x, y in tqdm(zip(images, masks), total=len(images)):
        """ extract the file name """
        file_name = x.split('/')[-1].split('.')[0]
        file_name_masks = y.split('/')[-1].split('.')[0]
        # print(name)

        image_raw = cv2.imread(x, cv2.IMREAD_COLOR)
        mask_raw = cv2.imread(y, cv2.IMREAD_COLOR)

        # image_raw = cv2.resize(image_raw, (640, 480))
        # mask_raw = cv2.resize(mask_raw, (640, 480))
        a = np.argmax(image_raw)
        a = np.max(image_raw)
        a2 = np.unique(image_raw)
        b = np.argmax(mask_raw)
        b = np.max(mask_raw)
        b2 = np.unique(mask_raw)

        processed_mask = process_mask(mask_raw, ENDOVIS_COLORMAP, file_name)   # one-hot encoding (True or False), 3-D channel
        grayscale_mask = np.argmax(processed_mask, axis=-1)     # 1-D channel, 2차원 numpy
        grayscale_mask = np.expand_dims(grayscale_mask, axis=-1)   # 3차원 numpy
        cv2.imwrite(f"{GLOBAL_GRAYMASK_NORMALIZE_DIR}/{file_name_masks}.png", grayscale_mask)

        ''' 원본 / 라벨 / 예측 이미지 저장'''
        # GLOBAL_TOTALMASK_DIR = './dataset/train'
        # create_dir(GLOBAL_TOTALMASK_DIR)

        # line = np.ones((image_raw.shape[1], image_raw.shape[0], 3))*255 # 경계선 넣고싶으면 씀
        # cat_images = np.concatenate([
        #     image_raw, mask_raw, np.concatenate([grayscale_mask, grayscale_mask, grayscale_mask], axis=-1) # 3차원이라 gray는 3개 써준거
        # ], axis=1)
        # cv2.imwrite(f"{GLOBAL_TOTALMASK_DIR}/{file_name}.png", cat_images)



























