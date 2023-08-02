
import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

GLOBAL_GRAYMASK_DIR = './dataset/train/graymasks'
GLOBAL_TOTALMASK_DIR = './dataset/train'
GLOBAL_GRAYMASK_NORMALIZE_DIR = './dataset/train/graymasks_normalized'

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
        cv2.imwrite(f"{GLOBAL_GRAYMASK_DIR}/{file_name}_{i}.png", cmap*255)
        output_mask.append(cmap)

    output_mask = np.stack(output_mask, axis=-1)

    return output_mask

if __name__ == "__main__":

    """ Create Directory """
    create_dir(GLOBAL_GRAYMASK_DIR)
    create_dir(GLOBAL_TOTALMASK_DIR)
    create_dir(GLOBAL_GRAYMASK_NORMALIZE_DIR)

    dataset_path = './dataset'
    images = sorted(glob(os.path.join(dataset_path, 'train/graymasks_normalized', '*.png')))

    print(f"Image(raw) : {len(images)}")

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
    for x in tqdm(zip(images), total=len(images)):
        """ extract the file name """
        x = ','.join(x)
        file_name = x.split('/')[-1].split('.')[0]
        # print(name)

        image_raw = cv2.imread(x, cv2.IMREAD_COLOR)

        # image_raw = cv2.resize(image_raw, (640, 480))
        # mask_raw = cv2.resize(mask_raw, (640, 480))
        a = np.argmax(image_raw)
        a = np.max(image_raw)
        a2 = np.unique(image_raw)
























