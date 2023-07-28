import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm

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

image_filter_dir = './dataset/train/masks_augmented_filter0'
image_file = './dataset/train/masks_augmented/aug__60_1.png'
src = cv2.imread(image_file, cv2.IMREAD_COLOR)

images_dir = sorted(glob(os.path.join('./dataset/train/masks_augmented', '*.png')))
print('total images : ', len(images_dir))
for idx, name in tqdm(enumerate(images_dir), total=len(images_dir)):


    img = cv2.imread(name, cv2.IMREAD_COLOR)
    img = img * (160.0 / 255)
    img = np.around(img)
    img = img.astype(np.int32)
    height = img.shape[0]
    width = img.shape[1]

    for h in range(height):
        for w in range(width):
            # print(img[h,w])
            if (img[h,w] != ENDOVIS_COLORMAP).all() :
                img[h,w] = [0, 0, 0]

    file_name = name.split('/')[-1].split('.')[0]
    cv2.imwrite(f"{image_filter_dir}/{file_name}.png", img)




print('filter done')