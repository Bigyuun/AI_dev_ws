import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

delta = 30

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


raw_image_path = './dataset/train/raw_augmented'
image_files_path = sorted(glob(os.path.join(raw_image_path, '*.png')))

save_path = './dataset/train/raw_augmented_randombrightness'
create_dir('./dataset/train/raw_augmented_randombrightness')
for i, file_path in tqdm(enumerate(image_files_path), total=len(image_files_path)):
    gamma = random.randint(-delta, delta)
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    image = cv2.add(image, gamma)

    file_name = file_path.split('/')[-1].split('.')[0]
    cv2.imwrite(f"{save_path}/{file_name}.png", image)


