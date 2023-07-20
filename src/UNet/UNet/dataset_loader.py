import glob

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image

'''
EndoVis datasets
'''
print(os.getcwd())

raw_dir = '../docs/Segmentation_Rigid_Training/Training/OP1/Raw'
mask_dir = '../docs/Segmentation_Rigid_Training/Training/OP1/Masks'

raw_file_list = glob.glob(f"{raw_dir}/*.png")
mask_file_list = glob.glob(f"{mask_dir}/*class*.png")

print(len(raw_file_list))
print(len(mask_file_list))
for f in raw_file_list:
    print(f)

n_img = len(raw_file_list)
img_input = Image.open(raw_file_list[0])
img_width, img_height = img_input.size

train_raw = np.zeros(n_img, img_width, img_height)
train_mask = np.zeros(n_img, img_width, img_height)
test_raw = np.zeros(n_img, img_width, img_height)
test_mask = np.zeros(n_img, img_width, img_height)




















if __name__ == "__main__":
    dir = '../docs/Segmentation_Rigid_Training/Training/OP1'








