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


mask_dir = '../docs/Segmentation_Rigid_Training/Training/OP1/Masks'
file_list = glob.glob(f"{mask_dir}/*class*.png")

print(len(file_list))
for f in file_list:
    print(f)

mask_filename = glob.glob(mask_dir)


if __name__ == "__main__":
    dir = '../docs/Segmentation_Rigid_Training/Training/OP1'








