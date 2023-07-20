import os
from os import path
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

base_dir = "../docs/Segmentation_Rigid_Training"
dir_train = './dataset/train'
dir_test = './dataset/test'

if not os.path.exists(dir_train):
    os.makedirs(dir_test)

if not os.path.exists(dir_test):
    os.makedirs(dir_test)

# dir_raw = os.path.join(base_dir, 'Training/OP1/Raw')
dir_raw = '../docs/Segmentation_Rigid_Training/Training/OP4/Raw'
dir_raw_list = os.listdir(dir_raw)

dir_mask = '../docs/Segmentation_Rigid_Training/Training/OP4/Masks'
dir_mask_list = os.listdir(dir_mask)

# for file in dir_raw:
#     if not path.exists(dir_train + '\\' + file):
#         shutil.copyfile(dir_raw + '\\' + file, dir_train + '\\' + file)

# shutil.copytree(dir_raw, dir_train, dirs_exist_ok=True)
for idx, file in enumerate(dir_raw_list):
    if 'raw' in file:
        shutil.copy(dir_raw + '\\' + file, dir_test + '\\raw' + '\\'  + 'img_op4_'+str(idx+1)+'.png')


for idx, file in enumerate(dir_mask_list):
    if 'class' in file:
        shutil.copy(dir_mask + '\\' + file, dir_test + '\\masks' + '\\'  + 'img_op4_'+str(int(idx/2+1))+'_class.png')
