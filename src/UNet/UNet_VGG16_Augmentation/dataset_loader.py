import glob
import os
import natsort

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
dir_raw = '../docs/Segmentation_Rigid_Training/Training/OP3/Raw'
dir_raw_list = os.listdir(dir_raw)
dir_raw_list = natsort.natsorted(dir_raw_list)

dir_mask = '../docs/Segmentation_Rigid_Training/Training/OP3/Masks'
dir_mask_list = os.listdir(dir_mask)
dir_mask_list = natsort.natsorted(dir_mask_list)

# for file in dir_raw:
#     if not path.exists(dir_train + '\\' + file):
#         shutil.copyfile(dir_raw + '\\' + file, dir_train + '\\' + file)

# shutil.copytree(dir_raw, dir_train, dirs_exist_ok=True)
for idx, file in enumerate(dir_raw_list):
    if 'raw' in file:
        shutil.copy(dir_raw + '/' + file, dir_train + '/raw' + '/' + 'img_op3_'+str(idx+1).zfill(4)+'.png')


for idx, file in enumerate(dir_mask_list):
    if 'class' in file:
        shutil.copy(dir_mask + '/' + file, dir_train + '/masks' + '/' + 'img_op3_'+str(int(idx/2+1)).zfill(4)+'_class.png')




raw_dir = '../docs/Segmentation_Rigid_Training/Training/OP1/Raw'
mask_dir = '../docs/Segmentation_Rigid_Training/Training/OP1/Masks'

raw_file_list = glob.glob(f"{raw_dir}/*.png")
mask_file_list = glob.glob(f"{mask_dir}/*class*.png")

print(len(raw_file_list))
print(len(mask_file_list))
for f in raw_file_list:
    print(f)

# n_img = len(raw_file_list)
# img_input = Image.open(raw_file_list[0])
# img_width, img_height = img_input.size()
#
# train_raw = np.zeros(n_img, img_width, img_height)
# train_mask = np.zeros(n_img, img_width, img_height)
# test_raw = np.zeros(n_img, img_width, img_height)
# test_mask = np.zeros(n_img, img_width, img_height)




















if __name__ == "__main__":
    dir = '../docs/Segmentation_Rigid_Training/Training/OP1'








