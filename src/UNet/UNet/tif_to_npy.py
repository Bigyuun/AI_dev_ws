## 라이브러리 불러오기
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

'''
@brief data loading
'''
# print(os.getcwd())
dir_data = '../docs/isbi-2012-master/data'
# dir_data = './docs/isbi-2012-master/data'

name_label = 'train-labels.tif'     # 512x512x30
name_input = 'train-volume.tif'     # 512x512x30

# open() function just identify image file(s)
img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

n_y, n_x = img_label.size
n_frame = img_label.n_frames

n_frame_train = int(0.8 * n_frame)
n_frame_val   = int(0.1 * n_frame)
n_frame_test  = int(0.1 * n_frame)

dir_save_train = os.path.join(dir_data, 'train')
dir_save_val   = os.path.join(dir_data, 'val')
dir_save_test  = os.path.join(dir_data, 'test')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)
if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)
if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)


# random하게 저장하기위해
id_frame = np.arange(n_frame)
np.random.shuffle(id_frame)

# save training set
offset_n_frame = 0
for i in range(n_frame_train):
    # load data
    img_label.seek(id_frame[i + offset_n_frame])
    img_input.seek(id_frame[i + offset_n_frame])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

# save validation set
offset_n_frame += n_frame_train
for i in range(n_frame_val):
    # load data
    img_label.seek(id_frame[i + offset_n_frame])
    img_input.seek(id_frame[i + offset_n_frame])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

# save test set
offset_n_frame += n_frame_val
for i in range(n_frame_test):
    # load data
    img_label.seek(id_frame[i + offset_n_frame])
    img_input.seek(id_frame[i + offset_n_frame])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

'''
show data to graph
'''
plt.subplot(121)
plt.imshow(label_, cmap='gray')
plt.title('label_')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('input_')

plt.show()























