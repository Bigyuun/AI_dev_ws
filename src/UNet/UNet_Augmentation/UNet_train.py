import platform
import subprocess
import os
if 'Linux' in platform.system():
    print(platform.system())
    os.system("echo 0 | sudo tee -a /sys/bus/pci/devices/0000\:01\:00.0/numa_node && qwe123")

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import PyQt5
from glob import glob
import matplotlib.pyplot as plt
import scipy.io
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from UNet_model import build_unet
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image



global IMG_H
global IMG_W
global NUM_CLASSES
global CLASSES
global RGB_CODES

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Load and split the dataset """
def load_dataset(path, split=0.2):
    """
    Dataset for train, valid, test
    """
    train_x = sorted(glob(os.path.join(path, "train", "raw_augmented", "*.png")))
    train_y = sorted(glob(os.path.join(path, "train", "graymasks_normalized_augmented", "*.png")))

    split_size = split
    train_x, valid_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(train_y, test_size=split_size, random_state=42)

    test_x = sorted(glob(os.path.join(path, "test", "raw", "*.png")))
    test_y = sorted(glob(os.path.join(path, "test", "graymasks_normalized", "*.png")))

    # print(train_x, train_y, valid_x, valid_y, test_x, test_y)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

# def get_colormap(path):
#     mat_path = os.path.join(path, "human_colormap.mat")
#     colormap = scipy.io.loadmat(mat_path)["colormap"]
#     colormap = colormap * 256
#     colormap = colormap.astype(np.uint8)
#     colormap = [[c[2], c[1], c[0]] for c in colormap]
#
#     classes = [
#         "Background",
#         "Hat",
#         "Hair",
#         "Glove",
#         "Sunglasses",
#         "UpperClothes",
#         "Dress",
#         "Coat",
#         "Socks",
#         "Pants",
#         "Torso-skin",
#         "Scarf",
#         "Skirt",
#         "Face",
#         "Left-arm",
#         "Right-arm",
#         "Left-leg",
#         "Right-leg",
#         "Left-shoe",
#         "Right-shoe"
#     ]
#
#     return classes, colormap

def read_image_mask(x, y):
    """ Reading """
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    # assert x.shape == y.shape, "size of raw image and mask image doesn't match"
    # print(x.shape, y.shape)

    # Raw Images
    x = cv2.resize(x, (IMG_W, IMG_H))
    x = x / 255.0
    x = x.astype(np.float32)
    # Mask Images
    # check y is normalized (if not, you can add normalzing code
    y = cv2.resize(y, (IMG_W, IMG_H))
    # y = y / 255.0
    y = y.astype(np.int32)

    # """ Mask processing """
    # output = []
    # for color in COLORMAP:
    #     cmap = np.all(np.equal(y, color), axis=-1)
    #     output.append(cmap)
    # output = np.stack(output, axis=-1)
    # output = output.astype(np.uint8)
    #
    # return x, output

    return x, y

def preprocess(x, y):
    def func(x, y):
        x = x.decode()
        y = y.decode()
        x, y = read_image_mask(x,y)
        return x, y

    image, mask = tf.numpy_function(func, [x, y], [tf.float32, tf.int32])
    mask = tf.one_hot(mask, NUM_CLASSES)

    image.set_shape([IMG_H, IMG_W, 3])  # 3 : RGB
    mask.set_shape([IMG_H, IMG_W, NUM_CLASSES])

    return image, mask

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Hyperparameters """
    IMG_H = 480
    IMG_W = 640
    NUM_CLASSES = 3
    input_shape = (IMG_H, IMG_W, 3)
    batch_size = 4
    lr = 1e-4
    num_epochs = 100
    dataset_path = "./dataset"

    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "data.csv")

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

    """ Loading the dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")
    print("")
    # print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)
    # """ Process the colormap """
    # # CLASSES, COLORMAP = get_colormap(dataset_path)

    """ Dataset Pipeline """
    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    # """ Model """
    model = build_unet(input_shape, NUM_CLASSES)
    # model.load_weights(model_path)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr)
    )
    # model.summary()

    # """ Training """
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path, append=True),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    model.fit(train_dataset,
        validation_data=valid_dataset,
        epochs=num_epochs,
        callbacks=callbacks
    )















