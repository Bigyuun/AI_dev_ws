
import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # debug level
import numpy as np
import pandas as pd
import cv2
from glob import glob
import scipy.io
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
# from UNet_keras import unet
from UNet_CHIPdataset import build_unet

global IMG_H
global IMG_W
global NUM_CLASSES
global CLASSES
global COLORMAP

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path, split=0.2):
    train_x = sorted(glob(os.path.join(path, "Training", "Images", "*")))[:10000]
    train_y = sorted(glob(os.path.join(path, "Training", "Categories", "*")))[:10000]
    print(f"images: {len(train_x)}")
    print(train_x[0])
    print(train_y[0])
    split_size = int(split * len(train_x))

    train_x, valid_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(train_y, test_size=split_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def get_colormap(path):
    """ BGR format """
    mat_path = os.path.join(path, "human_colormap.mat")
    colormap = scipy.io.loadmat(mat_path)["colormap"]
    print(colormap)
    print(f"num of classes is {len(colormap)}")
    colormap = colormap * 256
    colormap = colormap.astype(np.uint8)
    print(f"0 to 255 map is \n{colormap}")

    """ RGB format """
    colormap = [ [c[2], c[1], c[0]] for c in colormap]
    print(colormap)

    classes = [
        "Background",
        "Hat",
        "Hair",
        "Glove",
        "Sunglasses",
        "UpperClothes",
        "Dress",
        "Coat",
        "Socks",
        "Pants",
        "Torso-skin",
        "Scarf",
        "Skirt",
        "Face",
        "Left-arm",
        "Right-arm",
        "Left-leg",
        "Right-leg",
        "Left-shoe",
        "Right-shoe"
    ]

    return  classes, colormap

def read_image(x):
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMG_W, IMG_H))
    x = x/255.0
    x = x.astype(np.float32)

    return x

def read_mask(x):
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMG_W, IMG_H))

    """ Mask processing """
    """ 그러니까... 여러개의 클래스가 모두 마스킹된 이미지에서
        각 클래스 하나씩 마스킹한 이미지로 분리 (e.g. 마스킹이 20개면, 20개의 mask 이미지로 추출
    """
    output = []
    for i, color in enumerate(COLORMAP):
        cmap = np.all(np.equal(x, color), axis=-1)
        output.append(cmap)
        # cv2.imwrite(f"cmap_{i}.png", cmap*255)
        # print(cmap, cmap.shape)

    output = np.stack(output, axis=-1)
    output = output.astype(np.uint8)

    return output

def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()

        x = read_image(x)
        y = read_mask((y))

        return x, y

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.uint8])
    image.set_shape([IMG_H, IMG_W, 3])
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

    """ Directory to save files """
    create_dir("files")

    """ Hyperparameters """
    IMG_H = 480
    IMG_W = 360
    NUM_CLASSES = 20
    input_shape = (IMG_H, IMG_W, 3)

    batch_size = 1
    lr = 1e-4
    num_epochs = 100

    dataset_path = "./instance-level-human-parsing/instance-level_human_parsing/instance-level_human_parsing"
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "data.csv")

    """ Loading the dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    print(
        f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")
    print("")

    """ Process the colormap """
    CLASSES, COLORMAP = get_colormap(dataset_path)

    # read_mask(train_y[0])

    """ Dataset PipeLine """
    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    """ Model """
    model = build_unet(input_shape=input_shape, num_classes=NUM_CLASSES)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
    )

    model.summary()

    """ Training """
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























