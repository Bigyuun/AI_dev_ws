import platform
import subprocess
import os
if 'Linux' in platform.system():
    print(platform.system())
    os.system("echo 0 | sudo tee -a /sys/bus/pci/devices/0000\:01\:00.0/numa_node && qwe123")

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import f1_score, jaccard_score
from UNet_train import create_dir, load_dataset


global image_h
global image_w
global num_classes
global classes
global rgb_codes

def grayscale_to_rgb(mask, rgb_codes):
    h, w = mask.shape[0], mask.shape[1]
    mask = mask.astype(np.int32)
    output = []

    for i, pixel in enumerate(mask.flatten()):
        output.append(rgb_codes[pixel])

    output = np.reshape(output, (h, w, 3))
    return output

def save_results(image_x, mask, pred, save_image_path):
    mask = np.expand_dims(mask, axis=-1)
    mask = grayscale_to_rgb(mask, rgb_codes)

    pred = np.expand_dims(pred, axis=-1)
    pred = grayscale_to_rgb(pred, rgb_codes)

    line = np.ones((image_x.shape[0], 10, 3)) * 255

    cat_images = np.concatenate([image_x, line, mask, line, pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("results")

    """ Hyperparameters """
    image_h = 480
    image_w = 640
    num_classes = 3

    """ Paths """
    dataset_path = "./dataset"
    model_path = os.path.join("files", "model.h5")

    """ RGB Code and Classes """
    rgb_codes = [
        [0, 0, 0],
        [70, 70, 70],
        [160, 160, 160]
    ]

    classes = [
        'background',
        'manipulator',
        'shaft'
    ]

    """ Loading the dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")
    print("")

    """ Load the model """
    model = tf.keras.models.load_model(model_path)
    # model.summary()


    """ Prediction & Evaluation """
    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]
        # print(name)

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (image_w, image_h))
        image_x = image
        image = image/255.0     # (H, W, 3)
        image = np.expand_dims(image, axis=0)   # [1, H, W, 3] = [batch, H, W, rgb]
        image = image.astype(np.float32)

        """ Reading the mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (image_w, image_h))
        mask = mask.astype(np.int32)

        # print(image.shape)
        # print(mask.shape)


        """ Prediction """
        pred = model.predict(image, verbose=0)[0]
        pred = np.argmax(pred, axis=-1) ## [0.1, 0.2, 0.1, 0.6] -> 3
        pred = pred.astype(np.int32)

        # cv2.imwrite("pred.png", pred*(255/num_classes))
        # break

        ## cv2.imwrite("pred.png", pred * (255/11))

        rgb_mask = grayscale_to_rgb(pred, rgb_codes)
        # cv2.imwrite("pred.png", rgb_mask)

        """ Save the results """
        save_image_path = f"results/{name}.png"
        save_results(image_x, mask, pred, save_image_path)

        """ Flatten the array """
        mask = mask.flatten()
        pred = pred.flatten()

        labels = [i for i in range(num_classes)]

        """ Calculating the metrics values """
        f1_value = f1_score(mask, pred, labels=labels, average=None, zero_division=0)
        jac_value = jaccard_score(mask, pred, labels=labels, average=None, zero_division=0)

        SCORE.append([f1_value, jac_value])

    score = np.array(SCORE)
    score = np.mean(score, axis=0)
    print(score)


    """
    display 5 result images
    """
    plt.figure()
    img_result = sorted(glob(os.path.join("./results", "*.png")))
    for i, filename in enumerate(img_result):
        img = cv2.imread(filename)
        ax = plt.subplot(5, 1, i + 1)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if i is 4:
            break
    plt.show()

    #     for i in range(8):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.axis("off")
    #
    # plt.show()


    #
    # f = open("files/score.csv", "w")
    # f.write("Class,F1-score,Jaccard\n")

    # l = ["Class", "F1-score", "Jaccard"]
    # print(f"{l[0]:15s} {l[1]:10s} {l[2]:10s}")
    # print("-"*35)

    # for i in range(num_classes):
    #     class_name = classes[i]
    #     f1 = score[0, i]
    #     jac = score[1, i]
    #     dstr = f"{class_name:15s}: {f1:1.5f} - {jac:1.5f}"
    #     print(dstr)
    #     f.write(f"{class_name:15s},{f1:1.5f},{jac:1.5f}\n")
    #
    # print("-"*35)
    # class_mean = np.mean(score, axis=-1)
    # class_name = "Mean"
    #
    # f1 = class_mean[0]
    # jac = class_mean[1]
    #
    # dstr = f"{class_name:15s}: {f1:1.5f} - {jac:1.5f}"
    # print(dstr)
    # f.write(f"{class_name:15s},{f1:1.5f},{jac:1.5f}\n")
    #
    # f.close()








