from keras_unet.models import vanilla_unet

model = vanilla_unet(input_shape=(640, 480, 3))


import numpy as np
import matplotlib.pyplot as plt
import glob
import os, sys
from PIL import Image

orgs = sorted(glob.glob('./dataset/train/raw/*.png'))
masks = sorted(glob.glob('./dataset/train/masks/*.png'))

imgs_list = []
masks_list = []

for image, mask in zip(orgs, masks):
    imgs_list.append(np.array(Image.open(image).resize((640, 480))))
    masks_list.append(np.array(Image.open(mask).resize((640, 480))))

imgs_np = np.asarray(imgs_list)
masks_np = np.asarray(masks_list)

print(imgs_np.shape, masks_np.shape)

from keras_unet.utils import plot_imgs

plot_imgs(org_imgs=imgs_np, mask_imgs=masks_np, nm_img_to_plot=10, figsize=6)


print(imgs_np.max(), masks_np.max())
x = np.asarray(imgs_np, dtype=np.float32)/imgs_np.max()
y = np.asarray(masks_np, dtype=np.float32)/masks_np.max()
print(x.max(), y.max())
print(x.shape, y.shape)

y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)
print(x.shape, y.shape)

x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 3)
print(x.shape, y.shape)


from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.5, random_state=0)

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_val: ", x_val.shape)
print("y_val: ", y_val.shape)

from keras_unet.utils import get_augmented

train_gen = get_augmented(
    x_train, y_train, batch_size=4,
    data_gen_args = dict(
        rotation_range=15.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=50,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant'
    ))

sample_batch = next(train_gen)
xx, yy = sample_batch
print(xx.shape, yy.shape)
from keras_unet.utils import plot_imgs

plot_imgs(org_imgs=xx, mask_imgs=yy, nm_img_to_plot=2, figsize=6)


from keras_unet.models import custom_unet

input_shape = x_train[0].shape

model = custom_unet(
    input_shape,
    use_batch_norm=False,
    num_classes=2,
    filters=64,
    dropout=0.2,
    output_activation='softmax'
)

model.summary()

from keras.optimizers import Adam, SGD
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import jaccard_distance

model.compile(
    optimizer=Adam(),
    # optimizer=SGD(lr=0.001, momentum=0.99),
    loss='categorical_crossentropy',
    #loss=jaccard_distance,
    metrics=[iou, iou_thresholded]
)
history = model.fit_generator(
    train_gen,
    steps_per_epoch=100,
    epochs=10,

    validation_data=(x_val, y_val),
    # callbacks=[callback_checkpoint]
)

from keras_unet.utils import plot_segm_history

plot_segm_history(history)


# model.load_weights(model_filename)
y_pred = model.predict(x_val)

from keras_unet.utils import plot_imgs

plot_imgs(org_imgs=x_val, mask_imgs=y_val, pred_imgs=y_pred, nm_img_to_plot=9)