import tensorflow as tf
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from os.path import exists

# data set
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train/255.0 + 0.001
x_test = x_test/255.0 + 0.001

# pre-process data
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

# one-hot encoding
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# num_of_classes = y_test.shape[1]

# model config
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(16, kernel_size=5, padding='SAME', activation='relu', input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(8, kernel_size=5, padding='SAME', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()


# model fitting
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, verbose=1)

model.evaluate(x_test, y_test, verbose=0)

