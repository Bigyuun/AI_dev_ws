import platform
import os
if 'Linux' in platform.system():
    print(platform.system())
    os.system("echo 0 | sudo tee -a /sys/bus/pci/devices/0000\:01\:00.0/numa_node")

from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from keras.layers import Dropout
from keras.models import Model
from keras.applications import vgg16
from keras.applications import mobilenet
from keras.applications import imagenet_utils


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, activation='relu', padding="same")(input)
    x = BatchNormalization()(x) # train 속도를 올려줄거라고 생각함..
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, activation='relu', padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

# def encoder_block(input, num_filters):
#     x = conv_block(input, num_filters)
#     p = MaxPool2D((2, 2))(x)    # e.g. 128x128 -> 64x64
#     return x, p

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_vgg16_unet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    base_model = vgg16.VGG16(weights='imagenet',
                             include_top=False,
                             input_tensor=inputs)

    # Encoder
    s1 = base_model.get_layer('block1_conv2').output    # 64
    s2 = base_model.get_layer('block2_conv2').output    # 128
    s3 = base_model.get_layer('block3_conv3').output    # 256
    s4 = base_model.get_layer('block4_conv3').output    # 512

    # Bridge """
    b1 = base_model.get_layer('block5_conv3').output    # 32

    # Decoder
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs, name="U-Net-VGG16")
    model.summary()

    return model

if __name__ == "__main__":
    input_shape = (480, 640, 3)
    model = build_vgg16_unet(input_shape, 3)
    # model.summary()