import os
import numpy as np
import cv2
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.preprocessing.image import ImageDataGenerator

def unet(input_shape, num_classes):
    inputs = Input(input_shape)

    # Encoder 부분
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Decoder 부분
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    up1 = UpSampling2D(size=(2, 2))(conv3)

    merge1 = concatenate([conv2, up1], axis=3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(merge1)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)
    up2 = UpSampling2D(size=(2, 2))(conv4)

    merge2 = concatenate([conv1, up2], axis=3)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(merge2)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)

    outputs = Conv2D(num_classes, 1, activation='softmax')(conv5)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def preprocess_image(image_path, target_size):
    # 이미지를 읽어오기
    image = cv2.imread(image_path)

    # 이미지 크기 조정
    image = cv2.resize(image, target_size)

    # 이미지를 0~1 사이 값으로 정규화
    image = image / 255.0

    return image


def create_data_generator(data_dir, batch_size, target_size):
    data_gen_args = dict(rescale=1. / 255)

    # ImageDataGenerator를 사용하여 이미지와 라벨 데이터 생성
    image_data_gen = ImageDataGenerator(**data_gen_args)
    label_data_gen = ImageDataGenerator(**data_gen_args)

    image_generator = image_data_gen.flow_from_directory(
        data_dir + '/raw',  # RGB 이미지가 들어있는 폴더
        target_size=target_size,
        class_mode=None,  # None으로 설정하여 이미지 데이터만 반환
        batch_size=batch_size,
        seed=42
    )

    label_generator = label_data_gen.flow_from_directory(
        data_dir + '/masks',  # 흑백 라벨 이미지가 들어있는 폴더
        target_size=target_size,
        color_mode='grayscale',  # 흑백 이미지로 설정
        class_mode=None,  # None으로 설정하여 이미지 데이터만 반환
        batch_size=batch_size,
        seed=42
    )

    # 이미지 데이터와 라벨 데이터를 zip으로 묶어서 반환
    data_generator = zip(image_generator, label_generator)

    return data_generator


def train_unet(train_data_dir, batch_size, input_shape, num_classes, epochs):
    # U-Net 모델 생성
    model = unet(input_shape, num_classes)

    # 데이터셋 생성
    train_data_generator = create_data_generator(train_data_dir, batch_size, input_shape[:2])
    steps_per_epoch = len(os.listdir(train_data_dir + '/raw')) // batch_size

    # 모델 컴파일
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # 모델 학습
    model.fit(train_data_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)

    return model


def predict_unet(model, test_image_path):
    # 이미지 불러오기
    test_image = preprocess_image(test_image_path, model.input_shape[:2])
    test_image = np.expand_dims(test_image, axis=0)

    # 예측 수행
    predictions = model.predict(test_image)

    # 예측 결과 반환
    return predictions[0]

if __name__ == "__main__":
    train_data_dir = './dataset/train'  # 학습 데이터 폴더
    test_image_path = './dataset/test/raw/img_op4_1.png'  # 테스트할 이미지 경로

    batch_size = 16
    input_shape = (640, 480, 3)  # 입력 이미지 크기 (RGB 이미지)
    num_classes = 3  # 흑백 라벨 이미지 클래스 개수

    # U-Net 모델 학습
    model = train_unet(train_data_dir, batch_size, input_shape, num_classes, epochs=10)

    # 예측 수행
    predictions = predict_unet(model, test_image_path)
    print(predictions.shape)  # 출력 결과의 크기 확인
