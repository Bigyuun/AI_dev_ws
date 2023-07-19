from keras_unet.models import custom_unet

model = custom_unet(
    input_shape=(640, 480, 3),
    use_batch_norm=False,
    num_classes=3,
    filters=64,
    dropout=0.2,
    output_activation='sigmoid')

history = model.fit_generator()