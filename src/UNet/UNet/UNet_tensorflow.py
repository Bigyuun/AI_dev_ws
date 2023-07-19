from keras_unet.models import custom_unet

model = custom_unet(
    input_shape=(512,512,3),
    use_batch_norm=False,
    num_classes=1,
    filter=64,
    dropout=0.3,
    output_activation='relu'

)