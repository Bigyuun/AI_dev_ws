from UNet_keras import *

if __name__ == "__main__":
    # GPU

    unet(n_classes=3)
    ModelCheckpoint