import torch
import torch.nn as nn

class UNet(nn.Module):
    """
    Construction of Network
    """
    def __init__(self):
        super(UNet, self).__init__()

        # Convolution & Batch Nomalization & ReLu (CBR)
        def CBR2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(input_channels=input_channels,
                                 output_channels=output_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=output_channels)]
            layers += [nn.ReLU()]
            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting Path (Encoder)
        self.enc1_1 = CBR2d(input_channels=1, output_channels=64)
        self.enc1_2 = CBR2d(input_channels=64, output_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(input_channels=64, output_channels=128)
        self.enc2_2 = CBR2d(input_channels=128, output_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(input_channels=128, output_channels=256)
        self.enc3_2 = CBR2d(input_channels=256, output_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(input_channels=256, output_channels=512)
        self.enc4_2 = CBR2d(input_channels=512, output_channels=512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(input_channels=512, output_channels=1024)

        # Expansive Path (Decoder)
        self.dec5_1 = CBR2d(input_channels=1024, output_channels=512)
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(input_channels=2*512, output_channels=512)      # skip-connection
        self.dec4_1 = CBR2d(input_channels=512, output_channels=256)
        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(input_channels=2*256, output_channels=256)      # skip-connection
        self.dec3_1 = CBR2d(input_channels=256, output_channels=128)
        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(input_channels=2*128, output_channels=128)      # skip-connection
        self.dec2_1 = CBR2d(input_channels=128, output_channels=64)
        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(input_channels=2*64, output_channels=64)
        self.dec1_1 = CBR2d(input_channels=64, output_channels=64)


        # segmentation에 필요한 n 개의 클래스에 대한 output
        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    # 각 layer 연결
    def forward(self, x):   # x is input image

        # encoding
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc1_2(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        # decoding
        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        # Generally, 채널방향으로 연결하는 함수를 concatenation(cat)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)      # dim=[0:batch, 1:channel, 2:height, 3:width]
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x


































