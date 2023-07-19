
import os

import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets

# setup parameters
lr = 1e-3
batch_size = 4
num_epoch = 200

data_dir = '../docs/isbi-2012-master/data'
ckpt_dir = '../checkpoint'
log_dir = '../log'      # tensorboard log file
result_dir = '../results'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

class UNet(nn.Module):
    """
    Construction of Network
    """
    def __init__(self):
        super(UNet, self).__init__()

        # Convolution & Batch Nomalization & ReLu (CBR)
        def CBR2d(in_channels, out_channels, kernel_size=3,  stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]
            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting Path (Encoder)
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # Expansive Path (Decoder)
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2*512, out_channels=512)      # skip-connection
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)
        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2*256, out_channels=256)      # skip-connection
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)
        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2*128, out_channels=128)      # skip-connection
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)
        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2*64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)


        # segmentation에 필요한 n 개의 클래스에 대한 output
        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    # 각 layer 연결
    def forward(self, x):   # x is input image

        # encoding
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
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


class Dataset(torch.utils.data.Dataset):
    """
    Data Loader
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        list_data = os.listdir(self.data_dir)

        list_label = [f for f in list_data if f.startswith('label')]
        list_input = [f for f in list_data if f.startswith('input')]

        list_label.sort()
        list_input.sort()

        self.list_label = list_label
        self.list_input = list_input

    def __len__(self):
        return len(self.list_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.list_label[index]))
        input = np.load(os.path.join(self.data_dir, self.list_input[index]))

        label = label/255.0
        input = input/255.0

        # 2차원 데이터(e.g. 흑백)인 경우 새로운 채널(axis) 생성
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}     # dictionary 형태로 export

        # transform이 정의되어 있다면 transform을 거친 데이터를 불러옴
        if self.transform:
            data = self.transform(data)

        return data


class ToTensor(object):
    """
    Transform
    - Data Loader에 넣어 사용 (쉽게 수정 가능)
    - Data type conversion : Numpy -> Tensor
    """
    def __call__(self, data):
        label, input = data['label'], data['input']

        # Numpy : (Y,X,CH) / Tensor : (CH,Y,X)
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        # make dictionary
        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    '''
    Z-Score Normalization
    (x-mean) / (standard deviation)
    '''
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    """
    Up-down, Left-Right conversion (random)
    """
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data



###############
# 데이터 셋 확인
###############
#
# transform = transforms.Compose([Nomalization(mean=0.5, std=0.5),
#                                 RandomFlip(),
#                                 ToTensor()])
#
#
# # dataset_db = Dataset(data_dir='../docs/isbi-2012-master/data/train')
# print(os.getcwd())
# dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
# # dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'))
#
# # check the data set of first index
# data = dataset_train.__getitem__(0)
#
# input = data['input']
# label = data['label']
#
# plt.subplot(221)
# plt.imshow(input.squeeze())
#
# plt.subplot(222)
# plt.imshow(label.squeeze())
#
# plt.subplot(223)
# plt.hist(label.flatten(), bins=20)
# plt.title('label')
#
# plt.subplot(224)
# plt.hist(input.flatten(), bins=20)
# plt.title('input')
#
# plt.tight_layout()
# plt.show()
#

###############
## 네트워크 저장하기
###############
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

###############
## 네트워크 불러오기
###############
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch


###############
# 네트워크 학습하기
###############
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

# 네트워크 생성
net = UNet().to(device)     # domain is GPU

# Loss-Function 정의
fn_loss = nn.BCEWithLogitsLoss().to(device)

# Optimizer 설정
optim = torch.optim.Adam(net.parameters(), lr=lr)

# 그 외 부수적인 variables 설정
num_data_train = len(dataset_train)
num_data_val = len(dataset_val)

num_batch_train = np.ceil(num_data_train / batch_size)
num_batch_val = np.ceil(num_data_val / batch_size)

# 그 외 부수적인 functions 설정
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)  # tensor to numpy
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)    # output image를 binary class로 분류

# Tensorboard를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

# 네트워크 학습시키기
st_epoch = 0    # start position 0
# 학습된 모델이 있을 경우 모델 로드하기
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

def UNetTrain():
    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)
            output = net(input)

            # backward pass
            optim.zero_grad()
            loss = fn_loss(output, label)
            loss.backward()
            optim.step()

            # 손실함수 계산
            loss_arr += [loss.item()]
            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            # Tensorboard 저장하기
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            writer_train.add_image('label', label, num_batch_train * (epoch-1) + batch, dataformats='NHWC')
            writer_train.add_image('input', input, num_batch_train * (epoch-1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train * (epoch-1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        with torch.no_grad():   # backward pass 제외
            net.eval()
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)
                output = net(input)

                # 손실함수 계산
                loss = fn_loss(output, label)
                loss_arr += [loss.item()]
                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                # Tensorboard 저장
                label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_class(output))

                writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')


        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        # epoch 50 마다 모델 저장
        if epoch % 50 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()

#
# def UNetTest():
#     transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
#
#     dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
#     loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)
#
#     # 그 외 부수적인 variables 설정하기
#     num_data_test = len(dataset_test)
#     num_batch_test = np.ceil(num_data_test / batch_size)
#
#     # 결과 디렉토리 생성하기
#     result_dir = os.path.join(base_dir, 'result')
#     if not os.path.exists(result_dir):
#         os.makedirs(os.path.join(result_dir, 'png'))
#         os.makedirs(os.path.join(result_dir, 'numpy'))
#
#     net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
#
#     with torch.no_grad():
#         net.eval()
#         loss_arr = []
#
#         for batch, data in enumerate(loader_test, 1):
#             # forward pass
#             label = data['label'].to(device)
#             input = data['input'].to(device)
#
#             output = net(input)
#
#             # 손실함수 계산하기
#             loss = fn_loss(output, label)
#
#             loss_arr += [loss.item()]
#
#             print("TEST: BATCH %04d / %04d | LOSS %.4f" %
#                   (batch, num_batch_test, np.mean(loss_arr)))
#
#             # Tensorboard 저장하기
#             label = fn_tonumpy(label)
#             input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
#             output = fn_tonumpy(fn_class(output))
#
#             # 테스트 결과 저장하기
#             for j in range(label.shape[0]):
#                 id = num_batch_test * (batch - 1) + j
#
#                 # png type
#                 plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
#                 plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
#                 plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')
#
#                 # numpy type
#                 np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
#                 np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
#                 np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())
#
#     print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
#           (batch, num_batch_test, np.mean(loss_arr)))
#
#








if __name__ == '__main__':
    UNetTrain()
    # UNetTest()


































