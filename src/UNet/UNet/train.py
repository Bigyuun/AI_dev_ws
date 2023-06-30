
import os
# import Unet_structure
from data_loader import Dataset

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np


class Train():

    def train(self):
        # setup parameters
        lr = 1e-3
        batch_size = 4
        num_epoch = 100

        data_dir = '../docs/isbi-2012-master/data'
        ckpt_dir = '../checkpoint'
        log_dir = '../log'      # tensorboard log file

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # AB = Unet_structure.UNet
        #
        # AB.forward()

        transform = transforms.Compose([Nomalization(mean=0.5, std=0.5),
                                        RandomFlip(),
                                        ToTensor()])
        print(os.getcwd())
        dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'))
        dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)

        # check the data set of first index
        data = dataset_train.__getitem__(0)

        input = data['input']
        label = data['label']

        plt.subplot(121)
        plt.imshow(input)

        plt.subplot(122)
        plt.imshow(label)

        plt.show()

        return


if __name__ == '__main__':
    print('UNet Train Processing...')
    model = Train()
    model.train()

