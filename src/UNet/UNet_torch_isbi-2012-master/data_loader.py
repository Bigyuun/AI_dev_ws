import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}     # dictionary 형태로 export

        if self.transform:
            data = self.transform(data)

        return data
