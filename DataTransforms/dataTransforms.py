import torch
import torchvision
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset, dataloader
import os

from torchvision import transforms

CWD = os.path.dirname(os.path.realpath(__file__))
DLOADER_BATCH_SIZE = 4
EPOCH_NUM = 2


class WineDataset(Dataset):
    def __init__(self, transform):
        super().__init__()
        xy = np.loadtxt(f'{CWD}/wine.csv',
                        delimiter=',', dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class MulTransform:
    def __init__(self, factor) -> None:
        self.factor = factor
    
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets


tensor_dataset = WineDataset(transform=ToTensor())
tensor_data = tensor_dataset[0]
features, label = tensor_data
print(type(features), type(label))

numpy_dataset = WineDataset(transform=None)
numpy_data = numpy_dataset[0]
features, label = numpy_data
print(features)
print(type(features), type(label))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
comp_dataset = WineDataset(transform=composed)
comp_data = comp_dataset[0]
features, label = comp_data
print(features)
print(type(features), type(label))