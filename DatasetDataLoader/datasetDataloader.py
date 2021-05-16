import torch
import torchvision
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset, dataloader
import os

CWD = os.path.dirname(os.path.realpath(__file__))
DLOADER_BATCH_SIZE = 4
EPOCH_NUM = 2


class WineDataset(Dataset):
    def __init__(self):
        super().__init__()
        xy = np.loadtxt(f'{CWD}/wine.csv',
                        delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


# data loading
data = WineDataset()
dloder = DataLoader(dataset=data, batch_size=DLOADER_BATCH_SIZE, shuffle=True)
# training loop
epochs = EPOCH_NUM
total_samples = len(data)
n_iterations = math.ceil(total_samples/DLOADER_BATCH_SIZE)

for epoch in range(epochs):
    for i, (inputs, labels) in enumerate(dloder):
        # forward pass
        # loss
        # backward pass
        # update weights

        if (i+1) % 5 == 0:
            print(
                f"epoch:{epoch+1}/{epochs}, step:{i+1}/{n_iterations}, inputs:{inputs.shape}")
