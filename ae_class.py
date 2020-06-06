#%matplotlib inline
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from torchvision import datasets,transforms
import numpy as np
from PIL import Image
import copy
import time
trainLoader=torch.utils.data.DataLoader(datasets.FashionMNIST('./fmnist/',train=True,
                                                             download=True,
                                                             transform=transforms.Compose(
                                                                 [transforms.ToTensor()])),batch_size
                                       =1024,shuffle=True,num_workers=1)
testLoader=torch.utils.data.DataLoader(datasets.FashionMNIST('./fmnist/',train=False,
                                                             download=True,
                                                             transform=transforms.Compose(
                                                                 [transforms.ToTensor()])),batch_size
                                       =1024,shuffle=True,num_workers=1)


import torch
print(123)
a=torch.rand(3,3)
print(a) 

import os
import struct
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import datasets

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import copy


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 28*28),
            nn.ReLU())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


net = autoencoder()
print(net)
use_gpu = torch.cuda.is_available()
if use_gpu:
    net = net.double().cuda()
else:
    net = net.double()
init_weights = copy.deepcopy(net.encoder[0].weight.data)
