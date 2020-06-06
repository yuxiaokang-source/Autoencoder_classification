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