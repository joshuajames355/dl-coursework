import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from models import Generator

fileName = "dcgan_checkpoint.chkpt"
batch_size  = 64


params = torch.load(fileName)
G = Generator()
G.load_state_dict(params["G"])

# helper function to make getting another batch of data easier
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

#filters only elements with the corresponding labels
def cifarFilter(dataset, labels=[2,7]):
    idx = list(filter(lambda x: dataset.targets[x] in labels,range(len(dataset))))
    dataset.targets = [dataset.targets[x] for x in idx]
    dataset.data = [dataset.data[x] for x in idx]  
    return dataset 

dataset = torchvision.datasets.CIFAR10('cifar10', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ]))

dataset = cifarFilter(dataset)  
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

train_iterator = iter(cycle(train_loader))

x,t = next(train_iterator)

noise = torch.randn(batch_size, 128, 1, 1)
y = G(noise)

for i in range(batch_size):
    torchvision.utils.save_image(y[i],"images/"+str(i)+ ".png")

torchvision.utils.save_image(torchvision.utils.make_grid(y),"images/grid.png")

#A simple script to sample a batch of images from the generator.