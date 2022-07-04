import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from models import VAE

params = torch.load('save.chkpt')
batch_size  = 4
n_channels  = 3
latent_size = 256

A = VAE(n_channels, latent_size)

A.load_state_dict(params['A'])
epoch = params['epoch']

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

# plot some examples
x,t = next(train_iterator)
print(t)

plt.rcParams['figure.dpi'] = 100
plt.grid(False)
plt.imshow(torchvision.utils.make_grid(x[:8]).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
plt.show()
plt.pause(0.0001)

y, mu, logvar = A(x, device="cpu")
y = A.decode(mu)

plt.rcParams['figure.dpi'] = 100
plt.grid(False)
plt.imshow(torchvision.utils.make_grid(y[:8]).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
plt.show()
plt.pause(0.0001)



